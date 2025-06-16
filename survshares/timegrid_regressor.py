import numpy as np
import torch 

from sksurv.util import Surv
from sksurv.nonparametric import CensoringDistributionEstimator

import gplearn_clean.gplearn.genetic 
from gplearn_clean.gplearn.utils import check_random_state
from gplearn_clean.gplearn.genetic import SymbolicRegressor
from gplearn_clean.gplearn._program import _Program 



# Monkey patching gplearn.genetic._parallel_evolve to allow custom program classes
def _parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params):
    """Private function used to build a batch of programs within a job."""
    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']
    program_class = params['program_class'] # Relying on BaseEstimator.get_params() to pick this up 
    tgrid = params['tgrid']
    ghat = params['ghat']

    max_samples = int(max_samples * n_samples)

    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    # Build programs
    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()

            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program,
                                                             random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = program_class(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program,
                           tgrid=tgrid,
                           ghat=ghat)

        program.parents = genome

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight.copy()
        oob_sample_weight = curr_sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples,
                                                       max_samples,
                                                       random_state)

        curr_sample_weight[not_indices] = 0
        oob_sample_weight[indices] = 0

        program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)
        if max_samples < n_samples:
            # Calculate OOB fitness
            program.oob_fitness_ = program.raw_fitness(X, y, oob_sample_weight)

        programs.append(program)

    return programs

gplearn_clean.gplearn.genetic._parallel_evolve = _parallel_evolve


class _TimegridProgram(_Program): 
    def __init__(self,
                tgrid,
                ghat,
                function_set,
                arities,
                init_depth,
                init_method,
                n_features,
                const_range,
                metric,
                p_point_replace,
                parsimony_coefficient,
                random_state,
                transformer=None,
                feature_names=None,
                program=None):
        super().__init__(function_set, arities, init_depth, init_method, n_features,
                         const_range, metric, p_point_replace, parsimony_coefficient,
                         random_state, transformer, feature_names, program)
        self._feature_names = feature_names
        self._n_features_X = n_features
        self.tgrid = tgrid
        self.ghat = ghat # ghat is \hat{g}(t_max) - the non-parametric censoring survival probability at the end of the time scale
        if self.ghat < 0 or self.ghat >= 1:
            raise ValueError("ghat must be in the range [0, 1).")

    @property 
    def feature_names(self): 
        if self._feature_names is not None: 
            return list(self._feature_names) + ['time']
        else:
            return [f'X{i}' for i in range(self._n_features_X)] + ['time']
        
    @feature_names.setter
    def feature_names(self, value):
        self._feature_names = value

    @property 
    def n_features(self):
        return self._n_features_X + 1  # +1 for the time feature
    
    @n_features.setter
    def n_features(self, value):
        self._n_features_X = value 

    def execute(self, X, t = None):
        """Every row in X is evaluated at every time point in t. Returns a prediction matrix (n_samples, n_times)"""

        if t is None: # By default use the time grid
            t = self.tgrid
        if isinstance(t, (int, float)): # Handle single time points
            t = np.array([t])
        elif isinstance(t, list):
            t = np.array(t)
        elif not isinstance(t, np.ndarray):
            raise ValueError("t must be an int, float, list or numpy array")
        
        X_expanded = np.repeat(X, len(t), axis=0) # Repeat each row of X for each time point in t
        t_expanded = np.tile(t, X.shape[0]).reshape(-1, 1) # Corresponding time points for each row of X_expanded
        X_expanded = np.hstack((X_expanded, t_expanded)) # Combine X and t into a single matrix

        return super().execute(X_expanded).reshape(X.shape[0], len(t))
    

    def raw_fitness(self, X, T, E): 
        # EXPECTING T as IDX_DURATIONS - NOT AS DURATION TIMES
        # I.e. we have already discretised the time scale in the input data. Otherwise uncomment the transform below 
        from pycox.preprocessing.label_transforms import LabTransDiscreteTime
        from pycox.models.data import pair_rank_mat
        from pycox.models.loss import nll_pmf, rank_loss_deephit_single
        import torch

        # T, E = LabTransDiscreteTime(self.tgrid).fit_transform()

        # Alpha is weighting between likelihood and rank loss (so not like in paper):
        # loss = alpha * nll + (1 - alpha) rank_loss(sigma)
        alpha = 0.2 # Parameter that controls the linear combination between the nll and ranking loss
        sigma = 0.1 # Parameter used by the ranking loss 
        lamb = 1.0 # Parameter that weighs the ICP loss
        reduction = 'mean'

        # Required:
        # 1. phi: the predicted survival function at each time point in a matrix (n_samples, n_times)
        # 2. idx_durations: the time indices in the grid of the observed event/censoring times
        # 3. events: float indicator of event or censoring (1 is event)
        # 4. rank_mat: Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}. 
        #       So it takes value 1 if we observe that i has an event before j and zero otherwise.

        phi = self.execute(X, self.tgrid)
        rank_mat = pair_rank_mat(T, E) # Inputs must be numpy arrays 

        phi, T, E, rank_mat = torch.Tensor(phi), torch.tensor(T), torch.Tensor(E), torch.Tensor(rank_mat)

        # 3 parts: nll_pmf, rank_loss, and rescaling loss
        # Inputs must be tensors
        nll = nll_pmf(phi, T, E, reduction)

        # TODO: This internally calls a softmax as well. Should we include ICPW? 
        rank_loss = rank_loss_deephit_single(phi, T, E, rank_mat, sigma, reduction) 
        
        # ICPW loss, penalising difference of the CDF at t_max from 1 - Pr(censored after t_max)
        icp_loss = phi.sum(dim=1).sub(1-self.ghat[0]).abs().mean() 

        return alpha * nll + (1. - alpha) * rank_loss #+ lamb * icp_loss
    
class TimegridRegressor(SymbolicRegressor):
    def __init__(self,
                 *,
                 tgrid=None,
                 ghat=None,
                 program_class=_TimegridProgram,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicRegressor, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)
        
        self.program_class = program_class # Relying on BaseEstimator.get_params() to pass this to parallel_evolve
        self.tgrid = tgrid 
        self.ghat = ghat  # This will be set during fit

    def fit(self, X, T, E): 
        if self.ghat is None:
            censor_est = CensoringDistributionEstimator().fit(
                Surv.from_arrays(E, T)
            ) # KM estimator for the probability Pr(censored after t) of being censored after the given time point
            self.ghat = censor_est.predict_proba(np.unique(T)[-1:])  # Get Pr(censored after t_max) of being censored after the last time point in the grid
            return super().fit(X, T, E)

    def predict_proba(self, X):
        y_pred = super().predict(X) 
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.Tensor(y_pred)  # Convert to tensor if needed
        
        # y_pred_adjusted = torch.nn.LogSoftmax(dim=1)(y_pred)  # Apply log softmax to the predictions
        # Adjust so the CDF at t_max (the end of the time scale) is 1 - Pr(censored after t_max)
        # y_pred_adjusted = y_pred_adjusted + np.log(1-self.ghat) 

        y_pred_adjusted = torch.nn.LogSigmoid()(y_pred)

        return np.exp(y_pred_adjusted)  # Return as numpy array