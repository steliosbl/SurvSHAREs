import warnings
import wandb

from gplearn.gplearn.genetic import BaseSymbolic, SymbolicRegressor
from survshares.program import SurvProgram

class BaseSurvSymbolic(BaseSymbolic):
    """
    Extends BaseSymbolic with more advanced logging.
    """

    def __init__(self,
                 *,
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
                 random_state=None,
                 optim_dict=None,
                 categorical_variables=[],
                 logging_console=True,
                 logging_wandb=False,
                 wandb_run=None,
                 program_class=SurvProgram):
        
        super().__init__(
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
            verbose=True, # Always verbose so we can handle logging here
            random_state=random_state,
            optim_dict=optim_dict,
            categorical_variables=categorical_variables,
            program_class=program_class)

        self.logging_console, self.logging_wandb = logging_console, logging_wandb
        self.wandb_run = wandb_run

        if self.logging_wandb and wandb_run is None:
            warnings.warn("No wandb_run provided, setting 'logging_wandb' to False")
            self.logging_wandb = False

    def _verbose_reporter(self, run_details=None):
        if self.logging_console:
            super()._verbose_reporter(run_details)

        if (
            self.logging_wandb
            and self.wandb_run is not None
            and run_details is not None
        ):
            self.wandb_run.log({
                key: value[-1] for key, value in run_details.items()
            }, step=run_details.get('generation', [None])[-1])
