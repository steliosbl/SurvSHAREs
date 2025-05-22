import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import argparse
import os
from datetime import datetime

# Data
from survshares.datasets import Rossi, Metabric, GBSG2
from sklearn.model_selection import train_test_split

# Fitness
from lifelines.utils import concordance_index
from gplearn.gplearn.fitness import make_fitness
from pycox.evaluation.metrics import partial_log_likelihood_ph
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score

# Wrapper
from sklearn.base import BaseEstimator, RegressorMixin
from pycox.models.cox import _CoxPHBase
import torchtuples as tt

# SHAREs
from gplearn.gplearn.genetic import SymbolicRegressor
from gplearn.gplearn.model import ShapeNN
from experiments.utils import (
    load_share_from_checkpoint,
    get_n_shapes,
    get_n_variables,
)

##### Data #####
DATASETS = dict(
    rossi=Rossi(),
    metabric=Metabric(),
    gbsg2=GBSG2(),
)


##### Fitness Functions #####
def metric_c_index(y_true, y_pred, sample_weight):
    """
    Protected concordance score metric for gplearn. Greater is better.
    """
    # y_true is the event time, y_pred is the predicted risk
    # sample_weight is the event indicator
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            return concordance_index(y_true, np.exp(y_pred), sample_weight)
        except ZeroDivisionError:  # In case of no unambigous pairs
            return 0.5
        except RuntimeWarning:  # In case of invalid log or exp overflow
            return 0.5


def metric_partial_likelihood(y_true, y_pred, sample_weight):
    """
    Cox partial likelihood metric for gplearn. Less is better.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        # We want to minimise the negative partial log likelihood
        # y_pred should be the LOG partial hazards - i.e. we admit negative vaues
        try:
            pll = partial_log_likelihood_ph(y_pred, y_true, sample_weight, mean=True)
            return -pll  # Flip the sign to make it a minimization function
        except RuntimeWarning:  # In case of invalid log or exp overflow
            return np.inf


def metric_integrated_brier(surv_pred, E_train, T_train, E_test=None, T_test=None):
    """
    Integrated Brier score for pycox-type models
    """
    E_test = E_test if E_test is not None else E_train
    T_test = T_test if T_test is not None else T_train

    y_train, y_test = Surv.from_arrays(E_train, T_train), Surv.from_arrays(
        E_test, T_test
    )

    times = surv_pred.index.values[1:-1]
    times = times[(times > T_test.min()) & (times < T_test.max())]
    surv_pred = surv_pred.loc[times, :].T

    return integrated_brier_score(y_train, y_test, surv_pred, times)


def fitness_c_shrink(y_true, y_pred, sample_weight):
    """
    Concordance index with shrinkage penalty for gplearn. Greater is better.
    """
    return metric_c_index(y_true, y_pred, sample_weight) - 0.05 * np.abs(y_pred).mean()


def fitness_pll_shrink(y_true, y_pred, sample_weight):
    """
    Partial log-likelihood with shrinkage penalty for gplearn. Smaller is better.
    """
    pll = metric_partial_likelihood(y_true, y_pred, sample_weight)
    return pll + 0.05 * np.abs(y_pred).mean()


FITNESS = dict(
    c_index=make_fitness(function=metric_c_index, greater_is_better=True),
    c_shrink=make_fitness(function=fitness_c_shrink, greater_is_better=True),
    pll_shrink=make_fitness(function=fitness_pll_shrink, greater_is_better=False),
)


##### Wrapper #####
class SymRegPH(BaseEstimator, RegressorMixin, _CoxPHBase):
    """
    Wrapper for gplearn's SymbolicRegressor to use with pycox supporting functions
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X, *args, **kwargs):
        if isinstance(X, tt.TupleTree):
            X = X[0]
        return self.model.predict(X)


##### SHAREs #####
def init_share_regressor(metric, device, checkpoint_dir, categorical_variables={}):
    gp_config = {
        "population_size": 69,
        "generations": 10,
        "tournament_size": 10,
        "function_set": ("add", "mul", "div", "shape"),
        "verbose": True,
        "random_state": 42,
        "const_range": None,
        "n_jobs": 1,
        "p_crossover": 0.4,
        "p_subtree_mutation": 0.2,
        "p_point_mutation": 0.2,
        "p_hoist_mutation": 0.05,
        "p_point_replace": 0.2,
        "parsimony_coefficient": 0.0,
        "metric": metric,
        "parsimony_coefficient": 0.0,
        "optim_dict": {
            "alg": "adam",
            "lr": 1e-2,  # tuned automatically
            "max_n_epochs": 1000,
            "tol": 1e-3,
            "task": "regression",
            "device": device,
            "batch_size": 1000,
            "shape_class": ShapeNN,
            "constructor_dict": {
                "n_hidden_layers": 5,
                "width": 10,
                "activation_name": "ELU",
            },
            "num_workers_dataloader": 0,
            "seed": 42,
            "checkpoint_folder": checkpoint_dir,
            "keep_models": True,
        },
    }

    return SymbolicRegressor(**gp_config, categorical_variables=categorical_variables)


def test_share_ph(
    dataset_name, metric_name, device, checkpoint_dir, categorical_variables=False
):
    # Prepare dataset
    dataset = DATASETS[dataset_name]
    X, T, E = dataset.load(normalise=False)
    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        X, T, E, test_size=0.2, random_state=42
    )
    feature_names = dataset.features
    categoricals = dataset.categorical_dict if categorical_variables else {}

    # Initialise model
    fitness = FITNESS[metric_name]
    model = init_share_regressor(
        metric=fitness,
        device=device,
        checkpoint_dir=checkpoint_dir,
        categorical_variables=categoricals,
    )  # .set_params(feature_names=feature_names) # causes issues with reloading

    # Fit model
    print("Starting model fit")
    model.fit(X_train, T_train, sample_weight=E_train)
    print("Finished model fit")
    timestamp = model.timestamp
    # timestamp = max(
    #     os.listdir(checkpoint_dir),
    #     key=lambda x: datetime.strptime(x, "%Y-%m-%dT%H.%M.%S"),
    # )

    # Load results dataframe
    results_df = pd.read_csv(checkpoint_dir / timestamp / "dictionary.csv")

    print("Adding validation results")
    n_shapes, n_variables, loss_train, loss_test, c_test, brier_test = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for idx, id, eq, _, _ in results_df.itertuples():
        print(eq)
        n_shapes.append(get_n_shapes(eq))
        n_variables.append(get_n_variables(eq))

        esr = load_share_from_checkpoint(
            timestamp,
            eq,
            checkpoint_dir=checkpoint_dir,
            task="regression",
            n_features=len(feature_names),
            equation_id=id,
        )

        esr_wrap = SymRegPH(model=esr)
        h_train, h_test = esr_wrap.predict(X_train), esr_wrap.predict(X_test)
        h0 = esr_wrap.compute_baseline_hazards(X_train, (T_train, E_train))
        surv_test = esr_wrap.predict_surv_df(X_test)

        loss_train.append(fitness(T_train, h_train, E_train))
        loss_test.append(fitness(T_test, h_test, E_test))
        c_test.append(metric_c_index(T_test, h_test, E_test))

        brier = np.nan
        if not surv_test.isna().any().any():  # Junk models will yield NaN values
            try:
                brier = metric_integrated_brier(
                    surv_test, E_train, T_train, E_test, T_test
                )
            except Exception:
                pass
        brier_test.append(brier)

    results_df["n_shapes"] = n_shapes
    results_df["n_variables"] = n_variables
    results_df["loss_train"] = loss_train
    results_df["loss_test"] = loss_test
    results_df["c_test"] = c_test
    results_df["brier_test"] = brier_test

    # ad-hoc correction for flipped sign in y_pred
    if results_df["c_test"].mean() < 0.5:
        results_df["c_test"] = 1 - results_df["c_test"]

    out_path = checkpoint_dir / timestamp / "output.csv"
    print(f"Saving results to {out_path}")
    results_df.to_csv(out_path, index=False)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testing SHAREs for proportional hazards survival"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="rossi",
        choices=["rossi", "metabric", "gbsg2"],
        help="Dataset to use for test",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="c_index",
        choices=["c_index", "c_shrink", "pll_shrink"],
        help="Metric to use for fitness function",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for torch",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="data/checkpoints/rossi_c",
        help="Path to save working models",
    )
    parser.add_argument(
        "--categorical",
        action="store_true",
        help="Use categorical variables in SHARE",
    )

    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoints)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(args)

    test_share_ph(
        args.dataset, args.metric, args.device, checkpoint_dir, args.categorical
    )
