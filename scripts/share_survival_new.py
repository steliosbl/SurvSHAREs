import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import sys 
import wandb
import torch
sys.path.append('.')

from survshares.datasets import Rossi, Metabric, GBSG2
from survshares.metrics import negative_pll
from survshares.hazard_model import HazardModel
from survshares.loss import approximateNpllLoss, npllLoss
from survshares.symbolic_regressor import SurvSymbolicRegressor

from gplearn.gplearn.model import ShapeNN
from gplearn.gplearn.fitness import make_fitness
from experiments.utils import (
    load_share_from_checkpoint,
    get_n_shapes,
    get_n_variables,
)

##### Data #####
DATASETS = dict(
    rossi=Rossi(),
    metabric=Metabric(),
    gbsg2=GBSG2()
)

##### Fitness #####
def fitness_npll_shrink(y_true, y_pred, sample_weight):
    """
    Partial log-likelihood with shrink penalty for gplearn. Smaller is better.
    """
    pll = negative_pll(y_true, y_pred, sample_weight)
    return pll + 0.05 * np.abs(y_pred).mean()


FITNESS = dict(
    npll_shrink = make_fitness(function=fitness_npll_shrink, greater_is_better=False)
)

##### Loss #####
LOSS = dict(
    npll_approximate=approximateNpllLoss(),
    npll_breslow=npllLoss(ties_method="breslow"),
    npll_efron=npllLoss(ties_method="efron"),
)

##### SHAREs #####
def init_share_regressor(population_size, generations, metric, loss_fn, device, checkpoint_dir, wandb_run, categorical_variables={}):
    gp_config = {
        "logging_console": True,
        "logging_wandb": True, 
        "wandb_run": wandb_run,
        "population_size": population_size,
        "generations": generations,
        "tournament_size": 10,
        "function_set": ("add", "mul", "div", "shape"),
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
           # "lr": 1e-2,  # tuned automatically
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
            "loss_fn": loss_fn, 
        },
    }

    return SurvSymbolicRegressor(**gp_config, categorical_variables=categorical_variables)


def test_share_ph(
    dataset_name, metric_name, loss_name, population_size, generations, device, checkpoint_dir, wandb_run
): 
    dataset = DATASETS[dataset_name]
    dataset.load()
    X_train, X_test, T_train, T_test, E_train, E_test = dataset.split()
    feature_names = dataset.features 
    categorical_variables = dataset.categorical_values

    fitness = FITNESS[metric_name]
    loss_fn = LOSS[loss_name]

    model = init_share_regressor(
        population_size=population_size,
        generations=generations,
        metric=fitness,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
        wandb_run=wandb_run,
        categorical_variables=categorical_variables
    )

    print("Starting model fit")
    model.fit(torch.Tensor(X_train), torch.Tensor(T_train), sample_weight=torch.Tensor(E_train))
    print("Finished model fit")
    timestamp = model.timestamp

    results_df = pd.read_csv(checkpoint_dir / timestamp / "dictionary.csv")

    validation_results = []
    for idx, id, eq, _, _ in results_df.itertuples():

        esr = load_share_from_checkpoint(
            timestamp,
            eq,
            checkpoint_dir=checkpoint_dir,
            task="survival",
            n_features=len(feature_names),
            equation_id=id,
            loss_fn=loss_fn,
            categorical_variables_dict=categorical_variables,
        )

        esr_wrap = HazardModel(
            esr,
            categorical_variables=categorical_variables,
        ).prepare_estimands(X_train, T_train, E_train)

        scores = esr_wrap.score(X_test, T_test, E_test, extended=False)

        scores.update(dict(
            id=id,
            n_shapes=get_n_shapes(eq),
            n_variables=get_n_variables(eq),
        ))

        validation_results.append(scores)
        wandb.log(scores)

    score_df = pd.DataFrame(validation_results).set_index('id')
    results_df = results_df.set_index('id').join(score_df, how='left')
    results_df.to_csv(checkpoint_dir / timestamp / "results.csv")

    print('Done')


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
        default="npll_shrink",
        choices=["npll_shrink"],
        help="Metric to use for fitness function",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="npll_approximate",
        choices=["npll_approximate", "npll_breslow", "npll_efron"],
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
        "--population_size",
        type=int,
        default=69,
        help="Population size for the symbolic regressor",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations for the symbolic regressor",
    )

    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoints)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(args)

    wandb_run = wandb.init(
        project="share_survival",
        name=f"{args.dataset}_{args.metric}_{args.loss}",
        config={
            "dataset": args.dataset,
            "metric": args.metric,
            "loss": args.loss,
            "device": args.device,
            "checkpoints": str(checkpoint_dir),
            "population_size": args.population_size,
            "generations": args.generations,
        }
    )

    test_share_ph(
        args.dataset, args.metric, args.loss, args.population_size, args.generations, args.device, checkpoint_dir, wandb_run
    )

    wandb.finish()


