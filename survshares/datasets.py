import numpy as np
import pandas as pd
import warnings
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from SurvSet.data import SurvLoader

from pycox.datasets import metabric


def ohe_matrices(X, categorical_values, device):
    X = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    return {
        k: torch.from_numpy(
            OneHotEncoder(sparse_output=False, categories=[v]).fit_transform(
                X[:, [k]].astype("int")
            )
        )
        .float()
        .to(device)
        for k, v in categorical_values.items()
    }


class SurvivalDataset:
    def __init__(self, name):
        self.name = name
        self.X, self.T, self.E = None, None, None
        self._categorical_dict, self._numerical_ranges = None, None
        self._ordinal_encoder = OrdinalEncoder()

    def load_df(self):
        return SurvLoader().load_dataset(ds_name=self.name)["df"]

    def load(self, normalise=False, ordinal_encode=True):
        X = self.load_df()
        if "pid" in X:
            X = X.drop("pid", axis=1)

        # Ignore time-varying features for now
        if "time2" in X:
            X = X[X["time"] == 0]
            X["time"] = X.pop("time2")

        # Extract X, T, E
        self.T = X.pop("time").values
        self.E = X.pop("event").values
        self.features = X.columns.tolist()

        if normalise or ordinal_encode:
            transformers = []

            if normalise:
                transformers.append(
                    ("num", StandardScaler(), make_column_selector(pattern="^num\\_"))
                )
            if ordinal_encode:
                transformers.append(
                    (
                        "fac",
                        self._ordinal_encoder,
                        make_column_selector(pattern="^fac\\_"),
                    )
                )

            ct = ColumnTransformer(
                transformers, remainder="passthrough", verbose_feature_names_out=False
            ).set_output(transform="pandas")
            X = ct.fit_transform(X)[self.features]

        self.X = X.values
        return self.X, self.T, self.E

    def summary(self):
        df = self.load_df()

        summarise = (
            lambda s: s[
                (s >= (q1 := s.quantile(0.25)) - 1.5 * (iqr := s.quantile(0.75) - q1))
                & (s <= q1 + 1.5 * iqr)
            ]
            .agg(["min", "max"])
            .to_dict()
        )

        return df.select_dtypes(include="number").apply(summarise).to_dict()

    def _check_loaded(func):
        def wrapper(self, *args, **kwargs):
            if self.X is None:
                warnings.warn("X is not loaded. Loading now, with normalise=False.")
                self.load(normalise=False)
            return func(self, *args, **kwargs)

        return wrapper

    @property
    @_check_loaded
    def categorical_features(self):
        return [f for f in self.features if f.startswith("fac_")]

    @property
    @_check_loaded
    def categorical_values(self):
        if self._categorical_dict is None:
            self._categorical_dict = {
                self.features.index(col): tuple(
                    np.unique(  # Must be a tuple for get_argument_ranges_for_shape_functions
                        self.X[:, self.features.index(col)]
                    )
                    .astype(int)
                    .tolist()
                )
                for col in self.categorical_features
            }
        return self._categorical_dict

    @property
    @_check_loaded
    def numerical_ranges(self):
        if self._numerical_ranges is None:
            self._numerical_ranges = {
                i: (np.min(self.X[:, i]), np.max(self.X[:, i]))
                for i in range(self.X.shape[1])
                if not self.features[i] in self.categorical_features
            }

        return self._numerical_ranges

    @_check_loaded
    def split(self, train_size=0.8, random_state=None):
        X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
            self.X, self.T, self.E, train_size=train_size, random_state=random_state
        )

        # We have to ensure that the minimum and maximum event time exist in the training set
        if self.T.max() in T_test and not self.T.max() in T_train:
            idx_from = np.random.choice(np.where(T_test == self.T.max())[0])
            idx_to = np.random.randint(len(T_train))

            X_train[idx_to], X_test[idx_from] = (
                X_test[idx_from].copy(),
                X_train[idx_to].copy(),
            )
            T_train[idx_to], T_test[idx_from] = (
                T_test[idx_from].copy(),
                T_train[idx_to].copy(),
            )
            E_train[idx_to], E_test[idx_from] = (
                E_test[idx_from].copy(),
                E_train[idx_to].copy(),
            )

        return X_train, X_test, T_train, T_test, E_train, E_test

    def ohe_matrices(self, X=None, device="cpu"):
        if X is None:
            X = self.X
        return ohe_matrices(X, self.categorical_values, device)


class Metabric(SurvivalDataset):
    def __init__(self):
        self.name = "METABRIC"
        self.X, self.T, self.E = None, None, None
        self._categorical_dict, self._numerical_ranges = None, None
        self._ordinal_encoder = OrdinalEncoder()

    def load_df(self):
        return metabric.read_df().rename(
            columns={"duration": "time"}
            | {col: f"fac_{col}" for col in ["x4", "x5", "x6", "x7"]}
            | {col: f"num_{col}" for col in ["x1", "x2", "x3", "x8"]}
        )