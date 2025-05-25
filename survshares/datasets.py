import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from lifelines.datasets import load_rossi, load_gbsg2
from pycox.datasets import metabric


class SurvivalDataset:
    def __init__(self):
        self.X, self.T, self.E = None, None, None
        self._categorical_dict, self._numerical_ranges = None, None

    def load_df(self):
        raise NotImplementedError()

    def load(self, normalise=False):
        raise NotImplementedError()

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
    def categorical_values(self):
        if self._categorical_dict is None:
            self._categorical_dict = {
                self.features.index(col): np.unique(
                    self.X[:, self.features.index(col)]
                ).tolist()
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

    def _normalise_numericals(self, X: pd.DataFrame):
        columns_original = X.columns
        numerical_cols = [
            col for col in self.features if col not in self.categorical_features
        ]
        ct = ColumnTransformer(
            transformers=[("num", StandardScaler(), numerical_cols)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        return ct.fit_transform(X)[columns_original].values

    @_check_loaded
    def split(self, train_size=0.8, random_state=None):
        return train_test_split(
            self.X, self.T, self.E, train_size=train_size, random_state=random_state
        )


class Rossi(SurvivalDataset):
    features = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
    categorical_features = ["fin", "race", "wexp", "mar", "paro"]
    event = "arrest"
    time = "week"
    # categorical_dict = {0: 2, 2: 2, 3: 2, 4: 2, 5: 2}

    def load_df(self):
        return load_rossi()

    def load(self, normalise=False):
        X = self.load_df()
        self.T = X.pop("week").values
        self.E = X.pop("arrest").values

        if normalise:
            self.X = self._normalise_numericals(X)
        else:
            self.X = X.values

        return self.X, self.T, self.E


class Metabric(SurvivalDataset):
    features = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
    categorical_features = ["x4", "x5", "x6", "x7"]
    event = "event"
    time = "duration"
    # categorical_dict = {4: 2, 5: 2, 6: 2, 7: 2}

    def load_df(self):
        return metabric.read_df()

    def load(self, normalise=False):
        X = self.load_df()
        self.T = X.pop("duration").values
        self.E = X.pop("event").values

        if normalise:
            self.X = self._normalise_numericals(X)
        else:
            self.X = X.values

        return self.X, self.T, self.E


class GBSG2(SurvivalDataset):
    features = [
        "horTh",
        "age",
        "tsize",
        "tgrade",
        "pnodes",
        "progrec",
        "estrec",
        "postmen",
    ]
    categorical_features = ["horTh", "tgrade", "postmen"]
    event = "event"
    time = "time"
    # categorical_dict = {0: 2, 3: 3, 7: 2}

    def load_df(self):
        return load_gbsg2()

    def load(self, normalise=False):
        X = self.load_df()
        self.T = X.pop("time").values
        self.E = 1 - X.pop("cens").values

        X["horTh"] = X["horTh"].map({"yes": 1, "no": 0})
        X["postmen"] = X.pop("menostat").map({"Pre": 0, "Post": 1})
        X["tgrade"] = X["tgrade"].map({"I": 1, "II": 2, "III": 3})

        if normalise:
            self.X = self._normalise_numericals(X)
        else:
            self.X = X.values

        return self.X, self.T, self.E
