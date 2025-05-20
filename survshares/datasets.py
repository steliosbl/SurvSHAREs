import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from lifelines.datasets import load_rossi, load_gbsg2
from pycox.datasets import metabric


class Dataset:
    def __init__(self):
        pass

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
    
    def numerical_ranges(self, X):
        return {i: (np.min(X[:, i]), np.max(X[:,i])) for i in range(X.shape[1]) if not i in self.categorical_dict.keys()}



class Rossi(Dataset):
    features = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
    event = "arrest"
    time = "week"
    categorical_columns = ["fin", "race", "wexp", "mar", "paro"]
    categorical_dict = {0: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    tmax = 52

    def load_df(self):
        return load_rossi()

    def load(self, normalise=False):
        X = self.load_df()
        T = X.pop("week").values
        E = X.pop("arrest").values

        if normalise:
            X = StandardScaler().fit_transform(X)
        else:
            X = X.values

        return X, T, E


class Metabric(Dataset):
    features = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
    event = "event"
    time = "duration"
    categorical_columns = ["x4", "x5", "x6", "x7"]
    categorical_dict = {4: 2, 5: 2, 6: 2, 7: 2}
    tmax = 350  # Not entirely accurate

    def load_df(self):
        return metabric.read_df()

    def load(self, normalise=False):
        X = self.load_df()
        T = X.pop("duration").values
        E = X.pop("event").values

        if normalise:
            X = StandardScaler().fit_transform(X)
        else:
            X = X.values

        return X, T, E


class GBSG2(Dataset):
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
    event = "event"
    time = "time"
    categorical_columns = ["horTh", "tgrade", "postmen"]
    categorical_dict = {0: 2, 3: 3, 7: 2}
    tmax = 2296  # Not entirely accurate

    def load_df(self):
        return load_gbsg2()

    def load(self, normalise=False):
        X = self.load_df()
        T = X.pop("time").values
        E = 1 - X.pop("cens").values

        X["horTh"] = X["horTh"].map({"yes": 1, "no": 0})
        X["postmen"] = X.pop("menostat").map({"Pre": 0, "Post": 1})
        X["tgrade"] = X["tgrade"].map({"I": 1, "II": 2, "III": 3})

        if normalise:
            X = StandardScaler().fit_transform(X)
        else:
            X = X.values

        return X, T, E
