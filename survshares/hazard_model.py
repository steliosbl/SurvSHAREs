import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchtuples as tt

from sklearn.base import BaseEstimator, RegressorMixin
from pycox.models.cox import _CoxPHBase
from gplearn.gplearn._program import _Program as _ShareProgram 
from sksurv.util import Surv
from sksurv.util import check_y_survival

from survshares.datasets import ohe_matrices
from survshares.plot import calibration_plot_binned, calibration_plot_smoothed

from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from torchsurv.loss.cox import neg_partial_log_likelihood


class HazardModel(BaseEstimator, RegressorMixin, _CoxPHBase):
    def __init__(self, model, categorical_variables=None):
        self.model, self.categorical_variables = model, categorical_variables

    def fit(self, X, T, E):
        self.model.fit(X, T, sample_weight=E)

        return self.prepare_estimands(X, T, E)
    
    def prepare_estimands(self, X, T, E):
        self.compute_baseline_hazards(X, (T, E))
        self.y_train = Surv.from_arrays(E, T)
        return self
    
    def predict(self, X, *args, **kwargs):
        if isinstance(X, tt.TupleTree):
            X = X[0]

        if isinstance(self.model, _ShareProgram):
            ohe = {}
            if self.categorical_variables is not None: 
                ohe = ohe_matrices(X, self.categorical_variables, self.model.optim_dict.get("device"))

            return self.model.execute(torch.Tensor(X), ohe)
        elif hasattr(self.model, 'execute'):
            return self.model.execute(X)
        elif hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            raise ValueError(
                "Model must be a BaseSymbolic or _Program instance."
            )

    def score(self, X, T, E, extended=True):
        # Predict hazards and survival
        h_pred, surv_pred = self.predict(X), self.predict_surv_df(X)
        y_test = Surv.from_arrays(E, T)

        # Times for brier scores: min and max observed event times, within the range of validation T
        times = surv_pred.index.values[1:-1]
        times = times[(times > T.min().item()) & (times < T.max().item())]

        # Fix for IPCW estimation (c_ipcw and dynamic AUC)
        # We can't have observed events exactly at the final observed time
        # so we nudge the last time point slightly
        E_train, T_train = check_y_survival(self.y_train)

        y_train_adjusted = Surv.from_arrays(
            np.concat((E_train, [False])), np.concat((T_train, [T_train.max() + 1e-5]))
        )

        # Run metrics
        brier, integrated_brier = (None, None), None
        if not surv_pred.isna().any().any():
            try:
                brier = brier_score(self.y_train, y_test, surv_pred.loc[times, :].T, times)
                integrated_brier = integrated_brier_score(
                    self.y_train, y_test, surv_pred.loc[times, :].T, times
                )
            except ValueError as e:
                if "time must be smaller than largest observed time point" in str(e):
                    raise ValueError("Validation T contains times larger than the largest observed time point in training data. ")
                else:
                    raise e
        
        c_censored = concordance_index_censored((E == 1) if not E.dtype in (torch.bool, np.bool) else E, T, h_pred)
        c_ipcw = concordance_index_ipcw(y_train_adjusted, y_test, h_pred)
        auc = cumulative_dynamic_auc(y_train_adjusted, y_test, h_pred, times)

        get_npll = lambda ties: neg_partial_log_likelihood(
            torch.tensor(h_pred),
            torch.tensor(E, dtype=torch.bool),
            torch.tensor(T),
            ties_method=ties,
            reduction="mean",
            checks=True,
        ).item()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            npll_efron, npll_breslow = get_npll("efron"), get_npll("breslow")

        # Reformat
        brier = dict(zip(["time", "value"], brier))

        c_idx_names = ["value", "concordant", "discordant", "ties_risk", "ties_time"]
        c_censored, c_ipcw = dict(zip(c_idx_names, c_censored)), dict(
            zip(c_idx_names, c_ipcw)
        )

        auc = dict(zip(["values", "mean"], auc))

        npll = dict(efron=npll_efron, breslow=npll_breslow)

        if not extended:
            return dict(
                integrated_brier_score=integrated_brier,
                concordance_index_censored=c_censored["value"],
                concordance_index_ipcw=c_ipcw["value"],
                cumulative_dynamic_auc=auc["mean"],
                neg_partial_log_likelihood=npll["efron"],
            )
        
        return dict(
            brier_score=brier,
            intergrated_brier_score=integrated_brier,
            concordance_index_censored=c_censored,
            concordance_index_ipcw=c_ipcw,
            cumulative_dynamic_auc=auc,
            neg_partial_log_likelihood=npll,
        )
    
    def plot_calibration(self, X, T, E, t0=None): 
        surv_pred = self.predict_surv_df(X)
        t0 = t0 or surv_pred.index.values[-1]
        surv_pred_at_t0 = surv_pred.loc[t0, :].values.squeeze()

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        _, binned_ici, binned_e50 = calibration_plot_binned(surv_pred_at_t0, T, E, t0, ax=ax[0])
        _, smoothed_ici, smoothed_e50 = calibration_plot_smoothed(surv_pred_at_t0, T, E, t0, ax=ax[1])

        return dict(
            mean_absolute_difference_binned=binned_ici,
            mean_absolute_difference_smoothed=smoothed_ici,
            median_absolute_difference_binned=binned_e50,
            median_absolute_difference_smoothed=smoothed_e50,
        )