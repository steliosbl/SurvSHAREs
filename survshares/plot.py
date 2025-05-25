import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CRCSplineFitter, KaplanMeierFitter


def calibration_plot_binned(
    surv_pred_at_t0, T, E, t0, n_bins=10, ax=None, return_df=False
):
    predictions_at_t0 = np.clip(
        1 - surv_pred_at_t0, 1e-10, 1 - 1e-10
    )  # Go from survival proba to event proba and clip

    # Bin the samples based on event probability
    df = pd.DataFrame(dict(pred=predictions_at_t0, time=T, event=E)).assign(
        bin=pd.qcut(predictions_at_t0, q=n_bins, duplicates="drop")
    )

    groups = df.groupby("bin", observed=True)

    predicted = groups["pred"].mean()  # For each bin, represent with the mean event probability
    observed = groups[["time", "event"]].apply(  
        # For each bin, use Kaplan-Meier to get the actual event frequency for these samples
        lambda group: KaplanMeierFitter().fit(group["time"], group["event"]).predict(t0)
    )  # For good calibration, the observed event frequency should match the predicted event proba
    observed = 1-observed 

    if return_df:
        return pd.concat(
            (predicted.rename("predicted"), observed.rename("observed")), axis=1
        )
    
    print_scores = False 
    if ax is None:
        print_scores = True 
        ax = plt.gca()

    color = "tab:red"
    ax.plot(predicted, observed, "o-", label="Model", color=color)
    ax.set_title(
        "Smoothed calibration curve of \npredicted vs observed event probabilities by $T \leq t_0$"
    )
    ax.set_xlabel("Predicted event probability by $t_0$: $1-\hat S(t_0)$")
    ax.set_ylabel("Observed event frequency by $T \leq t_0$", color=color)
    ax.tick_params(axis="y", labelcolor=color)

    ax.plot([0, 1], [0, 1], ls="--", color="black", label="Perfect")

    _calibration_plot_histogram(predictions_at_t0, ax)
    ax.legend()
    plt.tight_layout()

    deltas = np.abs(observed.values - predicted.values)
    # Integrated Calibration Index (approx) and median absolute difference
    ICI, E50 = deltas.mean(), np.percentile(deltas, 50)  
    if print_scores:
        print(f"Mean Absolute Difference = {ICI}")
        print(f"Median Absolute Difference = {E50}")

    return ax, ICI, E50


# Minor changes from lifelines source - https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/calibration.py
def calibration_plot_smoothed(
    surv_pred_at_t0, T, E, t0: float, ax=None, return_df=False
):
    predictions_at_t0 = np.clip(1 - surv_pred_at_t0, 1e-10, 1 - 1e-10)

    T = T + 1e-2 # SBL: Add a small value to T to avoid non-positive durations as they mess up CRCSplineFitter

    # create new dataset with the predictions
    ccl = lambda p: np.log(-np.log(1 - p))
    prediction_df = pd.DataFrame(
        {"ccl_at_%d" % t0: ccl(predictions_at_t0), "T": T, "E": E}
    )

    # fit new dataset to flexible spline model
    # this new model connects prediction probabilities and actual survival. It should be very flexible, almost to the point of overfitting. It's goal is just to smooth out the data!
    regressors = {
        "beta_": ["ccl_at_%d" % t0],
        "gamma0_": "1",
        "gamma1_": "1",
        "gamma2_": "1",
    }

    # this model is from examples/royson_crowther_clements_splines.py
    crc = CRCSplineFitter(n_baseline_knots=3, penalizer=0.000001)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        crc.fit_right_censoring(prediction_df, "T", "E", regressors=regressors)

    # predict new model at values 0 to 1, but remember to ccl it!
    x = np.linspace(
        np.clip(predictions_at_t0.min() - 0.01, 0, 1),
        np.clip(predictions_at_t0.max() + 0.01, 0, 1),
        100,
    )
    y = crc.predict_survival_function(
        pd.DataFrame({"ccl_at_%d" % t0: ccl(x)}), times=[t0]
    )
    y = 1 - y.T.squeeze()

    if return_df:
        return pd.DataFrame(dict(predicted=y, observed=x))

    print_scores = False 
    if ax is None:
        print_scores = True 
        ax = plt.gca()

    # plot our results
    color = "tab:red"
    ax.plot(x, y, label="Model", color=color)
    ax.set_title(
        "Smoothed calibration curve of \npredicted vs observed event probabilities by $T \leq t_0$"
    )
    ax.set_xlabel("Predicted event probability by $t_0$: $1-\hat S(t_0)$")
    ax.set_ylabel("Observed event frequency by $T \leq t_0$", color=color)
    ax.tick_params(axis="y", labelcolor=color)

    # plot x=y line
    ax.plot(x, x, color="black", ls="--", label="Perfect")
    ax.legend()

    # plot histogram of our original predictions
    _calibration_plot_histogram(predictions_at_t0, ax)

    plt.tight_layout()

    deltas = (
        (1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze()
        - predictions_at_t0
    ).abs()
    ICI = deltas.mean()  # Integrated Calibration Index - mean absolute difference between pred and obs
    E50 = np.percentile(deltas, 50)  # The median absolute delta / 50th percentile of the error
    if print_scores:
        print("Mean Absolute Difference (ICI) = ", ICI)
        print("Median Absolute Difference (E50) = ", E50)

    return ax, ICI, E50

def _calibration_plot_histogram(predictions_at_t0, ax):
    color = "tab:blue"
    twin_ax = ax.twinx()
    twin_ax.set_ylabel(
        "Histogram of predicted event probabilities", color=color
    )  # we already handled the x-label with ax1
    twin_ax.tick_params(axis="y", labelcolor=color)
    twin_ax.hist(predictions_at_t0, alpha=0.3, bins="sqrt", color=color)