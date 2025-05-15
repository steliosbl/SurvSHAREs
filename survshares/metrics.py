import numpy as np 
import warnings 

from lifelines.utils import concordance_index
from pycox.evaluation.metrics import partial_log_likelihood_ph
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv 

def c_index(y_true, y_pred, sample_weight):
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


def partial_likelihood(y_true, y_pred, sample_weight):
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


def integrated_brier(surv_pred, E_train, T_train, E_test=None, T_test=None):
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