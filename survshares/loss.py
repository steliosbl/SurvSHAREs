import torch
import warnings

from torchsurv.loss.cox import neg_partial_log_likelihood
from pycox.models.loss import cox_ph_loss

# Anecdotally, on my machine, in a single attempt (no proper benchmarking) as of 2025-05-24:
# npllLoss with ties_method="efron" takes 20s to fit METABRIC
# npllLoss with ties_method="breslow" takes 8s to fit METABRIC
# approximateNpllLoss takes 3s to fit METABRIC

class npllLoss(torch.nn.Module):
    """
    Wrapper around `torchsurv.loss.cox.neg_partial_log_likelihood` to compute the
    negative partial log-likelihood loss for Cox Proportional Hazards models.
    
    """
    def __init__(self, reduction="mean", ties_method="breslow"):
        super().__init__()
        self.reduction = reduction
        self.ties_method = ties_method

    def forward(self, input, target, weight):
        if weight.dtype is torch.float: 
            # Because it's used as an index, (input[weight]), it must be int or bool typed
            weight = weight.int()
        
        result = neg_partial_log_likelihood(
            log_hz=input,
            event=weight,
            time=target,
            reduction=self.reduction,
            ties_method=self.ties_method,
            checks=False,
        )
        if result.isnan():
            result = torch.tensor(torch.inf, requires_grad=True)
        elif result.isneginf():
            result = -result
        return result


class approximateNpllLoss(torch.nn.Module):
    """
    Wrapper around `pycox.models.loss.cox_ph_loss` to compute the approximate
    negative partial log-likelihood loss for Cox Proportional Hazards models.

    This relies on an undocumented approximation that ignores ties. In DeepSurv, this is
    acceptable because of the random subsampling of risk sets during SGD batched training.  
    See: https://github.com/havakv/pycox/issues/27
    """
    def __init__(self, reduction=None):
        super().__init__()
        if reduction is not None:
            warnings.warn(f"{self.__class__.__name__} only supports mean reduction")

    def forward(self, input, target, weight):
        result = cox_ph_loss(log_h=input, durations=target, events=weight, eps=1e-7)
        if result.isnan():
            result = torch.tensor(torch.inf, requires_grad=True)
        elif result.isneginf():
            result = -result
        return result 
