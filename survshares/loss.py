import torch
import warnings

from torchsurv.loss.cox import neg_partial_log_likelihood
from pycox.models.loss import cox_ph_loss


class npllLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, weight):
        return neg_partial_log_likelihood(
            log_hz=input,
            event=weight,
            time=target,
            reduction=self.reduction,
            ties_method="efron",
            checks=False,
        )


class approximateNpllLoss(torch.nn.Module):
    def __init__(self, reduction=None):
        super().__init__()
        if self.reduction is not None:
            warnings.warn(f"{self.__class__.__name__} only supports mean reduction")

    def forward(self, input, target, weight):
        return cox_ph_loss(log_h=input, durations=target, events=weight, eps=1e-7)
