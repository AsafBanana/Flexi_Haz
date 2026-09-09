from __future__ import annotations

import torch


def negative_log_likelihood(eta, event, interval_length, reduction: str = "mean", clip: float = 80.0):
    eta_safe = torch.clamp(eta, -float(clip), float(clip))
    interval_length = torch.clamp(interval_length, min=0.0)
    nll = interval_length * torch.exp(eta_safe) - event * eta_safe
    if reduction == "sum":
        return torch.sum(nll)
    if reduction == "none":
        return nll
    return torch.mean(nll)
