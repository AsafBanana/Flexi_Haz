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


def cox_partial_likelihood_loss(eta, T, event, clip: float = 80.0):
    """Negative Breslow partial log-likelihood (no ties), averaged over events.

    Subjects are sorted by decreasing T, so the cumulative log-sum-exp at a
    subject is the log of its risk set. Used by model_type="zhong".
    """
    order = torch.argsort(T.reshape(-1), descending=True)
    eta_s = torch.clamp(eta.reshape(-1)[order], -float(clip), float(clip))
    d_s = event.reshape(-1)[order]
    log_denom = torch.logcumsumexp(eta_s, dim=0)
    n_events = torch.clamp(torch.sum(d_s), min=1.0)
    return -torch.sum(d_s * (eta_s - log_denom)) / n_events
