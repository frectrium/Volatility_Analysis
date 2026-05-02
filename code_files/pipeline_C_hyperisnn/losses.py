"""Loss functions for HyperISNN v3.

All losses operate on tensors of shape (B, M, 1) for predictions, targets, and
masks. The base interface is::

    Loss(pred, target, mask, batch) → (loss, dict_of_components)

`batch` carries auxiliary tensors (vega, m, T, r, ...) for losses that need them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .config import LossConfig


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over positions where mask == 1."""
    n = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / n


def _per_episode_variance(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x, mask: (B, M, 1) → (B,) per-episode variance over valid positions."""
    n = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / n
    var = (((x - mean) ** 2) * mask).sum(dim=1) / n
    return var.squeeze(-1)


# ----------------------------------------------------------------------------
# Soft constraint penalties (shared across all data losses)
#
# NOTE: the intrinsic-value lower bound C/S >= max(0, 1 - m*exp(-rT)) is now
# enforced by the architecture (see isnn.isnn2_forward), so the soft penalty
# that used to live here was removed. The upper-bound and variance-collapse
# safety nets are still useful as soft regularisers.
# ----------------------------------------------------------------------------

def upper_bound_penalty(pred: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """C/S <= 1."""
    return _masked_mean(F.relu(pred - 1.0) ** 2, batch["tmask"])


def variance_collapse_penalty(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Anti-collapse: penalise log(var_true / var_pred) when positive."""
    pv = _per_episode_variance(pred, mask)
    tv = _per_episode_variance(target, mask)
    log_ratio = torch.log(tv.clamp(min=1e-12)) - torch.log(pv.clamp(min=1e-12))
    return F.relu(log_ratio).mean()


# ----------------------------------------------------------------------------
# Data-fitting losses
# ----------------------------------------------------------------------------

class BaseLoss(ABC):
    """Base loss combining a data term with the standard soft constraints."""

    name: str = "base"

    def __init__(self, cfg: LossConfig):
        self.cfg = cfg

    @abstractmethod
    def data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor: ...

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mask = batch["tmask"]
        l_data = self.data_loss(pred, target, batch)
        l_upp  = upper_bound_penalty(pred, batch)
        l_var  = variance_collapse_penalty(pred, target, mask)

        loss = (
            l_data
            + self.cfg.lambda_upper * l_upp
            + self.cfg.lambda_var * l_var
        )
        components = dict(
            data=float(l_data.detach()),
            upper=float(l_upp.detach()),
            var=float(l_var.detach()),
            total=float(loss.detach()),
        )
        return loss, components


class MSELoss(BaseLoss):
    name = "mse"

    def data_loss(self, pred, target, batch):
        return _masked_mean((pred - target) ** 2, batch["tmask"])


class VegaWeightedMSELoss(BaseLoss):
    """1/vega^2 weighting → equivalent to MSE in IV space (first-order)."""
    name = "vega"

    def data_loss(self, pred, target, batch):
        # vega comes in as (B, M, 1) in normalized form (vega/S).
        w = 1.0 / torch.clamp(batch["vega"], min=1e-5)
        wsq = w * w
        num = (((pred - target) ** 2) * wsq * batch["tmask"]).sum()
        den = (wsq * batch["tmask"]).sum().clamp(min=1.0)
        return num / den


class HuberLoss(BaseLoss):
    """Huber on price for robustness to mispriced quotes."""
    name = "huber"
    delta: float = 0.005             # ~ 1.5x ATM std on C/S

    def data_loss(self, pred, target, batch):
        diff = pred - target
        ad = diff.abs()
        quad = torch.minimum(ad, torch.full_like(ad, self.delta)) ** 2 * 0.5
        lin  = self.delta * (ad - torch.minimum(ad, torch.full_like(ad, self.delta)))
        return _masked_mean(quad + lin, batch["tmask"])


class LogPriceMSELoss(BaseLoss):
    """MSE in log(price) space — emphasises relative error (good for OTM)."""
    name = "logprice"

    def data_loss(self, pred, target, batch):
        eps = 1e-4
        diff = torch.log(pred.clamp(min=eps)) - torch.log(target.clamp(min=eps))
        return _masked_mean(diff ** 2, batch["tmask"])


class HybridHuberVegaMSELoss(BaseLoss):
    """0.5 * Huber(price) + 0.5 * VegaWeightedMSE(price).

    Per suggestions.md Phase 2: pure VegaWeightedMSE explodes on deep OTM
    (vega clamped at 5e-3 → 1/vega^2 huge → gradient spikes). Huber on the raw
    price domain is robust on OTM but loses sensitivity ATM where vega-weighting
    behaves like IV-domain MSE. The 50/50 mix gets both.
    """
    name = "hybrid"
    delta: float = 0.005

    def data_loss(self, pred, target, batch):
        diff = pred - target
        ad = diff.abs()
        delta_t = torch.full_like(ad, self.delta)
        quad = torch.minimum(ad, delta_t) ** 2 * 0.5
        lin  = self.delta * (ad - torch.minimum(ad, delta_t))
        l_huber = _masked_mean(quad + lin, batch["tmask"])

        w = 1.0 / torch.clamp(batch["vega"], min=1e-5)
        wsq = w * w
        num = (((pred - target) ** 2) * wsq * batch["tmask"]).sum()
        den = (wsq * batch["tmask"]).sum().clamp(min=1.0)
        l_vega = num / den

        return 0.5 * l_huber + 0.5 * l_vega


# Registry for easy CLI selection
LOSS_REGISTRY = {
    MSELoss.name: MSELoss,
    VegaWeightedMSELoss.name: VegaWeightedMSELoss,
    HuberLoss.name: HuberLoss,
    LogPriceMSELoss.name: LogPriceMSELoss,
    HybridHuberVegaMSELoss.name: HybridHuberVegaMSELoss,
}


def build_loss(name: str, cfg: LossConfig) -> BaseLoss:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](cfg)
