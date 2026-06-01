"""
utils/loss.py
─────────────
Masked soft-label BCE with per-aspect positive weighting.

Why masked?
  Snorkel probabilities in [abstain_low, abstain_high] are not genuine
  uncertainty — they reflect LF abstention. We zero those positions in the
  loss so the model is not penalized for getting them "wrong".

Why pos_weight?
  Fiscal_Government has only 6.5% hard prevalence. Without weighting,
  the model learns to always predict negative for Fiscal (high accuracy,
  zero recall). pos_weight amplifies the gradient from false negatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from config import TrainConfig


class MaskedSoftBCE(nn.Module):
    """
    loss = mean over unmasked (aspect, sample) pairs of:
        pos_weight[j] * y_ij * log(σ(z_ij))
      + (1 - y_ij) * log(1 - σ(z_ij))

    Args:
        pos_weight  : 1-D tensor of length num_labels
        reduction   : 'mean' (default) or 'sum'
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            pos_weight if pos_weight is not None else torch.ones(1),
        )
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,       # [batch, num_labels]  raw (pre-sigmoid)
        targets: torch.Tensor,      # [batch, num_labels]  soft floats in [0,1]
        mask: torch.Tensor,         # [batch, num_labels]  1=use, 0=ignore
    ) -> torch.Tensor:

        # Element-wise BCE (no reduction yet)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),
            reduction="none",           # [batch, num_labels]
        )

        # Apply abstain mask
        bce = bce * mask                # zero out abstain positions

        if self.reduction == "mean":
            denom = mask.sum().clamp(min=1.0)
            return bce.sum() / denom
        elif self.reduction == "sum":
            return bce.sum()
        else:
            return bce


def build_loss_fn(cfg: TrainConfig) -> MaskedSoftBCE:
    pw = torch.tensor(cfg.pos_weight, dtype=torch.float32)
    return MaskedSoftBCE(pos_weight=pw)
