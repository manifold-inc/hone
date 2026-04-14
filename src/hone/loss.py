"""Standard cross-entropy loss for language modelling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss with float32 upcast for numerical stability.

    Args:
        logits: (B, S, V) model output logits.
        labels: (B, S) target token ids, -100 for ignored positions.
    """
    V = logits.size(-1)
    return F.cross_entropy(
        logits.float().reshape(-1, V),
        labels.reshape(-1),
        ignore_index=-100,
    )
