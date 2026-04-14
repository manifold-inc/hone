"""LoopLM training objectives.

Implements the three training stages from "Scaling Latent Reasoning via Looped
Language Models" (arXiv:2510.25741):

  Stage I  — Entropy-regularized pre-training (Eq. 4)
  Stage II — Focused adaptive gate training  (Eq. 6)
  SFT      — Standard cross-entropy on the final recurrent step

Gate logits are stored raw (pre-sigmoid) in LoopLMOutput to match the official
Ouro checkpoint convention.  Sigmoid is applied inside these loss functions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import LoopLMOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gate_logits_to_lambdas(
    step_gate_logits: list[torch.Tensor],
) -> torch.Tensor:
    """Convert raw gate logits to per-step exit probabilities lambda_t.

    All computation is in float32 for numerical stability during backward
    through the looped architecture.

    Returns:
        lam: (T, B, S) float32 tensor with values in (0, 1).
    """
    return torch.stack(
        [torch.sigmoid(g.squeeze(-1).float()) for g in step_gate_logits], dim=0
    )  # (T, B, S)


def _compute_exit_distribution(
    step_gate_logits: list[torch.Tensor],
) -> torch.Tensor:
    """Convert per-step raw gate logits into a valid discrete distribution
    p(t|x) over exit steps (Eq. 3).

    Args:
        step_gate_logits: List of T tensors, each (B, S, 1) raw logits.

    Returns:
        p_exit: (T, B, S) tensor where p_exit[t] is the probability of
                exiting at step t, summing to 1 over the T dimension.
    """
    lam = _gate_logits_to_lambdas(step_gate_logits)  # (T, B, S)

    # Clamp to avoid log(0) / division-by-zero
    lam = lam.clamp(1e-6, 1.0 - 1e-6)

    # Survival: S_t = prod_{j=1}^{t} (1 - lambda_j)
    log_survival = torch.cumsum(torch.log(1.0 - lam), dim=0)  # (T, B, S)
    # Shift right so prev_survival[t] = S_{t-1}
    prev_log_survival = torch.zeros_like(log_survival)
    prev_log_survival[1:] = log_survival[:-1]
    prev_survival = torch.exp(prev_log_survival)

    # p_tilde(t) = lambda_t * S_{t-1}  for t < T_max
    p = lam * prev_survival  # (T, B, S)

    # Assign remaining mass to final step: p(T_max) = S_{T_max - 1}
    p[-1] = torch.exp(prev_log_survival[-1])

    # Normalise for numerical safety (should already sum to ~1)
    p = p / (p.sum(dim=0, keepdim=True) + 1e-8)

    return p


def _per_step_cross_entropy(
    step_logits: list[torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-step cross-entropy losses.

    Args:
        step_logits: List of T tensors each (B, S, V).
        labels:      (B, S) with -100 for ignored positions.

    Returns:
        losses: (T,) float32 tensor of mean cross-entropy at each recurrent step.
    """
    losses = []
    for logits in step_logits:
        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.float().reshape(-1, V),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        losses.append(loss)
    return torch.stack(losses)  # (T,)


def _per_step_per_token_cross_entropy(
    step_logits: list[torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Per-step, per-token cross-entropy (unreduced).

    Logits are cast to float32 to avoid bf16 overflow in the softmax
    backward (especially harmful with large vocabularies like 256K).

    Returns:
        losses: (T, B, S) float32 tensor.
    """
    out = []
    for logits in step_logits:
        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.float().reshape(-1, V),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(B, S)
        out.append(loss)
    return torch.stack(out)  # (T, B, S)


# ---------------------------------------------------------------------------
# Stage I: Entropy-Regularized Objective  (Eq. 4)
# ---------------------------------------------------------------------------

def looplm_stage1_loss(
    output: LoopLMOutput,
    labels: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """Entropy-regularized pre-training loss.

    L = sum_t p(t|x) * L^(t) - beta * H(p(.|x))

    The exit distribution p(t|x) is derived from raw gate logits via Eq. 3.
    The entropy term prevents collapse to always using T_max.
    """
    # Per-step per-token CE: (T, B, S)
    per_token_losses = _per_step_per_token_cross_entropy(
        output.step_logits, labels
    )
    # Exit distribution: (T, B, S)
    p_exit = _compute_exit_distribution(output.step_gate_logits)

    # Expected task loss: sum_t p(t) * L(t)  — per-token then mean
    expected_loss = (p_exit * per_token_losses).sum(dim=0)  # (B, S)

    # Only count non-ignored positions
    valid_mask = (labels != -100).float()
    n_valid = valid_mask.sum().clamp(min=1.0)
    expected_loss = (expected_loss * valid_mask).sum() / n_valid

    # Entropy: H(p) = -sum_t p(t) * log(p(t))
    log_p = torch.log(p_exit + 1e-8)
    entropy = -(p_exit * log_p).sum(dim=0)  # (B, S)
    mean_entropy = (entropy * valid_mask).sum() / n_valid

    return expected_loss - beta * mean_entropy


# ---------------------------------------------------------------------------
# Stage II: Focused Adaptive Gate Training  (Eq. 6)
# ---------------------------------------------------------------------------

def looplm_stage2_loss(
    output: LoopLMOutput,
    labels: torch.Tensor,
    k: float = 50.0,
    gamma: float = 0.005,
) -> torch.Tensor:
    """Adaptive gate training loss (LM frozen, gate only).

    For each step t >= 2, computes the loss improvement I_t and derives an
    ideal continuation label w_t.  The gate is trained via BCE to match w_t.
    """
    T = len(output.step_logits)
    if T < 2:
        return torch.tensor(0.0, device=output.step_logits[0].device, requires_grad=True)

    # Per-step per-token CE with detached logits (frozen LM)
    detached_logits = [lg.detach() for lg in output.step_logits]
    per_token_losses = _per_step_per_token_cross_entropy(detached_logits, labels)
    # (T, B, S)

    # Convert raw gate logits to lambda_t
    lam = _gate_logits_to_lambdas(output.step_gate_logits)  # (T, B, S)

    valid_mask = (labels != -100).float()  # (B, S)
    n_valid = valid_mask.sum().clamp(min=1.0)

    total_adaptive_loss = torch.tensor(
        0.0, device=per_token_losses.device, dtype=per_token_losses.dtype
    )

    for t in range(1, T):
        # Loss improvement: I_t = max(0, L_{t-1} - L_t)
        improvement = torch.clamp(
            per_token_losses[t - 1] - per_token_losses[t], min=0.0
        )
        # Ideal continuation label: w_t = sigmoid(k * (I_t - gamma))
        w = torch.sigmoid(k * (improvement - gamma))

        # Gate continuation probability: 1 - lambda_t
        lam_t = lam[t]  # (B, S)
        cont_prob = (1.0 - lam_t).clamp(1e-6, 1.0 - 1e-6)
        lam_t_clamped = lam_t.clamp(1e-6, 1.0 - 1e-6)

        # BCE: w * log(1 - lambda) + (1 - w) * log(lambda)
        bce = -(w * torch.log(cont_prob) + (1.0 - w) * torch.log(lam_t_clamped))
        step_loss = (bce * valid_mask).sum() / n_valid
        total_adaptive_loss = total_adaptive_loss + step_loss

    return total_adaptive_loss / (T - 1)


# ---------------------------------------------------------------------------
# SFT: standard cross-entropy on final step
# ---------------------------------------------------------------------------

def looplm_sft_loss(
    output: LoopLMOutput,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Standard SFT loss using the final recurrent step's logits."""
    logits = output.step_logits[-1].float()  # (B, S, V)
    B, S, V = logits.shape
    return F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def compute_looplm_loss(
    output: LoopLMOutput,
    labels: torch.Tensor,
    stage: str = "pretrain",
    beta: float = 0.05,
    gate_k: float = 50.0,
    gate_gamma: float = 0.005,
) -> torch.Tensor:
    """Dispatch to the appropriate loss function based on training stage."""
    if stage == "pretrain":
        return looplm_stage1_loss(output, labels, beta=beta)
    elif stage == "gate":
        return looplm_stage2_loss(output, labels, k=gate_k, gamma=gate_gamma)
    elif stage == "sft":
        return looplm_sft_loss(output, labels)
    else:
        raise ValueError(f"Unknown training stage: {stage!r}")
