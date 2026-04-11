"""LoopLM — Looped Language Model with adaptive early exit.

Implements the architecture from "Scaling Latent Reasoning via Looped Language
Models" (arXiv:2510.25741).  A shared stack of N transformer layers is applied
recurrently T_max times.  At each recurrent step an exit gate produces a halting
probability and the LM head emits next-token logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LoopLMConfig:
    vocab_size: int = 49152
    dim: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int | None = None
    ffn_hidden_dim: int | None = None
    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq_len: int = 4096
    t_max: int = 4
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.ffn_hidden_dim is None:
            raw = int(8 * self.dim / 3)
            if self.ffn_dim_multiplier is not None:
                raw = int(raw * self.ffn_dim_multiplier)
            self.ffn_hidden_dim = self.multiple_of * (
                (raw + self.multiple_of - 1) // self.multiple_of
            )


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class LoopLMOutput:
    step_logits: list[torch.Tensor] = field(default_factory=list)
    step_gate_probs: list[torch.Tensor] = field(default_factory=list)
    final_hidden: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings
# ---------------------------------------------------------------------------

def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device).float() / dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # xq, xk: (B, S, H, D)  -> reshape last dim to complex pairs
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs = freqs_cis[: xq.shape[1]]  # (S, D/2)
    freqs = freqs.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)

    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        # GQA: expand KV heads
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                B, S, self.n_heads, self.head_dim
            )
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                B, S, self.n_heads, self.head_dim
            )

        # (B, H, S, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer Block with sandwich normalization
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm + post-norm ("sandwich") on both attention and FFN sub-layers,
    following Geiping et al. for recurrent-depth stability."""

    def __init__(self, config: LoopLMConfig):
        super().__init__()
        assert config.n_kv_heads is not None
        assert config.ffn_hidden_dim is not None

        self.pre_attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = Attention(config.dim, config.n_heads, config.n_kv_heads)
        self.post_attn_norm = RMSNorm(config.dim, config.norm_eps)

        self.pre_ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = FeedForward(config.dim, config.ffn_hidden_dim)
        self.post_ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.post_attn_norm(
            self.attention(self.pre_attn_norm(x), freqs_cis, mask)
        )
        out = h + self.post_ffn_norm(
            self.feed_forward(self.pre_ffn_norm(h))
        )
        return out


# ---------------------------------------------------------------------------
# Exit Gate
# ---------------------------------------------------------------------------

class ExitGate(nn.Module):
    """Produces per-token instantaneous exit probability lambda_t."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(-1)  # (B, S)


# ---------------------------------------------------------------------------
# LoopLM
# ---------------------------------------------------------------------------

class LoopLM(nn.Module):
    """Looped Language Model.

    A shared stack of ``n_layers`` transformer blocks is applied ``t_max``
    times.  Each recurrent step produces next-token logits and an exit-gate
    probability.
    """

    def __init__(self, config: LoopLMConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.exit_gate = ExitGate(config.dim)

        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.dim // config.n_heads,
                config.max_seq_len,
                config.rope_theta,
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        t_max: int | None = None,
    ) -> LoopLMOutput:
        B, S = input_ids.shape
        t_max = t_max if t_max is not None else self.config.t_max

        h = self.tok_embeddings(input_ids)

        freqs_cis = self.freqs_cis[:S].to(h.device)

        step_logits: list[torch.Tensor] = []
        step_gate_probs: list[torch.Tensor] = []

        for _t in range(t_max):
            for layer in self.layers:
                h = layer(h, freqs_cis)

            normed = self.norm(h)
            step_logits.append(self.lm_head(normed))
            step_gate_probs.append(self.exit_gate(normed))

        return LoopLMOutput(
            step_logits=step_logits,
            step_gate_probs=step_gate_probs,
            final_hidden=h,
        )

    # ------------------------------------------------------------------
    # Inference helper with early exit
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_with_early_exit(
        self,
        input_ids: torch.Tensor,
        q_threshold: float = 0.5,
        t_max: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Run forward with early exit based on cumulative exit probability.

        Returns the logits from the exit step and the step index used.
        """
        t_max = t_max if t_max is not None else self.config.t_max
        B, S = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:S].to(h.device)

        cdf = torch.zeros(B, S, device=h.device)
        survival = torch.ones(B, S, device=h.device)

        for t in range(t_max):
            for layer in self.layers:
                h = layer(h, freqs_cis)

            normed = self.norm(h)
            logits = self.lm_head(normed)
            lam = self.exit_gate(normed)  # (B, S)

            if t < t_max - 1:
                p_t = lam * survival
                survival = survival * (1 - lam)
            else:
                p_t = survival  # remaining mass at final step

            cdf = cdf + p_t

            if (cdf.mean() >= q_threshold):
                return logits, t + 1

        return logits, t_max

    def init_weights(self):
        """Xavier-uniform for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
