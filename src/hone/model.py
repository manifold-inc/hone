"""LoopLM — Looped Language Model with adaptive early exit.

Implements the architecture from "Scaling Latent Reasoning via Looped Language
Models" (arXiv:2510.25741).  A shared stack of N transformer layers is applied
recurrently T_max times.  At each recurrent step an exit gate produces a halting
logit and the LM head emits next-token logits.

Parameter naming matches the official Ouro HuggingFace checkpoint
(ByteDance/Ouro-1.4B) for weight-loading compatibility.
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
    intermediate_size: int | None = None
    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq_len: int = 4096
    t_max: int = 4
    tie_embeddings: bool = True
    hidden_act: str = "silu"

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.intermediate_size is None:
            raw = int(8 * self.dim / 3)
            if self.ffn_dim_multiplier is not None:
                raw = int(raw * self.ffn_dim_multiplier)
            self.intermediate_size = self.multiple_of * (
                (raw + self.multiple_of - 1) // self.multiple_of
            )


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class LoopLMOutput:
    step_logits: list[torch.Tensor] = field(default_factory=list)
    step_gate_logits: list[torch.Tensor] = field(default_factory=list)
    final_hidden: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# RMSNorm  (matches OuroRMSNorm)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings  (matches OuroRotaryEmbedding)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE.  q, k are (B, H, S, D); cos, sin are (B, S, D) or (1, S, D)."""
    cos = cos.unsqueeze(1)  # (B, 1, S, D)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Precomputes and caches cos/sin tables for RoPE."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Attention  (matches OuroAttention)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, config: LoopLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.dim // config.n_heads
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expand KV heads
        if self.num_key_value_groups > 1:
            k = k[:, :, None, :, :].expand(
                -1, -1, self.num_key_value_groups, -1, -1
            ).reshape(B, self.n_heads, S, self.head_dim)
            v = v[:, :, None, :, :].expand(
                -1, -1, self.num_key_value_groups, -1, -1
            ).reshape(B, self.n_heads, S, self.head_dim)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=(attention_mask is None)
        )
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward  (matches OuroMLP)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: LoopLMConfig):
        super().__init__()
        assert config.intermediate_size is not None
        self.gate_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer Block with sandwich normalization  (matches OuroDecoderLayer)
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Pre-norm + post-norm ("sandwich") on both attention and FFN sub-layers,
    following Geiping et al. for recurrent-depth stability.

    Norm naming matches the official Ouro checkpoint:
      input_layernorm / input_layernorm_2       — around attention
      post_attention_layernorm / post_attention_layernorm_2 — around FFN
    """

    def __init__(self, config: LoopLMConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)

        self.input_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.input_layernorm_2 = RMSNorm(config.dim, config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.post_attention_layernorm_2 = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with sandwich norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        # FFN with sandwich norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# LoopLM  (matches OuroModel + OuroForCausalLM structure)
# ---------------------------------------------------------------------------

class LoopLM(nn.Module):
    """Looped Language Model.

    A shared stack of ``n_layers`` transformer blocks is applied ``t_max``
    times.  Each recurrent step produces next-token logits and raw exit-gate
    logits (sigmoid applied downstream in loss computation).
    """

    def __init__(self, config: LoopLMConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.early_exit_gate = nn.Linear(config.dim, 1)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.rotary_emb = RotaryEmbedding(
            config.dim // config.n_heads,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        t_max: int | None = None,
    ) -> LoopLMOutput:
        B, S = input_ids.shape
        t_max = t_max if t_max is not None else self.config.t_max

        hidden_states = self.embed_tokens(input_ids)

        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        step_logits: list[torch.Tensor] = []
        step_gate_logits: list[torch.Tensor] = []

        for _t in range(t_max):
            for layer in self.layers:
                hidden_states = layer(hidden_states, position_embeddings)

            hidden_states = self.norm(hidden_states)
            step_logits.append(self.lm_head(hidden_states))
            step_gate_logits.append(self.early_exit_gate(hidden_states))

        return LoopLMOutput(
            step_logits=step_logits,
            step_gate_logits=step_gate_logits,
            final_hidden=hidden_states,
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
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cdf = torch.zeros(B, S, device=hidden_states.device)
        survival = torch.ones(B, S, device=hidden_states.device)

        for t in range(t_max):
            for layer in self.layers:
                hidden_states = layer(hidden_states, position_embeddings)

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            lam = torch.sigmoid(self.early_exit_gate(hidden_states).squeeze(-1))

            if t < t_max - 1:
                p_t = lam * survival
                survival = survival * (1 - lam)
            else:
                p_t = survival

            cdf = cdf + p_t

            if cdf.mean() >= q_threshold:
                return logits, t + 1

        return logits, t_max

    def init_weights(self):
        """Initialize weights following the Ouro convention.

        Embedding uses normal(0, 0.02).  Linear layers use Xavier-uniform,
        but the lm_head is skipped when weights are tied to the embedding
        (otherwise the lm_head Xavier init overwrites the embedding init,
        producing very small embedding norms that amplify gradients through
        the first RMSNorm by ~170x with a 256K vocabulary).
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name == "lm_head" and self.config.tie_embeddings:
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
