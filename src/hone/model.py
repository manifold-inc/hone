"""LoopLM — Llama-style decoder-only transformer.

Standard single-pass pre-norm transformer matching the TorchTitan Llama3
architecture used in templar.  Building blocks: RoPE, GQA attention, SwiGLU
feed-forward, RMSNorm.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    tie_embeddings: bool = True

    use_moe: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    moe_layers: list[int] | None = None
    moe_aux_loss_coeff: float = 0.01

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
# RMSNorm
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
# Rotary Position Embeddings
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
# Attention (GQA + SDPA)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, config: LoopLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.dim // config.n_heads
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads

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
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: LoopLMConfig, intermediate_size: int | None = None):
        super().__init__()
        ffn_dim = intermediate_size or config.intermediate_size
        assert ffn_dim is not None
        self.gate_proj = nn.Linear(config.dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEMLP(nn.Module):
    """Mixture-of-Experts MLP with grouped-GEMM dispatch over a stacked
    expert-weight tensor.

    The previous implementation kept one ``nn.Linear`` per expert in an
    ``nn.ModuleList`` and walked them with a Python ``for`` loop. For 96
    experts that meant ~13K small kernel launches per microbatch (24
    layers x 96 experts x 3 projections x 2 for fwd+bwd) plus a
    per-layer ``offsets.tolist()`` that synced the GPU back to the host
    every iteration. With ``(N*K)/E`` tokens per expert sitting around
    700 for our config, each of those GEMMs ran far below B200's
    tensor-core saturation point and the loop dominated the inner step.

    This rewrite:

    1. Replaces the ``nn.ModuleList(experts)`` with three stacked
       parameters keyed by expert index along ``dim 0``. Memory layout is
       chosen to match ``torch._grouped_mm`` directly so we avoid a
       transpose on every forward.
    2. Replaces the Python loop with three ``torch._grouped_mm`` calls
       over device-resident offsets. No host sync per layer.
    3. Drops the ``offsets.tolist()`` call so ``torch.compile`` can fuse
       the routed-expert path end-to-end without graph breaks.

    BREAKING CHANGE: parameter names move from
    ``mlp.experts.{e}.{gate,up,down}_proj.weight`` to
    ``mlp.{gate,up,down}_weight`` of shape ``(E, K, N)``. Old MoE
    checkpoints cannot be loaded; the gradient-compression wire keys
    change too. Validators and miners must be redeployed together.
    """

    def __init__(self, config: LoopLMConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        E = config.num_experts
        D = config.dim
        ffn = config.moe_intermediate_size or config.intermediate_size
        assert ffn is not None
        self.expert_dim = D
        self.expert_ffn = ffn

        self.gate = nn.Linear(D, E, bias=False)

        # ``torch._grouped_mm`` expects ``mat_b`` of shape
        # ``(num_groups, K, N)`` -- so for the up-projection ``K=D``,
        # ``N=ffn`` and for the down-projection ``K=ffn``, ``N=D``. We
        # store the tensors in that exact layout so the hot path needs no
        # ``.transpose()`` (which would otherwise force a contiguous copy
        # on every microbatch since FSDP-gathered DTensors aren't
        # transposed-contiguous).
        self.gate_weight = nn.Parameter(torch.empty(E, D, ffn))
        self.up_weight = nn.Parameter(torch.empty(E, D, ffn))
        self.down_weight = nn.Parameter(torch.empty(E, ffn, D))

        self.shared_expert: MLP | None = None
        if config.shared_expert_intermediate_size is not None:
            self.shared_expert = MLP(
                config, intermediate_size=config.shared_expert_intermediate_size
            )

    def _grouped_mm(
        self, a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        """Type-safe wrapper around ``torch._grouped_mm``.

        ``torch._grouped_mm`` is *not* autocast-aware (pytorch#174763),
        so we coerce ``a`` to the weight dtype before each call. Under
        FSDP with ``mixed_precision="bfloat16"`` ``b`` is already bf16
        and the coercion is a no-op; the explicit cast is the
        belt-and-suspenders that keeps the kernel happy when the
        upstream layer emitted float32 (e.g. RMSNorm, embedding lookup).
        """
        if a.dtype != b.dtype:
            a = a.to(b.dtype)
        return torch._grouped_mm(a, b, offs=offs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(output, aux_loss)``.

        Routing logic is unchanged from the previous permutation-based
        dispatch: tokens are repeated ``top_k`` times, stably sorted by
        chosen expert id, processed in expert order, and the inverse
        permutation reassembles per-token outputs that are then summed
        across the K choices. Only the per-expert compute step changes:
        the Python loop over ``self.experts`` is now three grouped
        matmuls over the stacked weights.
        """
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        N = x_flat.shape[0]
        K = self.top_k
        E = self.num_experts

        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(gate_probs, K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Load-balancing auxiliary loss (unchanged).
        tokens_per_expert = torch.zeros(E, device=x.device, dtype=x.dtype)
        tokens_per_expert.scatter_add_(
            0, topk_indices.view(-1),
            torch.ones(topk_indices.numel(), device=x.device, dtype=x.dtype),
        )
        f = tokens_per_expert / (N * K)
        p = gate_probs.mean(dim=0)
        aux_loss = self.aux_loss_coeff * E * (f * p).sum()

        # ---- Permutation dispatch ----
        expanded_input = x_flat.repeat_interleave(K, dim=0)
        expanded_indices = topk_indices.reshape(-1)
        expanded_weights = topk_weights.reshape(-1)

        sorted_indices, sort_perm = torch.sort(expanded_indices, stable=True)
        sorted_input = expanded_input.index_select(0, sort_perm)
        sorted_weights = expanded_weights.index_select(0, sort_perm)

        # ``torch._grouped_mm`` wants the *cumulative end indices* of each
        # group along the first axis of ``a`` as int32, length E. We keep
        # this entirely on-device -- no ``.tolist()`` host sync, so
        # torch.compile can capture the routed path as a single graph.
        counts = torch.bincount(sorted_indices, minlength=E)
        offs = counts.cumsum(0).to(torch.int32)

        gate_out = self._grouped_mm(sorted_input, self.gate_weight, offs)
        up_out = self._grouped_mm(sorted_input, self.up_weight, offs)
        sorted_output = self._grouped_mm(
            F.silu(gate_out) * up_out, self.down_weight, offs
        )

        sorted_output = sorted_output * sorted_weights.unsqueeze(-1)

        # Inverse permutation, then sum the K choices per original token.
        inv_perm = torch.argsort(sort_perm)
        output_expanded = sorted_output.index_select(0, inv_perm)
        output = output_expanded.view(N, K, -1).sum(dim=1)

        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        return output.view(orig_shape), aux_loss


# ---------------------------------------------------------------------------
# Transformer Block — standard pre-norm (matches TorchTitan TransformerBlock)
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, config: LoopLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, layer_idx)

        use_moe_here = config.use_moe and (
            config.moe_layers is None or layer_idx in config.moe_layers
        )
        self.use_moe = use_moe_here
        self.mlp = MoEMLP(config) if use_moe_here else MLP(config)

        self.input_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)

        self.weight_init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        if self.use_moe:
            mlp_out, aux_loss = self.mlp(self.post_attention_layernorm(h))
            out = h + mlp_out
            return out, aux_loss
        else:
            out = h + self.mlp(self.post_attention_layernorm(h))
            return out

    def init_weights(self):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            nn.init.ones_(norm.weight)
        for linear in (self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.self_attn.o_proj.weight, mean=0.0, std=self.weight_init_std)
        if self.use_moe:
            nn.init.trunc_normal_(self.mlp.gate.weight, mean=0.0, std=0.02)
            # Stacked-expert weights: every expert shares the same init
            # std, so we initialise the whole (E, K, N) tensor in one
            # shot. This works on FSDP DTensors because nn.init operates
            # on the local shard transparently.
            nn.init.trunc_normal_(self.mlp.gate_weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.mlp.up_weight, mean=0.0, std=self.weight_init_std)
            nn.init.trunc_normal_(self.mlp.down_weight, mean=0.0, std=self.weight_init_std)
            if self.mlp.shared_expert is not None:
                nn.init.trunc_normal_(self.mlp.shared_expert.gate_proj.weight, mean=0.0, std=0.02)
                nn.init.trunc_normal_(self.mlp.shared_expert.up_proj.weight, mean=0.0, std=self.weight_init_std)
                nn.init.trunc_normal_(self.mlp.shared_expert.down_proj.weight, mean=0.0, std=self.weight_init_std)
        else:
            nn.init.trunc_normal_(self.mlp.gate_proj.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.mlp.up_proj.weight, mean=0.0, std=self.weight_init_std)
            nn.init.trunc_normal_(self.mlp.down_proj.weight, mean=0.0, std=self.weight_init_std)


# ---------------------------------------------------------------------------
# LoopLM — single-pass Llama-style decoder
# ---------------------------------------------------------------------------

class LoopLM(nn.Module):
    """Decoder-only transformer language model.

    Single pass through ``n_layers`` transformer blocks, then RMSNorm and a
    linear head to vocabulary logits.
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

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.rotary_emb = RotaryEmbedding(
            config.dim // config.n_heads,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Returns logits (dense) or (logits, aux_loss) when MoE is active."""
        B, S = input_ids.shape
        h = self.embed_tokens(input_ids)

        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(h, position_ids)

        total_aux_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)
        has_moe = False

        for layer in self.layers:
            result = layer(h, position_embeddings)
            if isinstance(result, tuple):
                h, aux_loss = result
                total_aux_loss = total_aux_loss + aux_loss
                has_moe = True
            else:
                h = result

        h = self.norm(h)
        logits = self.lm_head(h)

        if has_moe:
            return logits, total_aux_loss
        return logits

    def init_weights(self):
        """Initialize weights following the TorchTitan Llama convention.

        Embedding: normal(0, 1.0).  Attention q/k/v: trunc_normal(0, 0.02).
        Output projections (o_proj, down_proj, up_proj): depth-scaled std.
        Final lm_head: trunc_normal(0, dim^-0.5), skipped when tied.
        """
        nn.init.normal_(self.embed_tokens.weight)
        for layer in self.layers:
            layer.init_weights()
        nn.init.ones_(self.norm.weight)
        if not self.config.tie_embeddings:
            final_std = self.config.dim ** -0.5
            cutoff = 3 * final_std
            nn.init.trunc_normal_(
                self.lm_head.weight, mean=0.0, std=final_std,
                a=-cutoff, b=cutoff,
            )
