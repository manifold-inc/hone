# The MIT License (MIT)
# © 2025 hone.training

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import asyncio
import contextlib
import gc
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor as DT
from torch.distributed.tensor import distribute_tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from wandb.sdk.wandb_run import Run

import hone
from hone.compress import unpack_12bit_indices
from hone.distributed import dist_helper

if TYPE_CHECKING:
    from neurons.miner import Miner
    from neurons.validator import Validator

NeuronT = TypeVar("NeuronT", "Miner", "Validator")


# PyTorch wrappers (torch.compile, checkpoint_wrapper, FSDP1) splice
# magic tokens into ``named_parameters()`` keys. They're orthogonal to
# the model's logical structure, but their position depends on the
# *order* of wrapping. Validator wraps root + AC + compile, so its
# keys look like ``_orig_mod.layers.5._checkpoint_wrapped_module.mlp.weight``.
# PP miner wraps per-layer + AC + per-leaf compile, so its keys look
# like ``stage.layers.5._orig_mod._checkpoint_wrapped_module.mlp.weight``.
# Different positions for the same logical param -> validator's
# expected set never matches the miner's upload, every UID gets
# rejected with a giant "extra keys" log spam ("skipping UID 174 due
# to validation failures"). Stripping these tokens uniformly on both
# sides gives both wrapping orders a single agreed namespace
# (``layers.5.mlp.weight``).
import re as _re

_WRAPPER_TOKENS = (
    "_orig_mod",  # torch.compile (Dynamo)
    "_checkpoint_wrapped_module",  # checkpoint_wrapper (selective AC)
    "_fsdp_wrapped_module",  # FSDP1 wrap (kept for safety; FSDP2 doesn't add it)
    "module",  # nn.parallel.DistributedDataParallel root
)
_WRAPPER_TOKEN_RE = _re.compile(
    r"(?:^|(?<=\.))(?:" + "|".join(_re.escape(t) for t in _WRAPPER_TOKENS) + r")\."
)


def _strip_wrapper_prefixes(name: str) -> str:
    """Remove every wrapper token (``_orig_mod.``, ``_checkpoint_wrapped_module.``,
    etc.) from ``name``, anywhere in the dotted path.

    The regex matches a wrapper token only when it sits at the start
    of the string OR immediately after a dot, so we don't accidentally
    chew up a real submodule that happens to *contain* one of these
    substrings as a suffix (e.g. ``my_module.``). Run repeatedly is
    safe (idempotent).
    """
    return _WRAPPER_TOKEN_RE.sub("", name)


def canonical_param_names(model: nn.Module) -> dict[str, str]:
    """Map ``model.named_parameters()`` keys to cross-rank-stable canonical
    names used in compressed gradient uploads.

    All returned canonical names are post-processed to strip PyTorch
    wrapper tokens (``_orig_mod.``, ``_checkpoint_wrapped_module.``,
    ``_fsdp_wrapped_module.``) so callers don't have to reason about
    whether the model was compiled / activation-checkpointed / FSDP-
    wrapped, in what order. Wrapping order silently broke gradient
    aggregation across topologies before this helper existed.
    """
    return {n: _strip_wrapper_prefixes(n) for n, _ in model.named_parameters()}


# Embedding-table substrings: ``LoopLM`` uses ``embed_tokens`` and
# ``lm_head`` (see ``hone/src/hone/model.py:419,424``). The other tokens
# cover common HF-Transformers names (``wte`` for GPT-2, ``embedding`` /
# ``embed`` as catch-alls) so the heuristic doesn't silently break if
# the architecture swaps out. Per Decoupled DiLoCo §D.2 Table 8: RDA
# helps non-embedding params and HURTS embedding params (norm of the
# vocab-row gradient is what carries token-frequency signal); plain Avg
# is preserved for embeddings.
_EMBEDDING_NAME_TOKENS = (
    "embed",
    "embd",
    "tok_embed",
    "wte",
    "lm_head",
    "embedding",
)


def _is_embedding_param(name: str) -> bool:
    """Return True when the canonical param name is an embedding /
    output-projection table that RDA should NOT rescale.

    Heuristic match against ``_EMBEDDING_NAME_TOKENS`` lowercased; see
    that constant's docstring for the rationale.
    """
    name_lower = name.lower()
    return any(token in name_lower for token in _EMBEDDING_NAME_TOKENS)


def prepare_gradient_dict(miner: "Miner", step_window: int, null_round: bool = False):
    """
    DTensor-deadlock-safe:
    - All ranks: rendezvous on DTensor grads (GFULL) and DTensor params (PFULL).
    - Only owning ranks: momentum update, encode, compress, estimate/decode, EF update.

    Args:
        miner: Miner instance containing model, compressor, transformer, etc.
        step_window: Current window number
        null_round: If True, this is a null/warmup round and error feedback should be cleared
    """

    # ------------ helpers ------------
    def ddp_initialized():
        return dist.is_available() and dist.is_initialized()

    def is_dtensor(x):
        try:
            from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]

            return isinstance(x, DTensor)
        except Exception:
            return type(x).__name__ in {"DTensor", "DistributedTensor", "DT"}

    def get_mesh_group(x):
        if not is_dtensor(x):
            return None
        mesh = getattr(x, "device_mesh", None)
        if mesh is None:
            spec = getattr(x, "_spec", None)
            mesh = getattr(spec, "mesh", None)
        if mesh is not None:
            try:
                return mesh.get_group()
            except Exception:
                pass
        return dist.group.WORLD if ddp_initialized() else None

    def barrier(group=None):
        if ddp_initialized() and group is not None:
            dist.barrier(group=group)

    # ------------ start ------------
    gradient, xshapes, totalks = {}, {}, {}
    use_dct = getattr(miner.hparams, "use_dct", False)
    topk = getattr(miner.hparams, "topk_compression", 32)

    # P6b TurboQuant gate (default-OFF). When ``turboquant_enabled`` is
    # true the branch below replaces the encode → compress → decompress
    # → decode chain with a Hadamard-rotation + Beta-Lloyd-Max
    # quantization + top-K-on-codes pipeline. THE BRANCH MUST STAY
    # DISABLED until ``hone/validator/turboquant_audit_cli.py`` exits 0
    # against a real production EF snapshot — see ``hone/docs/turboquant.md``
    # for the operator gating workflow.
    use_turboquant = getattr(miner.hparams, "turboquant_enabled", False)
    if use_turboquant:
        # Lazy import keeps the existing legacy path clean of the
        # turboquant module (which transitively pulls scipy on first
        # codebook fit). Importers that never enable the flag pay no
        # startup cost from this branch.
        from hone import turboquant as _turboquant

        tq_bits = int(getattr(miner.hparams, "turboquant_bits", 4))
        tq_outlier_k = int(getattr(miner.hparams, "turboquant_outlier_top_k", 32))
        tq_mode: Literal["mse", "prod"] = (
            "prod"
            if bool(getattr(miner.hparams, "turboquant_q_prod_mode", True))
            else "mse"
        )

    if isinstance(miner.model, torch.nn.parallel.DistributedDataParallel):
        inner_model = miner.model.module
    else:
        inner_model = miner.model
    model_iterator = inner_model.named_parameters()

    # Canonical name map keeps PP-wrapped miners and plain-LoopLM
    # validators speaking the same compressed-payload namespace. A None
    # entry means the param is local-only (PP boundary) and never shipped.
    canon_map = canonical_param_names(inner_model)

    # Build params dict once to avoid repeated iteration
    params_dict = dict(inner_model.named_parameters())

    # Batch load all error feedback tensors to GPU
    for n in miner.owned_params:
        if miner.error_feedback.get(n, None) is not None:
            if miner.error_feedback[n].is_cuda:
                continue
            # Get the device from the corresponding parameter
            param = params_dict.get(n)
            if param is not None:
                miner.error_feedback[n] = miner.error_feedback[n].to(
                    param.device, non_blocking=True
                )

    for _, (n, p) in enumerate(model_iterator, 1):
        owned = n in miner.owned_params
        p_is_dt = is_dtensor(p)
        g = getattr(p, "grad", None)
        g_is_dt = is_dtensor(g)

        # Resolve the canonical (cross-rank-stable) name we publish under.
        cname = canon_map.get(n, n)

        # --- 1) Grad full_tensor rendezvous (GFULL) ---
        if g_is_dt:
            grp_g = get_mesh_group(g)
            barrier(grp_g)
            assert g is not None
            grad_full = g.full_tensor().to(p.device)
            barrier(grp_g)
        else:
            if g is None and not p_is_dt:
                continue
            assert g is not None, f"p.grad is None for {n}"
            grad_full = g.to(p.device)

        # Non-owners: after participating in grad collective, drop grad and continue.
        if not owned:
            p.grad = None
            continue

        # --- 3) Momentum buffer update (owner only) ---
        # Handle DTensor error feedback by creating new regular tensor if needed
        error_feedback = miner.error_feedback[n]
        if error_feedback is None:
            error_feedback = torch.zeros_like(grad_full, device=p.device)
        elif error_feedback.device != p.device:
            # Should already be on GPU from batch load, but handle edge cases
            error_feedback = error_feedback.to(p.device)

        # Clear error feedback during null rounds to prevent accumulation of invalid gradients
        if null_round:
            error_feedback.zero_()
        else:
            error_feedback.mul_(miner.hparams.momentum_decay)
            error_feedback.add_(grad_full)

        # --- 4) Encode & compress (owner only) ---
        # ``ChunkingTransformer`` only knows 1D and 2D tensors. Stacked
        # MoE weights are 3D ``(E, D, ffn)`` after the grouped-GEMM
        # rewrite, so we collapse the leading expert dim into rows
        # giving a 2D ``(E*D, ffn)`` view that the codec can chunk
        # normally. The view shares storage with ``error_feedback`` so
        # any in-place op (``sub_`` below) propagates to the actual 3D
        # tensor we keep around between windows.
        ef_for_codec = (
            error_feedback.view(-1, error_feedback.shape[-1])
            if error_feedback.dim() == 3
            else error_feedback
        )

        if use_turboquant:
            # P6b dark-code path. Replaces the encode → compress →
            # decompress → decode chain with:
            #   1. Hadamard rotation on the FULL d-dim flat EF.
            #   2. Beta-Lloyd-Max scalar quantisation of every coord.
            #   3. Top-K **on the codebook indices** (NOT the float
            #      values) — the load-bearing fix to the original P6
            #      plan. Selecting on float values after rotation
            #      breaks the Beta distribution the codebook assumes.
            # The TurboQuant payload is written under ``cname + 'tq_*'``
            # keys; the legacy ``cname + {'idxs','vals','quant_params'}``
            # keys are deliberately NOT populated so a TurboQuant-
            # unaware validator skips this param outright instead of
            # mis-decoding it. Per the operator gating workflow in
            # ``hone/docs/turboquant.md``, ``turboquant_enabled`` MUST
            # remain false until both the matching validator-side
            # decoder lands and the P6a + A/B gates pass.
            ef_flat = ef_for_codec.flatten()
            d_flat = int(ef_flat.numel())
            rotated = _turboquant.hadamard_rotate(ef_flat.unsqueeze(0)).squeeze(0)
            tq_codes, tq_meta = _turboquant.quantize_turboquant(
                rotated,
                d=d_flat,
                b=tq_bits,
                mode=tq_mode,
                outlier_top_k=tq_outlier_k,
            )
            # Match the legacy budget so swapping codecs does not change
            # wire size: the legacy path keeps roughly
            # ``ef.numel() * topk / target_chunk`` coords (≈ 50% under
            # the default 32/64 ratio).
            target_chunk = int(getattr(miner.hparams, "target_chunk", 64))
            tq_keep_count = max(1, (d_flat * topk) // max(target_chunk, 1))
            tq_keep_count = min(tq_keep_count, d_flat)
            # Top-K on |codes| — rotated coords whose code magnitude is
            # largest carry the most reconstructed signal.
            topk_idx = tq_codes.long().abs().topk(tq_keep_count).indices

            # Reconstruct what the receiver would see (sparse rotated
            # vector + inverse rotation) so the local EF accumulates
            # the same residual the validator does. The WHT is its own
            # inverse on power-of-two padded length, so the second
            # ``hadamard_rotate`` call is the inverse rotation.
            #
            # We dequantize the FULL ``d_flat``-length codes tensor
            # before sparsifying because ``tq_meta`` carries outlier
            # indices in the full-d-space coordinate system; passing a
            # subset of codes would index outliers out-of-bounds. This
            # costs an O(d) lookup but keeps the codec API simple — the
            # alternative (re-indexing tq_meta at the kept positions)
            # would split the meta semantics into two flavours and
            # complicate the validator-side decoder we owe in the
            # follow-up PR.
            full_dequant = _turboquant.dequantize_turboquant(tq_codes, tq_meta)
            sparse_rotated = torch.zeros_like(rotated)
            sparse_rotated[topk_idx] = full_dequant[topk_idx].to(
                sparse_rotated.dtype
            )
            del full_dequant
            transmit_grad_flat = _turboquant.hadamard_rotate(
                sparse_rotated.unsqueeze(0)
            ).squeeze(0)
            transmit_grad = transmit_grad_flat.view(ef_for_codec.shape)

            alpha = getattr(miner.hparams, "momentum_subtraction_alpha", 1.0)
            if alpha == 1.0:
                ef_for_codec.sub_(transmit_grad)
            else:
                ef_for_codec.sub_(transmit_grad, alpha=alpha)
            miner.error_feedback[n] = error_feedback
            del transmit_grad, transmit_grad_flat, sparse_rotated, error_feedback

            # Stash the TurboQuant payload under TQ-prefixed keys. We
            # only ship the kept top-K subset to keep wire size in
            # parity with the legacy path; full-tensor codes would
            # roughly double payload at b=4. Validator-side TQ decode
            # (TBD) reads these keys; legacy decode skips them.
            gradient[cname + "tq_idxs"] = topk_idx.to("cpu")
            gradient[cname + "tq_codes"] = tq_codes[topk_idx].to("cpu")
            gradient[cname + "tq_meta"] = tq_meta

            p.grad = None
            continue

        encoded = miner.transformer.encode(ef_for_codec, use_dct=use_dct)

        idxs, vals, xshape, totalk, quant_params = miner.compressor.compress(
            encoded, topk
        )
        del encoded

        # --- 5) Decompress reference (owner only) ---
        # ``decompress`` only reads ``p.device``/``p.dtype``, so it
        # doesn't care about p's actual shape -- safe to pass the 3D
        # stacked tensor directly.
        decompressed = miner.compressor.decompress(
            p, idxs, vals, xshape, totalk, quant_params
        )

        # --- 6) Decode & error-feedback update (owner only) ---
        transmit_grad = miner.transformer.decode(decompressed, use_dct=use_dct)
        del decompressed
        alpha = getattr(miner.hparams, "momentum_subtraction_alpha", 1.0)
        # ``transmit_grad`` matches ``ef_for_codec``'s 2D shape; subtract
        # via the view so 3D ``error_feedback`` storage is updated
        # in-place too.
        if alpha == 1.0:
            ef_for_codec.sub_(transmit_grad)
        else:
            ef_for_codec.sub_(transmit_grad, alpha=alpha)
        # Keep error feedback on GPU for now, batch offload later
        miner.error_feedback[n] = error_feedback
        del transmit_grad, error_feedback

        # --- 7) Pack outputs (move compressed artifacts to CPU asynchronously) ---
        # Using non_blocking=True for async D2H transfers when CUDA is
        # available. Keys use ``cname`` (canonical / un-prefixed) so PP
        # miners and plain-LoopLM validators agree on the namespace.
        if isinstance(idxs, torch.Tensor):
            if torch.cuda.is_available():
                cpu_idxs = torch.empty_like(idxs, device="cpu", pin_memory=True)
                cpu_idxs.copy_(idxs, non_blocking=True)
                gradient[cname + "idxs"] = cpu_idxs
            else:
                gradient[cname + "idxs"] = idxs.cpu()
        else:
            gradient[cname + "idxs"] = idxs

        if isinstance(vals, torch.Tensor):
            if torch.cuda.is_available():
                cpu_vals = torch.empty_like(vals, device="cpu", pin_memory=True)
                cpu_vals.copy_(vals, non_blocking=True)
                gradient[cname + "vals"] = cpu_vals
            else:
                gradient[cname + "vals"] = vals.cpu()
        else:
            gradient[cname + "vals"] = vals
        gradient[cname + "quant_params"] = quant_params
        xshapes[cname] = xshape
        totalks[cname] = totalk

        # Clear per-param grad
        p.grad = None

    # Batch offload all error feedback tensors to CPU with pinned memory
    for name in miner.error_feedback:
        if (
            miner.error_feedback[name] is not None
            and miner.error_feedback[name].is_cuda
        ):
            # Copy to the pre-allocated pinned buffer
            miner.error_feedback_cpu_buffers[name].copy_(
                miner.error_feedback[name], non_blocking=True
            )
            miner.error_feedback[name] = miner.error_feedback_cpu_buffers[name]

    # Single synchronization at the end for all async operations
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gradient["metadata"] = {"window": step_window}
    return gradient, xshapes, totalks


def prepare_gradient_buckets(
    gradient: dict,
    num_buckets: int = 4,
) -> list[dict]:
    """Split a gradient dict into N buckets for streaming upload.

    Instead of uploading all compressed gradients at once at end-of-window,
    this lets us stream parameter subsets during training to reduce peak
    upload bandwidth by num_buckets-fold.

    Bucket assignment is **balanced greedy bin-packing**: tensors are sorted
    by total compressed payload size (idxs.numel + vals.numel) descending,
    then each tensor is placed in the currently smallest bucket. This is
    within 4/3 of the optimal max-bucket peak per Decoupled DiLoCo §C and
    replaces the original round-robin partition (which could leave one
    bucket holding the embedding while others held only attention weights).

    FU1 (P6b): TurboQuant-encoded params publish a ``tq_idxs`` /
    ``tq_codes`` / ``tq_meta`` triple INSTEAD of the legacy
    ``idxs`` / ``vals`` / ``quant_params`` triple (per the
    ``prepare_gradient_dict`` TurboQuant branch). The bucketing logic
    must keep the triple together — splitting any of the three across
    fragments would make the validator-side decode fail with a KeyError.
    We scan both suffix groups and collapse each param to a single
    canonical ``base`` before bin-packing.

    Args:
        gradient: The full gradient dict from prepare_gradient_dict
        num_buckets: Number of buckets to split into

    Returns:
        List of gradient sub-dicts, each containing a subset of parameters.
        Each bucket includes the metadata key with ``bucket_idx`` and
        ``num_buckets`` so the receiver can identify which fragment it is.
    """
    metadata = gradient.get("metadata", {})

    # FU1: canonicalise the per-param base name across both codec
    # layouts. A legacy param contributes ``base = cname`` via its
    # ``cname + "idxs"`` key; a TurboQuant param contributes the same
    # ``base = cname`` via its ``cname + "tq_idxs"`` key. Both land in
    # the same bucketing pool so a mixed-codec miner (currently
    # impossible per the encoder contract, but kept robust) would not
    # double-bucket a param.
    param_bases: set[str] = set()
    for key in gradient:
        if key.endswith("tq_idxs"):
            param_bases.add(key[: -len("tq_idxs")])
        elif key.endswith("idxs"):
            param_bases.add(key[:-4])

    def _payload_size(base: str) -> int:
        size = 0
        # Legacy top-K + 2-bit pipeline.
        for suffix in ("idxs", "vals"):
            t = gradient.get(base + suffix)
            if isinstance(t, torch.Tensor):
                size += t.numel()
        # FU1 TurboQuant triple. The meta dict's bytes are dominated
        # by the bulk_centroids + sign_bits tensors it carries, but
        # those are small per-param (K=16 bulk + d sign_bits ≤ param
        # numel) and don't warrant a full pickled byte count — we use
        # numel()-equivalents from the on-wire tensors (idxs int64 + codes
        # uint8) and accept a small under-count on meta. The LPT bound
        # is robust to small per-item weight errors.
        for suffix in ("tq_idxs", "tq_codes"):
            t = gradient.get(base + suffix)
            if isinstance(t, torch.Tensor):
                size += t.numel()
        return size

    # Sort tensors largest first, then place each into the smallest current
    # bucket. Greedy LPT (longest processing time first) — within 4/3 of
    # optimal makespan; the inline unit test below asserts this bound on
    # the canonical DiLoCo size distribution.
    tensors_by_size = sorted(param_bases, key=_payload_size, reverse=True)
    bucket_loads = [0] * num_buckets
    buckets: list[dict] = [{} for _ in range(num_buckets)]
    for base in tensors_by_size:
        # ``min(range(...), key=...)`` is O(num_buckets); the outer loop is
        # O(num_tensors) so the whole pack is O(N * B) — fine for B=24,
        # N≈7100 (<200µs on CPython).
        bucket_idx = min(range(num_buckets), key=lambda i: bucket_loads[i])
        # Keep each codec's triple co-resident in the same bucket —
        # splitting would break the validator-side decode. Both suffix
        # groups are scanned because in a mixed rollout window a single
        # gradient dict may carry either (but never both for the same
        # base, per prepare_gradient_dict's mutually-exclusive branches).
        for suffix in (
            "idxs",
            "vals",
            "quant_params",
            "tq_idxs",
            "tq_codes",
            "tq_meta",
        ):
            k = base + suffix
            if k in gradient:
                buckets[bucket_idx][k] = gradient[k]
        bucket_loads[bucket_idx] += _payload_size(base)

    for i, bucket in enumerate(buckets):
        bucket["metadata"] = {**metadata, "bucket_idx": i, "num_buckets": num_buckets}

    return buckets


class AsyncGatherBuffer:
    """Double-buffered gather for overlapping communication with training.

    Allows downloading peer gradients from window N-1 while training
    proceeds on window N, reducing wall-clock time by overlapping
    communication with compute.
    """

    def __init__(self):
        self._pending_result: asyncio.Future | None = None
        self._ready_result: SimpleNamespace | None = None
        self._ready_xshapes: dict | None = None
        self._ready_totalks: dict | None = None

    def submit_gather(
        self,
        comms,
        window: int,
        xshapes: dict,
        totalks: dict,
    ):
        """Start an async gather in the background.

        Args:
            comms: The Comms instance with gather capability
            window: Window number to gather for
            xshapes: Parameter shapes dict
            totalks: Total-k dict per parameter
        """
        self._ready_xshapes = xshapes
        self._ready_totalks = totalks

        async def _do_gather():
            return await comms.gather(window)

        loop = asyncio.get_event_loop()
        self._pending_result = asyncio.ensure_future(_do_gather())

    async def get_result(self) -> tuple[SimpleNamespace | None, dict | None, dict | None]:
        """Wait for and return the gathered result.

        Returns:
            (gather_result, xshapes, totalks) or (None, None, None) if nothing pending
        """
        if self._pending_result is None:
            return None, None, None

        try:
            result = await self._pending_result
        except Exception:
            result = None

        self._pending_result = None
        return result, self._ready_xshapes, self._ready_totalks

    @property
    def has_pending(self) -> bool:
        return self._pending_result is not None and not self._pending_result.done()


def _apply_k_safety_cap(
    peer_weights: dict[int, float] | None,
    k_safety: int,
) -> dict[int, float] | None:
    """Cap any single peer's weight at ``1 / k_safety`` of the total after normalisation.

    P3 deliverable: prevents a Byzantine peer who claims huge ``c_tokens``
    from dominating the aggregated update. Implemented as a rescale
    *after* the validator's raw-weight construction (``w = c_tokens *
    (c_tokens / c_steps)``) and *before* ``batch_decompress`` performs
    its own sum-to-one normalisation. Over-weight peers get clipped to
    the cap; the clipped mass is redistributed proportionally to the
    remaining under-cap peers so the total sum is preserved (modulo
    corner cases where every peer is at the cap).

    Two passes suffice for ``k_safety >= 2`` because: pass 1 lowers
    outliers, pass 2 re-checks after redistribution. For
    ``k_safety = 1`` (disable), ``cap_fraction = 1.0`` so nothing is
    ever over the cap and the function is a no-op. ``k_safety <= 0``
    disables the cap entirely (used when the hparam is flipped off).

    When ``peer_weights`` is ``None`` (default, i.e.
    ``token_weighted_aggregation=false``), the function returns
    ``None`` so downstream code takes the uniform-mean path. This is
    the guardrail that lets ``gather_safety_cap`` ship default-ON
    without activating any behaviour until the operator enables token
    weighting upstream.

    Returns the capped (and possibly redistributed) weights. Caller
    takes ownership.
    """
    if peer_weights is None or not peer_weights:
        return peer_weights
    if k_safety <= 0:
        return peer_weights
    cap_fraction = 1.0 / float(k_safety)
    total = sum(peer_weights.values())
    if total <= 0:
        return peer_weights
    for _ in range(2):
        new_weights = dict(peer_weights)
        total_local = sum(new_weights.values())
        if total_local <= 0:
            return new_weights
        cap_abs = cap_fraction * total_local
        spill = 0.0
        under_cap_peers: list[int] = []
        under_cap_sum = 0.0
        for uid, w in new_weights.items():
            if w > cap_abs:
                spill += w - cap_abs
                new_weights[uid] = cap_abs
            else:
                under_cap_peers.append(uid)
                under_cap_sum += w
        if spill == 0.0 or under_cap_sum == 0.0:
            return new_weights
        for uid in under_cap_peers:
            share = new_weights[uid] / under_cap_sum if under_cap_sum > 0 else 0.0
            new_weights[uid] += share * spill
        peer_weights = new_weights
    return peer_weights


@torch.no_grad()
def outer_step(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    gather_result: SimpleNamespace | None,
    transformer: hone.compress.ChunkingTransformer,
    compressor: hone.compress.TopKCompressor,
    xshapes: dict,
    totalks: dict,
    device: str,
    is_master: bool,
    world_size: int,
    use_dct: bool = False,
    wandb_run: Run | None = None,
    global_step: int | None = None,
    max_grad_norm: float | None = None,
    auto_clip_state: dict | None = None,
    auto_clip_factor: float = 1.5,
    auto_clip_ema_decay: float = 0.95,
    peer_weights: dict[int, float] | None = None,
    rda_merge: bool = False,
    rda_composes_with_clip_norm: bool = False,
    k_safety: int = 0,
    stream: torch.cuda.Stream | None = None,
) -> tuple[dict | None, dict[str, float]]:
    """
    Memory-minimizing variant:
      - Builds the dense gradient for each param one at a time.
      - Stages every param's ``p.grad`` and applies a single foreach
        optimizer step after the loop (was: ``optimizer.step()`` per
        param). On MoE 8B-A1B (~7100 trainable tensors) this removes
        5-15s of Python-loop overhead per outer step.

    P2 additions:
      - ``peer_weights``: optional ``{uid: w}`` mapping plumbed from
        the validator's clamped ``c_tokens`` / ``c_steps`` (see
        ``Validator.run`` token-weighted-aggregation block). Default
        ``None`` reproduces today's uniform ``scatter_reduce_(mean)``
        merge bit-for-bit. When provided, per-param weight lists are
        sliced from ``gather_result.uids_per_param`` so peers that
        contributed only some fragments are weighted only on those
        fragments. Validator-side gating: the
        ``token_weighted_aggregation`` hparam controls whether the
        validator builds this dict.
      - ``rda_merge``: enables the P2.1 light-RDA norm-matching rescale
        on non-embedding params (``_is_embedding_param`` heuristic).
        Per Decoupled DiLoCo §D.2 Table 8: separately averaging norms
        and unit directions stabilises the outer LR across varying
        peer counts. Light variant is constrained by the existing
        position-wise ``scatter_reduce`` aggregation in
        ``batch_decompress`` — full per-peer dense unit-dir averaging
        is O(P × d) memory and is deferred to P2.2. The light
        variant rescales the merged dense gradient to the
        ``peer_weights``-weighted mean of per-peer L2 norms (cheaply
        computed from the existing ``block_norms`` over sparse vals).
      - ``rda_composes_with_clip_norm``: safety guard. ``batch_decompress``
        already runs ``clip_norm=True`` (per-block rescale by
        ``median/||vals_i||``); stacking RDA's norm-match on top
        compounds the rescaling. When ``False`` (default), RDA is
        skipped whenever the caller also supplied ``max_grad_norm``
        (which triggers the 2-pass clip path). Operator must
        explicitly opt-in to composition via the
        ``rda_composes_with_clip_norm`` hparam after the A/B at 1.4B
        for 200 steps shows perplexity within 0.5%.

    P3 additions:
      - ``k_safety``: integer; when ``> 0`` and ``peer_weights`` is not
        ``None``, caps each peer's weight at ``1/k_safety`` of the
        total via ``_apply_k_safety_cap`` before any per-param
        slicing. Default ``0`` disables the cap (byte-for-byte parity
        with the pre-P3 outer_step). The validator gates this through
        the ``gather_safety_cap`` hparam, which is default-ON as a
        guardrail: when ``token_weighted_aggregation=false`` (the
        default),``peer_weights`` is ``None`` at the call site and the
        cap is a structural no-op; when operator flips token
        weighting on, the cap activates immediately without a
        separate hparam flip. Per Decoupled DiLoCo §4, the cap
        prevents a Byzantine peer who claims huge ``c_tokens`` from
        dominating the merged update.

    P4 additions:
      - ``stream``: optional ``torch.cuda.Stream`` to dispatch every
        kernel inside this function on. Default ``None`` keeps the
        legacy behaviour (current stream — implicit default stream
        from the validator main loop). When supplied, the function
        body runs inside ``torch.cuda.stream(stream)`` and every
        ``Event.record()`` call passes ``stream=stream`` so the P0a
        decode/merge/apply timing events are captured on the right
        queue (events recorded against the wrong stream silently
        record no-ops in CUDA when the work has been moved off the
        default stream).
        The validator uses this to overlap ``outer_step`` with peer
        evaluation: a snapshot of pre-step weights is taken on the
        default stream, ``outer_step`` is enqueued on a dedicated
        secondary stream (``hone.distributed.get_outer_step_stream``),
        and eval runs against the snapshot on the default stream.
        Caller is responsible for the cross-stream wait_stream/event
        synchronisation around this call (this function does NOT
        synchronise the supplied stream against the default stream
        — that would defeat the parallelism). NCCL collectives
        (``dist.broadcast``, ``distribute_tensor``) inherit the
        current stream, so all ranks must enter this function with
        the same stream choice (parallel-on or parallel-off, fleet
        wide). The hparam ``parallel_outer_step`` gates this at the
        call site.

    Returns:
      Tuple ``(fingerprint, outer_step_timings)``:
        - ``fingerprint``: gradient statistics dict on master rank, or
          ``None`` on non-master ranks.
        - ``outer_step_timings``: ``dict[str, float]`` with phase
          wall-clock contracts ``decode_seconds`` (decompress to dense
          gradient on master, includes both passes when
          ``outer_max_grad_norm`` is set), ``merge_seconds``
          (cross-rank distribute / broadcast into ``p.grad``;
          includes the P2 RDA norm-match rescale when ``rda_merge``
          is on), and ``apply_seconds`` (the foreach optimizer step).
          Always populated on every rank.
    """
    # P4: when ``stream`` is supplied, every CUDA kernel and NCCL
    # collective dispatched inside this body inherits that stream
    # via ``torch.cuda.stream(stream)``. ``contextlib.nullcontext()``
    # for the legacy ``stream is None`` path keeps the dispatch on
    # whatever stream the caller already had as current (today: the
    # implicit default stream) — byte-for-byte identical to pre-P4.
    stream_ctx: contextlib.AbstractContextManager = (
        torch.cuda.stream(stream)
        if stream is not None
        else contextlib.nullcontext()
    )
    with stream_ctx:
        model.train()
        outer_step_timings: dict[str, float] = {
            "decode_seconds": 0.0,
            "merge_seconds": 0.0,
            "apply_seconds": 0.0,
        }
        use_cuda_timing = torch.cuda.is_available()

        # Free any existing grads entirely (do not allocate zeros)
        optimizer.zero_grad(set_to_none=True)

        ddp = world_size > 1 and dist.is_available() and dist.is_initialized()
        src_rank = 0
        on_src = is_master or not ddp

        # Only master reads aggregated payload (others rely on broadcasts).
        # Accept both SimpleNamespace and plain dict payloads.
        src_sd: dict | None = None
        # P2: per-param contributing-UID list, populated by both gather paths
        # (legacy single-blob and ``_gather_fragmented``); see the gather
        # contract in ``hone/src/hone/comms.py:2071`` and L2433. Used to slice
        # ``peer_weights`` per param in case of fragmented mode where some
        # peers contributed only some params.
        uids_per_param: dict[str, list[int]] | None = None
        if (
            on_src
            and gather_result is not None
            and getattr(gather_result, "state_dict", None) is not None
        ):
            sd = gather_result.state_dict
            src_sd = vars(sd).copy() if isinstance(sd, SimpleNamespace) else dict(sd)
            upp = getattr(gather_result, "uids_per_param", None)
            if isinstance(upp, dict):
                uids_per_param = upp

        # P3 K_safety: cap any single peer's weight at ``1/k_safety`` of
        # the total BEFORE any per-param slicing so every downstream
        # ``_peer_weights_for_param`` call sees the capped weights. When
        # ``peer_weights is None`` (validator disabled token weighting)
        # this is a structural no-op; when ``k_safety <= 0`` the cap is
        # disabled. Applied once up top rather than per-param because the
        # cap is a property of the raw ``c_tokens``-derived weights, not
        # of each param's contributing-peer subset.
        if on_src and peer_weights is not None and k_safety > 0:
            peer_weights = _apply_k_safety_cap(peer_weights, k_safety)

        # P2 token-weighted aggregation: resolve ``peer_weights`` only on
        # master (the only rank that holds ``vals``). Helper closes over the
        # top-level ``peer_weights`` arg + ``uids_per_param``; falls back to
        # ``None`` (== uniform mean) per param when we don't have full
        # metadata. Per-param fan-out is deliberate: a peer may have shipped
        # only a subset of fragments, and the validator should weight them
        # only on the params they actually contributed to.
        #
        # ``key_suffix`` is the upload-dict suffix to look up contributor
        # UIDs under. The legacy top-K + 2-bit path uses ``"vals"`` (every
        # param-with-update ships a vals tensor). FU1's TurboQuant branch
        # uses ``"tq_idxs"`` instead because a TurboQuant-encoded param
        # doesn't populate ``vals`` / ``quant_params`` at all (see
        # ``prepare_gradient_dict`` turboquant branch). Contributor lists
        # are populated symmetrically across a peer's uploaded keys for
        # the same param, so any of the three tq_ suffixes would work;
        # ``tq_idxs`` is the canonical pick because it's the key the
        # decode branch below dispatches on.
        def _peer_weights_for_param(
            cname_local: str, n_peers: int, *, key_suffix: str = "vals"
        ) -> list[float] | None:
            if not on_src or peer_weights is None or uids_per_param is None:
                return None
            uids_for_key = uids_per_param.get(cname_local + key_suffix)
            if uids_for_key is None or len(uids_for_key) != n_peers:
                return None
            weights_local = [
                float(peer_weights.get(int(u), 0.0)) for u in uids_for_key
            ]
            if sum(weights_local) <= 0.0:
                # All peers missing from the weights dict — drop to uniform.
                return None
            return weights_local

        # P2 RDA composition guard. ``batch_decompress(clip_norm=True)``
        # already rescales each peer's vals by ``median/||vals_i||`` before
        # the cross-peer reduce. If the operator also enabled the static or
        # auto outer-grad clip via ``max_grad_norm``, RDA's norm-match
        # compounds those rescalings. Default-off composition: skip RDA
        # whenever ``max_grad_norm`` is set unless operator explicitly
        # opted in via ``rda_composes_with_clip_norm``.
        rda_active = bool(rda_merge) and (
            max_grad_norm is None or bool(rda_composes_with_clip_norm)
        )

        # compact flag broadcast
        def _bcast_flag(v: int) -> int:
            t = torch.tensor([v], device=device, dtype=torch.int32)
            if ddp:
                dist.broadcast(t, src_rank)
            return int(t.item())

        # optional stats
        min_median_norm = float("inf")
        max_median_norm = float("-inf")

        # Initialize fingerprint accumulator (master rank only)
        fingerprint: dict | None = None
        if on_src:
            fingerprint = {
                "param_norms": {},
                "param_means": {},
                "total_norm_sq": 0.0,
                "total_elements": 0,
                # Outer-grad clipping bookkeeping. ``pre_clip_norm`` is the
                # pre-scale global L2 of the incoming aggregated gradient
                # (computed in the pre-pass below); ``clip_scale`` is the
                # multiplier applied to every per-param ``full_grad_src``
                # tensor before it lands in ``p.grad``. ``1.0`` means
                # clipping didn't fire either because ``max_grad_norm`` is
                # None or because the global L2 was already under the
                # threshold.
                "pre_clip_norm": 0.0,
                "clip_scale": 1.0,
                # Self-tuning EMA threshold bookkeeping. When
                # ``auto_clip_state`` is supplied, ``effective_max`` is the
                # dynamic bound actually used this step
                # (= ``min(max_grad_norm, auto_clip_factor * ema)``) and
                # ``auto_ema`` is the post-step EMA value. Both default to
                # ``max_grad_norm`` / ``0.0`` when auto-clip is disabled or
                # the state dict is None, which matches the legacy fixed-cap
                # behaviour and is unambiguous in dashboards.
                "effective_max": float(max_grad_norm) if max_grad_norm is not None else 0.0,
                "auto_ema": 0.0,
            }

        def _idx_to_device(obj, dev: str):
            """
            Move indices to device, supporting:
              • Tensor
              • (packed_tensor, original_shape) for 12-bit packed indices
              • nested list/tuple containers of the above
            We only move the tensor parts; shapes/ints stay on CPU.
            """
            if torch.is_tensor(obj):
                return obj.to(device=dev, non_blocking=True)
            if isinstance(obj, tuple) and len(obj) == 2 and torch.is_tensor(obj[0]):
                return (obj[0].to(device=dev, non_blocking=True), obj[1])
            if isinstance(obj, list):
                return [_idx_to_device(x, dev) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_idx_to_device(x, dev) for x in obj)
            return obj

        # Walk our local model in the canonical (wrapper-stripped) namespace
        # so the keys we look up here match the keys peers publish in
        # ``prepare_gradient_dict``.
        canon_map = canonical_param_names(model)

        # ------------------------------------------------------------------
        # Outer-gradient clipping pre-pass (master rank only)
        # ------------------------------------------------------------------
        # We bound the global L2 of the dense post-decompress gradient (the
        # tensor actually written into ``p.grad`` and applied by the outer
        # optimizer) to ``max_grad_norm``. An earlier version of this pass
        # measured ``sum(||vals||^2)`` over the dequantised top-k blocks and
        # treated that as the global L2; that equality only holds under an
        # orthogonal sparse-into-zero scatter, but ``batch_decompress`` runs
        # with ``clip_norm=True`` (rescaling each block by
        # ``median/norm_i``) and ``transformer.decode`` runs an IDCT, so the
        # final ``full_grad_src`` has a substantially different L2 from the
        # vals. The discrepancy was large enough that ``clip_scale`` stayed
        # at 1.0 every window and the post-clip fingerprint global_l2 grew
        # unbounded (~1 -> 5 -> 11 -> ... -> 375 with threshold 5).
        #
        # The fix is a proper two-walk implementation:
        #   walk 1: fully decode every param, accumulate the dense
        #           ``||full_grad_src||^2``, free each per-param tensor
        #           before the next decode so peak memory grows by at most
        #           one decoded gradient at a time. Returns ``clip_scale``.
        #   walk 2: the existing main loop below redecodes each param,
        #           multiplies by ``clip_scale`` exactly once, then captures
        #           the fingerprint -- so the recorded global L2 is the
        #           post-clip norm and is bounded by ``max_grad_norm``.
        #
        # We pick option (a) from the recommended approaches: cache only
        # the scalar L2 squared (no dense tensors held across walks) and
        # accept a 2x decode cost in exchange for not bumping peak memory
        # by a full-model gradient slice. The ``vals_f32`` dequantised
        # cache *is* kept across walks because dequantisation is the
        # expensive sub-step; ``batch_decompress`` does not mutate its
        # input vals so re-using the cached list in walk 2 is safe.
        #
        # Per-stage clipping: each PP stage's ``outer_step`` only sees its
        # own param subset, so the bound is applied per-stage. That's the
        # right granularity -- spikes are per-stage param jumps too.
        vals_f32_cache: dict[str, list[torch.Tensor]] = {}
        clip_scale: float = 1.0
        pre_clip_norm: float = 0.0
        # Defaults reported in the fingerprint when auto-clip is off so
        # the keys are always present for downstream dashboards.
        effective_max: float = float(max_grad_norm) if max_grad_norm is not None else 0.0
        auto_ema_after: float = (
            float(auto_clip_state.get("ema", 0.0))
            if isinstance(auto_clip_state, dict) else 0.0
        )

        def _compute_pre_clip_norm() -> float:
            """Walk every param with an update, fully decode it, and accumulate
            the dense ``||full_grad_src||^2``. Returns the pre-clip global
            L2 of the aggregated outer gradient.

            Side effect: populates ``vals_f32_cache[cname]`` with the
            dequantised vals list for each visited param so the main walk
            can pop and reuse it. Each per-param dense gradient tensor is
            released before the next param is decoded so peak memory only
            grows by at most one full-model gradient slice on master.

            P2: the pre-clip walk MUST use the same ``peer_weights`` and
            RDA rescale as the main walk so the clip_scale derived from
            the pre-clip norm is the right scale for the actually-applied
            gradient. Otherwise pre-clip norm reflects an unweighted /
            unrescaled aggregation and the main walk's clip is mis-tuned.

            FU1 (P6b): the TurboQuant branch is walked here too so the
            clip_scale covers both legacy and TurboQuant params. Without
            this, a fleet that mixes TurboQuant and legacy encoders would
            derive clip_scale from legacy-only norms and under-clip the
            full aggregated gradient. TurboQuant params skip the RDA
            norm-match rescale (no per-peer block_norms to match against)
            — the post-inverse-rotation gradient's L2 is used directly.

            Caller must guarantee ``src_sd is not None`` (the only gate
            for this function -- the auto-clip path needs the pre-clip L2
            even when ``max_grad_norm`` is None to seed the EMA).
            """
            assert src_sd is not None
            total_sq_dev = torch.zeros((), device=device, dtype=torch.float32)
            for _name, _p in model.named_parameters():
                _cname = canon_map.get(_name, _name)

                # FU1 TurboQuant branch. Mutually exclusive with the
                # legacy path per prepare_gradient_dict contract: a
                # TurboQuant miner does NOT populate idxs/vals for
                # the same param (and vice versa). Checked FIRST so a
                # TQ-encoded peer doesn't accidentally dead-end in the
                # legacy lookup when some OTHER peer shipped legacy
                # keys for the same param (mixed fleets during rollout).
                _tq_idxs = src_sd.get(_cname + "tq_idxs")
                if _tq_idxs is not None:
                    _tq_codes = src_sd.get(_cname + "tq_codes")
                    _tq_metas = src_sd.get(_cname + "tq_meta")
                    if _tq_codes is None or _tq_metas is None:
                        continue
                    if not isinstance(_tq_idxs, (list, tuple)):
                        _tq_idxs = [_tq_idxs]
                    if not isinstance(_tq_codes, (list, tuple)):
                        _tq_codes = [_tq_codes]
                    if not isinstance(_tq_metas, (list, tuple)):
                        _tq_metas = [_tq_metas]
                    if not (
                        len(_tq_idxs) == len(_tq_codes) == len(_tq_metas)
                    ) or len(_tq_idxs) == 0:
                        continue
                    _pw_local = _peer_weights_for_param(
                        _cname, len(_tq_metas), key_suffix="tq_idxs"
                    )
                    _full = hone.turboquant.batch_decompress_turboquant(
                        _p,
                        _tq_idxs,
                        _tq_codes,
                        _tq_metas,
                        peer_weights=_pw_local,
                    )
                    _full = _full.to(
                        dtype=_p.dtype, device=_p.device, non_blocking=True
                    )
                    if _full.shape != _p.shape:
                        _full = _full.view(_p.shape)
                    # No RDA rescale for TurboQuant: the per-peer block
                    # norms the legacy light-RDA relies on are a
                    # top-K-per-chunk artefact that TurboQuant's
                    # single-flat-rotation encoding does not carry.
                    # Stabilising the outer LR across varying M for
                    # TurboQuant is a separate workstream; for now this
                    # stays a structural no-op.
                    _norm = torch.linalg.vector_norm(
                        _full, ord=2, dtype=torch.float32
                    )
                    total_sq_dev.add_(_norm * _norm)
                    del _full, _norm
                    continue

                _idxs = src_sd.get(_cname + "idxs")
                _vals = src_sd.get(_cname + "vals")
                _qps = src_sd.get(_cname + "quant_params")
                if _idxs is None or _vals is None:
                    continue
                if not isinstance(_idxs, (list, tuple)):
                    _idxs = [_idxs]
                if not isinstance(_vals, (list, tuple)):
                    _vals = [_vals]
                _vals_f32 = compressor.maybe_dequantize_values(_vals, _qps, device)
                if not _vals_f32:
                    continue
                vals_f32_cache[_cname] = _vals_f32
                _idxs_dev = _idx_to_device(_idxs, device)
                _block_norms = torch.stack([torch.norm(v, p=2) for v in _vals_f32])
                _ref = torch.empty_like(_p, device=device, dtype=_p.dtype)
                _pw_local = _peer_weights_for_param(_cname, len(_vals_f32))
                _decompressed = compressor.batch_decompress(
                    _ref,
                    _idxs_dev,
                    _vals_f32,
                    xshapes[_cname],
                    totalks[_cname],
                    quantize_params=None,
                    block_norms=_block_norms,
                    normalise=False,
                    clip_norm=True,
                    peer_weights=_pw_local,
                )
                _full = transformer.decode(_decompressed, use_dct=use_dct)
                _full = _full.to(
                    dtype=_p.dtype, device=_p.device, non_blocking=True
                )
                if _full.shape != _p.shape:
                    _full = _full.view(_p.shape)
                # P2 RDA rescale (mirror of the main-walk math) so the
                # pre-clip L2 reflects the post-RDA dense gradient. Skip
                # for embedding params per Decoupled DiLoCo §D.2 Table 8;
                # also a no-op when the composition guard tripped.
                if rda_active and not _is_embedding_param(_name):
                    if _pw_local is not None:
                        _ws = sum(_pw_local)
                        if _ws > 0.0:
                            _wn = [w / _ws for w in _pw_local]
                            _target = float(
                                sum(
                                    w * float(_block_norms[i].item())
                                    for i, w in enumerate(_wn)
                                )
                            )
                        else:
                            _target = float(_block_norms.mean().item())
                    else:
                        _target = float(_block_norms.mean().item())
                    _merged_norm = float(
                        torch.linalg.vector_norm(
                            _full, ord=2, dtype=torch.float32
                        ).item()
                    )
                    if _merged_norm > 1e-12 and _target > 0.0:
                        _full.mul_(_target / _merged_norm)
                # Single-pass fp32-accumulated L2; no full-size fp32 copy
                # of ``_full`` is materialised.
                _norm = torch.linalg.vector_norm(_full, ord=2, dtype=torch.float32)
                total_sq_dev.add_(_norm * _norm)
                del _decompressed, _full, _norm, _ref, _block_norms, _idxs_dev
            return float(total_sq_dev.sqrt().item())

        # Auto-clip threshold algorithm (master rank only). When the caller
        # provides an ``auto_clip_state`` dict we tighten the static cap
        # ``max_grad_norm`` toward ``auto_clip_factor * ema(post_clip_norm)``
        # so each outer step is bounded relative to its recent neighbours
        # rather than to a fixed magnitude. Tracking the EMA on the
        # post-clip value (not pre-clip) prevents one bad spike from
        # permanently inflating the threshold:
        #
        #   effective_max  = min(max_grad_norm, auto_clip_factor * ema)
        #   clip_scale     = min(1.0, effective_max / pre_clip_norm)
        #   post_clip_norm = pre_clip_norm * clip_scale
        #   ema_next       = decay*ema + (1-decay)*post_clip_norm
        #
        # On the first call ``state["ema"]`` is unset; we bootstrap it
        # with the static ``max_grad_norm`` floor so the very first step
        # behaves exactly like the legacy code path. Subsequent steps
        # tighten as data flows in.
        if (
            on_src
            and src_sd is not None
            and max_grad_norm is not None
            and max_grad_norm > 0.0
        ):
            # Pass 1 of the 2-pass decode for outer-grad clipping. The
            # work is GPU-bound (dequant + IDCT + scatter_reduce per
            # param); use a CUDA event pair so the contributing kernels
            # are measured rather than just the dispatch latency.
            #
            # P4: ``stream=stream`` is mandatory on .record(): when
            # ``stream is not None`` the body runs on the secondary
            # outer-step stream, so the events MUST land on that
            # stream too — recording on the implicit default stream
            # would silently capture zero work (the kernels live on
            # ``stream``, not on default). When ``stream is None``,
            # ``Event.record(stream=None)`` is identical to ``record()``
            # — both fall back to the current stream.
            if use_cuda_timing:
                _pre_decode_start_ev = torch.cuda.Event(enable_timing=True)
                _pre_decode_end_ev = torch.cuda.Event(enable_timing=True)
                _pre_decode_start_ev.record(stream=stream)
            _pre_t0 = time.perf_counter()
            pre_clip_norm = _compute_pre_clip_norm()
            _pre_t1 = time.perf_counter()
            if use_cuda_timing:
                _pre_decode_end_ev.record(stream=stream)
                _pre_decode_end_ev.synchronize()
                outer_step_timings["decode_seconds"] += (
                    _pre_decode_start_ev.elapsed_time(_pre_decode_end_ev) / 1000.0
                )
            else:
                outer_step_timings["decode_seconds"] += _pre_t1 - _pre_t0
            if (
                isinstance(auto_clip_state, dict)
                and auto_clip_factor > 0.0
                and 0.0 < auto_clip_ema_decay < 1.0
            ):
                ema_prev = auto_clip_state.get("ema")
                if ema_prev is None or float(ema_prev) <= 0.0:
                    effective_max = float(max_grad_norm)
                else:
                    effective_max = min(
                        float(max_grad_norm),
                        float(auto_clip_factor) * float(ema_prev),
                    )
                clip_scale = min(
                    1.0, effective_max / max(pre_clip_norm, 1e-8)
                )
                post_clip_norm = pre_clip_norm * clip_scale
                if ema_prev is None or float(ema_prev) <= 0.0:
                    auto_ema_after = float(post_clip_norm)
                else:
                    auto_ema_after = (
                        float(auto_clip_ema_decay) * float(ema_prev)
                        + (1.0 - float(auto_clip_ema_decay)) * float(post_clip_norm)
                    )
                auto_clip_state["ema"] = auto_ema_after
            else:
                effective_max = float(max_grad_norm)
                clip_scale = min(
                    1.0, effective_max / max(pre_clip_norm, 1e-8)
                )
                auto_ema_after = 0.0
            if fingerprint is not None:
                fingerprint["pre_clip_norm"] = pre_clip_norm
                fingerprint["clip_scale"] = clip_scale
                fingerprint["effective_max"] = effective_max
                fingerprint["auto_ema"] = auto_ema_after

        # Stage every param's ``p.grad`` across the loop and apply ONE
        # foreach SGD step after the loop finishes (was: ``optimizer.step()``
        # per-param, called ~7100 times on MoE 8B-A1B). The peak-memory
        # delta is one full-model gradient slice per rank (1/world_size
        # under FSDP2); the win is 5-15s of Python-loop overhead removed
        # per outer step.

        # P3A (2026-05-03): batched per-param ``has_update`` broadcast.
        # Pre-P3A the inline ``_bcast_flag(has_update)`` issued one
        # ``dist.broadcast(int32)`` per param for the ~7100 trainable
        # tensors of MoE 8B-A1B every outer step -- ~7100 NCCL launches
        # per step at ~100us launch + serialization stalls = 1-1.5s of
        # pure dispatch overhead. We hoist the master lookup into a
        # pre-pass that populates per-param ``has_update`` + payload
        # references, then ship every flag in a single int32 tensor
        # broadcast. The main loop below then dispatches on the
        # pre-computed flag (no inline NCCL) and uses the cached
        # payload tuple instead of re-reading from ``src_sd``.
        #
        # Memory cost is essentially zero: ``per_param_payload`` stores
        # tuples of references into ``src_sd`` (already allocated) plus
        # the dequantised ``vals_f32`` list (which previously also
        # lived in ``vals_f32_cache`` for the clipping pre-pass --
        # we're just holding it slightly longer). The full dense
        # ``full_grad_src`` is still built one-at-a-time and freed
        # after distribute_tensor / dist.broadcast, just like pre-P3A.
        named_params = list(model.named_parameters())
        n_params = len(named_params)

        per_param_has_update: list[int] = [0] * n_params
        per_param_payload: list[tuple | None] = [None] * n_params
        per_param_is_tq: list[bool] = [False] * n_params

        # Pass 1 (master only): per-param payload lookup. Non-master
        # ranks idle here -- they get the flags via the single
        # broadcast below. Folded into ``decode_seconds`` because pre-
        # P3A the equivalent lookup also lived inside the
        # decode-timed span of each per-param iteration.
        _p3a_lookup_t0 = time.perf_counter()
        if on_src and src_sd is not None:
            for _i, (_name, _p_unused) in enumerate(named_params):
                _cname = canon_map.get(_name, _name)
                if _cname is None:
                    continue

                # FU1 TurboQuant check FIRST -- mirrors pre-P3A inline
                # logic at the same call site. A TurboQuant-encoded
                # param has no idxs/vals/quant_params entries so the
                # legacy branch would silently drop the param
                # regardless; explicit dispatch is clearer and keeps
                # the ``has_update`` flag correct.
                _tq_idxs = src_sd.get(_cname + "tq_idxs")
                if _tq_idxs is not None:
                    _tq_codes = src_sd.get(_cname + "tq_codes")
                    _tq_metas = src_sd.get(_cname + "tq_meta")
                    if _tq_codes is not None and _tq_metas is not None:
                        if not isinstance(_tq_idxs, (list, tuple)):
                            _tq_idxs = [_tq_idxs]
                        if not isinstance(_tq_codes, (list, tuple)):
                            _tq_codes = [_tq_codes]
                        if not isinstance(_tq_metas, (list, tuple)):
                            _tq_metas = [_tq_metas]
                        if (
                            len(_tq_idxs) == len(_tq_codes) == len(_tq_metas)
                            and len(_tq_idxs) > 0
                        ):
                            per_param_payload[_i] = (
                                _tq_idxs,
                                _tq_codes,
                                _tq_metas,
                            )
                            per_param_is_tq[_i] = True
                            per_param_has_update[_i] = 1
                            continue

                # Legacy top-K branch
                _idxs = src_sd.get(_cname + "idxs")
                _vals = src_sd.get(_cname + "vals")
                _qps = src_sd.get(_cname + "quant_params")
                if _idxs is not None and _vals is not None:
                    if not isinstance(_idxs, (list, tuple)):
                        _idxs = [_idxs]
                    if not isinstance(_vals, (list, tuple)):
                        _vals = [_vals]
                    # Reuse the dequantised list from the clipping
                    # pre-pass if it ran; otherwise dequantise here.
                    # Saves one dequant per param when clipping is on.
                    _vals_f32 = vals_f32_cache.pop(_cname, None)
                    if _vals_f32 is None:
                        _vals_f32 = compressor.maybe_dequantize_values(
                            _vals, _qps, device
                        )
                    if _vals_f32:
                        _idxs_dev = _idx_to_device(_idxs, device)
                        per_param_payload[_i] = (_idxs_dev, _vals_f32)
                        per_param_has_update[_i] = 1
        outer_step_timings["decode_seconds"] += (
            time.perf_counter() - _p3a_lookup_t0
        )

        # Single batched broadcast of every per-param ``has_update``
        # flag in one int32 tensor. Mirrors the pre-P3A
        # ``_bcast_flag(has_update)`` semantics (no-op on single GPU,
        # one ``dist.broadcast`` in DDP mode) but pays the launch cost
        # exactly once instead of n_params times.
        _p3a_bcast_t0 = time.perf_counter()
        if ddp:
            _flags_tensor = torch.tensor(
                per_param_has_update, device=device, dtype=torch.int32
            )
            dist.broadcast(_flags_tensor, src_rank)
            per_param_has_update = _flags_tensor.cpu().tolist()
        outer_step_timings["decode_seconds"] += (
            time.perf_counter() - _p3a_bcast_t0
        )

        for _i, (name, p) in enumerate(named_params):
            cname = canon_map.get(name, name)

            # ---- has_update + payload come from the P3A pre-pass above ----
            # The inline ``_bcast_flag(has_update)`` cross-rank sync
            # was hoisted out of this loop; we just read the
            # pre-computed flag here. ``payload`` and ``is_tq_payload``
            # are populated only on master (the rank that ran the
            # pre-pass lookup); non-master ranks receive the flag and
            # then participate in the per-param ``distribute_tensor``
            # / ``dist.broadcast`` collective in the merge phase.
            _decode_t0 = time.perf_counter()

            has_update = per_param_has_update[_i]
            if has_update == 0:
                # Nothing to apply for this param; nothing more to
                # accumulate into decode_seconds either (the lookup +
                # broadcast time was already folded in pre-loop).
                continue

            payload = per_param_payload[_i] if on_src else None
            is_tq_payload = per_param_is_tq[_i] if on_src else False

            full_grad_src = torch.empty(1)
            decompressed = None
            block_norms = None

            # ------- build the full dense grad on the source rank only -------
            # P2 stash: per-peer L2 norms + per-peer weights snapshot
            # captured BEFORE the decode `finally` clears the intermediates,
            # so the merge-phase RDA rescale can read them without holding
            # the dense per-peer vals tensors in memory across the per-param
            # loop boundary.
            rda_target_norm: float | None = None
            if on_src:
                try:
                    if is_tq_payload:
                        # FU1 TurboQuant branch. The codec returns a dense
                        # tensor already shaped + dtype-cast to match
                        # ``p``, and already inverse-rotated, so there's
                        # no transformer.decode / DCT step here and no
                        # per-peer block_norms to accumulate (the per-peer
                        # gradient reconstruction does not expose chunk-
                        # level L2s the way the top-K path does). RDA is
                        # a structural no-op on this branch; see the
                        # rationale inline in ``_compute_pre_clip_norm``.
                        tq_idxs_local, tq_codes_local, tq_metas_local = (
                            payload  # type: ignore[misc]
                        )
                        pw_local = _peer_weights_for_param(
                            cname, len(tq_metas_local), key_suffix="tq_idxs"
                        )
                        full_grad_src = hone.turboquant.batch_decompress_turboquant(
                            p,
                            tq_idxs_local,
                            tq_codes_local,
                            tq_metas_local,
                            peer_weights=pw_local,
                        )
                        full_grad_src = full_grad_src.to(
                            dtype=p.dtype, device=p.device, non_blocking=True
                        )
                        if full_grad_src.shape != p.shape:
                            full_grad_src = full_grad_src.view(p.shape)
                        if clip_scale != 1.0:
                            full_grad_src.mul_(clip_scale)
                    else:
                        idxs_dev, vals_f32 = payload  # type: ignore[misc]
                        # Per-block norms for stats/optional clipping inside batch_decompress
                        block_norms = torch.stack(
                            [torch.norm(v, p=2) for v in vals_f32]
                        )

                        # stats
                        med = float(torch.median(block_norms).item())
                        min_median_norm = min(min_median_norm, med)
                        max_median_norm = max(max_median_norm, med)

                        # P2 token-weighted aggregation: per-param weight slice
                        # built from validator ``peer_weights`` dict +
                        # ``uids_per_param``. ``None`` here = uniform mean
                        # (legacy behaviour); see ``_peer_weights_for_param``
                        # for fallback semantics.
                        pw_local = _peer_weights_for_param(cname, len(vals_f32))

                        # P2 RDA target: the ``peer_weights``-weighted mean of
                        # per-peer L2 norms (or unweighted mean when
                        # ``peer_weights`` is None). Light-RDA rescales the
                        # post-IDCT merged gradient to this target norm
                        # downstream. We compute this BEFORE the rescale so we
                        # don't have to hold ``block_norms`` past the
                        # ``finally`` clear.
                        if rda_active and not _is_embedding_param(name):
                            if pw_local is not None:
                                ws = sum(pw_local)
                                if ws > 0.0:
                                    wn = [w / ws for w in pw_local]
                                    rda_target_norm = float(
                                        sum(
                                            w * float(block_norms[i].item())
                                            for i, w in enumerate(wn)
                                        )
                                    )
                                else:
                                    rda_target_norm = float(
                                        block_norms.mean().item()
                                    )
                            else:
                                rda_target_norm = float(block_norms.mean().item())

                        # Use empty_like to avoid copying the param; just provide dtype/device/shape
                        ref = torch.empty_like(p, device=device, dtype=p.dtype)
                        decompressed = compressor.batch_decompress(
                            ref,
                            idxs_dev,
                            vals_f32,
                            xshapes[cname],
                            totalks[cname],
                            quantize_params=None,
                            block_norms=block_norms,
                            normalise=False,
                            clip_norm=True,
                            peer_weights=pw_local,
                        )

                        full_grad_src = transformer.decode(
                            decompressed, use_dct=use_dct
                        )
                        # Single conversion to target dtype+device to avoid extra temporaries
                        full_grad_src = full_grad_src.to(
                            dtype=p.dtype, device=p.device, non_blocking=True
                        )

                        # Stacked MoE weights are stored as 3D ``(E, D, ffn)``
                        # but ``prepare_gradient_dict`` collapsed the leading
                        # expert dim into rows before encoding (the codec only
                        # speaks 1D / 2D). ``decode`` therefore returns a 2D
                        # tensor; reshape it back to ``p.shape`` so the
                        # subsequent ``distribute_tensor`` / ``p.grad =`` paths
                        # see a shape-compatible gradient.
                        if full_grad_src.shape != p.shape:
                            full_grad_src = full_grad_src.view(p.shape)

                        # Apply outer-grad clipping scale (computed in the
                        # pre-pass above). When ``clip_scale == 1.0`` this is a
                        # no-op the autograd graph won't even materialise.
                        if clip_scale != 1.0:
                            full_grad_src.mul_(clip_scale)
                finally:
                    # Free intermediate pieces ASAP (existence-guarded)
                    try:
                        del decompressed
                    except UnboundLocalError:
                        pass
                    # vals/idxs/qps live in src_sd; only local views should be dropped
                    try:
                        del vals_f32, idxs_dev, block_norms, ref
                    except UnboundLocalError:
                        pass

            # End of decode phase for this param; everything below is the
            # cross-rank distribute / broadcast that lands the merged
            # gradient into ``p.grad`` (the "merge" phase). Today this
            # is just Avg via ``distribute_tensor`` / ``dist.broadcast``;
            # P2 RDA norm-match also fires inside this span before the
            # dense tensor is broadcast / scattered, so the rescale is
            # measured against ``merge_seconds`` exactly like the eventual
            # full per-peer RDA in P2.2 will be.
            outer_step_timings["decode_seconds"] += time.perf_counter() - _decode_t0
            _merge_t0 = time.perf_counter()

            # P2 light-RDA: rescale the merged dense gradient on master to
            # the ``peer_weights``-weighted mean of per-peer L2 norms. This
            # is the reduced-tensor approximation of full RDA — we don't
            # have per-peer dense vectors after the position-wise
            # ``scatter_reduce`` so we cannot do per-peer unit-direction
            # averaging. Full per-peer RDA is O(P × d) memory and is
            # deferred until P2.2; the norm-match alone matches the leading
            # benefit (stabilising outer LR across varying M) per Table 8
            # of the paper. Fingerprint stats are recomputed AFTER this
            # rescale so the global L2 in the dashboard reflects the
            # actually-applied gradient magnitude.
            if on_src and rda_target_norm is not None and full_grad_src is not None:
                merged_norm = float(
                    torch.linalg.vector_norm(
                        full_grad_src, ord=2, dtype=torch.float32
                    ).item()
                )
                if merged_norm > 1e-12 and rda_target_norm > 0.0:
                    full_grad_src.mul_(rda_target_norm / merged_norm)

            # Accumulate fingerprint statistics POST-clip-scale + POST-RDA
            # so the recorded ``global_l2_norm`` matches the magnitude that
            # actually lands in ``p.grad``. Default-off RDA preserves
            # today's exact stat values for callers that don't enable it.
            if on_src and fingerprint is not None and full_grad_src is not None:
                param_norm = torch.norm(full_grad_src, p=2).item()
                fingerprint["param_norms"][cname] = param_norm
                fingerprint["total_norm_sq"] += param_norm**2
                fingerprint["total_elements"] += full_grad_src.numel()
                fingerprint["param_means"][cname] = full_grad_src.mean().item()

            # ------- distribute/broadcast directly into p.grad -------
            if isinstance(p, DT):
                # DTensor param: scatter shards from master
                src_tensor = (
                    full_grad_src
                    if on_src
                    else torch.empty(p.shape, device=p.device, dtype=p.dtype)
                )
                new_grad = distribute_tensor(
                    src_tensor,
                    device_mesh=p.device_mesh,
                    placements=p.placements,
                    src_data_rank=src_rank,
                )
                # master no longer needs the full dense grad
                if on_src:
                    del full_grad_src
                    full_grad_src = None

                # quick sanity (view, no extra big alloc)
                local_view = new_grad.to_local()
                if not torch.isfinite(local_view).all():
                    del new_grad, local_view
                    outer_step_timings["merge_seconds"] += (
                        time.perf_counter() - _merge_t0
                    )
                    continue

                p.grad = new_grad  # DTensor grad
                del new_grad, local_view

            else:
                # Replicated param: broadcast dense grad once.
                if ddp:
                    if on_src:
                        # Broadcast from the source tensor; then reuse it as grad
                        dist.broadcast(full_grad_src, src_rank)  # type: ignore[arg-type]
                        p.grad = full_grad_src
                        full_grad_src = None
                    else:
                        # Receive directly into p.grad to avoid an extra buffer
                        p.grad = torch.empty_like(p, device=p.device, dtype=p.dtype)
                        dist.broadcast(p.grad, src_rank)  # type: ignore[arg-type]
                else:
                    # Single process: just use the built tensor
                    p.grad = full_grad_src
                    full_grad_src = None

                if p.grad is not None and not torch.isfinite(p.grad).all():  # type: ignore[arg-type]
                    p.grad = None
                    outer_step_timings["merge_seconds"] += (
                        time.perf_counter() - _merge_t0
                    )
                    continue

            outer_step_timings["merge_seconds"] += time.perf_counter() - _merge_t0

        # ---- foreach SGD apply over every staged (p, p.grad) pair ----
        # ``optimizer.step()`` filters out params whose ``grad is None``
        # (those that had no peer update or failed the finite check above)
        # and routes the rest through the SGD foreach kernel because the
        # optimizer was constructed with ``foreach=True`` in
        # ``Trainer.init_optimizers_schedulers``. Wrapping with a CUDA
        # event pair gives a GPU-precise apply timing rather than just the
        # dispatch latency. P4: explicit ``stream=stream`` so the events
        # land on the secondary stream when parallel-outer-step is on
        # (see _pre_decode_start_ev block above for the full rationale).
        if use_cuda_timing:
            _apply_start_ev = torch.cuda.Event(enable_timing=True)
            _apply_end_ev = torch.cuda.Event(enable_timing=True)
            _apply_start_ev.record(stream=stream)
        _apply_t0 = time.perf_counter()
        optimizer.step()
        _apply_t1 = time.perf_counter()
        if use_cuda_timing:
            _apply_end_ev.record(stream=stream)
            _apply_end_ev.synchronize()
            outer_step_timings["apply_seconds"] = (
                _apply_start_ev.elapsed_time(_apply_end_ev) / 1000.0
            )
        else:
            outer_step_timings["apply_seconds"] = _apply_t1 - _apply_t0

        # optional W&B (master only)
        if (
            on_src
            and wandb_run is not None
            and global_step is not None
            and max_median_norm > float("-inf")
        ):
            wandb_run.log(
                {
                    "compress/min_median_block_norm": min_median_norm,
                    "compress/max_median_block_norm": max_median_norm,
                },
                step=global_step,
            )

        # Extra safety: ensure no grads are left allocated
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute final fingerprint (master rank only) and return phase
        # timings on every rank. P-1b consumes
        # ``outer_step_timings`` from the validator main loop and plumbs
        # the ``decode_seconds`` / ``merge_seconds`` / ``apply_seconds``
        # contract through ``DashboardReporter.report_window``.
        if on_src and fingerprint is not None:
            fingerprint["global_l2_norm"] = math.sqrt(fingerprint["total_norm_sq"])
            return fingerprint, outer_step_timings
        return None, outer_step_timings


async def update_peers(instance: NeuronT, window: int, peer_start: float) -> None:
    # Check if peers list is empty and fetch previous list if needed
    if len(instance.comms.peers) == 0:
        hone.logger.info(
            "Current peers list is empty, attempting to fetch previous peer list"
        )
        result = await instance.comms.get_peer_list(fetch_previous=True)
        if result is not None:
            prev_peers, prev_reserve, prev_update_window = result
            hone.logger.info(
                f"Got previous peer list with {len(prev_peers)} peers "
                f"and update window {prev_update_window}"
            )
            instance.comms.peers = prev_peers
            instance.comms.reserve_peers = prev_reserve

            # Don't set next_peers here, as we want the normal update process to continue
        else:
            hone.logger.warning(
                "Failed to fetch previous peer list, continuing with empty peers"
            )

    # Get next peers
    if (
        instance.next_peers is None  # next peers are not fetched yet
        and instance.peers_update_window  # they should be on bucket by now
        + instance.hparams.peer_replacement_frequency
        - window
        <= instance.hparams.peer_list_window_margin
    ):
        result = await instance.comms.get_peer_list()
        if result is None:
            hone.logger.info("Unable to get peer list from bucket")
        else:
            next_peers, reserve_peers, peers_update_window = result
            hone.logger.info(
                f"Got peer list {next_peers} and update window "
                f"{peers_update_window} from bucket"
            )
            if (
                instance.peers_update_window is None
                or peers_update_window > instance.peers_update_window
            ):
                instance.next_peers = next_peers
                instance.next_reserve_peers = reserve_peers
                instance.peers_update_window = peers_update_window
                hone.logger.info("This list is new, updating next_peers")

    # Update peers, if it's time
    if instance.next_peers is not None and window >= instance.peers_update_window:
        # ── atomic switch ─────────────────────────────────────────────
        instance.comms.peers = instance.next_peers
        instance.comms.reserve_peers = (
            instance.next_reserve_peers
            if instance.next_reserve_peers is not None
            else []
        )
        late_text = (
            f"{window - instance.peers_update_window} windows late"
            if window - instance.peers_update_window > 0
            else "on time"
        )
        hone.logger.info(
            f"{hone.P(window, hone.T() - peer_start)} Updated peers "
            f"{late_text} - gather:{len(instance.comms.peers)}, "
            f"reserve:{len(instance.comms.reserve_peers)}. Next update "
            f"expected on step window "
            f"{instance.peers_update_window + instance.hparams.peer_list_window_margin}"
        )
        instance.next_peers = None
    else:
        reason = (
            "next peers are not defined yet"
            if instance.next_peers is None
            else f"sync window is {window} and peers update window "
            f"is {instance.peers_update_window}"
        )
        hone.logger.info(f"Not time to replace peers: {reason}")


async def load_checkpoint_with_fallback(
    instance: NeuronT,
) -> tuple[bool, int, int, bool]:
    """
    Load checkpoint with fallback logic.

    1. First try loading from current version
    2. If not found, try bootstrap version if configured
    3. Return checkpoint status and metadata

    Returns:
        tuple of (checkpoint_ok, checkpoint_window, global_step, from_bootstrap)
    """
    ckpt_ok = False
    ckpt_sync_win = 0
    ckpt_global_step = 0
    from_bootstrap = False

    # First check if current version has any checkpoints
    latest_current_window = await instance.ckpt._discover_latest(
        prefer_highest_staked=True
    )

    if latest_current_window is not None:
        # Current version checkpoint exists, load it
        res = await instance.ckpt.download_and_load(
            model=instance.model,
            window=latest_current_window,
            shared_fs=True,
            process_group=None,
            prefer_highest_staked=True,
        )
        if res is not None:
            ckpt_ok = True
            ckpt_sync_win, ckpt_global_step = res
            instance.model_initialized = True  # Model now has real weights
            hone.logger.info(
                f"Loaded current version checkpoint (window={ckpt_sync_win}, "
                f"global_step={ckpt_global_step})"
            )

    # If no current version checkpoint and bootstrap is configured, try that
    if not ckpt_ok and instance.bootstrap_version:
        hone.logger.info(
            f"No current version checkpoint found, trying bootstrap version "
            f"{instance.bootstrap_version}"
        )
        # Try specific window if configured, otherwise latest
        bootstrap_window = getattr(instance.hparams, "checkpoint_init_window", None)
        bootstrap_ckpt = hone.DCPCheckpointer(
            instance.comms,
            uid=instance.uid,
            version=instance.bootstrap_version,
            repo_root=".",
        )

        # If no specific window configured, discover latest in bootstrap version
        if bootstrap_window is None:
            bootstrap_window = await bootstrap_ckpt._discover_latest(
                prefer_highest_staked=True
            )

        if bootstrap_window is not None:
            res = await bootstrap_ckpt.download_and_load(
                model=instance.model,
                window=bootstrap_window,
                shared_fs=True,
                process_group=None,
                prefer_highest_staked=True,
            )
            if res is not None:
                ckpt_ok = True
                from_bootstrap = True
                ckpt_sync_win, ckpt_global_step = res
                instance.model_initialized = True  # Model now has real weights
                hone.logger.info(
                    f"Loaded bootstrap checkpoint (version={instance.bootstrap_version}, "
                    f"window={ckpt_sync_win}, global_step={ckpt_global_step})"
                )

    # Handle global_step calculation if needed
    if ckpt_ok and ckpt_global_step == -1:
        if from_bootstrap:
            # For bootstrap checkpoints, try to get the start_window from that version
            bootstrap_start_window = await instance.comms.get_start_window(
                version=instance.bootstrap_version
            )
            if bootstrap_start_window is not None:
                ckpt_global_step = ckpt_sync_win - bootstrap_start_window
                hone.logger.info(
                    f"Bootstrap checkpoint has no global_step, calculated as {ckpt_global_step} "
                    f"(window {ckpt_sync_win} - bootstrap start {bootstrap_start_window})"
                )
            else:
                # Fallback if we can't get bootstrap start_window
                ckpt_global_step = 0
                hone.logger.info(
                    f"Bootstrap checkpoint has no global_step and couldn't fetch bootstrap start_window, "
                    f"setting to 0 (will be corrected during catch-up)"
                )
        else:
            # For current version checkpoints, calculate from window difference
            ckpt_global_step = ckpt_sync_win - instance.start_window
            hone.logger.info(
                f"No global_step in checkpoint, calculated as {ckpt_global_step} "
                f"(window {ckpt_sync_win} - start {instance.start_window})"
            )

    if ckpt_ok:
        instance.global_step = ckpt_global_step

    return ckpt_ok, ckpt_sync_win, ckpt_global_step, from_bootstrap


async def handle_checkpoint_catchup(
    instance: NeuronT,
    ckpt_ok: bool,
    ckpt_sync_win: int,
    ckpt_global_step: int,
    from_bootstrap: bool,
    aggregator_device: str | None = None,
) -> None:
    """
    Handle catch-up logic after checkpoint loading and replay scheduler steps.

    Args:
        instance: Miner or Validator instance
        ckpt_ok: Whether a checkpoint was successfully loaded
        ckpt_sync_win: Window number from checkpoint
        ckpt_global_step: Global step from checkpoint
        from_bootstrap: Whether checkpoint was from bootstrap version
        aggregator_device: which device to load aggregation results to
    """
    # Check for anneal mode - handles scheduler replay differently
    anneal_config = getattr(instance.hparams, "anneal_mode", {})
    anneal_enabled = anneal_config.get("enabled", False)

    if anneal_enabled:
        # Anneal mode: Calculate progress from start_global_step, not from 0
        anneal_start_global_step = anneal_config.get("start_global_step", 0)
        warmup_inner_steps = anneal_config.get("warmup_inner_steps", 100)

        # Calculate how far into anneal based on actual outer steps taken
        anneal_outer_steps = max(0, ckpt_global_step - anneal_start_global_step)
        anneal_inner_steps = anneal_outer_steps * instance.hparams.inner_steps

        # Set warmup tracking for anneal mode
        instance.warmup_inner_steps = warmup_inner_steps
        instance.warmup_steps_taken = min(anneal_inner_steps, warmup_inner_steps)

        # Replay anneal scheduler steps (NOT the old pre-anneal steps)
        instance.inner_scheduler_step_count = 0
        if (
            anneal_inner_steps > 0
            and getattr(instance, "inner_scheduler", None) is not None
        ):
            for _ in range(anneal_inner_steps):
                instance.inner_scheduler.step()
                instance.inner_scheduler_step_count += 1

        hone.logger.info(
            f"Anneal mode: start_global_step={anneal_start_global_step}, "
            f"ckpt_global_step={ckpt_global_step}, "
            f"replayed {anneal_inner_steps} anneal scheduler steps"
        )

        # Still need to do catchup for gradients
        if not ckpt_ok:
            hone.logger.info("No checkpoint found, will catch up from start_window")
            await catchup_with_aggregation_server(
                instance, instance.start_window, aggregator_device=aggregator_device
            )
        elif ckpt_sync_win < instance.current_window:
            catch_up_start = max(ckpt_sync_win, instance.start_window)
            hone.logger.info(
                f"Checkpoint at window {ckpt_sync_win} is behind current {instance.current_window}, "
                f"catching up from {catch_up_start}"
            )
            await catchup_with_aggregation_server(
                instance, catch_up_start, aggregator_device=aggregator_device
            )
        else:
            hone.logger.info(
                f"Checkpoint at window {ckpt_sync_win} is up to date with current window "
                f"{instance.current_window}"
            )
        return

    # Normal (non-anneal) mode below
    # Determine scheduler config and warmup settings
    optimizer_cfg = getattr(instance.hparams, "optimizer", {})
    opt_type = optimizer_cfg.get("type", "adamw").lower()
    opt_cfg = optimizer_cfg.get(opt_type, {})
    scheduler_cfg = opt_cfg.get("scheduler", {})

    default_warmup_inner = scheduler_cfg.get(
        "warmup_inner_steps", getattr(instance, "warmup_inner_steps", 0)
    )
    startup_warmup_inner = scheduler_cfg.get(
        "initial_warmup_inner_steps", default_warmup_inner
    )

    # Set warmup length:
    # - If resuming from bootstrap, use the longer startup warmup.
    # - If resuming from a regular checkpoint, use the default.
    # - If global_step is 0, leave as-is; the scheduler's own warmup covers this.
    if ckpt_global_step == 0:
        hone.logger.info("Global step is 0; leaving warmup settings unchanged.")
    elif from_bootstrap:
        instance.warmup_inner_steps = startup_warmup_inner
        hone.logger.info(
            f"Applying startup warmup_inner_steps={startup_warmup_inner} (bootstrap resume)"
        )
        instance.warmup_steps_taken = 0
    else:
        instance.warmup_inner_steps = default_warmup_inner
        hone.logger.info(
            f"Applying resumed warmup_inner_steps={default_warmup_inner} (checkpoint resume)"
        )
        instance.warmup_steps_taken = 0

    # Decide catch-up windows and run catch-up on ALL ranks
    # When loading from bootstrap, we always need to catch up from start_window
    # to ensure we're using current version's gradients
    if not ckpt_ok:
        # No checkpoint found, catch up from start_window
        hone.logger.info("No checkpoint found, will catch up from start_window")
        await catchup_with_aggregation_server(
            instance, instance.start_window, aggregator_device=aggregator_device
        )
    elif from_bootstrap:
        # Loading from bootstrap, catch up from start_window with current version gradients
        hone.logger.info(
            f"Loaded bootstrap checkpoint, catching up from start_window "
            f"{instance.start_window} to {instance.current_window}"
        )
        await catchup_with_aggregation_server(
            instance, instance.start_window, aggregator_device=aggregator_device
        )
    elif ckpt_sync_win < instance.current_window:
        # Current version checkpoint is behind, catch up from checkpoint window
        catch_up_start = max(ckpt_sync_win, instance.start_window)
        hone.logger.info(
            f"Checkpoint at window {ckpt_sync_win} is behind current {instance.current_window}, "
            f"catching up from {catch_up_start}"
        )
        await catchup_with_aggregation_server(
            instance, catch_up_start, aggregator_device=aggregator_device
        )
    else:
        hone.logger.info(
            f"Checkpoint at window {ckpt_sync_win} is up to date with current window "
            f"{instance.current_window}"
        )

    # Replay scheduler steps based on windows completed from checkpoint
    # ckpt_global_step tracks windows, scheduler needs inner_steps per window
    total_inner_steps = ckpt_global_step * instance.hparams.inner_steps

    # Apply configurable rewind before replaying scheduler to give slack on restarts
    rewind_inner_steps = scheduler_cfg.get("replay_rewind_inner_steps", 0)
    if rewind_inner_steps > 0:
        total_inner_steps = max(total_inner_steps - rewind_inner_steps, 0)
        hone.logger.info(
            f"Rewinding scheduler replay by {rewind_inner_steps} inner steps; "
            f"{total_inner_steps} steps remain to replay"
        )

    if total_inner_steps > 0 and getattr(instance, "inner_scheduler", None) is not None:
        for _ in range(total_inner_steps):
            # Respect flatten window during replay
            if not instance.should_skip_scheduler_step():
                instance.inner_scheduler.step()
            instance.inner_scheduler_step_count += 1
        hone.logger.info(
            f"Replayed {total_inner_steps} scheduler steps (checkpoint global_step="
            f"{ckpt_global_step} * {instance.hparams.inner_steps} inner_steps)"
        )


def _warn_missing_compose_params(
    instance: "NeuronT",
    gather_ns: SimpleNamespace,
    window: int,
) -> None:
    """FU2 post-compose sanity check.

    In fragmented-upload mode (``fragmented_uploads: true``) the
    aggregator blob at ``aggregator-{w}-v{V}.pt`` is the disjoint-key
    union of N per-fragment contributions assembled by
    ``Comms._gather_fragmented`` from the leader's gather. If a single
    fragment failed to upload or was dropped server-side, that
    fragment's params silently disappear from the composed dict and
    ``outer_step`` leaves them unchanged for this window (it iterates
    ``model.named_parameters()`` and ``continue``s when ``cname+"idxs"``
    is missing — see the main loop in ``outer_step``). Per FU2 spec
    item #4: walk every grad-bearing local param, WARN if neither
    ``cname+"idxs"`` (legacy compressed payload) nor ``cname+"tq_idxs"``
    (TurboQuant payload, default-OFF until P6b lands) is present in the
    composed dict, and survive — never raise. The catchup loop applies
    the partial update on the surviving keys; missing ones stay at
    their pre-step values.

    Master-only: only rank 0 holds the fetched/composed payload
    (non-master ranks receive broadcasts from ``outer_step``).
    """
    sd_obj = getattr(gather_ns, "state_dict", None)
    if sd_obj is None:
        return
    keys: set[str] = (
        set(vars(sd_obj).keys())
        if isinstance(sd_obj, SimpleNamespace)
        else set(sd_obj.keys())
    )
    inner_model = (
        instance.model.module
        if isinstance(instance.model, torch.nn.parallel.DistributedDataParallel)
        else instance.model
    )
    canon_map = canonical_param_names(inner_model)
    missing: list[str] = []
    for name, p in inner_model.named_parameters():
        if not p.requires_grad:
            continue
        cname = canon_map.get(name, name)
        if (cname + "idxs") in keys or (cname + "tq_idxs") in keys:
            continue
        missing.append(cname)
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        hone.logger.warning(
            f"[catchup w={window}] composed state_dict is missing "
            f"{len(missing)} param contribution(s) — outer_step will "
            f"leave these params unchanged for this window. Likely a "
            f"dropped fragment in fragmented-upload mode. First few: "
            f"{preview}{more}"
        )


async def catchup_with_aggregation_server(
    instance: NeuronT,
    checkpoint_current_window: int,
    aggregator_device: str | None = None,
) -> None:
    """
    Synchronise the local model with the chain with memory optimizations.

    For every window between the checkpoint and the current chain head:

    1. **Primary path** – download the pre-computed `aggregated_gradients`
       object uploaded by the *leader* validator and apply it via
       `hone.neurons.outer_step`.

    2. **Fallback for the final window only** – if the leader has not yet
       published an aggregator object for `target_window - 1`, perform a live
       `instance.comms.gather( ..., key="gradient", ... )` against the current
       peer-set and apply those gradients instead.

    After each application we advance the inner LR scheduler, aggressively clear
    memory including CUDA cache and CPU memory via garbage collection.

    The loop exits when `start_w` has caught up with `instance.current_window`
    (taking into account that the chain head may advance while we are replaying).
    """
    hone.logger.info(
        "Starting catch‑up using aggregated_gradients with memory optimization..."
    )
    assert instance.start_window is not None

    # Use provided device or default to instance's device
    catchup_device = (
        aggregator_device if aggregator_device is not None else instance.config.device
    )

    def log_memory_usage(prefix: str):
        """Log current memory usage statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            hone.logger.info(
                f"{prefix} - GPU Memory: Allocated={allocated:.2f}GB, "
                f"Reserved={reserved:.2f}GB, Max={max_memory:.2f}GB"
            )

    leader_uid: int = instance.comms.metagraph.S.argmax().item()

    start_w = checkpoint_current_window + 1
    target_w = instance.current_window
    hone.logger.info(f"Replaying windows {start_w} ... {target_w - 1}")

    # Log initial memory state
    log_memory_usage("Initial memory state")

    # Verify checkpoint loaded correctly before applying any gradients
    if checkpoint_current_window > 0 and instance.is_master:
        hone.logger.info(
            f"Verifying checkpoint state at window {checkpoint_current_window}"
        )
        debug_fetch = await instance.comms.get(
            uid=str(leader_uid),
            window=checkpoint_current_window,
            key="debug",
            local=False,
            stale_retention=10,
        )

        if debug_fetch.success and isinstance(debug_fetch.data, dict):
            debug_dict = debug_fetch.data  # validator's payload

            cmp = await compare_model_with_debug_dict(
                instance.model,
                debug_dict,
                param_avg_change={},  # Empty since we haven't started tracking yet
                learning_rate=instance.hparams.learning_rate,
            )
            if cmp["success"]:
                hone.logger.info(
                    f"✓ Checkpoint verification: model matches window {checkpoint_current_window} "
                    f"(l2_norm={cmp['l2_norm']:.4f}, avg_steps_behind={cmp['avg_steps_behind']:.3f})"
                )
                if cmp["l2_norm"] > 0.1:  # Threshold for acceptable difference
                    hone.logger.warning(
                        f"⚠️ Large L2 norm difference detected: {cmp['l2_norm']:.4f}. "
                        f"Checkpoint may not have loaded correctly."
                    )
            else:
                hone.logger.warning(
                    f"⚠️ Could not verify checkpoint state for window {checkpoint_current_window}"
                )
        else:
            hone.logger.info(
                f"No debug dict available for window {checkpoint_current_window}, skipping verification"
            )

    prev_param_state: dict[str, torch.Tensor] = {}
    param_avg_change: dict[str, torch.Tensor] = {}
    alpha: float = 0.20
    slice_idx = slice(0, 2)

    while start_w < target_w:
        hone.logger.info(f"  • window {start_w}")

        # ------------------------------------------------------------------
        # 1) Fetch the aggregated object dumped by the leader validator.
        # ------------------------------------------------------------------
        if instance.is_master:
            # Clear memory before fetching to maximize available space
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            fetch = await instance.comms.get(
                uid=str(leader_uid),
                window=start_w,
                key="aggregator",
                timeout=60,
                local=False,
                stale_retention=10,
                map_location=catchup_device,
            )

            # ── A. aggregated object exists → normal path ────────────────────
            if fetch.success and fetch.data is not None and "state_dict" in fetch.data:
                payload = fetch.data

                # ------------------------------------------------------------------
                # Re‑create the SimpleNamespace expected by `outer_step`.
                # ------------------------------------------------------------------
                gather_ns = SimpleNamespace(
                    state_dict=SimpleNamespace(**payload["state_dict"]),
                    uids=payload.get("uids", []),
                    skipped_uids=payload.get("skipped_uids", []),
                    success_rate=payload.get("success_rate", 0.0),
                )

                # Clear the original payload dict to free memory immediately
                del payload
                if hasattr(fetch, "data"):
                    fetch.data = None
                del fetch

            # ── B. aggregated object *missing* or *malformed* ────────────────
            else:
                gather_ns = None
                is_last_window = start_w == target_w - 1
                hone.logger.warning(
                    "    ↳ %s – %s",
                    "not available" if fetch is None else "malformed payload",
                    "attempting gather‑fallback" if is_last_window else "skipping",
                )

                if is_last_window:
                    sync_block = (start_w + 1) * instance.hparams.blocks_per_window
                    ts_value = await instance.loop.run_in_executor(
                        None, instance.query_block_timestamp, sync_block
                    )
                    if ts_value is None:
                        hone.logger.warning(
                            f"Could not get timestamp for sync block {sync_block}.",
                        )
                        time_min = time_max = None
                    else:
                        time_min = datetime.fromtimestamp(ts_value, tz=timezone.utc)
                        time_max = time_min + timedelta(
                            seconds=instance.hparams.time_window_delta_seconds
                        )

                    # FU2 (resolved): the catchup primary path stays
                    # single-blob because composition is handled by
                    # ``Comms.gather`` / ``Comms._gather_fragmented`` in
                    # fragmented mode and the leader's aggregator put
                    # publishes the already-composed
                    # ``gather_result.state_dict`` (see the
                    # ``upload_gather_results`` site in ``validator.py``).
                    # The fallback below DOES need to fan out per-fragment
                    # whenever the operator has flipped
                    # ``fragmented_uploads: true``, otherwise the catchup
                    # gather would issue a single ``gradient-{w}-{uid}``
                    # GET that fragmented miners never wrote.
                    _catchup_fragmented = bool(
                        getattr(instance.hparams, "fragmented_uploads", False)
                    )
                    _catchup_num_fragments = (
                        int(getattr(instance.hparams, "num_fragments", 24))
                        if _catchup_fragmented
                        else None
                    )

                    # ---- Gather fallback ----------------------------------------
                    gather_ns = await instance.comms.gather(
                        my_uid=instance.uid,
                        uids=instance.comms.peers,
                        window=start_w,
                        key="gradient",
                        timeout=45,
                        device=str(catchup_device),
                        local=False,
                        stale_retention=10,
                        totalks=instance.totalks,
                        compressor=instance.compressor,
                        time_min=time_min,
                        time_max=time_max,
                        num_fragments=_catchup_num_fragments,
                    )

                if gather_ns is None:
                    hone.logger.warning("    ↳ gather‑fallback failed – skipping")
                else:
                    hone.logger.info("    ↳ gather‑fallback succeeded – applying")

            # FU2 post-compose validator. Runs on master only, after
            # ``gather_ns`` is built (whether from the primary aggregator
            # blob or the per-fragment gather fallback) and BEFORE the
            # skip broadcast / outer_step. Logs WARN when fragmented-mode
            # composition dropped a param contribution; never raises.
            if gather_ns is not None:
                _warn_missing_compose_params(instance, gather_ns, start_w)
        else:
            gather_ns = None

        # Broadcast whether we should skip this window (master decides)
        if instance.is_master:
            skip_tensor = torch.tensor(
                [1 if gather_ns is None else 0],
                dtype=torch.int32,
                device=instance.config.device,
            )
        else:
            skip_tensor = torch.tensor(
                [0], dtype=torch.int32, device=instance.config.device
            )

        dist_helper.broadcast(skip_tensor, src=0)
        skip_window = bool(skip_tensor.item())

        # If skipping, continue to next window without updating scheduler
        if skip_window:
            # Don't increment global_step as no outer step was performed
            start_w += 1
            continue

        # ------------------------------------------------------------------
        # 2) All ranks apply the update.
        # ------------------------------------------------------------------
        # Synchronize all ranks before applying the outer step to ensure
        # they're processing the same window together
        dist_helper.safe_barrier("catchup_pre_outer_step", instance.local_rank)

        # Catchup replays old aggregator outputs to bring a restarted
        # node up to current global_step; the per-replay fingerprint
        # and timings aren't reported to the dashboard, so we discard
        # both halves of the new tuple return.
        _, _ = outer_step(
            instance.model,
            instance.outer_optimizer,
            gather_result=gather_ns,
            transformer=instance.transformer,
            compressor=instance.compressor,
            xshapes=instance.xshapes,
            totalks=instance.totalks,
            device=instance.config.device,
            is_master=instance.is_master,  # rank-0 handles logging
            world_size=instance.world_size,
            use_dct=instance.hparams.use_dct,
            wandb_run=instance.wandb
            if instance.is_master and isinstance(instance.wandb, Run)
            else None,
            global_step=instance.global_step,
            max_grad_norm=getattr(
                instance.hparams, "outer_max_grad_norm", None
            ),
        )

        # advance LR scheduler if one exists.
        inner_sched: LRScheduler | None = getattr(instance, "inner_scheduler", None)
        if inner_sched is not None:
            for _ in range(instance.hparams.inner_steps):
                # Respect flatten window during catch-up
                if not instance.should_skip_scheduler_step():
                    inner_sched.step()
                instance.inner_scheduler_step_count += 1

        # Aggressive memory cleanup after each window
        if instance.is_master and "gather_ns" in locals() and gather_ns is not None:
            # Clear the gather result to free memory
            if hasattr(gather_ns, "state_dict"):
                # Clear all attributes from state_dict namespace
                for key in list(vars(gather_ns.state_dict).keys()):
                    delattr(gather_ns.state_dict, key)
            del gather_ns

        # Force garbage collection to free CPU memory
        gc.collect()

        # Clear CUDA cache and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Log memory usage after cleanup
        if instance.is_master and (start_w - checkpoint_current_window) % 5 == 0:
            log_memory_usage(f"After window {start_w} cleanup")
        # ──────────────────────────────────────────────────────────────────────
        # 3) Debug‑dict comparison to estimate “how many steps behind” we are
        # ──────────────────────────────────────────────────────────────────────
        try:
            if instance.is_master:
                debug_fetch = await instance.comms.get(
                    uid=str(leader_uid),
                    window=start_w,
                    key="debug",
                    local=False,
                    stale_retention=10,
                )

                if debug_fetch.success and isinstance(debug_fetch.data, dict):
                    debug_dict = debug_fetch.data  # validator's payload

                    # --- update EMA of parameter‑slice changes ------------------
                    for name, p in instance.model.named_parameters():
                        if p.numel() < 2:
                            continue

                        # Handle DTensor parameters
                        if isinstance(p, DT):
                            curr_slice = (
                                p.to_local().detach().cpu().flatten()[slice_idx]
                            )
                        else:
                            curr_slice = p.detach().cpu().flatten()[slice_idx]

                        if name in prev_param_state:
                            delta = (curr_slice - prev_param_state[name]).abs()
                            if name not in param_avg_change:
                                param_avg_change[name] = delta.clone()
                            else:
                                param_avg_change[name].mul_(1 - alpha).add_(
                                    delta * alpha
                                )
                        prev_param_state[name] = curr_slice.clone()

                    # --- call shared comparison helper --------------------------
                    lr = instance.outer_optimizer.param_groups[0]["lr"]
                    cmp = await compare_model_with_debug_dict(
                        model=instance.model,
                        debug_dict=debug_dict,
                        learning_rate=lr,
                        index_range=(0, 2),
                        param_avg_change=param_avg_change,
                    )

                    if cmp["success"]:
                        hone.logger.info(
                            f"[catch‑up] window {start_w} "
                            f"avg_steps_behind={cmp['avg_steps_behind']:.3f}, "
                            f"l2_norm={cmp['l2_norm']:.4f}"
                        )
                    else:
                        hone.logger.warning(
                            f"[catch‑up] debug‑dict comparison failed for window {start_w}"
                        )
                else:
                    hone.logger.warning(
                        f"[catch‑up] no debug‑dict found for window {start_w}"
                    )
        except Exception as exc:
            hone.logger.warning(f"[catch‑up] debug‑dict processing error: {exc}")

        # Increment global_step since we performed an outer step
        instance.global_step += 1
        start_w += 1

        dist_helper.safe_barrier("catchup_post_window", instance.local_rank)

        # If the chain progressed while we were busy, extend the target.
        if instance.current_window > target_w:
            target_w = instance.current_window

    # Final aggressive memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Log final memory state
    log_memory_usage("Final memory state after catchup")
    hone.logger.info("Catch‑up finished – model now in sync.")


async def compare_model_with_debug_dict(
    model: nn.Module,
    debug_dict: dict[str, list[float]],
    learning_rate: float,
    param_avg_change: dict[str, torch.Tensor] | None = None,
    *,
    min_step_size: float = 1e-9,
    index_range: tuple[int, int] = (0, 2),
) -> dict[str, bool | float | int]:
    """
    Compare weights with published debug snippets and return sync metrics.
    """
    # Initialize metrics
    total_squared_diff = 0.0
    total_abs_diff = 0.0
    param_count = 0
    max_diff = 0.0  # largest raw parameter diff
    max_steps = 0.0

    # Collect per‑tensor step‑ratio vectors so we can take
    # a single global median later
    tensors = 0
    step_ratio_list: list[torch.Tensor] = []

    inner = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    named_params = inner.named_parameters()

    # Look up debug entries by *canonical* (wrapper-stripped) name to
    # match what writers emit -- the miner's debug_dict keys come from
    # ``cname + "_debug"`` (canonical, no ``_orig_mod`` /
    # ``_checkpoint_wrapped_module`` /``stage.`` prefixes), and the
    # validator's debug-PUT path is doing the same after the recent
    # fix. Comparing ``model.named_parameters()`` raw keys against the
    # debug dict here would always miss when the model is wrapped
    # (FSDP + AC + torch.compile), tensors counter would stay 0, and
    # every metric would default to ``math.inf`` -- which then crashes
    # ``log_sync_score`` on ``int(float("inf"))``. We also try the raw
    # key as a fallback so older miners that haven't deployed the
    # canonical-name change yet still get a score (better than crashing
    # the validator).
    canon_map = canonical_param_names(inner)

    for name, p in named_params:
        cname = canon_map.get(name, name)
        key = cname + "_debug"
        if key not in debug_dict or not isinstance(debug_dict[key], list):
            # Legacy fallback: try the raw wrapped name too so we can
            # still score peers running pre-fix code.
            legacy_key = name + "_debug"
            if legacy_key in debug_dict and isinstance(
                debug_dict[legacy_key], list
            ):
                key = legacy_key
            else:
                continue

        # --- grab the slice we care about --------------------------------
        if isinstance(p, DT):
            curr_slice = p.to_local().data.flatten()[index_range[0] : index_range[1]]
        else:
            curr_slice = p.data.flatten()[index_range[0] : index_range[1]]

        debug_slice = torch.tensor(
            debug_dict[key], dtype=p.dtype, device=curr_slice.device
        )

        diff_vec = curr_slice - debug_slice
        abs_vec = torch.abs(diff_vec)

        total_squared_diff += torch.sum(diff_vec**2).item()
        total_abs_diff += abs_vec.sum().item()
        raw_max = abs_vec.max().item()
        max_diff = max(max_diff, raw_max)
        param_count += abs_vec.numel()

        # --- element-wise steps-behind -----------------------------------
        if param_avg_change and name in param_avg_change:
            step_vec = torch.clamp(
                param_avg_change[name].to(curr_slice.device), min=min_step_size
            )
            if step_vec.numel() != abs_vec.numel():
                # fallback if stored slice has wrong length
                step_vec = abs_vec.new_full(abs_vec.size(), learning_rate)
        else:
            step_vec = abs_vec.new_full(abs_vec.size(), learning_rate)

        step_ratio = abs_vec / step_vec
        # Accumulate for global median
        step_ratio_list.append(step_ratio)
        max_steps = max(max_steps, step_ratio.max().item())
        tensors += 1

    l2_norm = math.sqrt(total_squared_diff)
    avg_l2_norm = math.inf if tensors == 0 else l2_norm / param_count
    avg_abs_diff = math.inf if tensors == 0 else total_abs_diff / param_count
    if not step_ratio_list:  # nothing compared
        median_steps = math.inf
        max_steps = math.inf
        interquartile_mean_steps = math.inf
    else:
        all_steps = torch.cat([t.flatten() for t in step_ratio_list])
        median_steps = all_steps.median().item()

        # Calculate interquartile mean (mean of values between Q1 and Q3)
        q1 = all_steps.quantile(0.25).item()
        q3 = all_steps.quantile(0.75).item()
        # Filter values in the interquartile range
        iqr_mask = (all_steps >= q1) & (all_steps <= q3)
        interquartile_mean_steps = all_steps[iqr_mask].mean().item()

    return {
        "success": True,
        "l2_norm": l2_norm,
        "avg_l2_norm": avg_l2_norm,
        "avg_abs_diff": avg_abs_diff,
        "max_diff": max_diff,
        "avg_steps_behind": interquartile_mean_steps,
        "interquartile_mean_steps_behind": interquartile_mean_steps,
        "max_steps_behind": max_steps,
        "param_count": param_count,
        "learning_rate": learning_rate,
    }


@torch.no_grad()
async def check_uid_index_overlap(
    neuron: NeuronT,
    gather_result: SimpleNamespace,
    window: int,
    *,
    overlap_threshold: float = 0.90,
) -> dict:
    """
    For every peer-pair compute the per-chunk *set* overlap of their top-k index
    lists on each parameter.  A pair is flagged **only if the size-weighted
    average across *all* checked parameters** is ≥ `overlap_threshold`.
    """

    # ── 0. basic sanity ───────────────────────────────────────────────────
    uids: list[int] = list(getattr(gather_result, "uids", []))
    Ptot = len(uids)
    if Ptot < 2:
        hone.logger.info("[overlap] <2 peers – skip")
        return dict(
            pairs_checked=0,
            pairs_high_ovlap=0,
            ratio_high_ovlap=0.0,
            mean_overlap=0.0,
            min_overlap=0.0,
            max_overlap=0.0,
            pairs_over_thresh=[],
            uids_over_thresh={},
        )

    ts_map = dict(
        zip(
            uids,
            await asyncio.gather(
                *[neuron.comms.gradient_timestamp(uid, window - 1) for uid in uids]
            ),
        )
    )

    # P1 fix 5a: prepare_gradient_dict writes under the canonical
    # (wrapper-stripped) name (``cname + "idxs"``) but this loop iterates
    # the model's *raw* ``named_parameters()`` keys which include
    # ``_orig_mod.``/``_checkpoint_wrapped_module.``/``_fsdp_wrapped_module.``
    # prefixes under FSDP+AC+compile. Pre-fix the lookup with
    # ``canonical_param_names()`` so the (idxs/vals) get-attr below
    # actually finds the uploaded tensors. Without this, every iteration
    # silently `continue`s and the function returned mean_overlap=0.0
    # regardless of real overlap state.
    inner_model = (
        neuron.model.module
        if isinstance(neuron.model, torch.nn.parallel.DistributedDataParallel)
        else neuron.model
    )
    canon_map = canonical_param_names(inner_model)

    # P1 fix 5b: when ``fragmented_uploads=true``, a peer may upload only
    # a subset of fragments so each ``state_dict.<param_idxs>`` list has
    # length ≤ len(uids), and the i-th tensor corresponds to
    # ``uids_per_param[idx_key][i]`` rather than ``uids[i]``. Legacy gather
    # populates ``uids_per_param`` with the same uid list for every key
    # (see ``Comms.gather`` legacy branch), so this lookup is correct in
    # both modes.
    uids_per_param: dict[str, list[int]] = getattr(
        gather_result, "uids_per_param", {}
    )

    # ── 1. bookkeeping ────────────────────────────────────────────────────
    # Pair keys are canonical (uid_a, uid_b) tuples with ``uid_a < uid_b``
    # so the same pair from different params accumulates into one bucket
    # regardless of the per-param contributor ordering.
    pair_acc: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    total_weighted_sum = 0.0
    total_weight = 0.0

    # ── 2. iterate over parameters that have compressed indices ───────────
    # FU1 (P6b): TurboQuant-encoded params publish ``cname + "tq_idxs"``
    # instead of ``cname + "idxs"``. The overlap-detection logic below
    # assumes 12-bit-packed top-K-per-chunk indices into the DCT domain
    # (``unpack_12bit_indices`` + per-chunk set intersection). TurboQuant
    # ships int64 top-K indices into a Hadamard-rotated flat vector —
    # the sparsity structure is completely different and "set overlap"
    # is not the right signal (the count-sketch-style similarity used in
    # P6 ("soft-sketch") is what catches TurboQuant overfit, not this
    # per-chunk index intersection). When a param has ONLY a tq_idxs key
    # and no legacy idxs, skip it outright; when it has both (a fleet
    # mid-rollout with some miners on each codec), we still use the
    # legacy idxs list for overlap and silently drop the TurboQuant-only
    # contributors' positions — this is a conservative under-estimate of
    # overlap for the mixed case, which is fine for a default-OFF
    # signalling path. Document in the returned stats that we dropped
    # TurboQuant params; operator can tail the log count to flag when
    # rollout progresses past a sensible audit coverage threshold.
    tq_params_skipped = 0
    for pname, _ in inner_model.named_parameters():
        cname = canon_map.get(pname, pname)
        idx_key = cname + "idxs"
        idxs_all = getattr(gather_result.state_dict, idx_key, None)
        if idxs_all is None:
            # If a TurboQuant variant is present, count it and skip.
            # Overlap detection was designed for the legacy top-K +
            # 12-bit-packed-index pipeline; TurboQuant's
            # rotated-flat-top-K indices are not comparable via
            # per-chunk set intersection.
            tq_idxs_all = getattr(
                gather_result.state_dict, cname + "tq_idxs", None
            )
            if tq_idxs_all is not None:
                tq_params_skipped += 1
            continue

        # Get values for unpacking shape
        vals_key = cname + "vals"
        vals_all = getattr(gather_result.state_dict, vals_key, None)
        if vals_all is None:
            continue

        # P0b (Rollout 2 fix, 2026-05-02): when senders opt into
        # ``pack_values_2bit``, ``vals`` is bit-packed 4× along the
        # last dim so ``val_data.shape`` is NOT the right input to
        # ``unpack_12bit_indices``. The original last dim lives in
        # ``qparams[6]`` of the 7-tuple ``PACK_VERSION_2BIT`` form;
        # fall back to ``val_data.shape`` for legacy 5-tuple qparams
        # or when no qparams travels alongside. Mirrors
        # ``Comms.check_compressed_indices`` exactly.
        qparams_key = cname + "quant_params"
        qparams_all = getattr(gather_result.state_dict, qparams_key, None)

        contributor_uids = uids_per_param.get(idx_key)
        if contributor_uids is None:
            # Legacy gather pre-P1 didn't populate ``uids_per_param``.
            # Fall back to the canonical positional assumption (one tensor
            # per gather UID, in ``uids`` order). Guard against partial
            # lists to avoid IndexError when a heterogeneous fleet is
            # in flight.
            contributor_uids = uids[
                : len(idxs_all) if isinstance(idxs_all, list) else 1
            ]

        # Need at least two peers contributing this param to compute an
        # overlap; skip otherwise (this naturally drops singletons that
        # arise when only one peer's fragment containing this param made
        # it through validation).
        n_contributors = len(contributor_uids)
        if n_contributors < 2:
            continue

        # Unpack all 12-bit packed indices using per-peer qparams
        # to recover the correct (pre-2-bit-pack) values shape.
        unpacked_indices = []
        for i in range(n_contributors):
            idx_data = idxs_all[i] if isinstance(idxs_all, list) else idxs_all
            val_data = vals_all[i] if isinstance(vals_all, list) else vals_all
            qp_data = (
                qparams_all[i]
                if isinstance(qparams_all, list)
                else qparams_all
            )

            # P0b: prefer the 7-tuple's ``original_last_dim`` when the
            # sender packed values as 2-bit. Legacy 5-tuple or
            # missing qparams → trust the wire last dim (pre-rollout
            # behaviour is preserved exactly).
            if (
                isinstance(qp_data, tuple)
                and len(qp_data) >= 7
                and int(qp_data[5]) == 1  # PACK_VERSION_2BIT
            ):
                values_shape = (*val_data.shape[:-1], int(qp_data[6]))
            else:
                values_shape = val_data.shape

            unpacked = unpack_12bit_indices(
                idx_data.to(neuron.config.device), values_shape
            )
            unpacked_indices.append(unpacked)

        idxs_tensor = torch.stack(unpacked_indices, dim=0)
        P, *chunk_dims, k = idxs_tensor.shape
        C = int(torch.prod(torch.tensor(chunk_dims)))  # num chunks
        idxs_flat = idxs_tensor.reshape(P, C, k)

        param_weight = C * k  # size weight

        for i in range(P):
            for j in range(i + 1, P):
                a = idxs_flat[i].unsqueeze(-1)  # (C,k,1)
                b = idxs_flat[j].unsqueeze(-2)  # (C,1,k)
                inter = (a == b).any(-1).sum(-1)  # (C,)
                mean_frac = (inter.float() / k).mean().item()

                total_weighted_sum += mean_frac * param_weight
                total_weight += param_weight

                # Use UIDs (not positional indexes) as the pair key so the
                # accumulator is robust to per-param contributor variance
                # in fragmented mode.
                uid_i = contributor_uids[i]
                uid_j = contributor_uids[j]
                pair_key = (
                    (uid_i, uid_j) if uid_i < uid_j else (uid_j, uid_i)
                )
                acc = pair_acc[pair_key]
                acc[0] += mean_frac * param_weight
                acc[1] += param_weight

    # ── 3. second pass – decide offenders & track min/max ─────────────────
    pairs_high, pairs_over, uids_with_slashing = 0, [], {}
    min_pair, min_val = None, 1.0
    max_pair, max_val = None, 0.0

    for (uid_i, uid_j), (w_sum, w_tot) in pair_acc.items():
        avg_overlap = w_sum / w_tot if w_tot > 0 else 0.0

        # --- track global min / max --------------------------------------
        if avg_overlap < min_val:
            min_val, min_pair = avg_overlap, (uid_i, uid_j)
        if avg_overlap > max_val:
            max_val, max_pair = avg_overlap, (uid_i, uid_j)
        # ------------------------------------------------------------------

        if avg_overlap >= overlap_threshold:
            pairs_high += 1
            # P1 fix 5b: use UIDs directly (not positional). The
            # ``ts_map`` lookup may miss for reserve UIDs whose timestamp
            # wasn't fetched; default to 0.0 so the comparison still
            # tie-breaks deterministically.
            ts_i = ts_map.get(uid_i, 0.0)
            ts_j = ts_map.get(uid_j, 0.0)
            offender = uid_i if ts_i >= ts_j else uid_j
            uids_with_slashing[offender] = determine_slash_egregiousness(avg_overlap)

            pairs_over.append((uid_i, uid_j, avg_overlap))
            hone.logger.debug(
                f"[overlap] peers {uid_i}/{uid_j} share "
                f"{avg_overlap * 100:.1f}% of indices (size-weighted avg)"
            )

    mean_overlap = total_weighted_sum / total_weight if total_weight else 0.0
    ratio_high = pairs_high / len(pair_acc) if pair_acc else 0.0

    # ── 4. summary log with min / max -------------------------------------
    hone.logger.info(
        f"[overlap] {len(pair_acc)} pairs, {pairs_high} ≥{overlap_threshold * 100:.0f}% "
        f"({ratio_high * 100:.2f}%), size-weighted mean {mean_overlap * 100:.1f}%"
    )
    if min_pair is not None and max_pair is not None:
        hone.logger.info(
            f"[overlap]   min {min_val * 100:.1f}%  (peers {min_pair[0]}/{min_pair[1]}) ; "
            f"max {max_val * 100:.1f}%  (peers {max_pair[0]}/{max_pair[1]})"
        )
    if uids_with_slashing:
        hone.logger.warning(
            f"[overlap] offenders: {sorted(list(uids_with_slashing.keys()))}"
        )
    if tq_params_skipped:
        # Surface the TurboQuant skip count so operators can monitor
        # rollout-era overlap coverage. Once every miner is on
        # TurboQuant this will equal the total trainable param count
        # and the overlap signal is structurally empty; the P6 soft-
        # sketch covenant signal (``hone.sketch``) is what flags
        # TurboQuant overfit.
        hone.logger.info(
            f"[overlap] skipped {tq_params_skipped} TurboQuant-encoded "
            "params (FU1); overlap signal only covers legacy-codec params. "
            "TurboQuant overfit is caught by the P6 soft count-sketch "
            "path, not this per-chunk set-intersection check."
        )

    return dict(
        pairs_checked=len(pair_acc),
        pairs_high_ovlap=pairs_high,
        ratio_high_ovlap=ratio_high,
        mean_overlap=mean_overlap,
        min_overlap=min_val if min_pair is not None else 0.0,
        max_overlap=max_val if max_pair is not None else 0.0,
        pairs_over_thresh=pairs_over,
        uids_over_thresh=uids_with_slashing,
        tq_params_skipped=tq_params_skipped,
    )


def determine_slash_egregiousness(overlap_pct: float) -> str:
    """
    Based on the overlap_pct, return a level corresponding
    to an action which will be taken

    Args:
        overlap_pct: The percentage of overlap in the grads with
             other miners

    Returns:
        Category of overlap pct
    """

    invalid_number = overlap_pct < 0.0 or overlap_pct > 1.0
    if invalid_number:
        raise ValueError(f"overlap_pct must be between 0.0 and 1.0, got {overlap_pct}")

    egregiousness = "high"
    if overlap_pct >= 0.5:
        egregiousness = "max"
    if overlap_pct >= 0.6:
        egregiousness = "mega"

    return egregiousness


def instantiate_slashing_multiplier():
    """Centralize slashing config

    We multiply these percentages against the base final_score
    """
    return {
        "high": 0.5,  # case when similarity high
        "max": 0.0,  # case when similarity >= 95%
        "mega": 0.0,  # case when similarity = 100%
    }


# ---------------------------------------------------------------------------
# Inline unit test for ``prepare_gradient_buckets`` (P1 deliverable).
# Run with: ``python hone/src/hone/neurons.py``
# Kept inline because the bin-packing balance bound is the only invariant
# this module needs to defend per-PR; a full pytest harness would be
# overkill for a single 30-line check.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys as _sys

    sizes = [1, 3, 9, 27, 81, 2, 6, 18, 54, 162]
    fake_gradient: dict = {"metadata": {"window": 1234}}
    for i, n in enumerate(sizes):
        # Two tensors per param to mirror the real (idxs, vals) shape; the
        # bucket-loader sums ``idxs.numel + vals.numel`` so the effective
        # per-param weight is ``2n``. Ratios stay invariant under the *2.
        fake_gradient[f"param{i:02d}idxs"] = torch.zeros(n, dtype=torch.uint8)
        fake_gradient[f"param{i:02d}vals"] = torch.zeros(n, dtype=torch.uint8)

    num_buckets = 4
    buckets = prepare_gradient_buckets(fake_gradient, num_buckets=num_buckets)

    # Compute per-bucket payload by summing element counts directly.
    bucket_loads: list[int] = []
    for b in buckets:
        load = 0
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                load += v.numel()
        bucket_loads.append(load)

    total = sum(2 * n for n in sizes)
    assert sum(bucket_loads) == total, (
        f"bucket loads must sum to total: got {sum(bucket_loads)}, expected {total}"
    )

    avg_load = total / num_buckets
    max_load = max(bucket_loads)
    max_item_load = 2 * max(sizes)
    # The textbook LPT bound: ``max_bucket ≤ 4/3 · OPT`` where
    # ``OPT ≥ max(total/m, max_item)``. For inputs where one item exceeds
    # ``total/m`` (the embedding tensor in real MoE gradients is exactly
    # this case), the max-bucket / avg ratio CANNOT be ≤ 4/3 — it's
    # bounded below by ``max_item / avg``. So we assert against the
    # proper OPT lower bound, not the simpler avg.
    lower_bound_opt = max(avg_load, float(max_item_load))
    bound = (4.0 / 3.0) * lower_bound_opt
    print(
        f"[prepare_gradient_buckets test] sizes={sizes} num_buckets={num_buckets}"
    )
    print(
        f"  bucket_loads={bucket_loads} total={total} avg={avg_load:.2f} "
        f"max={max_load} max_item={max_item_load}"
    )
    print(
        f"  lower_bound_opt={lower_bound_opt:.2f} 4/3-bound={bound:.2f} "
        f"max/avg={max_load / avg_load:.3f} "
        f"max/lb_opt={max_load / lower_bound_opt:.3f}"
    )
    if max_load > bound + 1e-6:
        print(
            f"FAIL: max bucket load {max_load} exceeds LPT 4/3 bound "
            f"{bound:.2f} (lower_bound_opt={lower_bound_opt:.2f})",
            file=_sys.stderr,
        )
        _sys.exit(1)

    # Sanity: every bucket carries metadata with the right num_buckets,
    # and each ``idxs`` key is present together with its companion ``vals``.
    for i, b in enumerate(buckets):
        assert b["metadata"]["bucket_idx"] == i
        assert b["metadata"]["num_buckets"] == num_buckets
        for k in b:
            if k.endswith("idxs"):
                assert k[:-4] + "vals" in b, f"vals missing for {k} in bucket {i}"

    print("OK: prepare_gradient_buckets balance bound holds.")

    # -----------------------------------------------------------------
    # P3 K_safety cap test — guards the per-peer 1/K cap that prevents
    # a Byzantine peer with huge claimed ``c_tokens`` from dominating
    # the aggregated update. Three asserted invariants:
    #   1. For a skewed dict where peer 1 is 10/13 ≈ 76.9% of mass,
    #      k_safety=4 reduces peer 1 to exactly 25% (= 1/4) and
    #      redistributes the spill to peers 2/3/4.
    #   2. ``_apply_k_safety_cap(None, k_safety=8)`` returns ``None``
    #      (preserves the "token-weighted aggregation off" contract).
    #   3. An empty dict round-trips (no peers, no work).
    # -----------------------------------------------------------------
    _skewed = {1: 10.0, 2: 1.0, 3: 1.0, 4: 1.0}
    _total = sum(_skewed.values())
    _capped = _apply_k_safety_cap(dict(_skewed), k_safety=4)
    assert _capped is not None
    _new_total = sum(_capped.values())
    _max_share = max(_capped.values()) / _new_total
    assert _max_share <= 0.25 + 1e-9, (
        f"K_safety=4 must cap at 25%; got {_max_share:.4f}"
    )
    # Peer 1 should be EXACTLY at the cap (1/4 of the total).
    assert abs(_capped[1] / _new_total - 0.25) < 1e-9, (
        f"peer 1 should land at exactly 25%; got {_capped[1] / _new_total:.6f}"
    )
    # Total mass preserved (cap redistributes, doesn't drop).
    assert abs(_new_total - _total) < 1e-6, (
        f"total mass must be preserved: before={_total} after={_new_total}"
    )

    # None input is a structural no-op — this is how ``outer_step`` sees
    # the default ``token_weighted_aggregation=false`` case.
    assert _apply_k_safety_cap(None, k_safety=8) is None

    # Empty dict round-trips unchanged.
    _empty_in: dict[int, float] = {}
    _empty_out = _apply_k_safety_cap(_empty_in, k_safety=8)
    assert _empty_out == {} and _empty_out is not None

    # k_safety <= 0 disables the cap (no-op even with skew).
    _same = _apply_k_safety_cap(dict(_skewed), k_safety=0)
    assert _same == _skewed

    print("OK: _apply_k_safety_cap 1/K cap holds under skew + no-op guards.")
