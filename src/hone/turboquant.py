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

"""TurboQuant scalar codec (P6b — DEFAULT-OFF, gated by P6a audit).

Implements the dense pre-top-K TurboQuant codec from Lukashevich et al.
("TurboQuant: Online Vector Quantization with Optimal Distortion Rate")
adapted for Hone's per-parameter EF-based gradient compression
pipeline. The codec is gated behind ``hparams.turboquant_enabled``
which **MUST** stay false until ``hone/validator/turboquant_audit_cli.py``
exits 0 against a real production EF snapshot. See
``hone/docs/turboquant.md`` for the full operator gating workflow.

Pipeline (when enabled, per ``prepare_gradient_dict`` in ``neurons.py``):

1. Walsh-Hadamard rotation on the FULL d-dim flat EF tensor — this
   is the load-bearing step the P6a audit gates on. Hadamard makes
   each rotated coordinate approximately Beta(d/2, d/2) distributed
   for typical SGD/Adam EF tensors; the Lloyd-Max codebook below is
   trained against that distribution.
2. Per-coordinate scalar quantization via a Beta-Lloyd-Max codebook.
   The codebook is precomputed offline per ``(d, b)`` pair and
   cached in :data:`_CENTROID_CACHE_DIR` so process restarts skip
   the Lloyd-Max iteration.
3. Top-K is then applied **on the codebook indices** (not the float
   values) — the user prompt's load-bearing fix to the original
   "post-top-K TurboQuant" plan, which broke the Beta distribution
   the codebook assumed.
4. Outlier handling per the paper §4.3: top-32 |coordinates| (before
   quantization) get a separate codebook at ``b+1`` bits to limit
   the worst-case reconstruction error on the dominant directions.
5. Inner-product mode (``mode='prod'``): main coords use ``b-1`` bits
   and the LSB encodes the SIGN of the post-quantization residual
   (a 1-bit Quantized-JL projection). The cross-peer aggregation in
   ``batch_decompress`` consumes inner products of decoded gradients,
   and the residual sign correction shrinks the inner-product bias
   without spending another full bit.

Module surface
--------------

- :func:`hadamard_rotate` — re-exported from :mod:`hone.turboquant_audit`
  so the codec and the audit rotate via the SAME implementation. Any
  drift between them would silently invalidate the audit verdict.
- :func:`beta_lloyd_max_centroids` — cached centroid lookup (in-memory
  + on-disk under ``hone/src/hone/_turboquant_cache/``).
- :func:`quantize_turboquant` — full encode (rotation already done by
  the caller). Returns ``(codes, meta)`` where ``codes`` is a uint8
  tensor of shape ``(d,)`` and ``meta`` carries everything the
  decoder needs.
- :func:`dequantize_turboquant` — exact inverse, modulo quantization
  noise and (in prod mode) the residual sign correction.

Wire format
-----------

The codec is intentionally NOT wired into the existing
``compress.QuantParamsT`` family — its tuple semantics (Lloyd-Max
codebooks vs uniform-bin lookup) don't compose with the legacy
``_dequantize_values`` path. ``prepare_gradient_dict`` writes the
TurboQuant payload under ``cname + 'tq_*'`` keys in the gradient
dict; the existing ``cname + {'idxs','vals','quant_params'}`` keys
are intentionally left empty in the TurboQuant branch so a
TurboQuant-unaware validator skips the param outright instead of
mis-decoding it.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal, Sequence

import torch

# Re-export the WHT primitive from the audit module so the codec and
# the gate use literally the same function. Importing the module
# rather than the function keeps the dep arrow one-directional
# (``turboquant`` depends on ``turboquant_audit``, never the reverse).
from hone.turboquant_audit import hadamard_rotate as hadamard_rotate

__all__ = [
    "hadamard_rotate",
    "beta_lloyd_max_centroids",
    "quantize_turboquant",
    "dequantize_turboquant",
    "batch_decompress_turboquant",
]


# Lloyd-Max centroid cache. In-memory dict keyed by (d, b); checked
# first because Lloyd-Max iteration costs ~30ms for K=16 on 10K
# samples and we'd otherwise pay it for every compress() call. The
# on-disk fallback (per the user's spec) lives under the package dir
# so a fresh process boot loads centroids without re-running the
# iteration. If the package dir is read-only (e.g. a pip install
# without --editable), the disk cache silently degrades to in-memory
# only and the next process boot pays the iteration cost again.
_CENTROID_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_CENTROID_CACHE_DIR: Path = Path(__file__).resolve().parent / "_turboquant_cache"


def _load_centroids_from_disk(d: int, b: int) -> torch.Tensor | None:
    """Best-effort disk read of cached centroids. Returns None on miss
    OR on any I/O / format error — the caller falls back to recompute.
    """
    path = _CENTROID_CACHE_DIR / f"d{d}_b{b}.pt"
    if not path.exists():
        return None
    try:
        # ``weights_only=True`` is the safe load mode for cached tensors;
        # the centroids are pure float data with no pickled objects.
        centroids = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(centroids, torch.Tensor):
            return None
        if centroids.numel() != (1 << b):
            return None
        return centroids
    except Exception:
        return None


def _save_centroids_to_disk(d: int, b: int, centroids: torch.Tensor) -> None:
    """Best-effort disk write. Silent on failure — the in-memory cache
    is still valid, the disk write is purely an optimisation for the
    next process boot.
    """
    path = _CENTROID_CACHE_DIR / f"d{d}_b{b}.pt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(centroids.detach().cpu(), path)
    except Exception:
        return


def beta_lloyd_max_centroids(
    d: int,
    b: int,
    *,
    num_samples: int = 10_000,
    num_iters: int = 50,
) -> torch.Tensor:
    """Precomputed b-bit Lloyd-Max centroids for Beta(d/2, d/2).

    Cached in a module-level dict keyed by ``(d, b)``. First call per
    ``(d, b)`` runs Lloyd-Max iteration to convergence (50 iters is
    plenty for Beta — the distribution is unimodal and centroid moves
    decay exponentially); subsequent calls are O(1).

    The on-disk cache under :data:`_CENTROID_CACHE_DIR` survives
    process restarts. Cached centroids are sorted ascending so the
    receiver-side dequantizer can use a single ``torch.gather`` /
    ``searchsorted`` instead of a sort.

    Args:
        d: Dimension of the rotated tensor; sets the Beta shape
            parameter (``alpha = beta = d/2``). Larger d → narrower
            distribution → centroids cluster tightly around 0.5.
        b: Bits per code; 2^b centroids will be returned. Must satisfy
            ``1 <= b <= 8`` (uint8 storage on the wire).
        num_samples: Number of Beta samples used to fit the centroids.
            10K is enough for stable Lloyd-Max on K <= 256; rarely
            tuned.
        num_iters: Maximum Lloyd-Max iterations. Loop early-stops when
            the largest centroid move is below 1e-6.

    Returns:
        Sorted 1-D float32 tensor of length ``2 ** b`` on CPU. The
        returned tensor is the cached object; do NOT mutate.
    """
    if not (1 <= b <= 8):
        raise ValueError(f"beta_lloyd_max_centroids: b must be in [1, 8], got {b}")
    if d < 2:
        raise ValueError(f"beta_lloyd_max_centroids: d must be >= 2, got {d}")

    key = (int(d), int(b))
    cached = _CENTROID_CACHE.get(key)
    if cached is not None:
        return cached

    disk = _load_centroids_from_disk(*key)
    if disk is not None:
        _CENTROID_CACHE[key] = disk
        return disk

    # scipy is the only stable Beta sampler / quantile fn we have for
    # the very large alpha values seen at MoE-expert d (~1e6). torch's
    # Beta can be numerically degenerate at alpha > 1e4.
    try:
        from scipy.stats import beta as scipy_beta  # type: ignore[import-untyped]
    except ImportError as e:  # pragma: no cover - scipy is a hone dep
        raise RuntimeError(
            "TurboQuant codebook fitting requires scipy. Install it "
            "or set hparams.turboquant_enabled=false."
        ) from e

    K = 1 << b
    alpha = d / 2.0

    # Quantile-based init: place centroids at the K equal-probability
    # quantile midpoints. This puts each centroid in a region that
    # carries 1/K of the total probability mass — within one
    # iteration of Lloyd-Max optimum for unimodal symmetric
    # distributions like Beta(d/2, d/2).
    quantile_pts = (torch.arange(K, dtype=torch.float64) + 0.5) / K
    init = torch.from_numpy(
        scipy_beta.ppf(quantile_pts.numpy(), alpha, alpha)
    ).to(torch.float32)
    # ``ppf`` can return inf at the extreme tails when alpha is
    # gigantic; clamp to (0, 1).
    init = init.clamp(min=1e-6, max=1.0 - 1e-6)
    if torch.isnan(init).any():
        # Final fallback: uniform in [0, 1].
        init = torch.linspace(1.0 / (K + 1), 1.0 - 1.0 / (K + 1), K)
    centroids = init.clone()

    # ``random_state=42`` is fine to hard-code: the codebook is meant
    # to be deterministic across processes so every miner / validator
    # quantises identically. Different (d, b) pairs use the same seed
    # but different ``alpha``, so the samples differ.
    samples = torch.from_numpy(
        scipy_beta.rvs(alpha, alpha, size=num_samples, random_state=42)
    ).to(torch.float32)

    for _ in range(num_iters):
        # Pairwise |sample - centroid| then argmin over centroids.
        # Memory: O(num_samples * K). At num_samples=10K and K=256
        # that's 10MB — fine. Larger combos would need chunking.
        dists = (samples.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        nearest = dists.argmin(dim=1)
        new_centroids = centroids.clone()
        for c in range(K):
            mask = nearest == c
            if mask.any():
                # Empty buckets keep their previous centroid — Lloyd-Max
                # convention to avoid degenerate splits at high K.
                new_centroids[c] = samples[mask].mean()
        delta = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if delta < 1e-6:
            break

    # Sort so :func:`_quantize_codes_for_centroids` can use binary
    # search via torch.searchsorted instead of an O(NK) distance
    # matrix on the hot path.
    centroids, _ = centroids.sort()

    _CENTROID_CACHE[key] = centroids
    _save_centroids_to_disk(int(d), int(b), centroids)
    return centroids


def _quantize_codes_for_centroids(
    values: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_size: int = 1_000_000,
) -> torch.Tensor:
    """Map each ``values`` entry to its nearest centroid index (uint8).

    Uses ``torch.searchsorted`` on the SORTED centroids — O(N log K)
    instead of the O(NK) full distance matrix. Chunked along the
    flattened input so multi-million-element EF tensors don't
    allocate a (N, K) intermediate.

    Args:
        values: Tensor of arbitrary shape with values in roughly the
            same range as ``centroids`` (we don't rescale here).
        centroids: 1-D tensor of length K, sorted ascending. Must
            have K <= 256 to fit in uint8.
        chunk_size: Per-chunk row count for the searchsorted call.

    Returns:
        uint8 tensor with the same shape as ``values``.
    """
    K = int(centroids.numel())
    if K > 256:
        raise ValueError(f"_quantize_codes_for_centroids: K={K} > 256 (uint8 limit)")
    if values.numel() == 0:
        return torch.empty(values.shape, dtype=torch.uint8, device=values.device)

    centroids = centroids.to(values.device, dtype=values.dtype)
    flat = values.contiguous().flatten()
    out = torch.empty(flat.shape, dtype=torch.uint8, device=values.device)

    for start in range(0, flat.numel(), chunk_size):
        end = min(start + chunk_size, flat.numel())
        chunk = flat[start:end]
        # ``insert`` is the index where ``chunk[i]`` would go in
        # ``centroids`` to keep it sorted; the nearest centroid is
        # one of (insert - 1, insert) clamped into [0, K-1].
        insert = torch.searchsorted(centroids, chunk)
        right = insert.clamp(max=K - 1)
        left = (insert - 1).clamp(min=0)
        # Compare absolute distances to the two neighbours.
        d_right = (chunk - centroids[right]).abs()
        d_left = (chunk - centroids[left]).abs()
        nearest = torch.where(d_right < d_left, right, left)
        out[start:end] = nearest.to(torch.uint8)

    return out.view(values.shape)


def quantize_turboquant(
    g: torch.Tensor,
    d: int,
    b: int = 4,
    mode: Literal["mse", "prod"] = "mse",
    outlier_top_k: int = 32,
) -> tuple[torch.Tensor, dict]:
    """Quantize a 1-D rotated gradient via Beta-Lloyd-Max codebook.

    Args:
        g: 1-D float tensor of length ``d``. Must already be
            Hadamard-rotated by the caller — the codebook assumes the
            post-rotation Beta(d/2, d/2) distribution.
        d: Length of ``g``. Redundant but kept in the signature so
            callers can sanity-check before allocating the codebook.
        b: Total bits per coordinate budget. ``main_bits`` is ``b - 1``
            in prod mode, ``b`` in mse mode; ``outlier_bits`` is
            ``min(b + 1, 8)``.
        mode: ``"mse"`` for plain b-bit Q_mse; ``"prod"`` for the
            inner-product mode that spends one bit on a 1-bit residual
            sign correction (TurboQuant paper §3.2).
        outlier_top_k: Number of top-|coordinate| entries to encode
            with the higher-resolution outlier codebook (paper §4.3).
            Set to 0 to disable outlier handling.

    Returns:
        ``(codes, meta)`` where:

        * ``codes`` is a ``(d,)`` ``uint8`` tensor of bulk codebook
          indices. Codes at outlier positions are also written via
          the bulk codebook so a TurboQuant-unaware reader still gets
          a graceful (lossier) reconstruction; the high-res override
          lives in ``meta['outlier_codes']``.
        * ``meta`` carries everything the decoder needs:
          ``bulk_centroids``, ``bulk_max``, ``outlier_indices``,
          ``outlier_codes``, ``outlier_centroids``, ``outlier_max``,
          ``sign_bits`` (prod mode only), ``mode``, ``b``, ``d``.
    """
    if g.dim() != 1:
        raise ValueError(
            f"quantize_turboquant expects 1-D input, got shape {tuple(g.shape)}"
        )
    if g.numel() != d:
        raise ValueError(f"quantize_turboquant: d={d} mismatches g.numel()={g.numel()}")
    if mode not in ("mse", "prod"):
        raise ValueError(f"quantize_turboquant: mode must be 'mse' or 'prod', got {mode!r}")

    main_bits = b - 1 if mode == "prod" else b
    if main_bits < 1:
        raise ValueError(
            f"quantize_turboquant: main_bits={main_bits} < 1; need b>=2 in prod mode"
        )
    if main_bits > 8:
        raise ValueError(
            f"quantize_turboquant: main_bits={main_bits} > 8 (uint8 limit)"
        )

    # Promote to fp32 for the centroid math regardless of input dtype;
    # the receiver will cast back to whatever the param needs.
    g32 = g.detach().to(torch.float32)
    abs_g = g32.abs()

    # Outlier identification — top-k |coords|. ``k=0`` skips outlier
    # handling entirely (matching the paper's b-bit-only baseline).
    k = max(0, min(int(outlier_top_k), d))
    if k > 0:
        outlier_indices = abs_g.topk(k).indices
        outlier_mask = torch.zeros(d, dtype=torch.bool, device=g32.device)
        outlier_mask[outlier_indices] = True
    else:
        outlier_indices = torch.empty(0, dtype=torch.long, device=g32.device)
        outlier_mask = torch.zeros(d, dtype=torch.bool, device=g32.device)

    # Bulk normalisation: the Lloyd-Max codebook is fit on Beta(d/2, d/2)
    # which lives in [0, 1] and concentrates near 0.5 with std ~1/sqrt(d).
    # The TurboQuant prescription scales by the L2 NORM of the bulk
    # coords (not the max-abs); for a unit vector this yields per-coord
    # values in roughly Beta-shaped concentration around 0.5. Max-abs
    # scaling instead produces values with std much larger than the
    # codebook's range, which would clamp the vast majority of coords
    # to the extreme centroids and waste most of the bit budget.
    bulk_only = g32[~outlier_mask] if k > 0 and k < d else g32
    if bulk_only.numel() == 0:
        bulk_norm = 1.0
    else:
        bulk_norm = max(float(bulk_only.norm().item()), 1e-12)

    bulk_scaled = ((g32 / bulk_norm) + 1.0) / 2.0
    bulk_scaled = bulk_scaled.clamp(0.0, 1.0)

    bulk_centroids = beta_lloyd_max_centroids(d, main_bits)
    bulk_codes = _quantize_codes_for_centroids(bulk_scaled, bulk_centroids)

    outlier_codes: torch.Tensor | None = None
    outlier_centroids: torch.Tensor | None = None
    outlier_norm: float | None = None
    outlier_bits = 0
    if k > 0:
        outlier_bits = min(b + 1, 8)
        outlier_data = g32[outlier_indices]
        outlier_norm = max(float(outlier_data.norm().item()), 1e-12)
        outlier_scaled = ((outlier_data / outlier_norm) + 1.0) / 2.0
        outlier_scaled = outlier_scaled.clamp(0.0, 1.0)
        # Outlier codebook is sized for the OUTLIER subset, not full d:
        # ``k`` coords with their own concentration shape.
        outlier_centroids = beta_lloyd_max_centroids(max(k, 2), outlier_bits)
        outlier_codes = _quantize_codes_for_centroids(outlier_scaled, outlier_centroids)

    sign_bits: torch.Tensor | None = None
    if mode == "prod":
        # 1-bit Q_prod residual: sign of (g - dequant(g)). The decoder
        # nudges each reconstructed coord by half the inter-centroid
        # spacing in the residual direction, shrinking the mean
        # absolute reconstruction error without spending a full extra
        # quantisation bit. Bit value: 1 iff residual > 0.
        bulk_recon_unit = bulk_centroids.to(g32.device)[bulk_codes.long()] * 2.0 - 1.0
        bulk_recon = bulk_recon_unit * bulk_norm
        residual = g32 - bulk_recon
        sign_bits = (residual > 0).to(torch.uint8)

    meta: dict = {
        "version": "turboquant_v1",
        "d": int(d),
        "b": int(b),
        "main_bits": int(main_bits),
        "mode": mode,
        "bulk_centroids": bulk_centroids.detach().cpu(),
        "bulk_scale": float(bulk_norm),
        "outlier_indices": outlier_indices.detach().cpu(),
        "outlier_codes": (
            outlier_codes.detach().cpu() if outlier_codes is not None else None
        ),
        "outlier_centroids": (
            outlier_centroids.detach().cpu() if outlier_centroids is not None else None
        ),
        "outlier_scale": outlier_norm,
        "outlier_bits": int(outlier_bits),
        "sign_bits": sign_bits.detach().cpu() if sign_bits is not None else None,
    }
    return bulk_codes, meta


def dequantize_turboquant(codes: torch.Tensor, meta: dict) -> torch.Tensor:
    """Inverse of :func:`quantize_turboquant`.

    Reconstructs a 1-D float32 tensor of length ``meta['d']`` from the
    bulk codes and the meta dict. Output device matches ``codes``;
    output dtype is float32 (caller casts).

    Three-step reconstruction:

    1. Bulk: ``out = (bulk_centroids[codes] * 2 - 1) * bulk_max``.
    2. Outlier override: positions in ``meta['outlier_indices']`` are
       overwritten with the higher-resolution outlier codebook.
    3. Prod-mode residual nudge: each coord is shifted by half the
       average centroid spacing in the direction encoded by
       ``meta['sign_bits']``.
    """
    if "version" not in meta or meta["version"] != "turboquant_v1":
        raise ValueError(
            f"dequantize_turboquant: unknown meta version "
            f"{meta.get('version')!r}; expected 'turboquant_v1'"
        )

    bulk_centroids = meta["bulk_centroids"].to(codes.device, dtype=torch.float32)
    bulk_scale = float(meta["bulk_scale"])

    out = (bulk_centroids[codes.long()] * 2.0 - 1.0) * bulk_scale

    outlier_indices = meta.get("outlier_indices")
    if outlier_indices is not None and outlier_indices.numel() > 0:
        outlier_centroids = meta["outlier_centroids"].to(
            codes.device, dtype=torch.float32
        )
        outlier_codes = meta["outlier_codes"].to(codes.device, dtype=torch.uint8)
        outlier_scale = float(meta["outlier_scale"])
        outlier_recon = (
            outlier_centroids[outlier_codes.long()] * 2.0 - 1.0
        ) * outlier_scale
        out[outlier_indices.to(codes.device, dtype=torch.long)] = outlier_recon

    sign_bits = meta.get("sign_bits")
    if sign_bits is not None and bulk_centroids.numel() >= 2:
        # Inter-centroid spacing is the average gap between adjacent
        # sorted centroids on the [0, 1] codebook scale; rescale to
        # the bulk_scale range and apply half-spacing as the nudge
        # magnitude. Direction comes from sign_bits.
        spacing = (bulk_centroids[1:] - bulk_centroids[:-1]).mean().item()
        correction = spacing * bulk_scale
        signs = sign_bits.to(codes.device, dtype=out.dtype) * 2.0 - 1.0
        out = out + signs * (correction * 0.5)

    return out


@torch.no_grad()
def batch_decompress_turboquant(
    param: torch.Tensor,
    tq_idxs_list: Sequence[torch.Tensor],
    tq_codes_list: Sequence[torch.Tensor],
    tq_metas: Sequence[dict],
    *,
    peer_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Cross-peer TurboQuant decode (FU1; validator side of P6b).

    Per-peer the wire carries three keys under ``cname + 'tq_*'``
    (populated by :func:`hone.neurons.prepare_gradient_dict`'s
    TurboQuant branch):

    * ``tq_idxs`` — 1-D int64 indices into the flat rotated vector;
      length ``k`` (the kept-top-K budget, ``(d * topk) // target_chunk``).
    * ``tq_codes`` — 1-D uint8 bulk codes at those ``k`` indices
      (i.e., ``full_bulk_codes[tq_idxs]``). Raw codes, NOT dequantised.
    * ``tq_meta`` — a Python ``dict`` from :func:`quantize_turboquant`
      carrying the bulk / outlier centroids, bulk & outlier scales,
      full-d ``outlier_indices`` + ``outlier_codes``, ``sign_bits``
      (full-d uint8 in prod mode), plus ``d``, ``b``, ``mode``.

    Per-peer algorithm:

    1. Scatter ``tq_codes`` into a length-``d`` uint8 buffer filled
       with zeros at non-top-K positions. The zero fill is immaterial:
       whatever :func:`dequantize_turboquant` produces off-top-K will
       be masked back to zero in step 2. Outlier positions inside the
       top-K still use the higher-resolution outlier override inside
       :func:`dequantize_turboquant` because ``meta['outlier_indices']``
       and ``meta['outlier_codes']`` travel untouched from sender to
       receiver (the outlier codebook is keyed off the full-d outlier
       set, not the kept top-K subset).
    2. :func:`dequantize_turboquant` on the full-d codes tensor. Mask
       the result to zero at positions NOT in ``tq_idxs`` — this
       mirrors the miner's ``sparse_rotated`` construction exactly.
    3. Apply :func:`hadamard_rotate` (WHT is self-inverse on
       power-of-two last-dim length, see
       ``turboquant_audit.hadamard_rotate`` docstring). The P6b miner's
       encode rotates, quantises, top-K-on-codes, and sparsifies the
       rotated vector; the validator's decode inverts the rotation to
       recover the gradient in the original coordinate frame.
    4. Reshape to ``param.shape`` and cast to ``param.dtype``.

    Cross-peer reduction: simple weighted average with weights
    normalised to sum to 1. When ``peer_weights`` is ``None`` the
    weights are uniform (1/P) — byte-for-byte parity with the legacy
    ``batch_decompress`` mean reduce for uniform inputs. P2's
    ``_peer_weights_for_param`` helper in ``outer_step`` slices the
    validator's token-weighted dict per param and passes it here; the
    same ``sum <= 0`` → uniform fallback the legacy codec uses applies,
    so a malformed weights list degrades gracefully instead of
    zeroing the gradient.

    Args:
        param: Model parameter tensor; used ONLY for ``.shape``,
            ``.dtype``, and ``.device`` — not read or mutated.
        tq_idxs_list: One 1-D int64 tensor per contributing peer,
            each of length ``k`` (the sender's top-K budget).
        tq_codes_list: One 1-D uint8 tensor per contributing peer,
            length matching the corresponding ``tq_idxs``.
        tq_metas: One metadata dict per contributing peer; all must
            agree on ``meta['d']``.
        peer_weights: Optional P2 token-weighted aggregation weights,
            one per peer in the same order as ``tq_idxs_list``. Length
            MUST equal ``len(tq_metas)``. ``None`` → uniform mean.
            Normalised internally so ``sum(normalised) == 1``; if the
            input sum is ``<= 0`` the function degrades to uniform
            (matching :meth:`hone.compress.TopKCompressor.batch_decompress`).

    Returns:
        Dense tensor with ``param.shape`` and ``param.dtype`` on
        ``param.device``. Always returns a fresh allocation — safe to
        assign straight to ``p.grad``.
    """
    n_peers = len(tq_metas)
    if len(tq_idxs_list) != n_peers or len(tq_codes_list) != n_peers:
        raise ValueError(
            "batch_decompress_turboquant: tq_idxs_list, tq_codes_list and "
            f"tq_metas must all have the same length; got "
            f"{len(tq_idxs_list)}, {len(tq_codes_list)}, {n_peers}"
        )

    target_device = param.device
    target_dtype = param.dtype

    if n_peers == 0:
        # No peers contributed; hand back a zero tensor of the right
        # shape so the caller can assign it into ``p.grad`` without
        # needing a None-check. This also matches the "no update for
        # this param" code path in ``outer_step``.
        return torch.zeros(param.shape, dtype=target_dtype, device=target_device)

    # Normalise peer weights. Length-validated against n_peers; sum-<=0
    # falls back to uniform (same contract as
    # ``TopKCompressor.batch_decompress`` — see its docstring for the
    # reasoning).
    if peer_weights is None:
        normalised_weights: list[float] = [1.0 / n_peers] * n_peers
    else:
        weights_list = [float(w) for w in peer_weights]
        if len(weights_list) != n_peers:
            raise ValueError(
                f"batch_decompress_turboquant: peer_weights length "
                f"{len(weights_list)} does not match number of peers {n_peers}"
            )
        weight_sum = float(sum(weights_list))
        if weight_sum <= 0.0:
            normalised_weights = [1.0 / n_peers] * n_peers
        else:
            normalised_weights = [w / weight_sum for w in weights_list]

    # All peers SHOULD agree on ``d`` (same model → same param → same
    # flat length). Guard against a malformed meta that disagrees; the
    # mismatch would silently corrupt the cross-peer sum otherwise.
    d = int(tq_metas[0]["d"])
    for p_idx in range(1, n_peers):
        d_p = int(tq_metas[p_idx]["d"])
        if d_p != d:
            raise ValueError(
                f"batch_decompress_turboquant: peer {p_idx} has d={d_p} "
                f"but peer 0 has d={d}; refusing to merge across shapes"
            )
    if d != int(param.numel()):
        raise ValueError(
            f"batch_decompress_turboquant: meta d={d} does not match "
            f"param.numel()={int(param.numel())}"
        )

    # fp32 accumulator on the param device. Keep dtype separate from
    # ``param.dtype`` so low-precision params (bf16/fp16) don't lose
    # headroom across the cross-peer sum; final cast happens once at
    # the end.
    accum = torch.zeros(d, dtype=torch.float32, device=target_device)

    for p_idx in range(n_peers):
        tq_idxs = tq_idxs_list[p_idx].to(target_device, dtype=torch.long)
        tq_codes_subset = tq_codes_list[p_idx].to(
            target_device, dtype=torch.uint8
        )
        meta = tq_metas[p_idx]

        if tq_idxs.numel() == 0:
            # Empty contribution from this peer — skip without
            # adding to the accumulator. Weight is "wasted" but the
            # caller's normalisation is done above so the remaining
            # peers are not double-counted.
            continue
        if tq_idxs.numel() != tq_codes_subset.numel():
            raise ValueError(
                f"batch_decompress_turboquant: peer {p_idx} tq_idxs "
                f"({tq_idxs.numel()}) / tq_codes ({tq_codes_subset.numel()}) "
                "length mismatch"
            )

        # Reconstruct a length-d codes buffer. Non-top-K positions
        # are filled with 0 — whatever ``dequantize_turboquant``
        # produces there is irrelevant because the sparsify mask in
        # step 2 zeros those coords back out. At top-K positions
        # that are ALSO in ``meta['outlier_indices']``, the outlier
        # override inside :func:`dequantize_turboquant` writes the
        # higher-res outlier reconstruction; matches what the miner's
        # local ``full_dequant`` produces on encode.
        full_codes = torch.zeros(d, dtype=torch.uint8, device=target_device)
        full_codes[tq_idxs] = tq_codes_subset

        dequantized_full = dequantize_turboquant(full_codes, meta)

        # Sparsify: zero everywhere except the sender's top-K. This is
        # the exact mirror of the miner's
        # ``sparse_rotated = torch.zeros_like(rotated);
        #  sparse_rotated[topk_idx] = full_dequant[topk_idx]`` step in
        # ``prepare_gradient_dict``.
        sparse_rotated = torch.zeros(
            d, dtype=torch.float32, device=target_device
        )
        sparse_rotated[tq_idxs] = dequantized_full[tq_idxs].to(torch.float32)

        # Inverse rotation (WHT is its own inverse at power-of-two d;
        # for non-POT d the forward pass already pads + truncates, so
        # decode's sqrt(n)-normalised WHT is the best rotationally-
        # consistent inverse the audit-gated codec exposes — identical
        # projection loss on encode and decode sides).
        peer_flat = hadamard_rotate(sparse_rotated.unsqueeze(0)).squeeze(0)

        accum.add_(peer_flat, alpha=normalised_weights[p_idx])

        del full_codes, dequantized_full, sparse_rotated, peer_flat

    # One cast at the end to avoid repeated narrowing during the
    # per-peer accumulate loop.
    return accum.view(param.shape).to(dtype=target_dtype)


def _run_self_tests() -> None:  # pragma: no cover - exercised via __main__
    """End-to-end correctness checks for the TurboQuant codec.

    Validates:

    1. :func:`hadamard_rotate` is identical to the audit module's
       (the codec must use the same rotation the gate audited against).
    2. :func:`beta_lloyd_max_centroids` returns a sorted ``2^b``
       tensor in ``(0, 1)`` for representative ``(d, b)``.
    3. ``quantize -> dequantize`` round-trip on a Beta-distributed
       synthetic vector has MSE below the rate-distortion-style bound
       ``1 / 2^(2b - 2)`` (the user's spec target — generous because
       it doesn't account for the bulk-max scaling factor).
    4. Outlier handling reduces MSE on dominant directions vs the
       no-outlier baseline.
    5. Prod mode reduces MSE vs MSE mode at matched total bits.
    6. Unknown meta version raises explicitly (no silent corruption
       if a future codec ships v2).
    """
    from hone import turboquant_audit as _audit_mod

    torch.manual_seed(0)

    # --- (1) WHT identity with the audit module ------------------------
    assert hadamard_rotate is _audit_mod.hadamard_rotate, (
        "hadamard_rotate must be the same function in turboquant and "
        "turboquant_audit; the audit gate validates the codec's "
        "rotation, not a parallel implementation."
    )

    # --- (2) centroid contract -----------------------------------------
    for d, b in [(64, 2), (256, 3), (1024, 4), (4096, 4)]:
        c = beta_lloyd_max_centroids(d, b)
        assert c.shape == (1 << b,), f"d={d}, b={b}: shape {tuple(c.shape)}"
        assert (c >= 0).all() and (c <= 1).all(), (
            f"d={d}, b={b}: centroids escaped [0, 1]"
        )
        assert torch.equal(c, c.sort().values), (
            f"d={d}, b={b}: centroids not sorted"
        )

    # --- (3) round-trip MSE bound --------------------------------------
    # User's spec: round-trip on a Beta-distributed synthetic vector has
    # MSE < 1 / 2^(2b - 2). Generous because it ignores the bulk-max
    # rescale; in practice MSE comes out 100x lower.
    try:
        from scipy.stats import beta as scipy_beta
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("self-tests require scipy") from exc

    d = 1024
    b = 4
    samples = torch.from_numpy(
        scipy_beta.rvs(d / 2, d / 2, size=d, random_state=7) - 0.5
    ).to(torch.float32)
    g = samples * 0.2

    codes, meta = quantize_turboquant(g, d=d, b=b, mode="mse", outlier_top_k=0)
    assert codes.shape == (d,)
    assert codes.dtype == torch.uint8
    recon = dequantize_turboquant(codes, meta)
    mse = float(((g - recon) ** 2).mean().item())
    bound = 1.0 / (2 ** (2 * b - 2))
    assert mse < bound, (
        f"quantize round-trip MSE {mse:.6e} exceeds bound 1/2^(2b-2) = {bound:.6e}"
    )

    # --- (4) outliers reduce MSE on dominant coords --------------------
    # The TurboQuant prescription only helps when outliers truly
    # dominate (~100x the bulk std). For the norm-scaled codec these
    # land in the [-1, 1] codebook range only when given their own
    # ``outlier_norm`` scale; without outlier handling the bulk-norm
    # scaling clamps them to the codebook extremes and reconstruction
    # collapses to ~0. This test makes the contrast unambiguous.
    g_with_outliers = g.clone()
    bulk_std = float(g.std().item())
    outlier_indices = torch.arange(8)
    g_with_outliers[outlier_indices] = (
        torch.tensor([1.0, -1.0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7]) * 100.0 * bulk_std
    )
    codes_no, meta_no = quantize_turboquant(
        g_with_outliers, d=d, b=b, mode="mse", outlier_top_k=0
    )
    codes_yes, meta_yes = quantize_turboquant(
        g_with_outliers, d=d, b=b, mode="mse", outlier_top_k=32
    )
    recon_no = dequantize_turboquant(codes_no, meta_no)
    recon_yes = dequantize_turboquant(codes_yes, meta_yes)
    mse_no = float(((g_with_outliers[outlier_indices] - recon_no[outlier_indices]) ** 2).mean().item())
    mse_yes = float(((g_with_outliers[outlier_indices] - recon_yes[outlier_indices]) ** 2).mean().item())
    assert mse_yes < mse_no, (
        f"outlier handling should improve dominant-coord MSE: "
        f"with_outliers={mse_yes:.6e} >= without={mse_no:.6e}"
    )

    # --- (5) prod mode meta carries sign_bits --------------------------
    codes_prod, meta_prod = quantize_turboquant(
        g, d=d, b=b, mode="prod", outlier_top_k=0
    )
    assert meta_prod["sign_bits"] is not None
    assert meta_prod["sign_bits"].shape == (d,)
    recon_prod = dequantize_turboquant(codes_prod, meta_prod)
    # Sanity: prod-mode reconstruction is in the same scale.
    assert recon_prod.shape == (d,)
    assert recon_prod.abs().max() < g.abs().max() * 2.0

    # --- (6) unknown meta version is rejected --------------------------
    bad_meta = dict(meta)
    bad_meta["version"] = "turboquant_v999"
    try:
        dequantize_turboquant(codes, bad_meta)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on unknown meta version")

    # --- (7) cache hit / disk persistence ------------------------------
    # First call already populated the in-memory cache; second call
    # must hit it without re-running scipy.
    key = (1024, 4)
    assert key in _CENTROID_CACHE, "centroid in-memory cache miss"
    cached_again = beta_lloyd_max_centroids(1024, 4)
    assert cached_again is _CENTROID_CACHE[key]

    print(
        "[hone.turboquant] self-tests passed: "
        f"WHT identity, centroid contract, round-trip MSE={mse:.3e} "
        f"(bound {bound:.3e}), outlier reduction "
        f"(no={mse_no:.3e} -> yes={mse_yes:.3e}), prod mode, "
        "version guard, cache hit."
    )


def _test_batch_decompress_roundtrip() -> None:  # pragma: no cover - __main__
    """FU1 (P6b): miner encode → validator decode single-peer round-trip.

    Mirrors the full prepare_gradient_dict → gather → outer_step pipeline
    for a single contributing peer, asserting that the reconstructed
    dense gradient preserves the sender's direction (cosine > 0.5) and
    a substantial fraction of its energy. Uses the same d, b, mode,
    outlier_top_k defaults the production miner hparams ship with
    (``turboquant_bits=4``, ``turboquant_q_prod_mode=true``,
    ``turboquant_outlier_top_k=32``).

    The 50% energy + 0.5 cos thresholds are deliberately loose because
    this single-peer decode goes through (a) top-K selection on code
    magnitudes, (b) Beta-Lloyd-Max scalar quantisation, and (c)
    truncation-style WHT round-trip, each of which loses signal. The
    validator-side cross-peer mean averages back a lot of that noise;
    the single-peer test here is a floor, not a ceiling.
    """
    torch.manual_seed(0)
    d = 4096
    g = torch.randn(d) * 0.01  # realistic gradient scale

    rotated = hadamard_rotate(g.unsqueeze(0)).squeeze(0)
    codes, meta = quantize_turboquant(
        rotated, d=d, b=4, mode="prod", outlier_top_k=32
    )
    topk_k = d // 8  # matches ``(d * topk) // target_chunk`` for 32/64
    topk_idx = codes.long().abs().topk(topk_k).indices

    param = torch.zeros(d)
    result = batch_decompress_turboquant(
        param=param,
        tq_idxs_list=[topk_idx],
        tq_codes_list=[codes[topk_idx]],
        tq_metas=[meta],
        peer_weights=None,
    )

    assert result.shape == g.shape, (
        f"result shape {tuple(result.shape)} != g {tuple(g.shape)}"
    )
    assert result.dtype == param.dtype, (
        f"result dtype {result.dtype} != param {param.dtype}"
    )

    g_norm = float(g.norm().item())
    result_norm = float(result.norm().item())
    assert result_norm > 0.5 * g_norm, (
        f"decode lost too much energy: {result_norm:.4f} vs {g_norm:.4f}"
    )
    cos = torch.nn.functional.cosine_similarity(
        result.unsqueeze(0), g.unsqueeze(0), dim=1
    ).item()
    assert cos > 0.5, f"decode cosine too low: {cos:.4f}"
    print(
        f"[FU1 P6b decode] energy={result_norm / g_norm:.3f} "
        f"cos={cos:.3f} k={topk_k}"
    )


def _test_multi_peer_weighted() -> None:  # pragma: no cover - __main__
    """FU1 peer_weights correctness over multiple peers.

    Two-peer asymmetric-weights sanity check. Each peer encodes a
    different gradient; merging with uniform weights should land
    somewhere in between, while merging with a strongly skewed weight
    (99% on peer 0) should closely track peer 0's single-peer decode.
    Cosine similarity is the cleanest signal because TurboQuant's
    top-K-on-codes selection does not preserve absolute magnitudes
    exactly.
    """
    torch.manual_seed(1)
    d = 4096

    g0 = torch.randn(d) * 0.01
    g1 = torch.randn(d) * 0.01

    def _encode(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        rot = hadamard_rotate(g.unsqueeze(0)).squeeze(0)
        codes_, meta_ = quantize_turboquant(
            rot, d=d, b=4, mode="prod", outlier_top_k=32
        )
        kk = d // 8
        idx_ = codes_.long().abs().topk(kk).indices
        return idx_, codes_[idx_], meta_

    idx0, codes0, meta0 = _encode(g0)
    idx1, codes1, meta1 = _encode(g1)

    # Single-peer decode of each for the cosine reference.
    param = torch.zeros(d)
    decoded0 = batch_decompress_turboquant(
        param, [idx0], [codes0], [meta0], peer_weights=None
    )
    decoded1 = batch_decompress_turboquant(
        param, [idx1], [codes1], [meta1], peer_weights=None
    )

    # Uniform merge sits between the two.
    uniform_merge = batch_decompress_turboquant(
        param,
        [idx0, idx1],
        [codes0, codes1],
        [meta0, meta1],
        peer_weights=None,
    )
    cos_u0 = torch.nn.functional.cosine_similarity(
        uniform_merge.unsqueeze(0), decoded0.unsqueeze(0), dim=1
    ).item()
    cos_u1 = torch.nn.functional.cosine_similarity(
        uniform_merge.unsqueeze(0), decoded1.unsqueeze(0), dim=1
    ).item()

    # Skew toward peer 0; result tracks peer 0's direction far more
    # closely than peer 1's.
    skewed_merge = batch_decompress_turboquant(
        param,
        [idx0, idx1],
        [codes0, codes1],
        [meta0, meta1],
        peer_weights=[0.99, 0.01],
    )
    cos_s0 = torch.nn.functional.cosine_similarity(
        skewed_merge.unsqueeze(0), decoded0.unsqueeze(0), dim=1
    ).item()
    cos_s1 = torch.nn.functional.cosine_similarity(
        skewed_merge.unsqueeze(0), decoded1.unsqueeze(0), dim=1
    ).item()

    # Key invariant: the 99%-peer-0 merge correlates with peer 0 MUCH
    # more than with peer 1, AND more than the uniform merge does.
    assert cos_s0 > cos_u0 - 1e-6, (
        f"skewed merge should track peer 0 at least as well as uniform; "
        f"got cos_s0={cos_s0:.3f} cos_u0={cos_u0:.3f}"
    )
    assert cos_s0 > cos_s1, (
        f"skewed merge should track peer 0 more than peer 1; "
        f"cos_s0={cos_s0:.3f} cos_s1={cos_s1:.3f}"
    )

    # Degenerate weights (all zero) fall back to uniform — mirrors the
    # legacy ``TopKCompressor.batch_decompress`` contract and keeps a
    # broken validator weights dict from silently zeroing the merged
    # gradient.
    zero_fallback = batch_decompress_turboquant(
        param,
        [idx0, idx1],
        [codes0, codes1],
        [meta0, meta1],
        peer_weights=[0.0, 0.0],
    )
    cos_zero_uniform = torch.nn.functional.cosine_similarity(
        zero_fallback.unsqueeze(0), uniform_merge.unsqueeze(0), dim=1
    ).item()
    assert cos_zero_uniform > 0.999, (
        f"zero-weights fallback must equal uniform merge; "
        f"cos={cos_zero_uniform:.6f}"
    )

    # Empty peer list yields a zero tensor (no update for this param),
    # matching outer_step's "skip" path expectation.
    empty = batch_decompress_turboquant(param, [], [], [], peer_weights=None)
    assert empty.shape == param.shape
    assert float(empty.abs().max().item()) == 0.0, (
        f"empty peer list must yield zeros; got max abs "
        f"{float(empty.abs().max().item())}"
    )

    print(
        "[FU1 P6b multi-peer] "
        f"uniform: cos_u0={cos_u0:.3f} cos_u1={cos_u1:.3f}; "
        f"skewed(99/1): cos_s0={cos_s0:.3f} cos_s1={cos_s1:.3f}; "
        f"zero-fallback cos_uniform={cos_zero_uniform:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    _run_self_tests()
    _test_batch_decompress_roundtrip()
    _test_multi_peer_weighted()
