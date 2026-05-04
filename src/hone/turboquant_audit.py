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

"""TurboQuant coordinate-distribution audit (P6a — hard gate for P6b).

The TurboQuant codec (``hone.turboquant``) only works when the per-coordinate
distribution of the post-Hadamard-rotation gradient looks like the
Beta(d/2, d/2) the Lloyd-Max codebook was fit against. EF tensors that
have been accumulated across many windows of biased gradient flow may
NOT satisfy this assumption (the structural-bias hypothesis flagged in
P6 of the throughput plan); shipping TurboQuant against a non-Beta
distribution silently degrades reconstruction quality without an
operator-visible failure mode.

This module is the gate. It exposes:

- :func:`hadamard_rotate` — the same Walsh-Hadamard butterfly the codec
  uses, kept here so the audit and the production codec rotate identically.
- :func:`classify_tensor` — a name-based heuristic that maps each
  parameter to one of {embedding, router, moe_expert, attention, mlp,
  other}; the audit aggregates pass-rates per class so an operator can
  see WHICH part of the model breaks the Beta assumption (e.g. routers
  almost always fail; embeddings almost always pass).
- :func:`audit_tensor` — runs WHT on a single tensor, normalises to
  unit length, KS-tests the squared coordinates against the Beta
  prescription per the plan spec, and returns a structured
  :class:`AuditResult`.
- :data:`PASS_RATE_THRESHOLD` — the canonical 80% per-class pass rate
  the CLI uses to gate the P6a → P6b transition.

The CLI wrapper lives in ``hone/validator/turboquant_audit_cli.py``;
this module is the reusable library it builds on.

Operator workflow (per ``hone/docs/turboquant.md``)::

    # On a running miner mid-training:
    kill -USR2 <miner_pid>          # dumps EF to /tmp/ef_snapshot_uid<U>_w<W>.pt

    # Then:
    python hone/validator/turboquant_audit_cli.py \\
        /tmp/ef_snapshot_uid<U>_w<W>.pt --json out.json

If the CLI exits 0, the per-class pass rate cleared the gate and
TurboQuant (``hparams.turboquant_enabled``) is safe to enable. If it
exits 1, the gate failed; keep the codec dark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# scipy is required for the KS / Beta test; it's already a hard
# dependency of ``hone`` (see ``pyproject.toml``), so this import is
# expected to succeed in any environment that can run a validator.
# Wrapped in a try so the module still IMPORTS in a hypothetical
# scipy-free environment — :func:`audit_tensor` raises explicitly at
# call time. This matters because ``hone/__init__.py`` exports the
# WHT primitives and we don't want a missing scipy to break unrelated
# code paths (e.g. the production codec doesn't need scipy).
try:
    from scipy.stats import kstest as _scipy_kstest  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - scipy is a hard dep
    _scipy_kstest = None


PASS_RATE_THRESHOLD: float = 0.80
"""Per-class fraction of audited tensors whose KS p-value must exceed
``BETA_KS_P_THRESHOLD`` for the class to be considered "Beta-conforming".

The CLI computes the worst class's pass rate and gates exit-code 0 on
``worst_class_pass_rate >= PASS_RATE_THRESHOLD``. 80% is the operator
contract from the P6a plan: a single odd-shaped tensor in a class of
hundreds shouldn't sink the gate, but a class where the majority fails
must (e.g. routers, where TurboQuant is known not to fit).
"""

BETA_KS_P_THRESHOLD: float = 0.05
"""KS-test p-value above which a single tensor counts as "Beta-passed"."""


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Apply the Walsh-Hadamard Transform along the last dim, normalised
    so the rotation is its own inverse on power-of-two last-dim lengths.

    For ``x`` with last-dim length ``d`` that is NOT a power of two, the
    last dim is zero-padded up to the next power of two ``n``, the WHT
    runs on the padded length, and the first ``d`` coordinates of the
    result are returned. The padding does not contaminate the kept
    coordinates in expectation (the synthetic zeros land in the
    discarded ``[d:n]`` slice after rotation), but the mapping is no
    longer a true rotation in the original ``d`` dim — it's a rotated
    projection. This is intentional and matches the production codec's
    behaviour: the audit and codec see the same coords.

    Implementation is the standard butterfly: ``log2(n)`` stages, each
    pairing every coordinate with its partner ``h`` slots away to form
    sum/difference outputs. Time is O(n log n), extra memory is O(n)
    for the working buffer (the input is not mutated). Per-stage view
    reshape uses ``view(-1, 2, h)`` on a contiguous buffer; PyTorch
    backs the slice assignments through to the underlying storage so
    the loop is in-place on the working buffer.

    Self-inverse identity: ``hadamard_rotate(hadamard_rotate(x)) == x``
    when ``d`` is a power of two (modulo float-precision noise; see the
    inline self-test). For non-power-of-two ``d`` the truncation is
    lossy and the identity does not hold.

    Args:
        x: Tensor of arbitrary shape; ``x.shape[-1]`` is the dim
            rotated. Other leading dims are batched (independent rotations).

    Returns:
        Tensor of the SAME shape as ``x`` (same last-dim length ``d``),
        rotated along the last dim and divided by ``sqrt(n)`` so the
        transform is orthogonal on the padded ``n``-dim space.
    """
    orig_shape = x.shape
    d = int(orig_shape[-1])
    if d == 0:
        raise ValueError("hadamard_rotate: last-dim length must be >= 1, got 0")
    if d == 1:
        # The trivial 1-dim "rotation" is identity. Skipping the WHT
        # avoids a division-by-1 and a useless allocation.
        return x.clone()

    # Round d up to the next power of two via the bit-length trick.
    # ``(d-1).bit_length()`` returns ``ceil(log2(d))`` for ``d >= 1``;
    # ``1 << k`` is ``2**k``. Equivalent to ``1 if d == 1 else 2**ceil(log2(d))``.
    n = 1 << (d - 1).bit_length()

    if n != d:
        pad_shape = (*orig_shape[:-1], n - d)
        x_padded = torch.cat(
            [x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)],
            dim=-1,
        )
    else:
        # ``view`` requires contiguous storage; if the caller handed us
        # a slice or transpose, materialise into a fresh buffer.
        x_padded = x.contiguous()

    # Working buffer the butterfly mutates in place. Cloning protects
    # the caller's input even when no padding was needed.
    out = x_padded.clone()

    h = 1
    while h < n:
        # Block layout for stage ``h``: pair coordinate ``2*j*h + k`` with
        # coordinate ``(2*j+1)*h + k`` for every j in [0, n/(2h)) and
        # every k in [0, h). The view exposes those two partners as the
        # ``[..., 0, k]`` and ``[..., 1, k]`` slices of every block.
        view = out.view(*orig_shape[:-1], n // (2 * h), 2, h)
        # Compute both partner outputs BEFORE any in-place write so the
        # ``b`` formula reads the original ``view[..., 0, :]`` (allocated
        # tensors don't share storage with the view).
        a = view[..., 0, :] + view[..., 1, :]
        b = view[..., 0, :] - view[..., 1, :]
        view[..., 0, :] = a
        view[..., 1, :] = b
        h *= 2

    # Normalise so ``H/sqrt(n)`` is orthogonal: ``(H/sqrt(n)) @ (H/sqrt(n))^T = I``.
    # The unnormalised butterfly gives ``H @ x``; dividing by sqrt(n)
    # makes the transform isometric and self-inverse.
    out = out / math.sqrt(n)

    # Discard the synthetic padding region. For power-of-two d this is
    # a no-op slice; for other d we hand back the rotated projection
    # of the original d coords (see docstring).
    return out[..., :d]


# Embedding-style names match an explicit allow-list rather than a
# generic regex because PyTorch / HF repos have used many conventions
# and a single substring like ``embed`` would over-match (e.g. an
# ``embed_dim`` config param surfaced via ``ParameterDict``). The
# canonical names below cover the six conventions seen in production:
# Hone's ``LoopLM`` (embed_tokens, lm_head), GPT-2 (wte), HF generic
# (embedding), and shorthand variants used by some external tokeniser
# stacks (embed, embd).
_EMBED_TOKENS = ("embed", "embd", "wte", "lm_head", "embedding")


def classify_tensor(param_name: str) -> str:
    """Bucket a parameter name into one of six TurboQuant audit classes.

    Order matters: ``embed`` is checked before ``router`` because some
    routing layers live INSIDE embedding modules and we want those
    counted as embeddings (the table-look-up dominates the gradient
    structure). ``moe_expert`` is checked before ``attention`` and
    ``mlp`` so MoE-specific tensors don't get diluted into the generic
    MLP class.

    Returns:
        One of ``"embedding"``, ``"router"``, ``"moe_expert"``,
        ``"attention"``, ``"mlp"``, ``"other"``.
    """
    name_lower = param_name.lower()
    if any(tok in name_lower for tok in _EMBED_TOKENS):
        return "embedding"
    # ``gate_proj`` is the SwiGLU gating in MLPs (NOT a router); the
    # ``and "proj" not in name_lower`` clause distinguishes the two
    # since SwiGLU gates always carry a ``proj`` suffix in HF naming.
    if "router" in name_lower or ("gate" in name_lower and "proj" not in name_lower):
        return "router"
    if "expert" in name_lower or "moe" in name_lower:
        return "moe_expert"
    if "attn" in name_lower or "attention" in name_lower:
        return "attention"
    if "mlp" in name_lower or "feed_forward" in name_lower or "ffn" in name_lower:
        return "mlp"
    return "other"


@dataclass(frozen=True)
class AuditResult:
    """Per-tensor outcome of one TurboQuant Beta-distribution audit."""

    tensor_class: str
    param_name: str
    d: int
    ks_stat: float
    ks_p_value: float
    beta_passed: bool
    mean_rel_error: float


def audit_tensor(
    t: torch.Tensor,
    param_name: str,
    tensor_class: str | None = None,
) -> AuditResult:
    """Rotate ``t`` with WHT, KS-test squared coords against Beta(d/2, d/2).

    The plan-spec test: after Hadamard rotation, the coordinate
    distribution of an EF tensor that the codec can safely quantize
    should match Beta(d/2, d/2) on the squared coordinates of the
    unit-normalised rotated vector. Tensors whose post-rotation
    coordinates are heavy-tailed, multi-modal, or otherwise non-Beta
    (common for router gradients in MoE; less common for embeddings)
    will fail the KS test and the operator will see a per-class
    pass-rate below ``PASS_RATE_THRESHOLD`` from the CLI.

    The KS p-value threshold is ``BETA_KS_P_THRESHOLD`` (0.05). The
    ``mean_rel_error`` field is reported as a sanity check — for any
    unit vector it is ~0 by construction (sum of squared coords is
    exactly 1, so the mean is exactly 1/d) and serves to detect
    rotation / normalisation bugs rather than distribution mismatches.

    Args:
        t: The tensor to audit. Flattened internally; arbitrary shape
            allowed. CUDA tensors are supported (rotation runs on the
            tensor's device; the KS test runs on CPU after a transfer).
        param_name: Used both for the result record and (when
            ``tensor_class`` is None) for the heuristic class lookup.
        tensor_class: Optional pre-classified bucket; pass to override
            :func:`classify_tensor`'s heuristic.

    Returns:
        :class:`AuditResult` with the per-tensor verdict and stats.
    """
    if _scipy_kstest is None:
        raise RuntimeError(
            "scipy is required for audit_tensor; install scipy to run "
            "the TurboQuant P6a audit (already a hone dependency)."
        )

    cls = tensor_class or classify_tensor(param_name)
    # fp32 keeps the KS statistic numerically stable on bf16 EF tensors
    # without changing the audit verdict.
    flat = t.detach().to(torch.float32).flatten()
    d = int(flat.numel())
    if d < 2:
        # KS is undefined on a single sample. Mark the tensor as failing
        # so it doesn't silently inflate the per-class pass rate.
        return AuditResult(
            tensor_class=cls,
            param_name=param_name,
            d=d,
            ks_stat=float("nan"),
            ks_p_value=0.0,
            beta_passed=False,
            mean_rel_error=0.0,
        )

    rotated = hadamard_rotate(flat.unsqueeze(0)).squeeze(0)
    norm = rotated.norm()
    rotated = rotated / (norm + 1e-12)
    squared_coords = rotated.pow(2).cpu().numpy()

    stat, p_value = _scipy_kstest(squared_coords, "beta", args=(d / 2, d / 2))

    # Sanity-only: for a unit vector the empirical mean of squared
    # coords is exactly 1/d (sum is 1 by construction), so this should
    # be ~1e-7 from float noise. Any non-trivial value here means the
    # WHT or normalisation is broken and the audit verdict should not
    # be trusted; surface it in the structured result.
    expected_mean = 1.0 / d
    mean_rel_error = abs(float(squared_coords.mean()) - expected_mean) / expected_mean

    return AuditResult(
        tensor_class=cls,
        param_name=param_name,
        d=d,
        ks_stat=float(stat),
        ks_p_value=float(p_value),
        beta_passed=bool(p_value > BETA_KS_P_THRESHOLD),
        mean_rel_error=float(mean_rel_error),
    )


def aggregate_by_class(
    results: list[AuditResult],
) -> dict[str, dict[str, float]]:
    """Roll up a list of per-tensor results into a per-class summary.

    Returns:
        ``{class_name: {"count": N, "passed": K, "pass_rate": K/N,
        "mean_p_value": ..., "median_p_value": ...}}``. Empty classes
        are omitted. The CLI uses ``pass_rate`` per class to compute
        the gate verdict.
    """
    by_class: dict[str, list[AuditResult]] = {}
    for r in results:
        by_class.setdefault(r.tensor_class, []).append(r)

    summary: dict[str, dict[str, float]] = {}
    for cls, rs in by_class.items():
        pvals = sorted(r.ks_p_value for r in rs)
        passed = sum(1 for r in rs if r.beta_passed)
        n = len(rs)
        summary[cls] = {
            "count": float(n),
            "passed": float(passed),
            "pass_rate": passed / n if n > 0 else 0.0,
            "mean_p_value": sum(pvals) / n if n > 0 else 0.0,
            "median_p_value": pvals[n // 2] if n > 0 else 0.0,
        }
    return summary


def overall_pass_rate(summary: dict[str, dict[str, float]]) -> float:
    """Return the WORST per-class pass rate across the audit summary.

    The gate is per-class, not aggregate: an audit that passes 100% of
    embeddings but 0% of routers must FAIL because TurboQuant would
    silently corrupt router gradients. The worst-class metric makes
    that failure visible.
    """
    if not summary:
        return 0.0
    return min(stats["pass_rate"] for stats in summary.values())


def _run_self_tests() -> None:  # pragma: no cover - exercised via __main__
    """Minimal correctness tests for the WHT + audit primitives.

    Validates:
      1. WHT is its own inverse on power-of-two ``d`` (the production
         contract the codec relies on).
      2. WHT preserves L2 norm (orthogonality after sqrt(n) normalisation).
      3. WHT runs without error on non-power-of-two ``d`` and returns
         the expected shape.
      4. ``audit_tensor`` runs end-to-end on a synthetic vector and
         returns a populated :class:`AuditResult`.
      5. ``classify_tensor`` resolves the canonical name conventions
         used by Hone's ``LoopLM`` and HF MoE checkpoints.
      6. ``aggregate_by_class`` and ``overall_pass_rate`` agree.
    """
    torch.manual_seed(0)

    # --- (1) self-inverse on power-of-two d ----------------------------
    for d in (2, 4, 16, 64, 1024):
        x = torch.randn(d, dtype=torch.float64)
        roundtrip = hadamard_rotate(hadamard_rotate(x))
        max_err = (roundtrip - x).abs().max().item()
        assert max_err < 1e-10, (
            f"WHT self-inverse failed at d={d}: max abs err = {max_err:.3e}"
        )

    # --- (2) norm preservation -----------------------------------------
    for d in (4, 64, 1024):
        x = torch.randn(d, dtype=torch.float64)
        rotated = hadamard_rotate(x)
        n_orig = x.norm().item()
        n_rot = rotated.norm().item()
        assert abs(n_orig - n_rot) < 1e-10, (
            f"WHT norm not preserved at d={d}: {n_orig:.6f} vs {n_rot:.6f}"
        )

    # --- (3) non-power-of-two d returns same shape ---------------------
    for d in (3, 5, 100, 1000):
        x = torch.randn(d)
        out = hadamard_rotate(x)
        assert out.shape == x.shape, (
            f"non-power-of-two shape changed: d={d}, got {tuple(out.shape)}"
        )

    # --- (3b) batched WHT independence ---------------------------------
    # Rotating a stacked batch must equal stacking row-rotations.
    batch = torch.randn(5, 16, dtype=torch.float64)
    batched = hadamard_rotate(batch)
    stacked = torch.stack([hadamard_rotate(row) for row in batch])
    assert torch.allclose(batched, stacked, atol=1e-10), (
        "batched WHT diverged from per-row WHT"
    )

    # --- (4) audit end-to-end ------------------------------------------
    if _scipy_kstest is not None:
        x = torch.randn(2048)
        res = audit_tensor(x, param_name="layers.0.mlp.gate_proj.weight")
        assert isinstance(res, AuditResult)
        assert res.d == 2048
        assert res.tensor_class == "mlp", (
            f"gate_proj should be mlp, got {res.tensor_class}"
        )
        # Sanity: mean_rel_error is float-noise on a unit vector.
        assert res.mean_rel_error < 1e-3, (
            f"mean_rel_error suspiciously high: {res.mean_rel_error:.3e}"
        )

    # --- (5) classify_tensor naming conventions ------------------------
    # Each case probes one branch of the classifier. Notable subtleties:
    #   * ``mlp.gate_proj`` must land in mlp (not router), because the
    #     SwiGLU gate is part of the MLP block, not the routing layer.
    #   * ``moe.router`` matches BOTH "router" and "moe", and the
    #     intentional ordering puts router first (we'd rather over-flag
    #     a router-shaped tensor as "router" than misclassify an MoE
    #     router as a generic expert weight).
    cases = {
        "model.embed_tokens.weight": "embedding",
        "lm_head.weight": "embedding",
        "transformer.wte.weight": "embedding",
        "layers.5.router.weight": "router",
        "layers.7.gate.weight": "router",
        "layers.3.moe.router.weight": "router",
        "layers.5.mlp.gate_proj.weight": "mlp",
        "layers.3.experts.0.w1": "moe_expert",
        "layers.2.self_attn.q_proj.weight": "attention",
        "layers.2.attention.k_proj.weight": "attention",
        "layers.4.feed_forward.up_proj.weight": "mlp",
        "norm.weight": "other",
    }
    for name, expected in cases.items():
        got = classify_tensor(name)
        assert got == expected, (
            f"classify_tensor({name!r}) = {got!r}, expected {expected!r}"
        )

    # --- (6) aggregate / overall_pass_rate -----------------------------
    fake = [
        AuditResult("embedding", "a", 100, 0.1, 0.5, True, 0.0),
        AuditResult("embedding", "b", 100, 0.2, 0.4, True, 0.0),
        AuditResult("router", "c", 100, 0.9, 0.001, False, 0.0),
        AuditResult("router", "d", 100, 0.8, 0.002, False, 0.0),
    ]
    summary = aggregate_by_class(fake)
    assert summary["embedding"]["pass_rate"] == 1.0
    assert summary["router"]["pass_rate"] == 0.0
    assert overall_pass_rate(summary) == 0.0, (
        "overall_pass_rate must take the WORST per-class rate"
    )

    print(
        "[hone.turboquant_audit] self-tests passed: "
        "WHT self-inverse, norm preservation, batched independence, "
        "audit end-to-end, classify_tensor, aggregate."
    )


if __name__ == "__main__":  # pragma: no cover
    _run_self_tests()
