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

"""Count-Sketch gradient fingerprint (P5 soft pre-filter).

Weinberger & Dasgupta 2009 style sign-hash + index-hash count sketch for
per-peer gradient fingerprinting in the validator's single-forward eval
path. Kept deliberately isolated from ``hone.compress`` so the
``QuantParamsT`` pipeline is not entangled with this coarse fingerprint
-- the sketch operates on *reconstructed* peer gradients (post top-K /
quantization), not on the wire representation.

Design notes:

- The per-dim operation ``sketch[h(i)] += sign(i) * g[i]`` is an
  unbiased, low-variance linear projection. Two similar gradients
  project to close sketch vectors; adversarial sign-flipped gradients
  project to near-opposite sketches. Cosine similarity over the b-dim
  sketch is the primary signal.
- The hashing tables ``idx`` and ``sign`` are **seeded** so every
  validator computes the same projection on the same window, and
  miners cannot guess which buckets their grads will land in (the
  seed is chain-bound per P5a).
- Output dimensionality ``b`` (default 1024) is fixed across all peers
  within a single window. ``d`` is the full flat gradient dim and can
  be enormous (8B params ~ 1e10); hashing tables are allocated once
  per window on CPU to avoid a multi-GB VRAM footprint.

Usage in the validator eval loop:

.. code-block:: python

    flat_dim = sum(p.numel() for p in model.parameters())
    ck = CountSketch(d=flat_dim, b=1024, seed=chain_bound_seed)
    per_peer_sketches = {
        uid: ck.sketch(flatten_peer_grad(uid))
        for uid in eval_uids
    }
    avg = torch.stack(list(per_peer_sketches.values())).mean(dim=0)
    for uid, s in per_peer_sketches.items():
        cos = F.cosine_similarity(s, avg, dim=0).item()
        weight = max(min_weight, cos)  # soft down-weight, never reject
        score[uid] *= weight
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CountSketch:
    """Sign-hash + index-hash count sketch for gradient fingerprinting.

    For a ``d``-dimensional flat gradient vector, output is a
    ``b``-dimensional sketch. Per-dim contribution:
    ``sketch[h(i)] += sign(i) * g[i]`` where ``h(i)`` maps dim ``i`` to
    a sketch bucket (uniform over ``[0, b)``) and ``sign(i)`` to ``±1``.

    Typical parameters: ``b = 1024``, ``d = `` full flat gradient dim.
    Insert is O(d), pairwise comparison is O(b).

    Hash tables live on CPU by default; :meth:`sketch` moves them to the
    input tensor's device lazily so large models still hash cheaply
    without a multi-GB up-front VRAM allocation for the tables.

    All randomness flows from ``seed`` exclusively -- call sites should
    pass a chain-bound seed so miners cannot predict which bucket their
    coords land in before gather close (P5a).
    """

    def __init__(
        self,
        d: int,
        b: int = 1024,
        seed: int = 0,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if d <= 0:
            raise ValueError(f"CountSketch d must be > 0, got {d}")
        if b <= 0:
            raise ValueError(f"CountSketch b must be > 0, got {b}")
        if b > d:
            # Nothing in the algorithm forbids b > d, but it signals the
            # caller has swapped arguments -- warn loudly via ValueError
            # rather than silently allocating a huge sketch.
            raise ValueError(
                f"CountSketch b ({b}) must be <= d ({d}); swapped args?"
            )

        self.d = int(d)
        self.b = int(b)
        self.seed = int(seed)

        rng = torch.Generator(device="cpu").manual_seed(self.seed)
        self.idx = torch.randint(
            0, self.b, (self.d,), generator=rng, dtype=torch.long
        )
        self.sign = torch.where(
            torch.rand(self.d, generator=rng) < 0.5,
            torch.tensor(1, dtype=torch.int8),
            torch.tensor(-1, dtype=torch.int8),
        )

        self._device = torch.device(device)
        if self._device.type != "cpu":
            self.idx = self.idx.to(self._device)
            self.sign = self.sign.to(self._device)

        self._cached_idx: dict[torch.device, torch.Tensor] = {}
        self._cached_sign: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _idx_on(self, device: torch.device) -> torch.Tensor:
        if device == self.idx.device:
            return self.idx
        cached = self._cached_idx.get(device)
        if cached is None:
            cached = self.idx.to(device)
            self._cached_idx[device] = cached
        return cached

    def _sign_on(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (device, dtype)
        cached = self._cached_sign.get(key)
        if cached is None:
            cached = self.sign.to(device=device, dtype=dtype)
            self._cached_sign[key] = cached
        return cached

    def sketch(self, flat_g: torch.Tensor) -> torch.Tensor:
        """Project ``flat_g`` (d-dim) to a b-dim count-sketch vector."""
        if flat_g.dim() != 1:
            raise ValueError(
                f"CountSketch.sketch expects 1D input, got shape {tuple(flat_g.shape)}"
            )
        if flat_g.numel() != self.d:
            raise ValueError(
                f"CountSketch.sketch dim mismatch: flat_g has {flat_g.numel()} elements, expected {self.d}"
            )

        idx = self._idx_on(flat_g.device)
        sign = self._sign_on(flat_g.device, flat_g.dtype)

        signed = flat_g * sign
        out = torch.zeros(self.b, dtype=flat_g.dtype, device=flat_g.device)
        out.scatter_add_(0, idx, signed)
        return out


def soft_weight_from_cosine(
    cosine: float,
    *,
    min_weight: float = 0.5,
) -> float:
    """Translate a peer-vs-average cosine similarity into a score weight.

    Implements the P5 incentive-critic requirement: low-cosine peers get
    *down-weighted*, never *rejected*. Honest minority gradient
    directions must still contribute. ``min_weight`` is the floor
    (``count_sketch_min_weight`` hparam; default 0.5) — peers whose
    sketch is orthogonal or opposite to the mean still pass 50% of
    their score through.
    """
    if min_weight < 0.0 or min_weight > 1.0:
        raise ValueError(
            f"min_weight must be in [0, 1], got {min_weight}"
        )
    return max(min_weight, float(cosine))


def _run_self_tests() -> None:  # pragma: no cover - exercised via __main__
    """Round-trip / discrimination checks for :class:`CountSketch`.

    Validates the core contract the P5 soft pre-filter relies on:

    1. **Near-identical gradients sketch to near-identical vectors.**
       A peer whose gradient matches the consensus direction should not
       be down-weighted by the count-sketch filter.
    2. **Adversarial sign-flipped gradients project to near-opposite
       sketches** (cosine ≈ -1). A sign-flip attack lands in the
       ``cos < min_weight`` regime and gets clipped to 0.5 of its
       score -- visibly detectable but not silently nuked.
    3. **Dim / seed contracts** hold across realistic sizes.
    4. **Soft-weight floor** is respected.
    """
    torch.manual_seed(0)

    # --- (1) near-identical grads cosine > 0.95 ------------------------
    d, b = 10_000, 1024
    g1 = torch.randn(d)
    g2 = g1 + 0.01 * torch.randn(d)

    ck = CountSketch(d=d, b=b, seed=42)
    s1 = ck.sketch(g1)
    s2 = ck.sketch(g2)
    assert s1.shape == (b,), f"sketch shape wrong: {s1.shape}"
    cos_close = F.cosine_similarity(s1, s2, dim=0).item()
    assert cos_close > 0.95, (
        f"near-identical grads should sketch similarly; got cos={cos_close:.4f}"
    )

    # --- (2) adversarial sign-flip projects to near-opposite -----------
    g3 = -g1
    s3 = ck.sketch(g3)
    cos_adv = F.cosine_similarity(s1, s3, dim=0).item()
    assert cos_adv < -0.95, (
        f"sign-flipped grad should sketch to opposite; got cos={cos_adv:.4f}"
    )

    # --- (3) determinism: same seed -> same idx/sign -------------------
    ck_a = CountSketch(d=d, b=b, seed=123)
    ck_b = CountSketch(d=d, b=b, seed=123)
    assert torch.equal(ck_a.idx, ck_b.idx), "idx tables not seed-deterministic"
    assert torch.equal(ck_a.sign, ck_b.sign), "sign tables not seed-deterministic"

    ck_c = CountSketch(d=d, b=b, seed=124)
    assert not torch.equal(ck_a.idx, ck_c.idx), (
        "idx tables should differ under a different seed"
    )

    # --- (4) soft-weight floor -----------------------------------------
    assert soft_weight_from_cosine(1.0) == 1.0
    assert soft_weight_from_cosine(0.7) == 0.7
    assert soft_weight_from_cosine(0.0) == 0.5, "cosine=0 should clip to floor"
    assert soft_weight_from_cosine(-1.0) == 0.5, "cosine=-1 should clip to floor"
    assert soft_weight_from_cosine(0.1, min_weight=0.25) == 0.25

    # --- (5) dim mismatch raises ---------------------------------------
    try:
        ck.sketch(torch.randn(d + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on dim mismatch")

    # --- (6) 2D input rejected -----------------------------------------
    try:
        ck.sketch(torch.randn(10, 100))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on non-1D input")

    # --- (7) b > d rejected --------------------------------------------
    try:
        CountSketch(d=16, b=64, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when b > d")

    print(
        "[hone.sketch] self-tests passed: "
        f"near-identical cos={cos_close:.3f}, adversarial cos={cos_adv:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    _run_self_tests()
