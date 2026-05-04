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

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Bucket(BaseModel):
    """Configuration for a bucket, including name and access credentials."""

    def __hash__(self):
        return hash(
            (self.name, self.account_id, self.access_key_id, self.secret_access_key)
        )

    def __eq__(self, other):
        if isinstance(other, Bucket):
            return self.model_dump() == other.model_dump()
        return False

    name: str = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)
    access_key_id: str = Field(..., min_length=1)
    secret_access_key: str = Field(..., min_length=1)

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )


class CommsGetResult(BaseModel):
    """A standard return type for the `get` function."""

    data: None | dict[str, Any] = Field(
        None, description="The data retrieved by the get function."
    )
    global_step: None | int = Field(
        None, description="The global step associated with the data."
    )
    status: Literal["OK", "TOO_EARLY", "TOO_LATE", "NOT_FOUND", "ERROR"] = Field(
        "OK", description="The status of the get operation."
    )

    @property
    def success(self) -> bool:
        """Returns True if the operation was successful and returned data."""
        return self.status == "OK" and self.data is not None


# ---------------------------------------------------------------------------
# P2: shared upload-metadata builder + validator-side guardrail
# ---------------------------------------------------------------------------
#
# Token-weighted aggregation (Decoupled DiLoCo Algorithm 2) needs every peer's
# ``c_tokens`` and ``c_steps`` for ``w_i = c_tokens * (c_tokens / c_steps)``.
# Both numbers are *miner-reported*: a malicious peer can multiply ``c_tokens``
# by 100x to inflate their merge weight 10000x. Two interlocked guardrails:
#
# 1) ``validate_upload_metadata`` clamps ``c_tokens`` to the hard upper bound
#    ``sample_count * seq_len`` (a peer cannot have trained on more tokens
#    than the sample pool times sequence length). Default-on guardrail.
# 2) ``UploadMetadata.digest_hex()`` binds ``c_tokens`` and ``c_steps`` into
#    the BLAKE2b-128 sample digest. Forging either field forces a hash
#    pre-image search against a window-locked sorted ``sample_ids`` tuple —
#    intractable on chain-window timing.
#
# This module stays dependency-light (stdlib + numpy) so any protocol re-
# implementation can produce identical digests without pulling torch.

@dataclass(frozen=True)
class UploadMetadata:
    """Canonical upload metadata emitted by the miner with each gradient.

    Binds sample identity (``sample_ids``) to throughput counters
    (``c_tokens``, ``c_steps``) so a miner cannot report inflated tokens
    without breaking the digest the validator independently reconstructs in
    ``Validator._training_pool_digest``. This is the on-wire shape consumed
    by ``hone.neurons.miner`` and clamped by ``validate_upload_metadata``.

    Fields:
        window: chain window the gradient was trained against.
        sample_ids: window-deterministic sample IDs the miner trained on.
            MUST be sorted ascending and de-duplicated; the digest input is
            order-sensitive so any deviation from the canonical sort
            breaks the validator-side reconstruction.
        c_tokens: total tokens trained on this window (post batch-mask /
            post -100 label exclusion). Linear factor in the merge weight.
        c_steps: inner-step count for this window. Quadratic factor in the
            throughput term ``c_tokens / c_steps`` of Algorithm 2.
    """

    window: int
    sample_ids: tuple[int, ...]
    c_tokens: int
    c_steps: int

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)

    def digest_hex(self) -> str:
        """BLAKE2b-128 hash over (sample_ids || c_tokens || c_steps).

        Layout is canonical and every field is little-endian uint64 so any
        protocol re-implementation produces identical digests:

            sample_ids[0] | sample_ids[1] | ... | c_tokens | c_steps

        The digest is stable across Python runs and platforms (numpy
        ``tobytes()`` is deterministic for fixed-dtype arrays).
        """
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray(self.sample_ids, dtype=np.uint64).tobytes())
        h.update(np.uint64(self.c_tokens).tobytes())
        h.update(np.uint64(self.c_steps).tobytes())
        return h.hexdigest()

    def to_wire(self) -> dict:
        """Serialise to the on-wire dict the miner attaches to the
        gradient under the ``metadata`` key. Validator reads exactly
        these fields in ``log_digest_match`` and ``validate_upload_metadata``.
        """
        return {
            "window": int(self.window),
            "sample_count": int(self.sample_count),
            "sample_digest": self.digest_hex(),
            "c_tokens": int(self.c_tokens),
            "c_steps": int(self.c_steps),
        }


def validate_upload_metadata(
    claimed: dict,
    seq_len: int,
) -> tuple[bool, int, str]:
    """Validator-side check on miner-reported metadata.

    Returns ``(valid, clamped_c_tokens, reason)`` where ``valid`` is True
    when the metadata is structurally usable (a clamp is NOT a rejection;
    we keep the peer's contribution but downweight via the clamped tokens).

    Guardrails:
        1) ``sample_count * seq_len`` is the hard upper bound for
           ``c_tokens``. Any larger claim is a lie and is silently clamped
           with a WARN-level reason string for the caller to surface.
        2) The digest binding (see ``UploadMetadata.digest_hex``) is what
           makes a ``c_tokens`` lie expensive: the miner would have to
           regenerate the entire sorted ``sample_ids`` tuple to match the
           hash, and they're committed on chain-window timing. This
           function does not re-verify the digest itself —
           ``Validator.log_digest_match`` already does that mirror check;
           clamp + digest mirror are independent guardrails that compose.

    Caller contract: legacy peers (pre-P2 miners) upload metadata WITHOUT
    ``c_tokens`` / ``c_steps``; in that case both default to 0, the clamp
    returns ``(True, 0, "ok")``, and the resulting weight is 0 (peer
    contributes uniformly via the default-uniform path in ``outer_step``).
    """
    sample_count = int(claimed.get("sample_count", 0))
    claimed_c_tokens = int(claimed.get("c_tokens", 0))
    c_tokens_max = sample_count * int(seq_len)
    clamped = min(claimed_c_tokens, c_tokens_max)
    if clamped < claimed_c_tokens:
        return (
            True,
            clamped,
            f"c_tokens clamped from {claimed_c_tokens} to {c_tokens_max} "
            f"(sample_count={sample_count} * seq_len={seq_len})",
        )
    return (True, clamped, "ok")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Inline unit tests — run with ``python -m hone.schemas``. These cover
    # the two invariants P2 needs: (a) digest stability across runs, and
    # (b) ``c_tokens`` clamp arithmetic. Anything beyond these two should
    # live in the proper test suite.
    # ------------------------------------------------------------------

    # (a) Digest stability: a fixed input MUST produce the same hex digest
    # across runs, Python versions, and platforms. The expected hex was
    # computed once with the canonical layout; any change to the digest
    # layout (field order, dtype, endianness) will trip this assertion.
    meta = UploadMetadata(
        window=42,
        sample_ids=(1, 7, 13, 21, 99),
        c_tokens=4096 * 5,
        c_steps=3,
    )
    expected = hashlib.blake2b(digest_size=16)
    expected.update(np.asarray((1, 7, 13, 21, 99), dtype=np.uint64).tobytes())
    expected.update(np.uint64(4096 * 5).tobytes())
    expected.update(np.uint64(3).tobytes())
    expected_hex = expected.hexdigest()
    actual_hex = meta.digest_hex()
    assert actual_hex == expected_hex, (
        f"digest_hex stability broken:\n  expected {expected_hex}\n  got      {actual_hex}"
    )
    # Re-running must be deterministic (no salt / no time-based mixing).
    assert meta.digest_hex() == actual_hex, "digest_hex is non-deterministic"

    # ``to_wire`` round-trips the same digest the validator will mirror.
    wire = meta.to_wire()
    assert wire["sample_digest"] == actual_hex
    assert wire["sample_count"] == 5
    assert wire["c_tokens"] == 4096 * 5
    assert wire["c_steps"] == 3
    assert wire["window"] == 42

    # Changing ANY field MUST change the digest (no collision in this trio).
    bumped_tokens = UploadMetadata(
        window=42, sample_ids=(1, 7, 13, 21, 99), c_tokens=4096 * 5 + 1, c_steps=3
    )
    bumped_steps = UploadMetadata(
        window=42, sample_ids=(1, 7, 13, 21, 99), c_tokens=4096 * 5, c_steps=4
    )
    bumped_ids = UploadMetadata(
        window=42, sample_ids=(1, 7, 13, 21, 100), c_tokens=4096 * 5, c_steps=3
    )
    assert bumped_tokens.digest_hex() != actual_hex, "c_tokens not in digest"
    assert bumped_steps.digest_hex() != actual_hex, "c_steps not in digest"
    assert bumped_ids.digest_hex() != actual_hex, "sample_ids not in digest"

    # (b) Clamp arithmetic. seq_len=4096, sample_count=10 → max=40960.
    valid, clamped, reason = validate_upload_metadata(
        {"sample_count": 10, "c_tokens": 100_000, "c_steps": 5},
        seq_len=4096,
    )
    assert valid is True, "clamp should keep peer valid (downweighted, not rejected)"
    assert clamped == 40960, f"expected clamp to 40960, got {clamped}"
    assert "clamped" in reason, f"reason should mention clamp, got: {reason!r}"

    # Below-cap claim: pass-through unchanged.
    valid, clamped, reason = validate_upload_metadata(
        {"sample_count": 10, "c_tokens": 30_000, "c_steps": 5},
        seq_len=4096,
    )
    assert valid is True
    assert clamped == 30_000
    assert reason == "ok"

    # Legacy / missing fields: defaults to (True, 0, "ok") so legacy peers
    # ride the uniform-weight default-off path without spurious warnings.
    valid, clamped, reason = validate_upload_metadata(
        {"sample_count": 10}, seq_len=4096
    )
    assert valid is True
    assert clamped == 0
    assert reason == "ok"

    print("schemas.py inline tests passed.")
