"""Unit tests for P4 (parallel ``outer_step`` on a 2nd CUDA stream).

Covers:
  - ``hone.distributed.use_snapshot`` context-manager semantics:
    * normal exit restores ``param.data`` to the live tensor.
    * exception inside the body still restores (the load-bearing
      property — the validator main loop has multiple ``continue``
      paths inside the eval block).
    * ``snapshot is None`` is a documented no-op.
    * Partial snapshot only swaps the keys it owns (the
      ``parallel_outer_snapshot_full=false`` path).
  - ``hone.distributed.get_outer_step_stream``:
    * returns ``None`` when CUDA is unavailable (so callers branch
      cleanly off ``stream is None``).
    * returns a ``torch.cuda.Stream`` distinct from the FSDP
      offload stream (when CUDA is available).
    * is idempotent across calls (same instance).
  - CUDA event ordering when CUDA is available:
    * Event recorded on the secondary stream serialises against
      that stream's queue, not the default stream — confirms the
      ``record(stream=)`` + ``synchronize()`` pattern the validator
      relies on.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hone.distributed import (
    get_outer_step_stream,
    use_snapshot,
)


# ─────────────────────────────────────────────────────────────────────
# ``use_snapshot``
# ─────────────────────────────────────────────────────────────────────


def _tiny_model() -> nn.Module:
    """Two-layer MLP with deterministic init so we can assert tensor
    identity / value equality precisely. Kept on CPU so the tests
    run in CI without GPU."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    return m


def test_use_snapshot_swaps_and_restores_on_normal_exit():
    model = _tiny_model()
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}
    live_ptrs = {n: p.data.data_ptr() for n, p in model.named_parameters()}

    snap_ptrs = {n: t.data_ptr() for n, t in snap.items()}

    with use_snapshot(model, snap):
        for n, p in model.named_parameters():
            assert p.data.data_ptr() == snap_ptrs[n], (
                f"Inside the with block, param {n!r} should point at the "
                f"snapshot tensor, got {p.data.data_ptr()} != "
                f"{snap_ptrs[n]}."
            )

    for n, p in model.named_parameters():
        assert p.data.data_ptr() == live_ptrs[n], (
            f"After the with block, param {n!r} should point back at "
            f"the live tensor, got {p.data.data_ptr()} != "
            f"{live_ptrs[n]}."
        )


def test_use_snapshot_restores_on_exception_inside_body():
    """Load-bearing: the validator's eval loop has several ``continue``
    paths inside the ``with _use_snapshot`` block. ``contextmanager``'s
    ``finally`` semantics MUST run on every exit, including when the
    body raises."""
    model = _tiny_model()
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}
    live_ptrs = {n: p.data.data_ptr() for n, p in model.named_parameters()}

    class _Sentinel(RuntimeError):
        pass

    with pytest.raises(_Sentinel):
        with use_snapshot(model, snap):
            raise _Sentinel("simulated eval failure")

    for n, p in model.named_parameters():
        assert p.data.data_ptr() == live_ptrs[n], (
            f"After exception, param {n!r} should be restored to the "
            f"live tensor, got data_ptr={p.data.data_ptr()} != "
            f"{live_ptrs[n]}."
        )


def test_use_snapshot_none_is_noop():
    """Default-OFF behaviour: ``snapshot=None`` (which is what the
    validator passes when ``parallel_outer_step=false``) must keep
    every ``param.data`` pointer untouched."""
    model = _tiny_model()
    live_ptrs = {n: p.data.data_ptr() for n, p in model.named_parameters()}

    with use_snapshot(model, None):
        for n, p in model.named_parameters():
            assert p.data.data_ptr() == live_ptrs[n]

    for n, p in model.named_parameters():
        assert p.data.data_ptr() == live_ptrs[n]


def test_use_snapshot_partial_keys_skip_silently():
    """Validator's ``parallel_outer_snapshot_full=false`` path builds a
    snapshot that omits embedding-class params. ``use_snapshot`` must
    only swap keys present in the snapshot dict; the rest keep their
    live ``.data``."""
    model = _tiny_model()
    all_names = [n for n, _ in model.named_parameters()]
    assert len(all_names) >= 2, "Need >=2 params for a meaningful subset test"

    # Snapshot only the FIRST half of params — mirrors the embedding-skip case.
    head = all_names[: len(all_names) // 2 or 1]
    tail = all_names[len(all_names) // 2 or 1 :]
    snap = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n in head
    }

    head_snap_ptrs = {n: snap[n].data_ptr() for n in head}
    tail_live_ptrs = {
        n: p.data.data_ptr()
        for n, p in model.named_parameters()
        if n in tail
    }
    head_live_ptrs = {
        n: p.data.data_ptr()
        for n, p in model.named_parameters()
        if n in head
    }

    with use_snapshot(model, snap):
        for n, p in model.named_parameters():
            if n in head:
                assert p.data.data_ptr() == head_snap_ptrs[n], (
                    f"Inside body, head param {n!r} should point at "
                    f"snapshot, got {p.data.data_ptr()}"
                )
            else:
                assert p.data.data_ptr() == tail_live_ptrs[n], (
                    f"Inside body, tail param {n!r} should still "
                    f"point at live tensor (it was not snapshotted)"
                )

    for n, p in model.named_parameters():
        if n in head:
            assert p.data.data_ptr() == head_live_ptrs[n]
        else:
            assert p.data.data_ptr() == tail_live_ptrs[n]


def test_use_snapshot_writes_to_snapshot_inside_body_dont_touch_live():
    """The whole point of P4: while inside the ``with`` block, an
    in-place write to ``param.data`` lands on the SNAPSHOT tensor,
    not the live one. The validator's per-UID
    ``update_model_with_gradient`` apply / revert relies on this so
    the live weights stay clean for the parallel ``outer_step`` to
    write into."""
    model = _tiny_model()
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}
    # Capture the original live VALUES so we can assert they didn't move.
    live_before = {
        n: p.detach().clone() for n, p in model.named_parameters()
    }

    with use_snapshot(model, snap):
        # Mutate every param IN-PLACE. Under P4 semantics this writes
        # to the snapshot tensors (the redirected ``p.data``), NOT the
        # live ones.
        for _, p in model.named_parameters():
            p.data.fill_(42.0)

    # After exiting the with-block, ``param.data`` is back to live;
    # the live tensors must still hold their original values.
    for n, p in model.named_parameters():
        assert torch.equal(p.data, live_before[n]), (
            f"Param {n!r} was mutated through the snapshot redirect: "
            f"live tensor must be unchanged. live diff norm = "
            f"{(p.data - live_before[n]).norm().item()}"
        )

    # The snapshot tensors should now contain the 42.0 fill.
    for n, t in snap.items():
        assert torch.all(t == 42.0).item(), (
            f"Snapshot for {n!r} should hold the in-place write "
            f"(42.0); got {t}"
        )


# ─────────────────────────────────────────────────────────────────────
# ``get_outer_step_stream``
# ─────────────────────────────────────────────────────────────────────


def test_get_outer_step_stream_none_when_no_cuda():
    if torch.cuda.is_available():
        pytest.skip("Run only when CUDA is unavailable")
    assert get_outer_step_stream() is None


def test_get_outer_step_stream_idempotent_under_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    s1 = get_outer_step_stream()
    s2 = get_outer_step_stream()
    assert s1 is not None and s2 is not None
    # Same singleton (lazy global init).
    assert s1 is s2, (
        "get_outer_step_stream must return the same Stream object on "
        "every call (lazy singleton). Re-creating per call would defeat "
        "the priority-(-1) reservation we set at first init."
    )


def test_get_outer_step_stream_low_priority_under_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    s = get_outer_step_stream()
    assert s is not None
    # Priority is -1 (low); eval on default stream gets GPU cycles
    # first per the P4 design.
    assert s.priority == -1, (
        f"Outer-step stream should have priority=-1 (low) so eval on "
        f"default stream pre-empts; got {s.priority}"
    )


# ─────────────────────────────────────────────────────────────────────
# CUDA event ordering across streams (CUDA-only smoke test)
# ─────────────────────────────────────────────────────────────────────


def test_cuda_event_ordering_serialises_against_secondary_stream():
    """The validator's parallel-outer-step path relies on the
    `event.record(stream=outer_stream)` / `event.synchronize()` pair
    to serialise the eval-side exit against the outer-step's
    secondary-stream tail. This test confirms the ordering: a kernel
    enqueued on the secondary stream BEFORE the event must be
    visible after the event synchronises."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    secondary = get_outer_step_stream()
    assert secondary is not None

    x = torch.zeros(1024, device=device)

    secondary.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(secondary):
        x.add_(1.0)
    event = torch.cuda.Event()
    event.record(stream=secondary)

    # Without synchronize, the default stream might read x before
    # the secondary's add_ landed; the event sync makes this race
    # impossible.
    event.synchronize()
    val = float(x[0].item())
    assert val == 1.0, (
        f"After event.synchronize(), the secondary stream's add_ "
        f"must be visible on default; got x[0]={val}"
    )
