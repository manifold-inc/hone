#!/usr/bin/env python
# The MIT License (MIT)
# © 2025 hone.training
"""P6a TurboQuant audit CLI — coordinate-distribution gate for P6b rollout.

Standalone operator tool — load a snapshot of a miner's
``self.error_feedback`` dict (dumped via the SIGUSR2 hook in
``hone/neurons/miner.py``) and run the per-tensor Beta(d/2, d/2)
KS-test from :func:`hone.turboquant_audit.audit_tensor`. Aggregate
results by tensor class (embedding / router / moe_expert / attention /
mlp / other), print a human-readable summary, optionally dump JSON,
and exit 0/1 based on whether every class clears
:data:`hone.turboquant_audit.PASS_RATE_THRESHOLD` (80%).

The entire P6b TurboQuant codec is gated on this CLI exiting 0 against
a real production snapshot. If exit code is 1, do NOT enable
``hparams.turboquant_enabled``.

Snapshot format
---------------

The script expects a ``torch.save``-d dict where keys are parameter
names (str) and values are tensors (or ``None``). DTensors and other
non-plain tensors are skipped with a warning so a partial snapshot
(e.g. taken before all owned params have a non-zero EF) still produces
useful output.

Operator workflow
-----------------

::

    # On the miner host, with the miner running:
    kill -USR2 <miner_pid>
    # ...look in the miner's stdout for the snapshot path, or:
    ls -lt /tmp/ef_snapshot_uid*_w*.pt | head -1

    # Then on any host with hone + scipy installed:
    python hone/validator/turboquant_audit_cli.py \\
        /tmp/ef_snapshot_uid42_w12345.pt \\
        --json /tmp/ef_audit_uid42_w12345.json

Exit codes
----------

0
    Every audited tensor class cleared the per-class pass rate
    threshold. P6b TurboQuant is safe to enable (after the matching
    A/B perplexity gate from ``turboquant_ab_compare.py``).

1
    At least one tensor class fell below the per-class pass rate
    threshold. Do NOT flip ``turboquant_enabled`` true; the codec
    would silently corrupt that class of gradients.

2
    Usage / file error. Nothing was audited.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# Importing the audit module via importlib bypasses ``hone/__init__.py``,
# which has hard-required env vars (R2 secrets) and triton imports the
# CLI does not need. The audit module itself is a leaf with only
# torch + scipy as runtime deps.
_AUDIT_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "hone" / "turboquant_audit.py"
)


def _load_audit_module():
    """Side-load ``hone.turboquant_audit`` without importing the parent
    package. Same trick the inline self-test uses; lets the CLI run on
    any host with torch + scipy even if R2 secrets are not configured.
    """
    if not _AUDIT_MODULE_PATH.exists():
        print(
            f"ERROR: cannot find audit module at {_AUDIT_MODULE_PATH}",
            file=sys.stderr,
        )
        sys.exit(2)
    spec = importlib.util.spec_from_file_location(
        "hone_turboquant_audit", str(_AUDIT_MODULE_PATH)
    )
    if spec is None or spec.loader is None:
        print(
            f"ERROR: failed to construct module spec for {_AUDIT_MODULE_PATH}",
            file=sys.stderr,
        )
        sys.exit(2)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass + frozen=True can resolve cls.__module__.
    sys.modules["hone_turboquant_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_snapshot(path: Path):
    """Load a ``torch.save``-d EF dict from ``path``. Returns the raw
    dict so the caller can iterate over it; we don't pre-filter so the
    operator can see the full key set in the verbose output.
    """
    try:
        import torch  # local import; deferred until after CLI parsing
    except ImportError:
        print(
            "ERROR: torch is required to load the EF snapshot.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not path.exists():
        print(f"ERROR: snapshot file not found: {path}", file=sys.stderr)
        sys.exit(2)

    try:
        # ``weights_only=True`` is the safe default but doesn't allow
        # arbitrary objects; EF snapshots are dict[str, Tensor] so it's
        # fine. Fall back to weights_only=False with a warning if the
        # snapshot uses an unsupported pickled object (e.g. DTensor
        # mesh references).
        try:
            obj = torch.load(path, weights_only=True, map_location="cpu")
        except Exception:
            print(
                f"WARN: weights_only=True load failed for {path}; "
                "retrying weights_only=False (trust the snapshot source)",
                file=sys.stderr,
            )
            obj = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as e:
        print(f"ERROR: failed to load snapshot {path}: {e}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(obj, dict):
        print(
            f"ERROR: snapshot must be a dict[str, Tensor], got {type(obj).__name__}",
            file=sys.stderr,
        )
        sys.exit(2)

    return obj


def _run_audit(audit_mod, ef_dict, *, verbose: bool) -> tuple[list, dict]:
    """Run :func:`audit_tensor` on every tensor in ``ef_dict``, skipping
    None / non-tensor values with a warning. Returns the raw list of
    AuditResult objects and the per-class summary dict.
    """
    import torch  # available after _load_snapshot

    results: list = []
    skipped: dict[str, int] = {"none": 0, "non_tensor": 0, "empty": 0, "errored": 0}
    for name, t in sorted(ef_dict.items()):
        if t is None:
            skipped["none"] += 1
            if verbose:
                print(f"  SKIP {name}: None (param never compressed)", file=sys.stderr)
            continue
        if not isinstance(t, torch.Tensor):
            skipped["non_tensor"] += 1
            if verbose:
                print(
                    f"  SKIP {name}: not a torch.Tensor (got {type(t).__name__})",
                    file=sys.stderr,
                )
            continue
        if t.numel() < 2:
            skipped["empty"] += 1
            if verbose:
                print(
                    f"  SKIP {name}: numel={t.numel()} (too small for KS)",
                    file=sys.stderr,
                )
            continue
        try:
            res = audit_mod.audit_tensor(t, param_name=name)
        except Exception as e:
            skipped["errored"] += 1
            print(
                f"  WARN {name}: audit_tensor raised {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            continue
        results.append(res)
        if verbose:
            print(
                f"  {name}: class={res.tensor_class} d={res.d} "
                f"ks_p={res.ks_p_value:.4g} passed={res.beta_passed}",
                file=sys.stderr,
            )

    summary = audit_mod.aggregate_by_class(results)
    summary["_skipped"] = {
        "none": float(skipped["none"]),
        "non_tensor": float(skipped["non_tensor"]),
        "empty": float(skipped["empty"]),
        "errored": float(skipped["errored"]),
    }
    return results, summary


def _format_summary_table(summary: dict) -> str:
    """Render the per-class summary as a fixed-width table for the
    operator's terminal. Pure str; no rich/colour deps.
    """
    classes = sorted(c for c in summary if not c.startswith("_"))
    if not classes:
        return "(no classes audited)"
    header = f"{'class':<14}{'count':>8}{'passed':>8}{'pass_rate':>11}{'mean_p':>10}{'median_p':>11}"
    rows = [header, "-" * len(header)]
    for cls in classes:
        s = summary[cls]
        rows.append(
            f"{cls:<14}{int(s['count']):>8}{int(s['passed']):>8}"
            f"{s['pass_rate']:>11.3f}{s['mean_p_value']:>10.4f}{s['median_p_value']:>11.4f}"
        )
    skipped = summary.get("_skipped", {})
    if any(int(v) > 0 for v in skipped.values()):
        rows.append("")
        rows.append(
            "skipped: "
            + ", ".join(f"{k}={int(v)}" for k, v in skipped.items() if int(v) > 0)
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "P6a TurboQuant audit gate: KS-test post-Hadamard EF coords "
            "against Beta(d/2, d/2) per tensor class. Exit 0 iff every "
            "class clears the per-class pass-rate threshold."
        )
    )
    parser.add_argument(
        "snapshot",
        type=Path,
        help=(
            "Path to a torch.save-d miner.error_feedback dict. Produce "
            "via the miner's SIGUSR2 hook: `kill -USR2 <miner_pid>` "
            "writes /tmp/ef_snapshot_uid<U>_w<W>.pt."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "Write the full per-tensor result list and per-class "
            "summary to this JSON file (in addition to the human "
            "summary on stdout)."
        ),
    )
    parser.add_argument(
        "--pass-rate-threshold",
        type=float,
        default=None,
        help=(
            "Minimum per-class pass rate; default reads "
            "PASS_RATE_THRESHOLD from hone.turboquant_audit (0.80)."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help=(
            "Print per-tensor audit lines to stderr (otherwise only "
            "the aggregate summary is shown)."
        ),
    )
    args = parser.parse_args()

    audit_mod = _load_audit_module()
    threshold = (
        args.pass_rate_threshold
        if args.pass_rate_threshold is not None
        else audit_mod.PASS_RATE_THRESHOLD
    )

    ef_dict = _load_snapshot(args.snapshot)
    print(
        f"Loaded EF snapshot: {args.snapshot} "
        f"({len(ef_dict)} keys; size {os.path.getsize(args.snapshot) / 1e6:.2f} MB)"
    )

    results, summary = _run_audit(audit_mod, ef_dict, verbose=args.verbose)

    print()
    print(_format_summary_table(summary))
    print()

    worst_rate = audit_mod.overall_pass_rate(
        {k: v for k, v in summary.items() if not k.startswith("_")}
    )
    print(
        f"worst-class pass rate = {worst_rate:.3f} "
        f"(gate threshold = {threshold:.3f})"
    )

    if args.json is not None:
        # JSON dump uses simple types so other tooling (jq, dashboards)
        # can consume it without scipy/torch installed.
        out = {
            "snapshot_path": str(args.snapshot),
            "pass_rate_threshold": float(threshold),
            "worst_class_pass_rate": float(worst_rate),
            "passed": bool(worst_rate >= threshold),
            "summary_by_class": {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in summary.items()
                if not k.startswith("_")
            },
            "skipped": {
                k: int(v) for k, v in summary.get("_skipped", {}).items()
            },
            "results": [
                {
                    "tensor_class": r.tensor_class,
                    "param_name": r.param_name,
                    "d": int(r.d),
                    "ks_stat": float(r.ks_stat),
                    "ks_p_value": float(r.ks_p_value),
                    "beta_passed": bool(r.beta_passed),
                    "mean_rel_error": float(r.mean_rel_error),
                }
                for r in results
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"Wrote JSON report: {args.json}")

    if worst_rate >= threshold:
        print(
            "PASS: every tensor class cleared the per-class pass rate. "
            "TurboQuant (P6b) is safe to enable; flip "
            "`hparams.turboquant_enabled: true` after the matching A/B "
            "perplexity gate from turboquant_ab_compare.py."
        )
        return 0

    failing = [
        cls
        for cls, s in summary.items()
        if not cls.startswith("_") and s["pass_rate"] < threshold
    ]
    print(
        "FAIL: P6a coordinate-distribution gate did NOT pass. "
        f"Failing classes: {', '.join(sorted(failing))}. "
        "Do NOT enable hparams.turboquant_enabled — the codec would "
        "silently corrupt these tensor classes.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
