#!/usr/bin/env python
# The MIT License (MIT)
# © 2025 hone.training
"""P5 Spearman validation gate for the single-forward eval rollout.

Standalone operator tool — compare two per-UID score files captured by
the Hone validator under the legacy 4-forward eval path vs the new P5b
single-forward delta-loss path, and gate the rollout on the Spearman
rank-correlation between them.

The plan (P5b validation) requires Spearman ≥ 0.85 on a 1.4B model
across 200 outer steps before defaulting ``eval_single_forward`` to
true. This script is the final decision gate: operators capture two
sibling runs (one with the flag off, one with it on), export the
per-UID score maps per window, and pipe them through this tool.

NOTE: the full 200-step soak is expensive (hours of GPU time) and is
EXPLICITLY NOT run by this script. Operators run it offline; this
script just reads the resulting score logs.

Score file format (flexible — pick the easier one to dump from your
validator's metrics pipeline):

1. ``{"<uid>": <score>, ...}`` — single-window, UID-keyed JSON map.
2. ``[{"uid": <int>, "score": <float>, "window": <int>}, ...]`` — list
   of records across windows. When ``--window`` is omitted the tool
   averages each UID's score across windows before correlating.

Exit codes:
    0 — Spearman ≥ threshold (default 0.85). Single-forward is safe to
        default on.
    1 — Spearman < threshold OR score files don't overlap on enough
        UIDs (< ``--min-uids``). Keep the multi-forward path.
    2 — Usage / file error. Nothing was compared.

Example::

    python -m hone.validator.score_spearman_check \\
        --legacy scores_four_forward.json \\
        --candidate scores_single_forward.json \\
        --threshold 0.85 --min-uids 8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scipy.stats import spearmanr  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    print(
        "ERROR: scipy is required for this script. Install with `pip install scipy`.",
        file=sys.stderr,
    )
    sys.exit(2)


def _load_score_map(path: Path, window: int | None) -> dict[int, float]:
    """Load ``path`` and collapse to ``{uid: score}``.

    Accepts either the single-window dict form or the list-of-records
    form (see module docstring). When ``window`` is ``None`` and the
    input is list-form, scores are averaged across windows per UID.
    """
    try:
        raw: Any = json.loads(path.read_text())
    except Exception as e:
        print(f"ERROR: failed to read {path}: {e}", file=sys.stderr)
        sys.exit(2)

    if isinstance(raw, dict):
        try:
            return {int(uid): float(s) for uid, s in raw.items()}
        except Exception as e:
            print(
                f"ERROR: {path} top-level dict must be {{uid: score}}: {e}",
                file=sys.stderr,
            )
            sys.exit(2)

    if isinstance(raw, list):
        per_uid: dict[int, list[float]] = defaultdict(list)
        for rec in raw:
            if not isinstance(rec, dict):
                continue
            try:
                uid = int(rec["uid"])
                score = float(rec["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if window is not None:
                rec_window = rec.get("window")
                if rec_window is None or int(rec_window) != window:
                    continue
            per_uid[uid].append(score)
        if not per_uid:
            print(
                f"ERROR: {path} contained zero matching records"
                + (f" for window {window}" if window is not None else ""),
                file=sys.stderr,
            )
            sys.exit(2)
        return {uid: sum(xs) / len(xs) for uid, xs in per_uid.items()}

    print(
        f"ERROR: {path} must be a JSON object or list of records, got {type(raw).__name__}",
        file=sys.stderr,
    )
    sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "P5 Spearman gate: compare legacy 4-forward scores vs single-forward "
            "delta-loss scores and decide whether single-forward is safe to default on."
        )
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        required=True,
        help="Score file from the legacy 4-forward eval path.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Score file from the P5b single-forward eval path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum Spearman rank-correlation to pass the gate (default 0.85).",
    )
    parser.add_argument(
        "--min-uids",
        type=int,
        default=8,
        help=(
            "Minimum number of UIDs that must appear in BOTH files "
            "before Spearman is meaningful (default 8)."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help=(
            "If the score file is a list-of-records, restrict to this window "
            "number. When omitted, per-UID scores are averaged across all "
            "windows in the file."
        ),
    )
    args = parser.parse_args()

    legacy = _load_score_map(args.legacy, args.window)
    candidate = _load_score_map(args.candidate, args.window)

    shared = sorted(set(legacy) & set(candidate))
    if len(shared) < args.min_uids:
        print(
            "FAIL: only "
            f"{len(shared)} UIDs overlap between files (need >= {args.min_uids}).",
            file=sys.stderr,
        )
        return 1

    x = [legacy[uid] for uid in shared]
    y = [candidate[uid] for uid in shared]
    try:
        res = spearmanr(x, y)
    except Exception as e:  # pragma: no cover
        print(f"ERROR: scipy spearmanr failed: {e}", file=sys.stderr)
        return 2

    # scipy returns a SignificanceResult (>=1.9) or (corr, pval) tuple.
    rho: float
    pval: float
    if hasattr(res, "correlation"):
        rho = float(res.correlation)  # type: ignore[attr-defined]
        pval = float(res.pvalue)  # type: ignore[attr-defined]
    else:  # pragma: no cover — older scipy
        rho, pval = float(res[0]), float(res[1])

    if rho >= args.threshold:
        print(
            f"PASS: Spearman rho={rho:.4f} (p={pval:.4g}) on n={len(shared)} UIDs "
            f">= threshold {args.threshold:.2f}. Single-forward eval is safe to default on."
        )
        return 0

    # Loud, explicit failure message for CI logs + human eyes.
    print(
        "=" * 72
        + "\n"
        + "FAIL: P5b single-forward Spearman gate DID NOT PASS.\n"
        + f"  rho = {rho:.4f} (p={pval:.4g}) on n={len(shared)} UIDs\n"
        + f"  threshold = {args.threshold:.2f}\n"
        + "\n"
        + "Do NOT set ``eval_single_forward: true`` in hparams.json.\n"
        + "The legacy 4-forward eval path is the load-bearing data-drift\n"
        + "defense until this correlation clears the bar.\n"
        + "=" * 72,
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
