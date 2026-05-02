#!/usr/bin/env python
# The MIT License (MIT)
# © 2025 hone.training
"""P6b TurboQuant A/B perplexity gate (matching ``rda_ab_compare.py``).

Standalone operator tool — compare two windowed score files captured by
the Hone validator: one run with ``turboquant_enabled=true`` and one
with ``turboquant_enabled=false`` (otherwise identical hparams). Decide
whether the TurboQuant codec degrades validation perplexity / re-ranks
peers beyond the per-phase tolerance before the operator flips
``turboquant_enabled`` default-on.

The plan (P6b validation hook) requires:

* validation perplexity regression ≤ 0.5% (TQ-on perplexity ≤ TQ-off
  perplexity × 1.005 over the 200-step soak), AND
* Spearman rank-correlation of the per-UID score streams ≥ 0.85
  (TurboQuant must not silently re-rank peers — that's a sign the
  scalar codebook is fighting the existing per-block clip and
  producing arbitrary magnitudes).

Both gates must pass IN ADDITION to the P6a coordinate-distribution
gate (``turboquant_audit_cli.py``). The flow is:

    P6a audit → if pass → P6b A/B (this script) → if pass → flip on.

NOTE: the full 200-step soak at 1.4B is expensive (hours of GPU time)
and is EXPLICITLY NOT run by this script. Operators run it offline;
this script just reads the resulting score / perplexity logs.

Score file format (flexible — pick the easier one to dump from your
validator's metrics pipeline):

1. ``{"window": <int>, "perplexity": <float>, "scores": {"<uid>": <float>}}``
   single-window summary.
2. ``[{"window": <int>, "perplexity": <float>, "scores": {...}}, ...]``
   list-of-window-records. Perplexity is averaged over windows; scores
   are averaged per-UID across windows before correlating.

Exit codes:
    0 — Both gates passed (Spearman ≥ ``--corr-threshold`` AND
        perplexity regression ≤ ``--ppl-regression-threshold``). Safe
        to flip ``turboquant_enabled`` true (after P6a audit also
        passes; this script does NOT re-check that).
    1 — At least one gate failed. Keep ``turboquant_enabled: false``
        and re-tune the codec (smaller ``turboquant_outlier_top_k``,
        more bits, prod-mode toggle, ...).
    2 — Usage / file error. Nothing was compared.

Example::

    python -m hone.validator.turboquant_ab_compare \\
        --tq-off scores_tq_off_1.4b.json \\
        --tq-on  scores_tq_on_1.4b.json \\
        --corr-threshold 0.85 \\
        --ppl-regression-threshold 0.005
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


def _load_run(path: Path) -> tuple[dict[int, float], float]:
    """Load ``path`` and collapse to ``({uid: avg_score}, avg_perplexity)``.

    Mirrors ``rda_ab_compare._load_run`` so operators can reuse the
    same dump format across both gates. Per-UID scores are averaged
    across windows; perplexity is averaged across windows.
    """
    try:
        raw: Any = json.loads(path.read_text())
    except Exception as e:
        print(f"ERROR: failed to read {path}: {e}", file=sys.stderr)
        sys.exit(2)

    records: list[dict[str, Any]]
    if isinstance(raw, dict):
        records = [raw]
    elif isinstance(raw, list):
        records = [r for r in raw if isinstance(r, dict)]
    else:
        print(
            f"ERROR: {path} must be a JSON object or list of records, "
            f"got {type(raw).__name__}",
            file=sys.stderr,
        )
        sys.exit(2)

    if not records:
        print(f"ERROR: {path} contained zero usable records", file=sys.stderr)
        sys.exit(2)

    per_uid: dict[int, list[float]] = defaultdict(list)
    ppls: list[float] = []
    for rec in records:
        scores = rec.get("scores")
        if isinstance(scores, dict):
            for uid, s in scores.items():
                try:
                    per_uid[int(uid)].append(float(s))
                except (TypeError, ValueError):
                    continue
        ppl = rec.get("perplexity")
        if ppl is not None:
            try:
                ppls.append(float(ppl))
            except (TypeError, ValueError):
                continue

    if not per_uid:
        print(
            f"ERROR: {path} contained no parseable per-UID scores",
            file=sys.stderr,
        )
        sys.exit(2)
    if not ppls:
        print(
            f"ERROR: {path} contained no parseable perplexity values",
            file=sys.stderr,
        )
        sys.exit(2)

    avg_scores = {uid: sum(xs) / len(xs) for uid, xs in per_uid.items()}
    avg_ppl = sum(ppls) / len(ppls)
    return avg_scores, avg_ppl


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "P6b TurboQuant A/B gate: compare TQ-on vs TQ-off score / "
            "perplexity logs and decide whether the TurboQuant codec "
            "degrades validation perplexity or re-ranks peers beyond the "
            "per-phase tolerance. Run AFTER turboquant_audit_cli.py exits 0."
        )
    )
    parser.add_argument(
        "--tq-off",
        type=Path,
        required=True,
        help="Score file from the run with turboquant_enabled=false (baseline).",
    )
    parser.add_argument(
        "--tq-on",
        type=Path,
        required=True,
        help=(
            "Score file from the run with turboquant_enabled=true "
            "(everything else identical to the baseline)."
        ),
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.85,
        help=(
            "Minimum Spearman rank-correlation of per-UID scores "
            "(default 0.85). Below this, TurboQuant is silently "
            "re-ranking peers and is unsafe to default on."
        ),
    )
    parser.add_argument(
        "--ppl-regression-threshold",
        type=float,
        default=0.005,
        help=(
            "Maximum allowed relative validation-perplexity regression "
            "(TQ-on / TQ-off - 1). Default 0.005 = 0.5%%, matching the "
            "plan's per-phase validation gate."
        ),
    )
    parser.add_argument(
        "--min-uids",
        type=int,
        default=8,
        help=(
            "Minimum number of UIDs that must appear in BOTH score files "
            "before Spearman is meaningful (default 8)."
        ),
    )
    args = parser.parse_args()

    off_scores, off_ppl = _load_run(args.tq_off)
    on_scores, on_ppl = _load_run(args.tq_on)

    # ---- Perplexity gate ------------------------------------------------
    # ``off_ppl`` is the reference; flag if TQ-on regresses past the
    # threshold. Negative regression (TQ-on perplexity LOWER than off)
    # is fine — that's a win and never trips the gate.
    ppl_delta_rel = (on_ppl / off_ppl - 1.0) if off_ppl > 0 else float("inf")
    ppl_pass = ppl_delta_rel <= args.ppl_regression_threshold

    # ---- Spearman score gate -------------------------------------------
    shared = sorted(set(off_scores) & set(on_scores))
    if len(shared) < args.min_uids:
        print(
            "FAIL: only "
            f"{len(shared)} UIDs overlap between score files "
            f"(need >= {args.min_uids}).",
            file=sys.stderr,
        )
        return 1

    x = [off_scores[uid] for uid in shared]
    y = [on_scores[uid] for uid in shared]
    try:
        res = spearmanr(x, y)
    except Exception as e:  # pragma: no cover
        print(f"ERROR: scipy spearmanr failed: {e}", file=sys.stderr)
        return 2

    rho: float
    pval: float
    if hasattr(res, "correlation"):
        rho = float(res.correlation)  # type: ignore[attr-defined]
        pval = float(res.pvalue)  # type: ignore[attr-defined]
    else:  # pragma: no cover — older scipy
        rho, pval = float(res[0]), float(res[1])

    corr_pass = rho >= args.corr_threshold

    # ---- Verdict --------------------------------------------------------
    summary = (
        f"TQ-off perplexity = {off_ppl:.6f}\n"
        f"TQ-on  perplexity = {on_ppl:.6f}  "
        f"({100.0 * ppl_delta_rel:+.3f}% vs off; "
        f"threshold {100.0 * args.ppl_regression_threshold:.3f}%)\n"
        f"Spearman rho = {rho:.4f} (p={pval:.4g}) on n={len(shared)} UIDs "
        f"(threshold {args.corr_threshold:.2f})"
    )

    if ppl_pass and corr_pass:
        print(
            "PASS: both A/B gates cleared. TurboQuant codec is "
            "perplexity-neutral and does not re-rank peers.\n"
            + summary
            + "\n\n"
            "REMINDER: this script ONLY checks the perplexity / "
            "Spearman gates. Confirm `turboquant_audit_cli.py` ALSO "
            "exited 0 against a recent EF snapshot before flipping "
            "`turboquant_enabled: true`."
        )
        return 0

    failed_gates: list[str] = []
    if not ppl_pass:
        failed_gates.append("perplexity-regression")
    if not corr_pass:
        failed_gates.append("score-Spearman")
    print(
        "=" * 72
        + "\n"
        + "FAIL: P6b TurboQuant A/B gate DID NOT PASS — failed: "
        + ", ".join(failed_gates)
        + "\n"
        + summary
        + "\n"
        + "\n"
        + "Do NOT set ``turboquant_enabled: true`` in hparams.json.\n"
        + "The codec produces measurable perplexity / ranking drift on\n"
        + "this snapshot; re-tune ``turboquant_bits`` (try 5),\n"
        + "``turboquant_outlier_top_k`` (try 64), or revisit P6a to\n"
        + "confirm the Beta assumption still holds at the current\n"
        + "training stage.\n"
        + "=" * 72,
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
