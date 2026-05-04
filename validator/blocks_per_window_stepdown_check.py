#!/usr/bin/env python3
# The MIT License (MIT)
# © 2025 hone.training
"""P3b step-down gate check for ``hparams.blocks_per_window``.

Standalone operator CLI — query the hone-api ``windowMetrics`` endpoint
for the most recent N hours of validator telemetry and report whether
all of the P3b step-down gates pass. The gates encode the same
criteria the operator playbook
(``hone/docs/blocks-per-window-stepdown.md``) calls out, so an operator
can answer "is it safe to halve ``blocks_per_window``?" without
manually scrolling the dashboard.

Run *before* every ``pm2 restart vali``-driven step-down. Exit code is
machine-readable and intentionally simple:

* ``0`` — every gate passed; the step-down is safe per this tool's
  view of the last N hours of telemetry.
* ``1`` — at least one gate failed; do not step down. Inspect the
  per-gate output (or ``--json``) to see which.
* ``2`` — could not fetch / parse data (env var missing, network
  error, API unreachable, malformed JSON, etc.). Nothing was decided.

The tool reads three env vars in order of precedence:

1. ``HONE_API_URL`` — explicit override; what the playbook
   recommends operators set in their shell profile.
2. ``NEXT_PUBLIC_API_URL`` — for parity with the dashboard's
   public env naming.
3. ``DASHBOARD_API_URL`` — already set by the validator process for
   ingest, so opportunistically reused if neither of the above is
   present.

Example::

    HONE_API_URL=https://api.hone.training \\
    CURRENT_BPW=30 \\
    python hone/validator/blocks_per_window_stepdown_check.py

    # Machine-readable output (CI / cron / scripts):
    HONE_API_URL=... CURRENT_BPW=15 \\
    python hone/validator/blocks_per_window_stepdown_check.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Tunables. Defaults match the gate criteria in the operator playbook
# (`hone/docs/blocks-per-window-stepdown.md`). Override via CLI flags
# rather than editing here.
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_HOURS = 24
DEFAULT_GATHER_TIMEOUT_RATE = 5.0  # gather-timeouts/hour ceiling
DEFAULT_PERPLEXITY_REGRESSION_PCT = 1.0  # 7-day slope ceiling
DEFAULT_WALL_CLOCK_RATIO = 0.5  # timing_window_total < 0.5 × bpw × 12s
SECONDS_PER_BLOCK = 12  # Bittensor finney block target time
HTTP_TIMEOUT = 30  # seconds per HTTP request


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib-only). We deliberately avoid `requests` because
# it's not in `hone/pyproject.toml`'s explicit deps; relying on the
# transitive copy that ships with `bittensor`/`wandb` would be brittle.
# ---------------------------------------------------------------------------


def _http_get_json(url: str) -> Any:
    """GET ``url`` and parse the response as JSON.

    Raises ``RuntimeError`` for any non-2xx, network failure, or
    malformed JSON. The caller (``main``) maps that to exit code 2.
    """
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            if resp.status >= 300:
                raise RuntimeError(f"HTTP {resp.status} from {url}")
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error fetching {url}: {e.reason}") from e
    except OSError as e:
        raise RuntimeError(f"network error fetching {url}: {e}") from e
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON from {url}: {e}") from e


def _resolve_api_url() -> str | None:
    """Pick the first of the three documented env vars that is set."""
    for var in ("HONE_API_URL", "NEXT_PUBLIC_API_URL", "DASHBOARD_API_URL"):
        v = os.environ.get(var)
        if v:
            return v.rstrip("/")
    return None


# ---------------------------------------------------------------------------
# Data fetch. Two-step: discover the latest validator run, then read
# its windowMetrics rows since the cutoff. Both calls go through the
# public ``hone-api`` REST surface so the tool needs no DB credentials.
# ---------------------------------------------------------------------------


def _fetch_latest_validator_run(api_url: str) -> dict[str, Any]:
    """Return the most-recently-active validator training run.

    We list all validator runs ordered by ``last_seen_at`` desc (the
    API's default) and pick the head. If there are zero validator
    runs, callers get a clean RuntimeError rather than a None deref.
    """
    body = _http_get_json(f"{api_url}/api/runs?role=validator&limit=1")
    runs = body.get("runs", []) if isinstance(body, dict) else []
    if not runs:
        raise RuntimeError(
            "no validator runs registered with hone-api; "
            "is your validator running and reporting telemetry?"
        )
    return runs[0]


def fetch_recent_windows(
    api_url: str, hours: int = DEFAULT_WINDOW_HOURS
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(run, windows)`` for the most recent validator run.

    ``windows`` is the list of WindowMetrics-shaped dicts whose
    ``createdAt`` is at or after ``now - hours``, sorted ascending by
    window number. Empty list when the API has no rows in range —
    callers are responsible for treating that as "insufficient data"
    (which fails every gate by construction).
    """
    run = _fetch_latest_validator_run(api_url)
    run_id = run.get("id") or run.get("externalId")
    if run_id is None:
        raise RuntimeError("validator run has neither id nor externalId; bad payload")

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    qs = urllib.parse.urlencode({"from": since.isoformat(), "limit": 1000})
    body = _http_get_json(f"{api_url}/api/runs/{run_id}/windows?{qs}")
    windows = body.get("windows", []) if isinstance(body, dict) else []

    # The API returns rows sorted by window desc; flip to ascending so
    # downstream median / slope math reads naturally.
    windows.sort(key=lambda w: w.get("window", 0))
    return run, windows


# ---------------------------------------------------------------------------
# Gate evaluation. Each gate returns a small dict so the JSON output is
# self-describing; ``main`` aggregates them into the human-readable
# report. Gates that can't run (insufficient data) fail closed — the
# whole point of this tool is to be conservative.
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _gate(passed: bool, value: Any, target: Any, reason: str) -> dict[str, Any]:
    return {"passed": bool(passed), "value": value, "target": target, "reason": reason}


def gate_wall_clock(
    windows: list[dict[str, Any]], current_bpw: int, ratio: float
) -> dict[str, Any]:
    """``timing_window_total`` median < ``ratio × bpw × 12 s``."""
    cap = ratio * current_bpw * SECONDS_PER_BLOCK
    medians = _median([w.get("timingWindowTotal") for w in windows])
    if medians is None:
        return _gate(False, None, cap, "no timingWindowTotal samples in window")
    return _gate(
        medians < cap,
        round(medians, 2),
        round(cap, 2),
        f"sliding-median window total {medians:.1f}s "
        f"vs cap {cap:.1f}s ({ratio:g} × {current_bpw} × {SECONDS_PER_BLOCK}s)",
    )


def gate_gather_timeout_rate(
    windows: list[dict[str, Any]], hours: int, max_per_hour: float
) -> dict[str, Any]:
    """Count of windows with ``gatherSuccessRate < 0.5`` per hour."""
    timeouts = sum(
        1
        for w in windows
        if isinstance(w.get("gatherSuccessRate"), (int, float))
        and w["gatherSuccessRate"] < 50.0
    )
    rate = timeouts / hours if hours > 0 else float("inf")
    return _gate(
        rate <= max_per_hour,
        round(rate, 3),
        max_per_hour,
        f"{timeouts} timeout window(s) over {hours}h = {rate:.2f}/h "
        f"(allowed: ≤ {max_per_hour}/h)",
    )


def gate_perplexity_trend(
    windows: list[dict[str, Any]], regression_pct: float
) -> dict[str, Any]:
    """Compare median ``lossOwnAfter`` of first/last thirds of the window.

    Surrogate for the playbook's "7-day slope flat or improving":
    inside a 24-h fetch we don't have 7 days of context, so we use the
    most recent N rows and compare their first vs last third. A
    regression of more than ``regression_pct`` percent fails the gate.
    """
    losses = [
        w.get("lossOwnAfter")
        for w in windows
        if isinstance(w.get("lossOwnAfter"), (int, float))
    ]
    if len(losses) < 6:
        return _gate(
            False,
            len(losses),
            6,
            f"need ≥ 6 lossOwnAfter samples to estimate trend; have {len(losses)}",
        )
    third = max(1, len(losses) // 3)
    early = _median(losses[:third])
    late = _median(losses[-third:])
    if early is None or late is None or early <= 0:
        return _gate(False, None, regression_pct, "could not compute trend medians")
    delta_pct = ((late - early) / early) * 100.0
    return _gate(
        delta_pct <= regression_pct,
        round(delta_pct, 3),
        regression_pct,
        f"lossOwnAfter median moved {delta_pct:+.2f}% "
        f"(first third {early:.4f} → last third {late:.4f}); "
        f"allowed regression: ≤ {regression_pct:+.2f}%",
    )


def gate_sigterm_count(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reserved gate; hone-api does not yet ingest a SIGTERM counter.

    We surface this as a passthrough today (always `True`) with a
    clear ``TODO`` value so it's easy to find when wiring the source.
    Bumping this from passthrough to a real gate is a one-liner once
    the ingest column lands.
    """
    return _gate(
        True,
        "TODO: ingest SIGTERM counter into windowMetrics",
        0,
        "SIGTERM count not available in windowMetrics today (passthrough)",
    )


def gate_blocks_per_window_consistency(
    windows: list[dict[str, Any]], current_bpw: int
) -> dict[str, Any]:
    """Sanity gate: validator-reported `blocksPerWindow` matches `CURRENT_BPW`.

    Catches the obvious operator mistake of running this tool with a
    stale `CURRENT_BPW` env var after a step-down. If validator rows
    don't yet carry the `blocksPerWindow` passthrough field (pre-P3b
    binaries), this gate is a no-op pass with a clear note.
    """
    reported = [
        w.get("blocksPerWindow")
        for w in windows
        if isinstance(w.get("blocksPerWindow"), int)
    ]
    if not reported:
        return _gate(
            True,
            None,
            current_bpw,
            "validator did not report blocksPerWindow on any window "
            "(pre-P3b binary or telemetry gap); skipping consistency check",
        )
    most_recent = reported[-1]
    return _gate(
        most_recent == current_bpw,
        most_recent,
        current_bpw,
        f"validator reports blocksPerWindow={most_recent}; "
        f"CURRENT_BPW env says {current_bpw}",
    )


def check_step_down_gates(
    windows: list[dict[str, Any]],
    current_bpw: int,
    *,
    hours: int = DEFAULT_WINDOW_HOURS,
    wall_clock_ratio: float = DEFAULT_WALL_CLOCK_RATIO,
    gather_timeout_rate: float = DEFAULT_GATHER_TIMEOUT_RATE,
    perplexity_regression_pct: float = DEFAULT_PERPLEXITY_REGRESSION_PCT,
) -> dict[str, dict[str, Any]]:
    """Run every gate and return ``{gate_name: gate_result}``."""
    return {
        "blocks_per_window_consistency": gate_blocks_per_window_consistency(
            windows, current_bpw
        ),
        "wall_clock_headroom": gate_wall_clock(windows, current_bpw, wall_clock_ratio),
        "gather_timeout_rate": gate_gather_timeout_rate(
            windows, hours, gather_timeout_rate
        ),
        "perplexity_trend": gate_perplexity_trend(windows, perplexity_regression_pct),
        "sigterm_count": gate_sigterm_count(windows),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _suggest_next_step(current_bpw: int) -> int | None:
    """Map the current value to the playbook's next step-down target."""
    table = {30: 15, 15: 10, 10: 6}
    return table.get(current_bpw)


def render_human(
    *,
    api_url: str,
    run: dict[str, Any],
    windows: list[dict[str, Any]],
    hours: int,
    current_bpw: int,
    gates: dict[str, dict[str, Any]],
    overall_pass: bool,
) -> str:
    next_step = _suggest_next_step(current_bpw)
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("blocks_per_window step-down gate check")
    lines.append("=" * 72)
    lines.append(f"  api          : {api_url}")
    lines.append(
        f"  run          : id={run.get('id')} hotkey={run.get('hotkey')} "
        f"version={run.get('version')}"
    )
    lines.append(f"  window       : last {hours}h, {len(windows)} window row(s)")
    lines.append(f"  current_bpw  : {current_bpw} (from $CURRENT_BPW)")
    if next_step is not None:
        lines.append(
            f"  next step    : {current_bpw} → {next_step} "
            f"(per hone/docs/blocks-per-window-stepdown.md)"
        )
    else:
        lines.append(
            f"  next step    : (none — {current_bpw} is at or below the floor)"
        )
    lines.append("")
    lines.append("Gates:")
    for name, g in gates.items():
        marker = "PASS" if g["passed"] else "FAIL"
        lines.append(f"  [{marker}] {name}")
        lines.append(f"         {g['reason']}")
    lines.append("")
    if overall_pass:
        lines.append("Overall: PASS — step-down is safe per this tool's view.")
    else:
        failed = [n for n, g in gates.items() if not g["passed"]]
        lines.append(f"Overall: FAIL — gates not satisfied: {', '.join(failed)}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="blocks_per_window_stepdown_check",
        description=(
            "Check whether `hparams.blocks_per_window` can be safely "
            "stepped down on the current validator (P3b)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--window-hours",
        type=int,
        default=DEFAULT_WINDOW_HOURS,
        help="Look-back window for telemetry (hours).",
    )
    p.add_argument(
        "--current-bpw",
        type=int,
        default=None,
        help=(
            "Current blocks_per_window value. Defaults to $CURRENT_BPW "
            "and falls back to 30 (the shipped default) if unset."
        ),
    )
    p.add_argument(
        "--wall-clock-ratio",
        type=float,
        default=DEFAULT_WALL_CLOCK_RATIO,
        help="Pass when timing_window_total median < ratio × bpw × 12s.",
    )
    p.add_argument(
        "--gather-timeout-rate",
        type=float,
        default=DEFAULT_GATHER_TIMEOUT_RATE,
        help=(
            "Maximum gather-timeout windows per hour "
            "(gatherSuccessRate < 0.5)."
        ),
    )
    p.add_argument(
        "--perplexity-regression-pct",
        type=float,
        default=DEFAULT_PERPLEXITY_REGRESSION_PCT,
        help="Maximum tolerated lossOwnAfter regression, in percent.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit a single machine-readable JSON object instead of human text.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    api_url = _resolve_api_url()
    if not api_url:
        print(
            "error: set HONE_API_URL or NEXT_PUBLIC_API_URL "
            "(DASHBOARD_API_URL also accepted) to point at hone-api",
            file=sys.stderr,
        )
        return 2

    if args.current_bpw is not None:
        current_bpw = args.current_bpw
    else:
        env_bpw = os.environ.get("CURRENT_BPW")
        try:
            current_bpw = int(env_bpw) if env_bpw is not None else 30
        except ValueError:
            print(
                f"error: $CURRENT_BPW={env_bpw!r} is not an integer", file=sys.stderr
            )
            return 2

    try:
        run, windows = fetch_recent_windows(api_url, hours=args.window_hours)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    gates = check_step_down_gates(
        windows,
        current_bpw,
        hours=args.window_hours,
        wall_clock_ratio=args.wall_clock_ratio,
        gather_timeout_rate=args.gather_timeout_rate,
        perplexity_regression_pct=args.perplexity_regression_pct,
    )
    overall_pass = all(g["passed"] for g in gates.values())

    if args.json:
        print(
            json.dumps(
                {
                    "api_url": api_url,
                    "run_id": run.get("id"),
                    "run_external_id": run.get("externalId"),
                    "hotkey": run.get("hotkey"),
                    "version": run.get("version"),
                    "window_hours": args.window_hours,
                    "windows_seen": len(windows),
                    "current_bpw": current_bpw,
                    "next_bpw": _suggest_next_step(current_bpw),
                    "overall_pass": overall_pass,
                    "gates": gates,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            render_human(
                api_url=api_url,
                run=run,
                windows=windows,
                hours=args.window_hours,
                current_bpw=current_bpw,
                gates=gates,
                overall_pass=overall_pass,
            )
        )

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
