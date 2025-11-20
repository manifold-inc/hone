#!/usr/bin/env python3
"""
Enhanced Sandbox Runner Dashboard using Gradio

Features:
- Real-time job monitoring with proper error handling
- Failed jobs tracking with logs persistence
- Success jobs metrics page
- Disk space monitoring
- Interactive gauges and visualizations
"""

import gradio as gr
import httpx
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time
import plotly.graph_objects as go
import json
import shutil
from pathlib import Path
import threading

# ============================================================
# Configuration
# ============================================================

API_BASE_URL = "http://localhost:8080"
API_KEY = "dev-key-12345"

# Local storage for failed jobs
FAILED_JOBS_DIR = Path("/tmp/sandbox_failed_jobs")
SUCCESS_JOBS_DIR = Path("/tmp/sandbox_success_jobs")
FAILED_JOBS_DIR.mkdir(parents=True, exist_ok=True)
SUCCESS_JOBS_DIR.mkdir(parents=True, exist_ok=True)

MAX_DISK_USAGE_PERCENT = 50.0

# Cache for reducing API calls
_status_cache = {"data": None, "timestamp": 0.0}
_CACHE_TTL = 1.0

_client: Optional[httpx.Client] = None


def get_client() -> httpx.Client:
    """Get a shared HTTP client with proper headers."""
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=API_BASE_URL,
            headers={"X-API-Key": API_KEY},
            timeout=10.0,
        )
    return _client


# ============================================================
# Disk Space Management
# ============================================================

def get_disk_usage() -> Dict:
    """Get disk usage statistics"""
    try:
        total, used, free = shutil.disk_usage("/tmp")
        
        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "used_percent": (used / total) * 100
        }
    except Exception as e:
        print(f"Error getting disk usage: {e}")
        return {
            "total_gb": 0,
            "used_gb": 0,
            "free_gb": 0,
            "used_percent": 0
        }


def cleanup_old_jobs_if_needed():
    """Remove oldest job logs if disk usage exceeds threshold"""
    disk = get_disk_usage()
    
    if disk["used_percent"] < MAX_DISK_USAGE_PERCENT:
        return
    
    print(f"Disk usage at {disk['used_percent']:.1f}%, cleaning up old jobs...")
    
    # cleanup failed jobs first
    failed_jobs = sorted(FAILED_JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    
    removed_count = 0
    for job_file in failed_jobs[:len(failed_jobs)//4]:  # remove oldest 25%
        try:
            job_file.unlink()
            removed_count += 1
        except Exception as e:
            print(f"Error removing {job_file}: {e}")
    
    # cleanup success jobs if still needed
    disk = get_disk_usage()
    if disk["used_percent"] >= MAX_DISK_USAGE_PERCENT:
        success_jobs = sorted(SUCCESS_JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        
        for job_file in success_jobs[:len(success_jobs)//4]:
            try:
                job_file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Error removing {job_file}: {e}")
    
    print(f"Cleaned up {removed_count} old job files")


# ============================================================
# Job Persistence
# ============================================================

def save_failed_job(job_id: str, job_data: Dict, logs: str):
    """Save failed job with logs to disk"""
    try:
        cleanup_old_jobs_if_needed()
        
        job_file = FAILED_JOBS_DIR / f"{job_id}.json"
        
        data = {
            "job_id": job_id,
            "saved_at": datetime.now().isoformat(),
            "job_data": job_data,
            "logs": logs
        }
        
        with open(job_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved failed job: {job_id}")
    except Exception as e:
        print(f"Error saving failed job {job_id}: {e}")


def save_success_job(job_id: str, job_data: Dict, metrics: Dict):
    """Save successful job with metrics"""
    try:
        cleanup_old_jobs_if_needed()
        
        job_file = SUCCESS_JOBS_DIR / f"{job_id}.json"
        
        data = {
            "job_id": job_id,
            "saved_at": datetime.now().isoformat(),
            "job_data": job_data,
            "metrics": metrics
        }
        
        with open(job_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved success job: {job_id}")
    except Exception as e:
        print(f"Error saving success job {job_id}: {e}")


def load_failed_jobs() -> List[Dict]:
    """Load all failed jobs from disk"""
    jobs = []
    
    for job_file in sorted(FAILED_JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(job_file, 'r') as f:
                jobs.append(json.load(f))
        except Exception as e:
            print(f"Error loading {job_file}: {e}")
    
    return jobs


def load_success_jobs() -> List[Dict]:
    """Load all success jobs from disk"""
    jobs = []
    
    for job_file in sorted(SUCCESS_JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(job_file, 'r') as f:
                jobs.append(json.load(f))
        except Exception as e:
            print(f"Error loading {job_file}: {e}")
    
    return jobs


# ============================================================
# Core API helpers
# ============================================================

def get_runner_status(use_cache: bool = True) -> Dict:
    """Get runner status including active job IDs with caching."""
    global _status_cache

    now = time.time()
    if (
        use_cache
        and _status_cache["data"] is not None
        and now - _status_cache["timestamp"] < _CACHE_TTL
    ):
        return _status_cache["data"]

    try:
        client = get_client()
        resp = client.get("/v1/status")
        if resp.status_code == 200:
            data = resp.json()
            _status_cache = {"data": data, "timestamp": now}
            return data
        return _status_cache.get("data") or {}
    except Exception as e:
        print(f"Error fetching status: {e}")
        return _status_cache.get("data") or {}


def get_active_jobs_table(status: Dict) -> Tuple[List[List[str]], List[str]]:
    """Get active jobs with proper error handling"""
    try:
        active_job_ids = status.get("active_job_ids", []) or []
        if not active_job_ids:
            return [], []

        client = get_client()
        table_data: List[List[str]] = []

        for job_id in active_job_ids:
            try:
                r = client.get(f"/v1/jobs/{job_id}")
                
                if r.status_code == 404:
                    # job no longer exists - might be completed, check if we should save it
                    continue
                
                if r.status_code != 200:
                    table_data.append(
                        [job_id, "❌ error", "0%", "unknown", "N/A", "N/A", "error"]
                    )
                    continue

                job = r.json()
                gpus = job.get("assigned_gpus", []) or []
                gpu_str = f"GPU {','.join(map(str, gpus))}" if gpus else "N/A"

                started_at = job.get("started_at")
                if started_at:
                    try:
                        dt = datetime.fromisoformat(started_at)
                        time_str = dt.strftime("%H:%M:%S")
                    except Exception:
                        time_str = "N/A"
                else:
                    time_str = "N/A"

                phase = job.get("current_phase", "unknown")
                phase_emoji = {
                    "build": "🔨",
                    "prep": "📦",
                    "inference": "🧠",
                    "vllm": "🚀",
                    "vllm_pipeline": "🚀",
                }
                phase_display = f"{phase_emoji.get(phase, '📝')} {phase}"

                table_data.append(
                    [
                        job_id,
                        phase_display,
                        f"{job.get('progress_percentage', 0.0):.1f}%",
                        job.get("weight_class", "unknown"),
                        gpu_str,
                        time_str,
                        (job.get("miner_hotkey") or "")[:12] + "...",
                    ]
                )
                
                # check if job is completed/failed and save it
                job_status = job.get("status", "").lower()
                if job_status in ["completed", "failed", "timeout", "cancelled"]:
                    if job_status == "completed":
                        # try to get metrics
                        try:
                            metrics_resp = client.get(f"/v1/jobs/{job_id}/metrics")
                            if metrics_resp.status_code == 200:
                                metrics = metrics_resp.json().get("metrics", {})
                                save_success_job(job_id, job, metrics)
                        except Exception as e:
                            print(f"Could not fetch metrics for {job_id}: {e}")
                    else:
                        # failed job - try to get logs
                        try:
                            logs_resp = client.get(f"/v1/logs/{job_id}/all")
                            if logs_resp.status_code == 200:
                                logs_data = logs_resp.json()
                                logs = "\n".join([
                                    f"[{entry.get('timestamp')}] [{entry.get('phase')}] {entry.get('message')}"
                                    for entry in logs_data.get("entries", [])
                                ])
                                save_failed_job(job_id, job, logs)
                        except Exception as e:
                            print(f"Could not fetch logs for {job_id}: {e}")
                
            except Exception as e:
                print(f"Error fetching job {job_id}: {e}")
                table_data.append(
                    [job_id, "❌ error", "0%", "unknown", "N/A", "N/A", "error"]
                )

        return table_data, active_job_ids
    except Exception as e:
        print(f"Error in get_active_jobs_table: {e}")
        return [], []


def get_job_logs_tail(job_id: str, lines: int = 100, phase_filter: str = "all") -> str:
    """Get last N lines of logs for a job using /tail endpoint."""
    if not job_id:
        return ""

    try:
        client = get_client()
        params = {"lines": lines}
        if phase_filter != "all":
            params["phase"] = phase_filter

        resp = client.get(f"/v1/logs/{job_id}/tail", params=params)
        if resp.status_code != 200:
            return f"[{job_id}] Error: Status {resp.status_code}\n"

        data = resp.json()
        entries = data.get("entries", [])
        if not entries:
            return f"[{job_id}] No logs available yet...\n"

        lines_out: List[str] = []
        for entry in entries:
            timestamp = entry.get("timestamp", "")
            phase = entry.get("phase", "")
            level = (entry.get("level") or "info").lower()
            message = entry.get("message", "")

            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                ts_str = dt.strftime("%H:%M:%S")
            except Exception:
                ts_str = timestamp[:8] if len(timestamp) >= 8 else "??:??:??"

            phase_emoji = {
                "build": "🔨",
                "prep": "📦",
                "inference": "🧠",
                "vllm": "🚀",
                "vllm_pipeline": "🚀",
            }
            level_emoji = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}

            phase_icon = phase_emoji.get(phase, "📝")
            level_icon = level_emoji.get(level, "")

            lines_out.append(f"[{ts_str}] {phase_icon} {level_icon} {message}")

        return "\n".join(lines_out)
    except Exception as e:
        return f"[{job_id}] Error: {e}\n"


def get_all_active_logs(
    selected_job_ids: List[str],
    phase_filter: str = "all",
) -> str:
    """Get logs for all selected jobs."""
    if not selected_job_ids:
        return "No active jobs selected. Click 'Refresh' to load active jobs."

    out: List[str] = []
    out.append("=" * 100)
    out.append(f"LOGS FOR {len(selected_job_ids)} ACTIVE JOB(S)")
    out.append(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append("=" * 100)
    out.append("")

    for job_id in selected_job_ids:
        out.append("\n" + "─" * 100)
        out.append(f"📋 JOB: {job_id}")
        out.append("─" * 100 + "\n")
        out.append(get_job_logs_tail(job_id, lines=60, phase_filter=phase_filter))
        out.append("")

    return "\n".join(out)


# ============================================================
# Plotly visualizations
# ============================================================

def create_gpu_gauge(total: int, allocated: int, available: int) -> go.Figure:
    """Gauge chart for GPU utilization."""
    total = max(total, 1)
    allocated = max(0, min(allocated, total))
    utilization_pct = (allocated / total) * 100.0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=allocated,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"GPU Allocation<br><sub>{available} free · {utilization_pct:.0f}% used</sub>",
                "font": {"size": 16},
            },
            number={"suffix": f" / {total}", "font": {"size": 22}},
            gauge={
                "axis": {"range": [0, total], "tickwidth": 1},
                "bar": {"color": "#4299e1"},
                "bgcolor": "rgba(15, 23, 42, 1)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, total * 0.5], "color": "rgba(56, 189, 248, 0.2)"},
                    {"range": [total * 0.5, total * 0.8], "color": "rgba(250, 204, 21, 0.25)"},
                    {"range": [total * 0.8, total], "color": "rgba(248, 113, 113, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.6,
                    "value": total * 0.9,
                },
            },
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig


def create_job_stats_chart(
    active: int,
    completed: int,
    failed: int,
    queued: int,
) -> go.Figure:
    """Bar chart for job statistics."""
    labels = ["Active", "Queued", "Completed", "Failed"]
    values = [active, queued, completed, failed]
    colors = ["#4299e1", "#fbbf24", "#10b981", "#ef4444"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=values,
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title={"text": "Job Statistics", "font": {"size": 16}},
        height=260,
        margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,1)",
        font={"color": "#e2e8f0"},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
    )
    return fig


def create_disk_usage_gauge() -> go.Figure:
    """Gauge chart for disk usage."""
    disk = get_disk_usage()
    
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=disk["used_percent"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"Disk Usage<br><sub>{disk['free_gb']:.1f} GB free</sub>",
                "font": {"size": 16},
            },
            number={"suffix": "%", "font": {"size": 22}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#4299e1"},
                "bgcolor": "rgba(15, 23, 42, 1)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [50, 75], "color": "rgba(250, 204, 21, 0.25)"},
                    {"range": [75, 100], "color": "rgba(248, 113, 113, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.6,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig


def create_runner_info_fig(status: Dict) -> Tuple[go.Figure, str]:
    """Create a small donut chart for success vs failed + summary text."""
    total_submitted = status.get("total_submitted", 0) or 0
    total_completed = status.get("total_completed", 0) or 0
    total_failed = status.get("total_failed", 0) or 0
    total_done = total_completed + total_failed

    if total_done == 0:
        success_rate = 0.0
    else:
        success_rate = (total_completed / max(total_done, 1)) * 100.0

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Completed", "Failed"],
                values=[total_completed, total_failed],
                hole=0.55,
                textinfo="label+percent",
                marker=dict(colors=["#10b981", "#ef4444"]),
            )
        ]
    )
    fig.update_layout(
        title={"text": "Runner Success Rate", "font": {"size": 14}},
        showlegend=False,
        height=230,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )

    summary = (
        f"ID: {status.get('runner_id', 'unknown')}  ·  "
        f"Mode: {status.get('execution_mode', 'unknown')}  ·  "
        f"Status: {str(status.get('status', 'unknown')).upper()}\n"
        f"Completed: {total_completed}  ·  Failed: {total_failed}  ·  "
        f"Success rate: {success_rate:.1f}%"
    )
    return fig, summary


def create_queue_fig(status: Dict) -> Tuple[go.Figure, str]:
    """Create a gauge or bar showing queue depth."""
    queue_depth = status.get("queue_depth", 0) or 0
    active_jobs = status.get("active_jobs", 0) or 0
    max_depth = max(queue_depth, 1) * 2

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=queue_depth,
            title={"text": "Queue Depth", "font": {"size": 14}},
            number={"font": {"size": 22}},
            gauge={
                "axis": {"range": [0, max_depth]},
                "bar": {"color": "#fbbf24"},
                "steps": [
                    {"range": [0, max_depth * 0.4], "color": "rgba(56,189,248,0.2)"},
                    {"range": [max_depth * 0.4, max_depth * 0.7], "color": "rgba(250,204,21,0.25)"},
                    {"range": [max_depth * 0.7, max_depth], "color": "rgba(248,113,113,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )

    summary = (
        f"Queue depth: {queue_depth}  ·  "
        f"Active jobs: {active_jobs}  ·  "
        f"Updated: {datetime.now().strftime('%H:%M:%S')}"
    )
    return fig, summary


# ============================================================
# Gradio glue
# ============================================================

def refresh_all(phase_filter: str, job_id_filter: str = ""):
    """Refresh all dashboard data with a single /status call."""
    status = get_runner_status(use_cache=False)

    table_data, active_job_ids = get_active_jobs_table(status)

    job_id_filter = (job_id_filter or "").strip()
    if job_id_filter:
        logs = get_all_active_logs([job_id_filter], phase_filter)
    else:
        logs = get_all_active_logs(active_job_ids, phase_filter)

    gpu_gauge = create_gpu_gauge(
        status.get("total_gpus", 0) or 0,
        status.get("allocated_gpus", 0) or 0,
        status.get("available_gpus", 0) or 0,
    )

    job_chart = create_job_stats_chart(
        status.get("active_jobs", 0) or 0,
        status.get("total_completed", 0) or 0,
        status.get("total_failed", 0) or 0,
        status.get("queue_depth", 0) or 0,
    )

    runner_fig, runner_summary = create_runner_info_fig(status)
    queue_fig, queue_summary = create_queue_fig(status)
    disk_gauge = create_disk_usage_gauge()

    return (
        table_data,
        logs,
        gpu_gauge,
        job_chart,
        runner_fig,
        runner_summary,
        queue_fig,
        queue_summary,
        disk_gauge,
    )


def refresh_failed_jobs():
    """Refresh failed jobs page"""
    failed_jobs = load_failed_jobs()
    
    if not failed_jobs:
        return [], "No failed jobs recorded", create_disk_usage_gauge()
    
    table_data = []
    for job in failed_jobs:
        job_data = job.get("job_data", {})
        table_data.append([
            job.get("job_id", "unknown"),
            job.get("saved_at", "unknown"),
            job_data.get("status", "unknown"),
            job_data.get("current_phase", "unknown"),
            (job_data.get("miner_hotkey", "") or "")[:12] + "...",
            job_data.get("error_message", "")[:50] + "..." if job_data.get("error_message") else "N/A"
        ])
    
    summary = f"Total failed jobs: {len(failed_jobs)}"
    
    return table_data, summary, create_disk_usage_gauge()


def get_failed_job_logs(evt: gr.SelectData, table_data):
    """Get logs for selected failed job"""
    try:
        if hasattr(table_data, "iloc"):
            job_id = str(table_data.iloc[evt.index[0], 0])
        elif isinstance(table_data, list) and evt.index[0] < len(table_data):
            job_id = str(table_data[evt.index[0]][0])
        else:
            return "Error loading logs"
        
        job_file = FAILED_JOBS_DIR / f"{job_id}.json"
        if not job_file.exists():
            return f"Logs not found for job {job_id}"
        
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        
        logs = job_data.get("logs", "No logs available")
        return f"=== LOGS FOR {job_id} ===\n\n{logs}"
        
    except Exception as e:
        return f"Error loading logs: {e}"


def refresh_success_jobs():
    """Refresh success jobs page"""
    success_jobs = load_success_jobs()
    
    if not success_jobs:
        return [], "No successful jobs recorded", create_disk_usage_gauge()
    
    table_data = []
    for job in success_jobs:
        job_data = job.get("job_data", {})
        metrics = job.get("metrics", {})
        
        # calculate execution time
        started_at = job_data.get("started_at")
        completed_at = job_data.get("completed_at")
        exec_time = "N/A"
        if started_at and completed_at:
            try:
                start_dt = datetime.fromisoformat(started_at)
                end_dt = datetime.fromisoformat(completed_at)
                exec_time = f"{(end_dt - start_dt).total_seconds():.1f}s"
            except:
                pass
        
        exact_match = metrics.get("exact_match", False)
        partial = metrics.get("partial_correctness", 0.0)
        similarity = metrics.get("grid_similarity", 0.0)
        
        table_data.append([
            job.get("job_id", "unknown"),
            job.get("saved_at", "unknown"),
            (job_data.get("miner_hotkey", "") or "")[:12] + "...",
            job_data.get("weight_class", "unknown"),
            exec_time,
            "✅" if exact_match else "❌",
            f"{partial:.3f}",
            f"{similarity:.3f}"
        ])
    
    summary = f"Total successful jobs: {len(success_jobs)}"
    
    return table_data, summary, create_disk_usage_gauge()


def get_success_job_details(evt: gr.SelectData, table_data):
    """Get detailed metrics for selected success job"""
    try:
        if hasattr(table_data, "iloc"):
            job_id = str(table_data.iloc[evt.index[0], 0])
        elif isinstance(table_data, list) and evt.index[0] < len(table_data):
            job_id = str(table_data[evt.index[0]][0])
        else:
            return "Error loading details"
        
        job_file = SUCCESS_JOBS_DIR / f"{job_id}.json"
        if not job_file.exists():
            return f"Details not found for job {job_id}"
        
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        
        metrics = job_data.get("metrics", {})
        job_info = job_data.get("job_data", {})
        
        details = f"=== JOB DETAILS: {job_id} ===\n\n"
        details += f"Miner Hotkey: {job_info.get('miner_hotkey', 'N/A')}\n"
        details += f"Weight Class: {job_info.get('weight_class', 'N/A')}\n"
        details += f"Started: {job_info.get('started_at', 'N/A')}\n"
        details += f"Completed: {job_info.get('completed_at', 'N/A')}\n\n"
        
        details += "=== METRICS ===\n\n"
        details += f"Exact Match: {'✅ Yes' if metrics.get('exact_match') else '❌ No'}\n"
        details += f"Partial Correctness: {metrics.get('partial_correctness', 0.0):.4f}\n"
        details += f"Grid Similarity: {metrics.get('grid_similarity', 0.0):.4f}\n"
        details += f"Efficiency Score: {metrics.get('efficiency_score', 0.0):.4f}\n\n"
        
        if metrics.get("problem_id"):
            details += f"Problem ID: {metrics.get('problem_id')}\n"
        if metrics.get("base_task_num"):
            details += f"Base Task: {metrics.get('base_task_num')}\n"
        if metrics.get("chain_length"):
            details += f"Chain Length: {metrics.get('chain_length')}\n"
        
        return details
        
    except Exception as e:
        return f"Error loading details: {e}"


def clear_job_filter():
    """Clear job filter and reset refresh rate."""
    return "", False, 5


def select_job_from_table(evt: gr.SelectData, table_data):
    """Handle job selection from table."""
    try:
        if hasattr(table_data, "iloc"):
            job_id = str(table_data.iloc[evt.index[0], 0])
        elif isinstance(table_data, list) and evt.index[0] < len(table_data):
            job_id = str(table_data[evt.index[0]][0])
        else:
            return "", False, 2
        return job_id, True, 2
    except Exception as e:
        print(f"Error selecting job: {e}")
        return "", False, 2


def update_timer(enabled: bool, interval: int):
    """Update timer active flag + interval."""
    return gr.update(active=enabled, value=interval)


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(title="Sandbox Runner Dashboard", theme=gr.themes.Soft()) as app:
    gr.HTML(
        """
        <div style="text-align: center; padding: 18px 0 10px;">
            <h1 style="
                font-size: 2.4em;
                margin: 0;
                background: linear-gradient(90deg, #4299e1 0%, #667eea 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
                🚀 Sandbox Runner Dashboard
            </h1>
            <p style="color: #a0aec0; margin-top: 8px; font-size: 1.05em;">
                Real-time GPU job monitoring & analytics
            </p>
        </div>
        """
    )

    with gr.Tabs():
        # ===== TAB 1: MAIN DASHBOARD =====
        with gr.Tab("📊 Main Dashboard"):
            with gr.Row():
                # LEFT COLUMN: Jobs & Logs
                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.Markdown("### 📋 Active Jobs")
                        with gr.Column(scale=2):
                            with gr.Row():
                                refresh_btn = gr.Button("🔄 Refresh", variant="primary", size="sm")
                                auto_refresh = gr.Checkbox(label="Auto", value=False)
                                refresh_interval = gr.Radio(
                                    choices=[2, 5],
                                    value=5,
                                    show_label=False,
                                )

                    jobs_table = gr.Dataframe(
                        headers=[
                            "Job ID",
                            "Phase",
                            "Progress",
                            "Weight",
                            "GPUs",
                            "Started",
                            "Miner",
                        ],
                        value=[],
                        interactive=False,
                        wrap=True,
                        column_widths=["22%", "14%", "10%", "12%", "10%", "10%", "22%"],
                    )

                    gr.Markdown("### ⚙️ Filters")
                    with gr.Row():
                        job_id_filter = gr.Textbox(
                            label="Job ID",
                            placeholder="Click a job row or enter ID…",
                            scale=3,
                        )
                        phase_filter = gr.Dropdown(
                            label="Phase",
                            choices=["all", "build", "prep", "inference", "vllm", "vllm_pipeline"],
                            value="all",
                            scale=2,
                        )
                        clear_filter_btn = gr.Button("✕", scale=1, size="sm")

                    gr.Markdown("### 📜 Live Logs")
                    log_output = gr.Textbox(
                        label="",
                        lines=25,
                        max_lines=40,
                        show_label=False,
                        show_copy_button=True,
                        interactive=False,
                        container=True,
                        elem_id="srd-log-output",
                    )

                # RIGHT COLUMN: Stats & Visuals
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 System Overview")

                    with gr.Row():
                        gpu_gauge_plot = gr.Plot(label="GPU Allocation")

                    with gr.Row():
                        job_stats_plot = gr.Plot(label="Job Statistics")
                    
                    with gr.Row():
                        disk_gauge_plot = gr.Plot(label="Disk Usage")

                    gr.Markdown("### 📈 Runner & Queue")

                    with gr.Row():
                        with gr.Column():
                            with gr.Accordion("🖥️ Runner Info", open=True):
                                runner_fig_plot = gr.Plot(show_label=False)
                                runner_info_output = gr.Markdown(
                                    value="",
                                    elem_id="runner-summary",
                                )
                        with gr.Column():
                            with gr.Accordion("📊 Queue Status", open=True):
                                queue_fig_plot = gr.Plot(show_label=False)
                                queue_info_output = gr.Markdown(
                                    value="",
                                    elem_id="queue-summary",
                                )

        # ===== TAB 2: FAILED JOBS =====
        with gr.Tab("❌ Failed Jobs"):
            gr.Markdown("### Failed Jobs History")
            
            with gr.Row():
                failed_refresh_btn = gr.Button("🔄 Refresh", variant="primary", size="sm")
            
            with gr.Row():
                failed_disk_gauge = gr.Plot(label="Disk Usage")
            
            failed_summary = gr.Markdown(value="")
            
            failed_jobs_table = gr.Dataframe(
                headers=[
                    "Job ID",
                    "Saved At",
                    "Status",
                    "Phase",
                    "Miner",
                    "Error"
                ],
                value=[],
                interactive=False,
                wrap=True,
            )
            
            gr.Markdown("### 📜 Job Logs")
            gr.Markdown("*Click a row above to view logs*")
            
            failed_logs_output = gr.Textbox(
                label="",
                lines=30,
                max_lines=50,
                show_label=False,
                show_copy_button=True,
                interactive=False,
            )

        # ===== TAB 3: SUCCESS JOBS =====
        with gr.Tab("✅ Success Jobs"):
            gr.Markdown("### Successful Jobs History")
            
            with gr.Row():
                success_refresh_btn = gr.Button("🔄 Refresh", variant="primary", size="sm")
            
            with gr.Row():
                success_disk_gauge = gr.Plot(label="Disk Usage")
            
            success_summary = gr.Markdown(value="")
            
            success_jobs_table = gr.Dataframe(
                headers=[
                    "Job ID",
                    "Saved At",
                    "Miner",
                    "Weight",
                    "Exec Time",
                    "Exact Match",
                    "Partial",
                    "Similarity"
                ],
                value=[],
                interactive=False,
                wrap=True,
            )
            
            gr.Markdown("### 📊 Job Details")
            gr.Markdown("*Click a row above to view metrics*")
            
            success_details_output = gr.Textbox(
                label="",
                lines=30,
                max_lines=50,
                show_label=False,
                show_copy_button=True,
                interactive=False,
            )

    # ===== MAIN DASHBOARD EVENTS =====
    refresh_btn.click(
        fn=lambda pf, jf: refresh_all(pf, jf),
        inputs=[phase_filter, job_id_filter],
        outputs=[
            jobs_table,
            log_output,
            gpu_gauge_plot,
            job_stats_plot,
            runner_fig_plot,
            runner_info_output,
            queue_fig_plot,
            queue_info_output,
            disk_gauge_plot,
        ],
    )

    phase_filter.change(
        fn=lambda pf, jf: refresh_all(pf, jf),
        inputs=[phase_filter, job_id_filter],
        outputs=[
            jobs_table,
            log_output,
            gpu_gauge_plot,
            job_stats_plot,
            runner_fig_plot,
            runner_info_output,
            queue_fig_plot,
            queue_info_output,
            disk_gauge_plot,
        ],
    )

    job_id_filter.change(
        fn=lambda pf, jf: refresh_all(pf, jf),
        inputs=[phase_filter, job_id_filter],
        outputs=[
            jobs_table,
            log_output,
            gpu_gauge_plot,
            job_stats_plot,
            runner_fig_plot,
            runner_info_output,
            queue_fig_plot,
            queue_info_output,
            disk_gauge_plot,
        ],
    )

    clear_filter_btn.click(
        fn=clear_job_filter,
        inputs=[],
        outputs=[job_id_filter, auto_refresh, refresh_interval],
    )

    jobs_table.select(
        fn=select_job_from_table,
        inputs=[jobs_table],
        outputs=[job_id_filter, auto_refresh, refresh_interval],
    )

    # Auto-refresh timer
    timer = gr.Timer(value=5, active=False)

    auto_refresh.change(
        fn=update_timer,
        inputs=[auto_refresh, refresh_interval],
        outputs=timer,
    )

    refresh_interval.change(
        fn=update_timer,
        inputs=[auto_refresh, refresh_interval],
        outputs=timer,
    )

    timer.tick(
        fn=lambda pf, jf: refresh_all(pf, jf),
        inputs=[phase_filter, job_id_filter],
        outputs=[
            jobs_table,
            log_output,
            gpu_gauge_plot,
            job_stats_plot,
            runner_fig_plot,
            runner_info_output,
            queue_fig_plot,
            queue_info_output,
            disk_gauge_plot,
        ],
    )

    # ===== FAILED JOBS EVENTS =====
    failed_refresh_btn.click(
        fn=refresh_failed_jobs,
        outputs=[failed_jobs_table, failed_summary, failed_disk_gauge]
    )
    
    failed_jobs_table.select(
        fn=get_failed_job_logs,
        inputs=[failed_jobs_table],
        outputs=[failed_logs_output]
    )

    # ===== SUCCESS JOBS EVENTS =====
    success_refresh_btn.click(
        fn=refresh_success_jobs,
        outputs=[success_jobs_table, success_summary, success_disk_gauge]
    )
    
    success_jobs_table.select(
        fn=get_success_job_details,
        inputs=[success_jobs_table],
        outputs=[success_details_output]
    )

    # Initial load
    app.load(
        fn=lambda pf, jf: refresh_all(pf, jf),
        inputs=[phase_filter, job_id_filter],
        outputs=[
            jobs_table,
            log_output,
            gpu_gauge_plot,
            job_stats_plot,
            runner_fig_plot,
            runner_info_output,
            queue_fig_plot,
            queue_info_output,
            disk_gauge_plot,
        ],
    )

    # Custom CSS
    app.css = """
    body { background: radial-gradient(circle at top, #0f172a 0, #020617 55%, #000 100%); }

    #srd-log-output textarea {
        font-family: 'SF Mono', 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace !important;
        background-color: #0b1120 !important;
        color: #e2e8f0 !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        border-radius: 10px !important;
        border: 1px solid #1e293b !important;
    }

    .dataframe {
        font-size: 13px !important;
        font-family: 'SF Mono', 'Monaco', 'Menlo', monospace !important;
    }

    .dataframe table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
    }

    .dataframe th {
        background: linear-gradient(180deg, #4299e1 0%, #3182ce 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 10px 8px !important;
        text-align: left !important;
        border: none !important;
    }

    .dataframe td {
        padding: 8px 8px !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }

    .dataframe tbody tr:hover {
        background-color: rgba(56, 189, 248, 0.12) !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease-out !important;
    }

    .gr-button-primary {
        background: linear-gradient(90deg, #4299e1 0%, #667eea 100%) !important;
        border: none !important;
        color: white !important;
    }

    .gr-button-primary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.45) !important;
    }

    .gr-accordion {
        border-radius: 10px !important;
        border: 1px solid #1f2933 !important;
        background-color: rgba(15, 23, 42, 0.7) !important;
    }

    .plotly {
        border-radius: 10px !important;
        border: 1px solid #1f2933 !important;
        background-color: #020617 !important;
    }

    .gr-box {
        padding: 0.8rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        background-color: rgba(15, 23, 42, 0.7) !important;
    }

    #runner-summary, #queue-summary {
        font-size: 13px !important;
        color: #cbd5f5 !important;
    }
    """


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=9090,
        share=False,
        inbrowser=True,
    )