"""
Hone Subnet Sandbox Runner - Dashboard

Real-time monitoring dashboard for GPU sandbox runner
"""

import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import json
from typing import Dict, List, Optional

# Configuration
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "dev-key-12345"  # TODO: Load from env or config

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Page config
st.set_page_config(
    page_title="Sandbox Runner Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-job {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .failed-job {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .running-job {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .job-detail {
        font-family: monospace;
        font-size: 0.9em;
    }
    .error-message {
        color: #dc3545;
        font-family: monospace;
        background-color: #fff5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: bold;
        color: #0066cc;
    }
    .log-container {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 0.85em;
        max-height: 500px;
        overflow-y: auto;
    }
    .log-line {
        margin: 2px 0;
        line-height: 1.4;
    }
    .log-error {
        color: #f48771;
    }
    .log-warning {
        color: #dcdcaa;
    }
    .log-info {
        color: #4fc1ff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API Functions
# ============================================================================

def fetch_dashboard_summary() -> Optional[Dict]:
    """Fetch dashboard summary from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/dashboard/summary",
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch dashboard summary: {e}")
        return None


def fetch_active_jobs() -> List[Dict]:
    """Fetch active jobs from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/dashboard/jobs/active",
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch active jobs: {e}")
        return []


def fetch_completed_jobs(limit: int = 50, offset: int = 0) -> Dict:
    """Fetch completed jobs from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/dashboard/jobs/completed",
            params={"limit": limit, "offset": offset},
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch completed jobs: {e}")
        return {"jobs": [], "total": 0}


def fetch_job_details(job_id: str) -> Optional[Dict]:
    """Fetch detailed job information"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/jobs/{job_id}",
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch job details: {e}")
        return None


def fetch_job_metrics(job_id: str) -> Optional[Dict]:
    """Fetch job metrics"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/jobs/{job_id}/metrics",
            headers=HEADERS,
            timeout=10
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def fetch_job_logs(job_id: str, lines: int = 1000) -> Optional[Dict]:
    """Fetch job logs - tries both log endpoints"""
    # Try the log streaming endpoint first
    try:
        response = requests.get(
            f"{API_BASE_URL}/logs/{job_id}/all",
            headers=HEADERS,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback to dashboard logs endpoint
    try:
        response = requests.get(
            f"{API_BASE_URL}/dashboard/jobs/{job_id}/logs",
            params={"lines": lines},
            headers=HEADERS,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    return None


def fetch_gpu_status() -> List[Dict]:
    """Fetch GPU status from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/dashboard/gpus",
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch GPU status: {e}")
        return []


def fetch_queue_breakdown() -> List[Dict]:
    """Fetch queue breakdown from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/dashboard/queue",
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch queue breakdown: {e}")
        return []


# ============================================================================
# UI Components
# ============================================================================

def render_overview_tab():
    """Render the Overview tab"""
    st.header("📊 Overview")
    
    summary = fetch_dashboard_summary()
    if not summary:
        st.warning("Unable to fetch dashboard data")
        return
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Jobs",
            summary.get("active_jobs", 0),
            help="Currently executing jobs"
        )
    
    with col2:
        st.metric(
            "Queued Jobs",
            summary.get("queued_jobs", 0),
            help="Jobs waiting in queue"
        )
    
    with col3:
        success_rate = summary.get("success_rate", 0)
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            help="Percentage of jobs completed successfully"
        )
    
    with col4:
        st.metric(
            "Free GPUs",
            f"{summary.get('free_gpus', 0)}/{summary.get('total_gpus', 0)}",
            help="Available GPUs / Total GPUs"
        )
    
    st.divider()
    
    # GPU Status
    st.subheader("🎮 GPU Status")
    
    gpu_status = fetch_gpu_status()
    if gpu_status:
        gpu_cols = st.columns(min(len(gpu_status), 4))
        for idx, gpu in enumerate(gpu_status):
            col_idx = idx % 4
            with gpu_cols[col_idx]:
                status_color = {
                    "free": "🟢",
                    "allocated": "🔴",
                    "error": "⚠️",
                    "offline": "⚫"
                }.get(gpu["status"], "❓")
                
                st.markdown(f"""
                **GPU {gpu['gpu_id']}** {status_color}
                - Status: {gpu['status']}
                - Util: {gpu['utilization_percent']:.1f}%
                - Temp: {gpu['temperature_celsius']:.1f}°C
                - Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB
                """)
                
                if gpu.get("allocated_to_job"):
                    st.caption(f"Job: {gpu['allocated_to_job'][:12]}...")
    
    st.divider()
    
    # Queue Breakdown
    st.subheader("📋 Queue Breakdown")
    
    queue_data = fetch_queue_breakdown()
    if queue_data:
        queue_df = pd.DataFrame([
            {
                "Weight Class": item["weight_class"],
                "Jobs in Queue": item["count"]
            }
            for item in queue_data
        ])
        
        if not queue_df.empty:
            st.dataframe(queue_df, use_container_width=True, hide_index=True)
        else:
            st.info("No jobs in queue")
    
    # Recent Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Statistics")
        st.markdown(f"""
        - **Total Submitted**: {summary.get('total_submitted', 0)}
        - **Total Completed**: {summary.get('total_completed', 0)}
        - **Total Failed**: {summary.get('total_failed', 0)}
        - **Avg GPU Utilization**: {summary.get('avg_gpu_utilization', 0):.1f}%
        - **Avg GPU Temperature**: {summary.get('avg_gpu_temperature', 0):.1f}°C
        """)
    
    with col2:
        st.subheader("⚙️ System Info")
        st.markdown(f"""
        - **Runner ID**: {summary.get('runner_id', 'unknown')}
        - **Execution Mode**: {summary.get('execution_mode', 'unknown')}
        - **Estimated Queue Time**: {summary.get('estimated_queue_time_seconds', 0) / 60:.1f} min
        """)


def render_active_jobs_tab():
    """Render the Active Jobs tab"""
    st.header("🔄 Active Jobs")
    
    active_jobs = fetch_active_jobs()
    
    if not active_jobs:
        st.info("No active jobs currently running")
        return
    
    for job in active_jobs:
        status_class = {
            "cloning": "running-job",
            "building": "running-job",
            "prep": "running-job",
            "inference": "running-job"
        }.get(job.get("status", "").lower(), "running-job")
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="{status_class}">
                <strong>Job ID:</strong> <code>{job['job_id']}</code><br>
                <strong>Status:</strong> {job['status']} - {job.get('current_phase', 'N/A')}<br>
                <strong>Miner:</strong> <code>{job['miner_hotkey'][:16]}...</code><br>
                <strong>Weight Class:</strong> {job['weight_class']} | 
                <strong>Priority:</strong> {job['priority']}<br>
                <strong>Progress:</strong> {job.get('progress_percentage', 0):.1f}%<br>
                <strong>GPUs:</strong> {', '.join(map(str, job.get('assigned_gpus', [])))}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.progress(job.get('progress_percentage', 0) / 100.0)
                
                if st.button(f"View Details", key=f"active_{job['job_id']}"):
                    st.session_state['selected_job'] = job['job_id']
                    st.session_state['active_tab'] = "Job Details"
                    st.rerun()


def render_completed_jobs_tab():
    """Render the Completed Jobs tab"""
    st.header("✅ Completed Jobs")
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        limit = st.selectbox("Jobs per page", [10, 25, 50, 100], index=1, key="completed_limit")
    with col2:
        page = st.number_input("Page", min_value=1, value=1, key="completed_page")
    
    offset = (page - 1) * limit
    
    # Fetch completed jobs
    result = fetch_completed_jobs(limit=limit, offset=offset)
    jobs = result.get("jobs", [])
    total = result.get("total", 0)
    
    if not jobs:
        st.info("No completed jobs found")
        return
    
    st.caption(f"Showing {len(jobs)} of {total} jobs")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            ["completed", "failed", "timeout"],
            default=["completed"],
            key="status_filter"
        )
    with col2:
        weight_filter = st.multiselect(
            "Filter by Weight Class",
            ["1xH200", "2xH200", "4xH200", "8xH200"],
            key="weight_filter"
        )
    
    # Apply filters
    filtered_jobs = jobs
    if status_filter:
        filtered_jobs = [j for j in filtered_jobs if j["status"] in status_filter]
    if weight_filter:
        filtered_jobs = [j for j in filtered_jobs if j.get("weight_class") in weight_filter]
    
    # Display jobs
    for job in filtered_jobs:
        status_class = {
            "completed": "success-job",
            "failed": "failed-job",
            "timeout": "failed-job"
        }.get(job["status"], "success-job")
        
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="{status_class}">
                <strong>Job ID:</strong> <code>{job['job_id']}</code><br>
                <strong>Status:</strong> {job['status'].upper()}<br>
                <strong>Miner:</strong> <code>{job['miner_hotkey'][:16]}...</code><br>
                <strong>Weight Class:</strong> {job.get('weight_class', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if job.get('has_metrics') and job.get('metrics'):
                    metrics = job['metrics']
                    st.markdown(f"""
                    **Metrics:**
                    - Exact Matches: {metrics.get('num_exact_matches', 0)}/{metrics.get('total_problems', 0)}
                    - Exact Match Rate: {metrics.get('exact_match_rate', 0)*100:.1f}%
                    - Avg Correctness: {metrics.get('avg_partial_correctness', 0)*100:.1f}%
                    - Success Rate: {metrics.get('success_rate', 0)*100:.1f}%
                    """)
                else:
                    if job.get('error_message'):
                        st.markdown(f"""
                        <div class="error-message">
                        <strong>Error:</strong> {job['error_message'][:100]}...
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.caption("No metrics available")
                
                if job.get('execution_time'):
                    st.caption(f"⏱️ Execution time: {job['execution_time']/60:.1f} min")
            
            with col3:
                if st.button("View Details", key=f"completed_{job['job_id']}"):
                    st.session_state['selected_job'] = job['job_id']
                    st.session_state['active_tab'] = "Job Details"
                    st.rerun()


def render_failed_jobs_tab():
    """Render the Failed Jobs tab - CRITICAL FOR DEBUGGING"""
    st.header("❌ Failed Jobs")
    
    st.markdown("""
    This tab shows all failed jobs with detailed error information for debugging.
    Use this to identify and fix issues with miner submissions.
    """)
    
    # Pagination controls
    col1, col2 = st.columns([1, 3])
    with col1:
        limit = st.selectbox("Jobs per page", [10, 25, 50, 100], index=1, key="failed_limit")
    with col2:
        page = st.number_input("Page", min_value=1, value=1, key="failed_page")
    
    offset = (page - 1) * limit
    
    # Fetch completed jobs and filter for failures
    result = fetch_completed_jobs(limit=limit*2, offset=offset)  # Fetch more to ensure we get failures
    all_jobs = result.get("jobs", [])
    
    # Filter for failed/timeout jobs only
    failed_jobs = [
        job for job in all_jobs 
        if job["status"] in ["failed", "timeout"]
    ]
    
    if not failed_jobs:
        st.success("🎉 No failed jobs! Everything is running smoothly.")
        return
    
    st.warning(f"Found {len(failed_jobs)} failed jobs")
    
    # Group by error type for quick analysis
    with st.expander("📊 Failure Analysis", expanded=True):
        error_types = {}
        for job in failed_jobs:
            error = job.get('error_message', 'Unknown error')
            # Extract error type (first part before colon or full message)
            error_type = error.split(':')[0] if ':' in error else error[:50]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if error_types:
            error_df = pd.DataFrame([
                {"Error Type": k, "Count": v}
                for k, v in sorted(error_types.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(error_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Display each failed job with detailed information
    for job in failed_jobs:
        with st.expander(
            f"🔴 {job['job_id']} - {job['status'].upper()} - {job.get('miner_hotkey', 'N/A')[:16]}...",
            expanded=False
        ):
            # Job metadata
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                **Job Information:**
                - **Job ID**: `{job['job_id']}`
                - **Status**: {job['status'].upper()}
                - **Weight Class**: {job.get('weight_class', 'N/A')}
                """)
            
            with col2:
                st.markdown(f"""
                **Timing:**
                - **Submitted**: {job.get('submitted_at', 'N/A')[:19] if job.get('submitted_at') else 'N/A'}
                - **Started**: {job.get('started_at', 'N/A')[:19] if job.get('started_at') else 'N/A'}
                - **Completed**: {job.get('completed_at', 'N/A')[:19] if job.get('completed_at') else 'N/A'}
                """)
                
                if job.get('execution_time'):
                    st.caption(f"⏱️ Runtime: {job['execution_time']/60:.1f} minutes")
            
            with col3:
                st.markdown(f"""
                **Parties:**
                - **Miner**: `{job.get('miner_hotkey', 'N/A')[:16]}...`
                - **Validator**: `{job.get('validator_hotkey', 'N/A')[:16] if job.get('validator_hotkey') else 'N/A'}...`
                """)
            
            st.divider()
            
            # Error message - THE MOST IMPORTANT PART
            if job.get('error_message'):
                st.markdown("### 🚨 Error Message")
                st.markdown(f"""
                <div class="error-message">
                <pre>{job['error_message']}</pre>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No error message recorded")
            
            st.divider()
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"📋 View Full Details", key=f"failed_details_{job['job_id']}"):
                    st.session_state['selected_job'] = job['job_id']
                    st.session_state['active_tab'] = "Job Details"
                    st.rerun()
            
            with col2:
                if st.button(f"📜 View Logs", key=f"failed_logs_{job['job_id']}"):
                    with st.spinner("Fetching logs..."):
                        logs_data = fetch_job_logs(job['job_id'])
                        if logs_data and logs_data.get('entries'):
                            st.markdown("### Execution Logs")
                            render_logs(logs_data['entries'])
                        else:
                            st.info("No logs available for this job")
            
            with col3:
                # Copy job ID to clipboard
                st.code(job['job_id'], language=None)


def render_job_details_tab():
    """Render detailed job information"""
    st.header("🔍 Job Details")
    
    if 'selected_job' not in st.session_state:
        st.info("No job selected. Select a job from Active Jobs or Completed Jobs tabs.")
        return
    
    job_id = st.session_state['selected_job']
    
    # Back button
    if st.button("← Back"):
        del st.session_state['selected_job']
        st.rerun()
    
    st.subheader(f"Job: {job_id}")
    
    # Fetch job details
    with st.spinner("Loading job details..."):
        job_details = fetch_job_details(job_id)
        job_metrics = fetch_job_metrics(job_id)
        logs_data = fetch_job_logs(job_id)
    
    if not job_details:
        st.error(f"Failed to load job details for {job_id}")
        return
    
    # Job Status
    st.markdown("### 📊 Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", job_details.get("status", "unknown").upper())
    with col2:
        st.metric("Progress", f"{job_details.get('progress_percentage', 0):.1f}%")
    with col3:
        st.metric("Phase", job_details.get("current_phase", "N/A"))
    with col4:
        st.metric("Priority", job_details.get("priority", 0))
    
    # Job Information
    st.markdown("### ℹ️ Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Job Details:**
        - **Job ID**: `{job_details['job_id']}`
        - **Weight Class**: {job_details.get('weight_class', 'N/A')}
        - **Assigned GPUs**: {', '.join(map(str, job_details.get('assigned_gpus', []))) or 'N/A'}
        """)
    
    with col2:
        st.markdown(f"""
        **Parties:**
        - **Miner Hotkey**: `{job_details.get('miner_hotkey', 'N/A')}`
        - **Validator Hotkey**: `{job_details.get('validator_hotkey', 'N/A')}`
        """)
    
    # Timing
    st.markdown("### ⏱️ Timing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        submitted = job_details.get('submitted_at')
        st.markdown(f"**Submitted**: {submitted[:19] if submitted else 'N/A'}")
    
    with col2:
        started = job_details.get('started_at')
        st.markdown(f"**Started**: {started[:19] if started else 'N/A'}")
    
    with col3:
        completed = job_details.get('completed_at')
        st.markdown(f"**Completed**: {completed[:19] if completed else 'N/A'}")
    
    # Error Message (if any)
    if job_details.get('error_message'):
        st.markdown("### 🚨 Error")
        st.markdown(f"""
        <div class="error-message">
        <pre>{job_details['error_message']}</pre>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics (if available)
    if job_metrics and job_metrics.get('metrics'):
        st.markdown("### 📈 Metrics")
        
        metrics = job_metrics['metrics']
        
        if isinstance(metrics, dict) and 'aggregate' in metrics:
            agg = metrics['aggregate']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Problems", agg.get('total_problems', 0))
            with col2:
                st.metric("Solved", agg.get('num_solved', 0))
            with col3:
                st.metric("Exact Matches", agg.get('num_exact_matches', 0))
            with col4:
                st.metric(
                    "Exact Match Rate",
                    f"{agg.get('exact_match_rate', 0)*100:.1f}%"
                )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Avg Partial Correctness",
                    f"{agg.get('avg_partial_correctness', 0)*100:.1f}%"
                )
            with col2:
                st.metric(
                    "Avg Grid Similarity",
                    f"{agg.get('avg_grid_similarity', 0)*100:.1f}%"
                )
            with col3:
                st.metric(
                    "Success Rate",
                    f"{agg.get('success_rate', 0)*100:.1f}%"
                )
            
            # Per-problem metrics
            if 'per_problem' in metrics and metrics['per_problem']:
                with st.expander("📋 Per-Problem Metrics", expanded=False):
                    per_problem = metrics['per_problem']
                    
                    df = pd.DataFrame([
                        {
                            "Problem": p.get('problem_index', i),
                            "Exact Match": "✅" if p.get('exact_match') else "❌",
                            "Shape Match": "✅" if p.get('shape_match') else "❌",
                            "Partial Correctness": f"{p.get('partial_correctness', 0)*100:.1f}%",
                            "Grid Similarity": f"{p.get('grid_similarity', 0)*100:.1f}%",
                        }
                        for i, p in enumerate(per_problem)
                    ])
                    
                    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Logs
    st.markdown("### 📜 Execution Logs")
    
    if logs_data and logs_data.get('entries'):
        render_logs(logs_data['entries'])
    else:
        st.info("No logs available for this job")


def render_logs(log_entries: List[Dict]):
    """Render logs with syntax highlighting"""
    
    # Phase filter
    phases = list(set(entry.get('phase', 'unknown') for entry in log_entries))
    selected_phases = st.multiselect(
        "Filter by phase",
        phases,
        default=phases,
        key=f"log_phase_filter_{id(log_entries)}"
    )
    
    # Level filter
    levels = ["info", "warning", "error", "debug"]
    selected_levels = st.multiselect(
        "Filter by level",
        levels,
        default=["info", "warning", "error"],
        key=f"log_level_filter_{id(log_entries)}"
    )
    
    # Filter logs
    filtered_logs = [
        entry for entry in log_entries
        if entry.get('phase') in selected_phases
        and entry.get('level') in selected_levels
    ]
    
    st.caption(f"Showing {len(filtered_logs)} of {len(log_entries)} log entries")
    
    # Render logs
    log_html = '<div class="log-container">'
    
    for entry in filtered_logs:
        level = entry.get('level', 'info')
        level_class = f"log-{level}"
        timestamp = entry.get('timestamp', '')[:19] if entry.get('timestamp') else ''
        phase = entry.get('phase', 'unknown')
        message = entry.get('message', '')
        
        log_html += f'''
        <div class="log-line {level_class}">
            <span style="color: #858585;">[{timestamp}]</span>
            <span style="color: #569cd6;">[{phase}]</span>
            <span style="color: #ce9178;">[{level.upper()}]</span>
            {message}
        </div>
        '''
    
    log_html += '</div>'
    
    st.markdown(log_html, unsafe_allow_html=True)


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main dashboard application"""
    
    # Sidebar
    with st.sidebar:
        st.title("🚀 Sandbox Runner")
        st.caption("GPU Job Execution Dashboard")
        
        st.divider()
        
        # Refresh controls
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=5,
            max_value=60,
            value=10,
            disabled=not auto_refresh
        )
        
        if st.button("🔄 Refresh Now") or auto_refresh:
            st.rerun()
        
        st.divider()
        
        # API Configuration
        st.subheader("⚙️ Configuration")
        st.text_input("API Base URL", value=API_BASE_URL, disabled=True)
        st.text_input("API Key", value="***", type="password", disabled=True)
        
        st.divider()
        
        # Navigation hint
        st.caption("💡 Tip: Use Failed Jobs tab to debug submission issues")
    
    # Main content
    st.title("🚀 Hone Subnet Sandbox Runner Dashboard")
    
    # Initialize session state
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = "Overview"
    
    # Tab selection
    tabs = st.tabs([
        "📊 Overview",
        "🔄 Active Jobs",
        "✅ Completed Jobs",
        "❌ Failed Jobs",
        "🔍 Job Details"
    ])
    
    with tabs[0]:
        render_overview_tab()
    
    with tabs[1]:
        render_active_jobs_tab()
    
    with tabs[2]:
        render_completed_jobs_tab()
    
    with tabs[3]:
        render_failed_jobs_tab()
    
    with tabs[4]:
        render_job_details_tab()
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()