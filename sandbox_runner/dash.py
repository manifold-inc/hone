"""
Hone Subnet Sandbox Runner - Real-Time Monitoring Dashboard

Features:
- Real-time queue visualization
- Running jobs monitoring with log access
- GPU utilization metrics
- Job history and statistics
"""

import streamlit as st
import httpx
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="Sandbox Runner Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8080")
API_KEY = st.secrets.get("API_KEY", "dev-key-12345")

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


# ============================================================================
# API Client
# ============================================================================

class APIClient:
    """Client for interacting with the Sandbox Runner API."""
    
    def __init__(self, base_url: str, headers: Dict[str, str]):
        self.base_url = base_url.rstrip('/')
        self.headers = headers
        self.client = httpx.Client(timeout=30.0)
    
    def get_runner_status(self) -> Dict:
        """Get overall runner status."""
        try:
            response = self.client.get(
                f"{self.base_url}/v1/status",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch runner status: {e}")
            return {}
    
    def get_dashboard_summary(self) -> Dict:
        """Get complete dashboard summary."""
        try:
            response = self.client.get(
                f"{self.base_url}/v1/dashboard/summary",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch dashboard summary: {e}")
            return {}
    
    def get_gpu_details(self) -> List[Dict]:
        """Get detailed GPU status."""
        try:
            response = self.client.get(
                f"{self.base_url}/v1/dashboard/gpus",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch GPU details: {e}")
            return []
    
    def get_queue_breakdown(self) -> List[Dict]:
        """Get queue breakdown by weight class."""
        try:
            response = self.client.get(
                f"{self.base_url}/v1/dashboard/queue",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch queue breakdown: {e}")
            return []
    
    def get_active_jobs(self) -> List[Dict]:
        """Get active jobs with details."""
        try:
            response = self.client.get(
                f"{self.base_url}/v1/dashboard/jobs/active",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch active jobs: {e}")
            return []
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a specific job."""
        try:
            response = self.client.get(
                f"{self.base_url}/v1/jobs/{job_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch job {job_id}: {e}")
            return None
    
    def get_job_logs(self, job_id: str, lines: int = 100, offset: int = 0) -> Optional[Dict]:
        """Get job logs."""
        try:
            response = self.client.get(
                f"{self.base_url}/v1/dashboard/jobs/{job_id}/logs",
                headers=self.headers,
                params={"lines": lines, "offset": offset}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch logs for job {job_id}: {e}")
            return None
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        try:
            response = self.client.get(
                f"{self.base_url}/metrics",
                headers={"Accept": "text/plain"}
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            st.error(f"Failed to fetch metrics: {e}")
            return ""
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        try:
            response = self.client.delete(
                f"{self.base_url}/v1/jobs/{job_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def submit_test_job(self) -> Optional[Dict]:
        """Submit a test job."""
        try:
            payload = {
                "repo_url": "https://github.com/manifold-inc/hone",
                "repo_branch": "main",
                "repo_path": "miner-sr",
                "weight_class": "1xH200",
                "miner_hotkey": "test_miner_hotkey",
                "priority": 5
            }
            response = self.client.post(
                f"{self.base_url}/v1/jobs/submit",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to submit test job: {e}")
            return None


# ============================================================================
# Session State Initialization
# ============================================================================

if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient(API_BASE_URL, HEADERS)

if 'selected_job' not in st.session_state:
    st.session_state.selected_job = None

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5

if 'log_offset' not in st.session_state:
    st.session_state.log_offset = 0

if 'log_lines' not in st.session_state:
    st.session_state.log_lines = 100


# ============================================================================
# Helper Functions
# ============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_status_color(status: str) -> str:
    """Get color for job status."""
    colors = {
        "pending": "#FFA500",
        "cloning": "#4169E1",
        "building": "#4169E1",
        "prep": "#1E90FF",
        "inference": "#00CED1",
        "completed": "#32CD32",
        "failed": "#DC143C",
        "timeout": "#FF6347",
        "cancelled": "#808080"
    }
    return colors.get(status.lower(), "#808080")


def create_gpu_utilization_chart(gpu_data: List[Dict]) -> go.Figure:
    """Create GPU utilization bar chart."""
    if not gpu_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No GPU data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    gpu_ids = [f"GPU {gpu['gpu_id']}" for gpu in gpu_data]
    utilization = [gpu.get('utilization_percent', 0) for gpu in gpu_data]
    status = [gpu.get('status', 'unknown') for gpu in gpu_data]
    
    colors = ['#32CD32' if s == 'free' else '#DC143C' for s in status]
    
    fig = go.Figure(data=[
        go.Bar(
            x=gpu_ids,
            y=utilization,
            marker_color=colors,
            text=[f"{u:.1f}%" for u in utilization],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="GPU Utilization",
        xaxis_title="GPU ID",
        yaxis_title="Utilization %",
        yaxis=dict(range=[0, 100]),
        height=300,
        showlegend=False
    )
    
    return fig


def create_queue_visualization(queue_data: List[Dict]) -> go.Figure:
    """Create queue depth visualization by weight class."""
    if not queue_data:
        fig = go.Figure()
        fig.add_annotation(
            text="Queue is empty",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    weight_classes = [item['weight_class'] for item in queue_data]
    counts = [item['count'] for item in queue_data]
    
    colors = ['#1E90FF', '#4169E1', '#0000CD', '#00008B']
    
    fig = go.Figure(data=[
        go.Bar(
            x=weight_classes,
            y=counts,
            marker_color=colors[:len(weight_classes)],
            text=counts,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Queue Depth by Weight Class",
        xaxis_title="Weight Class",
        yaxis_title="Number of Jobs",
        height=300,
        showlegend=False
    )
    
    return fig


def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("🚀 Sandbox Runner")
    st.markdown("---")
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.checkbox(
        "Auto Refresh",
        value=st.session_state.auto_refresh
    )
    
    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=30,
            value=st.session_state.refresh_interval
        )
    
    # Manual refresh button
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    # Test job submission
    st.subheader("Testing")
    if st.button("🧪 Submit Test Job", use_container_width=True):
        with st.spinner("Submitting test job..."):
            result = st.session_state.api_client.submit_test_job()
            if result:
                st.success(f"Job submitted: {result.get('job_id', 'Unknown')}")
            else:
                st.error("Failed to submit test job")
    
    st.markdown("---")
    
    # API Status
    st.subheader("API Status")
    try:
        response = st.session_state.api_client.client.get(
            f"{API_BASE_URL}/v1/health",
            headers=HEADERS,
            timeout=5.0
        )
        if response.status_code == 200:
            st.success("✅ Connected")
        else:
            st.error("❌ Disconnected")
    except:
        st.error("❌ Disconnected")
    
    st.text(f"URL: {API_BASE_URL}")


# ============================================================================
# Main Dashboard
# ============================================================================

st.title("📊 Sandbox Runner Dashboard")

# Fetch current status
dashboard_summary = st.session_state.api_client.get_dashboard_summary()

if not dashboard_summary:
    st.error("Failed to connect to Sandbox Runner API. Please check your connection.")
    st.stop()

# ============================================================================
# Overview Metrics Row
# ============================================================================

st.subheader("Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total GPUs",
        dashboard_summary.get('total_gpus', 0)
    )

with col2:
    free_gpus = dashboard_summary.get('free_gpus', 0)
    st.metric(
        "Free GPUs",
        free_gpus
    )

with col3:
    allocated_gpus = dashboard_summary.get('allocated_gpus', 0)
    st.metric(
        "Allocated GPUs",
        allocated_gpus
    )

with col4:
    active_jobs = dashboard_summary.get('active_jobs', 0)
    st.metric(
        "Active Jobs",
        active_jobs
    )

with col5:
    queued_jobs = dashboard_summary.get('queued_jobs', 0)
    st.metric(
        "Queued Jobs",
        queued_jobs
    )

# Additional metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_util = dashboard_summary.get('avg_gpu_utilization', 0)
    st.metric(
        "Avg GPU Util",
        f"{avg_util:.1f}%"
    )

with col2:
    success_rate = dashboard_summary.get('success_rate', 0)
    st.metric(
        "Success Rate",
        f"{success_rate:.1f}%"
    )

with col3:
    total_completed = dashboard_summary.get('total_completed', 0)
    st.metric(
        "Completed",
        total_completed
    )

with col4:
    total_failed = dashboard_summary.get('total_failed', 0)
    st.metric(
        "Failed",
        total_failed
    )

st.markdown("---")

# ============================================================================
# GPU Utilization Section
# ============================================================================

st.subheader("GPU Utilization")

gpu_data = st.session_state.api_client.get_gpu_details()

col1, col2 = st.columns([2, 1])

with col1:
    fig = create_gpu_utilization_chart(gpu_data)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### GPU Status")
    if gpu_data:
        for gpu in gpu_data:
            status = gpu.get('status', 'unknown')
            status_icon = "🟢" if status == 'free' else "🔴"
            gpu_id = gpu.get('gpu_id', 'unknown')
            util = gpu.get('utilization_percent', 0)
            temp = gpu.get('temperature_celsius', 0)
            
            st.markdown(f"{status_icon} **GPU {gpu_id}**: {status.upper()}")
            st.caption(f"Util: {util:.1f}% | Temp: {temp:.1f}°C")
    else:
        st.info("No GPU data available")

st.markdown("---")

# ============================================================================
# Queue Visualization Section
# ============================================================================

st.subheader("Job Queue")

queue_breakdown = st.session_state.api_client.get_queue_breakdown()

col1, col2 = st.columns([2, 1])

with col1:
    fig = create_queue_visualization(queue_breakdown)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Queue Summary")
    
    total_queued = sum(item['count'] for item in queue_breakdown)
    st.metric("Total Queued", total_queued)
    
    if active_jobs > 0 and total_queued > 0:
        est_wait = dashboard_summary.get('estimated_queue_time_seconds', 0)
        st.metric("Est. Wait Time", format_duration(est_wait))
    
    # Show breakdown details
    if queue_breakdown:
        st.markdown("#### By Weight Class")
        for item in queue_breakdown:
            wc = item['weight_class']
            count = item['count']
            if count > 0:
                st.text(f"{wc}: {count} jobs")

st.markdown("---")

# ============================================================================
# Active Jobs Section
# ============================================================================

st.subheader("Active Jobs")

active_jobs_list = st.session_state.api_client.get_active_jobs()

if active_jobs_list:
    for job in active_jobs_list:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 1])
            
            with col1:
                st.markdown(f"**{job['job_id']}**")
                miner = job.get('miner_hotkey', 'Unknown')
                if len(miner) > 20:
                    miner = miner[:8] + "..." + miner[-8:]
                st.caption(f"Miner: {miner}")
            
            with col2:
                status = job.get('status', 'unknown')
                status_color = get_status_color(status)
                st.markdown(
                    f"<span style='color: {status_color}'>●</span> {status.upper()}",
                    unsafe_allow_html=True
                )
                phase = job.get('current_phase', '')
                if phase:
                    st.caption(f"Phase: {phase}")
            
            with col3:
                st.text(job.get('weight_class', 'Unknown'))
                gpus = job.get('assigned_gpus', [])
                if gpus:
                    st.caption(f"GPUs: {gpus}")
            
            with col4:
                progress = job.get('progress_percentage', 0)
                st.progress(progress / 100)
                st.caption(f"{progress:.1f}% complete")
            
            with col5:
                if st.button("📋 Logs", key=f"logs_{job['job_id']}"):
                    st.session_state.selected_job = job['job_id']
                    st.session_state.log_offset = 0
                    st.rerun()
            
            st.markdown("---")
else:
    st.info("No active jobs")

st.markdown("---")

# ============================================================================
# Statistics Section
# ============================================================================

st.subheader("Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Submitted",
        dashboard_summary.get('total_submitted', 0)
    )

with col2:
    st.metric(
        "Total Completed",
        dashboard_summary.get('total_completed', 0)
    )

with col3:
    st.metric(
        "Total Failed",
        dashboard_summary.get('total_failed', 0)
    )

with col4:
    execution_mode = dashboard_summary.get('execution_mode', 'unknown')
    st.metric(
        "Execution Mode",
        execution_mode
    )

# ============================================================================
# Job Details Modal
# ============================================================================

if st.session_state.selected_job:
    st.markdown("---")
    st.subheader(f"📋 Job Details: {st.session_state.selected_job}")
    
    job_details = st.session_state.api_client.get_job_status(
        st.session_state.selected_job
    )
    
    if job_details:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Job Information")
            
            info_data = {
                "Job ID": job_details.get('job_id', 'Unknown'),
                "Status": job_details.get('status', 'Unknown'),
                "Weight Class": job_details.get('weight_class', 'Unknown'),
                "Priority": job_details.get('priority', 0),
                "Miner Hotkey": job_details.get('miner_hotkey', 'Unknown'),
                "Current Phase": job_details.get('current_phase', 'N/A'),
                "Progress": f"{job_details.get('progress_percentage', 0):.1f}%",
                "Assigned GPUs": str(job_details.get('assigned_gpus', [])),
                "Submitted At": format_timestamp(job_details.get('submitted_at', '')),
                "Started At": format_timestamp(job_details.get('started_at', '')) if job_details.get('started_at') else 'N/A',
            }
            
            for key, value in info_data.items():
                st.text(f"{key}: {value}")
            
            if job_details.get('error_message'):
                st.error(f"Error: {job_details['error_message']}")
            
            # Action buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔙 Back to Dashboard", use_container_width=True):
                    st.session_state.selected_job = None
                    st.rerun()
            
            with col_b:
                if job_details.get('status') not in ['completed', 'failed', 'cancelled']:
                    if st.button("❌ Cancel Job", type="secondary", use_container_width=True):
                        if st.session_state.api_client.cancel_job(st.session_state.selected_job):
                            st.success("Job cancelled successfully")
                            time.sleep(1)
                            st.session_state.selected_job = None
                            st.rerun()
        
        with col2:
            st.markdown("#### Job Logs")
            
            # Log controls
            col_log1, col_log2 = st.columns([3, 1])
            with col_log1:
                st.session_state.log_lines = st.selectbox(
                    "Lines to display",
                    [50, 100, 200, 500, 1000],
                    index=1
                )
            with col_log2:
                if st.button("🔄 Refresh Logs"):
                    st.rerun()
            
            # Fetch logs
            log_data = st.session_state.api_client.get_job_logs(
                st.session_state.selected_job,
                lines=st.session_state.log_lines,
                offset=st.session_state.log_offset
            )
            
            if log_data and log_data.get('logs'):
                logs_list = log_data.get('logs', [])
                
                # Display logs
                log_text = ""
                for log_entry in logs_list:
                    if isinstance(log_entry, dict):
                        timestamp = log_entry.get('timestamp', '')
                        level = log_entry.get('level', 'INFO')
                        message = log_entry.get('message', '')
                        log_text += f"[{timestamp}] {level}: {message}\n"
                    else:
                        log_text += str(log_entry) + "\n"
                
                st.code(log_text, language="log")
                
                # Pagination controls
                total_lines = log_data.get('total_lines', 0)
                has_more = log_data.get('has_more', False)
                
                st.text(f"Showing lines {st.session_state.log_offset + 1} - {st.session_state.log_offset + len(logs_list)} of {total_lines}")
                
                col_prev, col_next = st.columns(2)
                with col_prev:
                    if st.session_state.log_offset > 0:
                        if st.button("⬅️ Previous", use_container_width=True):
                            st.session_state.log_offset = max(0, st.session_state.log_offset - st.session_state.log_lines)
                            st.rerun()
                
                with col_next:
                    if has_more:
                        if st.button("Next ➡️", use_container_width=True):
                            st.session_state.log_offset += st.session_state.log_lines
                            st.rerun()
            else:
                st.info("No logs available for this job yet")

# ============================================================================
# Auto-refresh
# ============================================================================

if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()