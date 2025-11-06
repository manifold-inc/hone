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
import asyncio


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
        self.client = httpx.Client(timeout=10.0)
    
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
    st.session_state.refresh_interval = 5  # seconds

if 'job_history' not in st.session_state:
    st.session_state.job_history = []


# ============================================================================
# Helper Functions
# ============================================================================

def parse_metrics(metrics_text: str) -> Dict[str, float]:
    """Parse Prometheus metrics text into a dictionary."""
    parsed = {}
    for line in metrics_text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        try:
            parts = line.split()
            if len(parts) >= 2:
                metric_name = parts[0].split('{')[0]
                value = float(parts[-1])
                parsed[metric_name] = value
        except:
            continue
    return parsed


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
        return go.Figure()
    
    gpu_ids = [gpu['gpu_id'] for gpu in gpu_data]
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


def create_queue_visualization(queue_stats: Dict) -> go.Figure:
    """Create queue depth visualization by weight class."""
    if not queue_stats:
        return go.Figure()
    
    weight_classes = list(queue_stats.keys())
    counts = list(queue_stats.values())
    
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
    
    # API Status
    st.subheader("API Status")
    try:
        response = st.session_state.api_client.client.get(
            f"{API_BASE_URL}/health",
            headers=HEADERS
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
runner_status = st.session_state.api_client.get_runner_status()

if not runner_status:
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
        runner_status.get('total_gpus', 0),
        delta=None
    )

with col2:
    available = runner_status.get('available_gpus', 0)
    st.metric(
        "Available GPUs",
        available,
        delta=None,
        delta_color="normal"
    )

with col3:
    allocated = runner_status.get('allocated_gpus', 0)
    st.metric(
        "Allocated GPUs",
        allocated,
        delta=None
    )

with col4:
    active = runner_status.get('active_jobs', 0)
    st.metric(
        "Active Jobs",
        active,
        delta=None
    )

with col5:
    queue_depth = runner_status.get('queue_depth', 0)
    st.metric(
        "Queued Jobs",
        queue_depth,
        delta=None
    )

st.markdown("---")

# ============================================================================
# GPU Utilization Section
# ============================================================================

st.subheader("GPU Utilization")

# Mock GPU data (in production, this would come from the API)
# You'll need to extend your API to return GPU status
gpu_data = []
for i in range(runner_status.get('total_gpus', 8)):
    gpu_data.append({
        'gpu_id': f"GPU {i}",
        'utilization_percent': 0 if i >= allocated else 85,
        'status': 'allocated' if i < allocated else 'free',
        'memory_used_mb': 0 if i >= allocated else 65536,
        'memory_total_mb': 81920,
        'temperature_celsius': 65 if i < allocated else 45
    })

col1, col2 = st.columns([2, 1])

with col1:
    fig = create_gpu_utilization_chart(gpu_data)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### GPU Status")
    for gpu in gpu_data:
        status_icon = "🟢" if gpu['status'] == 'free' else "🔴"
        st.markdown(f"{status_icon} **{gpu['gpu_id']}**: {gpu['status'].upper()}")

st.markdown("---")

# ============================================================================
# Queue Visualization Section
# ============================================================================

st.subheader("Job Queue")

# Mock queue data (extend API to provide this)
queue_by_weight = {
    "1xH200": 3,
    "2xH200": 2,
    "4xH200": 1,
    "8xH200": 0
}

col1, col2 = st.columns([2, 1])

with col1:
    fig = create_queue_visualization(queue_by_weight)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Queue Summary")
    total_queued = sum(queue_by_weight.values())
    st.metric("Total Queued", total_queued)
    
    if active > 0 and total_queued > 0:
        avg_job_duration = 1800  # 30 minutes
        eta_seconds = (total_queued / max(active, 1)) * avg_job_duration
        st.metric("Est. Wait Time", format_duration(eta_seconds))

st.markdown("---")

# ============================================================================
# Active Jobs Section
# ============================================================================

st.subheader("Active Jobs")

# Mock active jobs (extend API to provide detailed job list)
active_jobs = [
    {
        "job_id": "job_001",
        "status": "inference",
        "weight_class": "2xH200",
        "miner_hotkey": "5Abc...xyz",
        "progress": 75,
        "started_at": "2025-01-01T10:30:00Z",
        "gpus": [0, 1]
    },
    {
        "job_id": "job_002",
        "status": "prep",
        "weight_class": "1xH200",
        "miner_hotkey": "5Def...uvw",
        "progress": 45,
        "started_at": "2025-01-01T10:45:00Z",
        "gpus": [2]
    }
]

if active_jobs:
    for job in active_jobs:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 1])
            
            with col1:
                st.markdown(f"**{job['job_id']}**")
                st.caption(f"Miner: {job['miner_hotkey']}")
            
            with col2:
                status_color = get_status_color(job['status'])
                st.markdown(
                    f"<span style='color: {status_color}'>●</span> {job['status'].upper()}",
                    unsafe_allow_html=True
                )
            
            with col3:
                st.text(job['weight_class'])
            
            with col4:
                st.progress(job['progress'] / 100)
                st.caption(f"{job['progress']}% complete")
            
            with col5:
                if st.button("📋 Logs", key=f"logs_{job['job_id']}"):
                    st.session_state.selected_job = job['job_id']
                    st.rerun()
            
            st.markdown("---")
else:
    st.info("No active jobs")

st.markdown("---")

# ============================================================================
# Statistics Section
# ============================================================================

st.subheader("Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Submitted",
        runner_status.get('total_submitted', 0)
    )

with col2:
    st.metric(
        "Total Completed",
        runner_status.get('total_completed', 0)
    )

with col3:
    st.metric(
        "Total Failed",
        runner_status.get('total_failed', 0)
    )

# Success rate calculation
total_finished = runner_status.get('total_completed', 0) + runner_status.get('total_failed', 0)
if total_finished > 0:
    success_rate = (runner_status.get('total_completed', 0) / total_finished) * 100
    st.metric(
        "Success Rate",
        f"{success_rate:.1f}%"
    )

# ============================================================================
# Job Details Modal
# ============================================================================

if st.session_state.selected_job:
    with st.expander(f"📋 Job Details: {st.session_state.selected_job}", expanded=True):
        job_details = st.session_state.api_client.get_job_status(
            st.session_state.selected_job
        )
        
        if job_details:
            col1, col2 = st.columns(2)
            
            with col1:
                st.json(job_details)
            
            with col2:
                st.subheader("Logs")
                
                # Mock logs (implement log fetching in your API)
                logs = f"""
[2025-01-01 10:30:00] Job started
[2025-01-01 10:30:05] Cloning repository...
[2025-01-01 10:30:30] Building Docker image...
[2025-01-01 10:32:00] Starting prep phase...
[2025-01-01 10:35:00] Downloading model weights...
[2025-01-01 10:40:00] Prep phase complete
[2025-01-01 10:40:05] Starting inference phase...
[2025-01-01 10:42:00] Processing data...
"""
                st.code(logs, language="log")
                
                if st.button("🔙 Back to Dashboard"):
                    st.session_state.selected_job = None
                    st.rerun()
                
                if st.button("❌ Cancel Job", type="secondary"):
                    if st.session_state.api_client.cancel_job(st.session_state.selected_job):
                        st.success("Job cancelled successfully")
                        st.session_state.selected_job = None
                        time.sleep(1)
                        st.rerun()

# ============================================================================
# Auto-refresh
# ============================================================================

if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()