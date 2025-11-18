"""
Enhanced Log Streaming Dashboard using Gradio

Features:
- Auto-fetches active jobs from /v1/status endpoint
- Real-time log streaming with /v1/logs/{job_id}/tail
- Interactive gauges and visualizations
- Performance optimized with caching
- Multi-job log viewing
- Phase filtering and auto-refresh
- GPU and queue statistics
"""

import gradio as gr
import httpx
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time
import plotly.graph_objects as go
from functools import lru_cache

# Configuration
API_BASE_URL = "http://localhost:8080"
API_KEY = "dev-key-12345"

# Cache for reducing API calls
_status_cache = {"data": None, "timestamp": 0}
_cache_ttl = 1.0  # 1 second cache


def get_client():
    """Get HTTP client with proper headers"""
    return httpx.Client(
        base_url=API_BASE_URL,
        headers={"X-API-Key": API_KEY},
        timeout=10.0
    )


def get_runner_status(use_cache: bool = True) -> Dict:
    """Get runner status including active job IDs with caching"""
    global _status_cache
    
    if use_cache and _status_cache["data"] and (time.time() - _status_cache["timestamp"] < _cache_ttl):
        return _status_cache["data"]
    
    try:
        client = get_client()
        response = client.get("/v1/status")
        if response.status_code == 200:
            data = response.json()
            _status_cache = {"data": data, "timestamp": time.time()}
            return data
        return {}
    except Exception as e:
        print(f"Error fetching status: {e}")
        return _status_cache.get("data", {})


def create_gpu_gauge(total: int, allocated: int, available: int) -> go.Figure:
    """Create a gauge chart for GPU utilization"""
    utilization = (allocated / max(total, 1)) * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=allocated,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"GPU Allocation<br><sub>{available} available</sub>", 'font': {'size': 16}},
        delta={'reference': total, 'increasing': {'color': "orange"}},
        number={'suffix': f" / {total}", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, total], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#4299e1"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, total * 0.5], 'color': '#e6f7ff'},
                {'range': [total * 0.5, total * 0.8], 'color': '#fffbe6'},
                {'range': [total * 0.8, total], 'color': '#fff1f0'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': total * 0.9
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=220,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_job_stats_chart(active: int, completed: int, failed: int, queued: int) -> go.Figure:
    """Create bar chart for job statistics"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Active', 'Queued', 'Completed', 'Failed'],
            y=[active, queued, completed, failed],
            marker_color=['#4299e1', '#fbbf24', '#10b981', '#ef4444'],
            text=[active, queued, completed, failed],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=220,
        margin=dict(l=20, r=20, t=40, b=40),
        title={'text': "Job Statistics", 'font': {'size': 16}},
        xaxis={'showgrid': False},
        yaxis={'showgrid': True, 'gridcolor': 'rgba(255,255,255,0.1)'}
    )
    
    return fig


def get_queue_info() -> str:
    """Get queue information in formatted text"""
    try:
        status = get_runner_status()
        
        if not status:
            return "Failed to fetch queue status"
        
        queue_depth = status.get("queue_depth", 0)
        
        queue_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                      QUEUE STATUS                            ║
╚══════════════════════════════════════════════════════════════╝

📊 QUEUE DEPTH: {queue_depth}

"""
        
        if queue_depth == 0:
            queue_text += "   ✓ Queue is empty - all systems ready\n"
        else:
            queue_text += f"   ⏳ {queue_depth} job(s) waiting in queue\n"
        
        queue_text += f"\n⏰ Last Updated: {datetime.now().strftime('%H:%M:%S')}\n"
        
        return queue_text
    except Exception as e:
        return f"Error fetching queue info: {str(e)}"


def get_active_jobs_table() -> Tuple[List[List[str]], List[str]]:
    """
    Get active jobs from status endpoint
    Returns: (table_data, active_job_ids)
    """
    try:
        status = get_runner_status()
        active_job_ids = status.get("active_job_ids", [])
        
        if not active_job_ids:
            return [], []
        
        client = get_client()
        table_data = []
        
        for job_id in active_job_ids:
            try:
                response = client.get(f"/v1/jobs/{job_id}")
                if response.status_code == 200:
                    job = response.json()
                    
                    gpus = job.get("assigned_gpus", [])
                    gpu_str = f"GPU {','.join(map(str, gpus))}" if gpus else "N/A"
                    
                    started_at = job.get("started_at")
                    if started_at:
                        try:
                            dt = datetime.fromisoformat(started_at)
                            time_str = dt.strftime("%H:%M:%S")
                        except:
                            time_str = "N/A"
                    else:
                        time_str = "N/A"
                    
                    phase = job.get("current_phase", "unknown")
                    phase_emoji = {
                        "build": "🔨",
                        "prep": "📦",
                        "inference": "🧠",
                        "vllm": "🚀"
                    }
                    phase_display = f"{phase_emoji.get(phase, '📝')} {phase}"
                    
                    table_data.append([
                        job_id,
                        phase_display,
                        f"{job.get('progress_percentage', 0):.1f}%",
                        job.get("weight_class", "unknown"),
                        gpu_str,
                        time_str,
                        job.get("miner_hotkey", "")[:12] + "..."
                    ])
                else:
                    table_data.append([
                        job_id, "❌ error", "0%", "unknown", "N/A", "N/A", "error"
                    ])
            except Exception as e:
                print(f"Error fetching job {job_id}: {e}")
                table_data.append([
                    job_id, "❌ error", "0%", "unknown", "N/A", "N/A", "error"
                ])
        
        return table_data, active_job_ids
    except Exception as e:
        print(f"Error in get_active_jobs_table: {e}")
        return [], []


def get_job_logs_tail(job_id: str, lines: int = 100, phase_filter: str = "all") -> str:
    """Get last N lines of logs for a job using /tail endpoint"""
    if not job_id:
        return ""
    
    try:
        client = get_client()
        params = {"lines": lines}
        
        if phase_filter != "all":
            params["phase"] = phase_filter
        
        response = client.get(f"/v1/logs/{job_id}/tail", params=params)
        
        if response.status_code == 200:
            data = response.json()
            entries = data.get("entries", [])
            
            if not entries:
                return f"[{job_id}] No logs available yet...\n"
            
            log_lines = []
            for entry in entries:
                timestamp = entry.get("timestamp", "")
                phase = entry.get("phase", "")
                level = entry.get("level", "info")
                message = entry.get("message", "")
                
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp_str = dt.strftime("%H:%M:%S")
                except:
                    timestamp_str = timestamp[:8] if len(timestamp) >= 8 else "??:??:??"
                
                phase_emoji = {
                    "build": "🔨",
                    "prep": "📦",
                    "inference": "🧠",
                    "vllm": "🚀"
                }
                
                level_emoji = {
                    "error": "❌",
                    "warning": "⚠️",
                    "info": "ℹ️"
                }
                
                phase_icon = phase_emoji.get(phase, "📝")
                level_icon = level_emoji.get(level, "")
                
                log_lines.append(
                    f"[{timestamp_str}] {phase_icon} {level_icon} {message}"
                )
            
            return "\n".join(log_lines)
        else:
            return f"[{job_id}] Error: Status {response.status_code}\n"
            
    except Exception as e:
        return f"[{job_id}] Error: {str(e)}\n"


def get_all_active_logs(selected_job_ids: List[str], phase_filter: str = "all") -> str:
    """Get logs for all selected jobs"""
    if not selected_job_ids:
        return "No active jobs selected. Click 'Refresh' to load active jobs."
    
    all_logs = []
    all_logs.append("=" * 100)
    all_logs.append(f"LOGS FOR {len(selected_job_ids)} ACTIVE JOB(S)")
    all_logs.append(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_logs.append("=" * 100)
    all_logs.append("")
    
    for job_id in selected_job_ids:
        all_logs.append(f"\n{'─' * 100}")
        all_logs.append(f"📋 JOB: {job_id}")
        all_logs.append(f"{'─' * 100}\n")
        
        logs = get_job_logs_tail(job_id, lines=50, phase_filter=phase_filter)
        all_logs.append(logs)
        all_logs.append("")
    
    return "\n".join(all_logs)


def get_runner_info() -> str:
    """Get runner info in formatted text"""
    try:
        status = get_runner_status()
        
        if not status:
            return "Failed to fetch runner status"
        
        total_submitted = status.get('total_submitted', 0)
        total_completed = status.get('total_completed', 0)
        success_rate = (total_completed / max(total_submitted, 1)) * 100
        
        info_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    RUNNER INFO                               ║
╚══════════════════════════════════════════════════════════════╝

🖥️  RUNNER:
   • ID: {status.get('runner_id', 'unknown')}
   • Status: {status.get('status', 'unknown').upper()}
   • Mode: {status.get('execution_mode', 'unknown')}

📈 PERFORMANCE:
   • Success Rate: {success_rate:.1f}%
   • Total Completed: {total_completed}
   • Total Failed: {status.get('total_failed', 0)}

⏰ Updated: {datetime.now().strftime('%H:%M:%S')}
"""
        
        return info_text
    except Exception as e:
        return f"Error fetching runner info: {str(e)}"


def select_job_from_table(evt: gr.SelectData, table_data):
    """Handle job selection from table"""
    try:
        if hasattr(table_data, 'iloc'):
            job_id = str(table_data.iloc[evt.index[0], 0])
        elif isinstance(table_data, list) and evt.index[0] < len(table_data):
            job_id = table_data[evt.index[0]][0]
        else:
            return "", False, 2
        
        return job_id, True, 2
    except Exception as e:
        print(f"Error selecting job: {e}")
        return "", False, 2


def refresh_all(phase_filter: str, job_id_filter: str = ""):
    """Refresh all dashboard data"""
    status = get_runner_status(use_cache=False)
    
    table_data, active_job_ids = get_active_jobs_table()
    
    if job_id_filter and job_id_filter.strip():
        logs = get_all_active_logs([job_id_filter.strip()], phase_filter)
    else:
        logs = get_all_active_logs(active_job_ids, phase_filter)
    
    gpu_gauge = create_gpu_gauge(
        status.get("total_gpus", 0),
        status.get("allocated_gpus", 0),
        status.get("available_gpus", 0)
    )
    
    job_chart = create_job_stats_chart(
        status.get("active_jobs", 0),
        status.get("total_completed", 0),
        status.get("total_failed", 0),
        status.get("queue_depth", 0)
    )
    
    runner_info = get_runner_info()
    queue_info = get_queue_info()
    
    return table_data, logs, gpu_gauge, job_chart, runner_info, queue_info


def clear_job_filter():
    """Clear job filter and reset refresh rate"""
    return "", False, 5


# Create Gradio interface
with gr.Blocks(title="Sandbox Runner Dashboard", theme=gr.themes.Soft()) as app:
    gr.HTML("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 2.5em; margin: 0; background: linear-gradient(90deg, #4299e1 0%, #667eea 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🚀 Sandbox Runner Dashboard
            </h1>
            <p style="color: #718096; margin-top: 10px; font-size: 1.1em;">Real-time GPU job monitoring & analytics</p>
        </div>
    """)
    
    with gr.Row():
        # Left column - Jobs & Logs
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 📋 Active Jobs")
                with gr.Column(scale=2):
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh", variant="primary", size="sm")
                        auto_refresh = gr.Checkbox(label="Auto", value=False)
                        refresh_interval = gr.Radio(
                            label="",
                            choices=[2, 5],
                            value=5,
                            show_label=False
                        )
            
            jobs_table = gr.Dataframe(
                headers=[
                    "Job ID", 
                    "Phase", 
                    "Progress", 
                    "Weight", 
                    "GPUs",
                    "Started",
                    "Miner"
                ],
                value=[],
                interactive=False,
                wrap=True,
                column_widths=["22%", "14%", "10%", "12%", "10%", "10%", "22%"]
            )
            
            gr.Markdown("### ⚙️ Filters")
            with gr.Row():
                job_id_filter = gr.Textbox(
                    label="Job ID",
                    placeholder="Click job or enter ID...",
                    scale=3
                )
                phase_filter = gr.Dropdown(
                    label="Phase",
                    choices=["all", "build", "prep", "inference", "vllm"],
                    value="all",
                    scale=2
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
                container=False
            )
        
        # Right column - Stats & Visualizations
        with gr.Column(scale=2):
            gr.Markdown("### 📊 System Overview")
            
            with gr.Row():
                gpu_gauge_plot = gr.Plot(label="GPU Allocation")
            
            with gr.Row():
                job_stats_plot = gr.Plot(label="Job Statistics")
            
            gr.Markdown("### 📈 Details")
            
            with gr.Accordion("🖥️ Runner Info", open=True):
                runner_info_output = gr.Textbox(
                    label="",
                    lines=12,
                    max_lines=15,
                    show_label=False,
                    interactive=False
                )
            
            with gr.Accordion("📊 Queue Status", open=True):
                queue_info_output = gr.Textbox(
                    label="",
                    lines=8,
                    max_lines=10,
                    show_label=False,
                    interactive=False
                )
    
    # Event handlers
    refresh_btn.click(
        fn=lambda pf, jf: refresh_all(pf, jf),
        inputs=[phase_filter, job_id_filter],
        outputs=[jobs_table, log_output, gpu_gauge_plot, job_stats_plot, runner_info_output, queue_info_output]
    )
    
    phase_filter.change(
        fn=lambda pf, jf: refresh_all(pf, jf),
        inputs=[phase_filter, job_id_filter],
        outputs=[jobs_table, log_output, gpu_gauge_plot, job_stats_plot, runner_info_output, queue_info_output]
    )
    
    job_id_filter.change(
        fn=lambda pf, jf: refresh_all(pf, jf),
        inputs=[phase_filter, job_id_filter],
        outputs=[jobs_table, log_output, gpu_gauge_plot, job_stats_plot, runner_info_output, queue_info_output]
    )
    
    clear_filter_btn.click(
        fn=clear_job_filter,
        inputs=[],
        outputs=[job_id_filter, auto_refresh, refresh_interval]
    )
    
    jobs_table.select(
        fn=select_job_from_table,
        inputs=[jobs_table],
        outputs=[job_id_filter, auto_refresh, refresh_interval]
    )
    
    # Auto-refresh timer
    timer = gr.Timer(value=5, active=False)
    
    def update_timer(enabled, interval):
        return gr.update(active=enabled, value=interval)
    
    auto_refresh.change(
        fn=update_timer,
        inputs=[auto_refresh, refresh_interval],
        outputs=timer
    )
    
    refresh_interval.change(
        fn=update_timer,
        inputs=[auto_refresh, refresh_interval],
        outputs=timer
    )
    
    timer.tick(
        fn=lambda pf, jf: refresh_all(pf, jf),
        inputs=[phase_filter, job_id_filter],
        outputs=[jobs_table, log_output, gpu_gauge_plot, job_stats_plot, runner_info_output, queue_info_output]
    )
    
    # Initial load
    app.load(
        fn=lambda pf, jf: refresh_all(pf, jf),
        inputs=[phase_filter, job_id_filter],
        outputs=[jobs_table, log_output, gpu_gauge_plot, job_stats_plot, runner_info_output, queue_info_output]
    )
    
    # Custom CSS
    app.css = """
    /* Terminal-style log output */
    textarea {
        font-family: 'SF Mono', 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace !important;
    }
    
    /* Log output specific */
    #component-* textarea {
        background-color: #1a1d23 !important;
        color: #e2e8f0 !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
        border: 1px solid #2d3748 !important;
    }
    
    /* Table styling */
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
        padding: 12px 8px !important;
        text-align: left !important;
        border: none !important;
    }
    
    .dataframe td {
        padding: 10px 8px !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(66, 153, 225, 0.1) !important;
        cursor: pointer !important;
        transition: background-color 0.2s ease !important;
    }
    
    /* Button styling */
    .gr-button-primary {
        background: linear-gradient(90deg, #4299e1 0%, #667eea 100%) !important;
        border: none !important;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.4) !important;
    }
    
    /* Accordion styling */
    .gr-accordion {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Plotly charts */
    .plotly {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Compact spacing */
    .gr-box {
        padding: 0.75rem !important;
    }
    
    /* Info boxes */
    .gr-form {
        border: none !important;
    }
    """


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=9090,
        share=False,
        inbrowser=True
    )