"""
Enhanced Log Streaming Dashboard using Gradio

Features:
- Auto-fetches active jobs from /v1/status endpoint
- Real-time log streaming with /v1/logs/{job_id}/tail
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

# Configuration
API_BASE_URL = "http://localhost:8080"
API_KEY = "dev-key-12345"

# Global state
log_cursors = {}
selected_jobs = set()  # Track which jobs user wants to monitor


def get_client():
    """Get HTTP client with proper headers"""
    return httpx.Client(
        base_url=API_BASE_URL,
        headers={"X-API-Key": API_KEY},
        timeout=30.0
    )


def get_runner_status() -> Dict:
    """Get runner status including active job IDs"""
    try:
        client = get_client()
        response = client.get("/v1/status")
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"Error fetching status: {e}")
        return {}


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
                    
                    # Format GPU allocation
                    gpus = job.get("assigned_gpus", [])
                    gpu_str = f"GPU {','.join(map(str, gpus))}" if gpus else "N/A"
                    
                    # Format timestamp
                    started_at = job.get("started_at")
                    if started_at:
                        try:
                            dt = datetime.fromisoformat(started_at)
                            time_str = dt.strftime("%H:%M:%S")
                        except:
                            time_str = "N/A"
                    else:
                        time_str = "N/A"
                    
                    table_data.append([
                        job_id,
                        job.get("status", "unknown"),
                        job.get("current_phase", "unknown"),
                        f"{job.get('progress_percentage', 0):.1f}%",
                        job.get("weight_class", "unknown"),
                        gpu_str,
                        time_str,
                        job.get("miner_hotkey", "")[:12] + "..."
                    ])
                else:
                    table_data.append([
                        job_id, "error", "unknown", "0%", "unknown", "N/A", "N/A", "error"
                    ])
            except Exception as e:
                print(f"Error fetching job {job_id}: {e}")
                table_data.append([
                    job_id, "error", "error", "0%", "unknown", "N/A", "N/A", "error"
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
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp_str = dt.strftime("%H:%M:%S")
                except:
                    timestamp_str = timestamp[:8] if len(timestamp) >= 8 else "??:??:??"
                
                # Color coding
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
        return "No active jobs selected. Click 'Refresh Jobs' to load active jobs."
    
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


def get_runner_stats() -> str:
    """Get runner statistics in formatted text"""
    try:
        status = get_runner_status()
        
        if not status:
            return "Failed to fetch runner status"
        
        gpu_stats = status.get("gpu_stats", {})
        queue_stats = status.get("queue_stats", {})
        
        stats_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                  SANDBOX RUNNER STATUS                       ║
╚══════════════════════════════════════════════════════════════╝

🖥️  RUNNER INFO:
   • ID: {status.get('runner_id', 'unknown')}
   • Status: {status.get('status', 'unknown')}
   • Execution Mode: {status.get('execution_mode', 'unknown')}

🎮 GPU STATUS:
   • Total GPUs: {gpu_stats.get('total_gpus', 0)}
   • Available: {gpu_stats.get('free_gpus', 0)}
   • Allocated: {gpu_stats.get('allocated_gpus', 0)}
   • Utilization: {gpu_stats.get('allocated_gpus', 0) / max(gpu_stats.get('total_gpus', 1), 1) * 100:.1f}%

📊 JOB STATISTICS:
   • Active Jobs: {status.get('active_jobs', 0)}
   • Queue Depth: {status.get('queue_depth', 0)}
   • Total Submitted: {status.get('total_submitted', 0)}
   • Total Completed: {status.get('total_completed', 0)}
   • Total Failed: {status.get('total_failed', 0)}
   • Success Rate: {status.get('total_completed', 0) / max(status.get('total_submitted', 1), 1) * 100:.1f}%

🔢 QUEUE BREAKDOWN:
"""
        
        queue_by_weight = queue_stats.get("by_weight_class", {})
        for weight_class, count in queue_by_weight.items():
            if count > 0:
                stats_text += f"   • {weight_class}: {count} job(s)\n"
        
        if not queue_by_weight or all(c == 0 for c in queue_by_weight.values()):
            stats_text += "   • Queue is empty\n"
        
        stats_text += f"\n⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return stats_text
    except Exception as e:
        return f"Error fetching stats: {str(e)}"


def select_job_from_table(evt: gr.SelectData, table_data):
    """Handle job selection from table"""
    if evt.index[0] < len(table_data):
        job_id = table_data[evt.index[0]][0]
        return job_id
    return ""


def refresh_all(phase_filter: str):
    """Refresh jobs table and logs"""
    table_data, active_job_ids = get_active_jobs_table()
    logs = get_all_active_logs(active_job_ids, phase_filter)
    stats = get_runner_stats()
    
    # Format job IDs for display
    job_ids_text = "\n".join(active_job_ids) if active_job_ids else "No active jobs"
    
    return table_data, logs, stats, job_ids_text


# Create Gradio interface
with gr.Blocks(title="Sandbox Runner Dashboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🚀 Sandbox Runner Dashboard")
    gr.Markdown("Real-time monitoring for GPU job execution")
    
    with gr.Row():
        # Left column - Active Jobs
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Active Jobs")
            
            jobs_table = gr.Dataframe(
                headers=[
                    "Job ID", 
                    "Status", 
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
                column_widths=["20%", "10%", "10%", "8%", "10%", "8%", "8%", "16%"]
            )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh All", variant="primary", scale=2)
                auto_refresh = gr.Checkbox(label="Auto-refresh (5s)", value=False, scale=1)
        
        # Right column - Controls & Stats
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Controls")
            
            phase_filter = gr.Dropdown(
                label="Phase Filter",
                choices=["all", "build", "prep", "inference", "vllm"],
                value="all"
            )
            
            gr.Markdown("### 📊 Runner Stats")
            stats_output = gr.Textbox(
                label="",
                lines=20,
                max_lines=25,
                show_label=False,
                interactive=False
            )
    
    # Job IDs display (hidden, for internal use)
    active_job_ids_state = gr.Textbox(
        label="Active Job IDs",
        visible=True,
        lines=3,
        interactive=False
    )
    
    # Log viewer
    gr.Markdown("### 📜 Live Logs")
    log_output = gr.Textbox(
        label="",
        lines=30,
        max_lines=50,
        show_label=False,
        show_copy_button=True,
        interactive=False
    )
    
    # Event handlers
    refresh_btn.click(
        fn=refresh_all,
        inputs=[phase_filter],
        outputs=[jobs_table, log_output, stats_output, active_job_ids_state]
    )
    
    # Phase filter change
    phase_filter.change(
        fn=refresh_all,
        inputs=[phase_filter],
        outputs=[jobs_table, log_output, stats_output, active_job_ids_state]
    )
    
    # Job selection
    jobs_table.select(
        fn=select_job_from_table,
        inputs=[jobs_table],
        outputs=[]  # Could add single job view if needed
    )
    
    # Auto-refresh timer
    timer = gr.Timer(value=5, active=False)
    
    auto_refresh.change(
        fn=lambda x: gr.update(active=x),
        inputs=auto_refresh,
        outputs=timer
    )
    
    timer.tick(
        fn=refresh_all,
        inputs=[phase_filter],
        outputs=[jobs_table, log_output, stats_output, active_job_ids_state]
    )
    
    # Initial load on app start
    app.load(
        fn=refresh_all,
        inputs=[phase_filter],
        outputs=[jobs_table, log_output, stats_output, active_job_ids_state]
    )
    
    # Custom CSS
    app.css = """
    /* Terminal-style log output */
    .log-output textarea {
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace !important;
        background-color: #1e1e1e !important;
        color: #d4d4d4 !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
    }
    
    /* Stats output styling */
    .stats-output textarea {
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace !important;
        font-size: 12px !important;
        line-height: 1.4 !important;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 12px !important;
    }
    
    /* Compact spacing */
    .gr-box {
        padding: 0.5rem !important;
    }
    """


# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=9090,
        share=False,
        inbrowser=True
    )