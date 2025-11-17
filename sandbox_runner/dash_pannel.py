"""
Simple Log Streaming Dashboard using Gradio

Lightweight and fast dashboard for viewing job logs.
"""

import gradio as gr
import httpx
import json
from datetime import datetime
from typing import Dict, List, Optional
import time

# Configuration
API_BASE_URL = "http://localhost:8080"
API_KEY = "dev-key-12345"

# Global state
log_cursors = {}
log_cache = {}

def get_client():
    """Get HTTP client"""
    return httpx.Client(
        base_url=API_BASE_URL,
        headers={"X-API-Key": API_KEY},
        timeout=30.0
    )

def get_active_jobs() -> List[List[str]]:
    """Get list of active jobs as table data"""
    try:
        client = get_client()
        response = client.get("/v1/logs/active")
        if response.status_code == 200:
            jobs = response.json().get("active_jobs", [])
            
            # Get status for each job
            table_data = []
            for job_id in jobs:
                try:
                    status_response = client.get(f"/v1/jobs/{job_id}")
                    if status_response.status_code == 200:
                        status = status_response.json()
                        table_data.append([
                            job_id,
                            status.get("status", "unknown"),
                            status.get("current_phase", "unknown"),
                            f"{status.get('progress_percentage', 0):.1f}%",
                            status.get("weight_class", "unknown"),
                            status.get("miner_hotkey", "")[:12] + "..."
                        ])
                    else:
                        table_data.append([job_id, "unknown", "unknown", "0%", "unknown", "unknown"])
                except:
                    table_data.append([job_id, "error", "error", "0%", "unknown", "unknown"])
            
            return table_data
        return []
    except Exception as e:
        print(f"Error fetching jobs: {e}")
        return []

def get_job_logs(job_id: str, phase_filter: str = "all", limit: int = 100) -> str:
    """Get logs for a specific job"""
    global log_cursors, log_cache
    
    if not job_id:
        return "Please enter a Job ID or select from the active jobs table."
    
    try:
        client = get_client()
        params = {"limit": limit}
        
        # Use cursor if available
        if job_id in log_cursors:
            params["cursor_id"] = log_cursors[job_id]
        
        # Apply phase filter
        if phase_filter != "all":
            params["phase"] = phase_filter
        
        response = client.get(f"/v1/logs/{job_id}", params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Update cursor
            if data.get("cursor_id"):
                log_cursors[job_id] = data["cursor_id"]
            
            # Cache logs
            if job_id not in log_cache:
                log_cache[job_id] = []
            
            for entry in data.get("entries", []):
                log_cache[job_id].append(entry)
            
            # Format logs for display
            log_text = f"=== LOGS FOR {job_id} ===\n"
            log_text += f"Total Entries: {data.get('total_entries', 0)} | "
            log_text += f"Loaded: {len(log_cache[job_id])} | "
            log_text += f"Has More: {'Yes' if data.get('has_more') else 'No'}\n"
            log_text += "=" * 80 + "\n\n"
            
            # Display logs
            for entry in log_cache[job_id]:
                timestamp = entry.get("timestamp", "")
                phase = entry.get("phase", "")
                level = entry.get("level", "info")
                message = entry.get("message", "")
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp_str = dt.strftime("%H:%M:%S")
                except:
                    timestamp_str = timestamp[:19]
                
                # Color coding for terminal (ANSI codes)
                phase_colors = {
                    "build": "\033[94m",      # Blue
                    "prep": "\033[92m",       # Green
                    "inference": "\033[93m",  # Yellow
                    "vllm": "\033[95m"        # Magenta
                }
                
                level_colors = {
                    "error": "\033[91m",      # Red
                    "warning": "\033[93m",    # Yellow
                    "info": "\033[0m"         # Normal
                }
                
                phase_color = phase_colors.get(phase, "\033[0m")
                level_color = level_colors.get(level, "\033[0m")
                reset_color = "\033[0m"
                
                log_text += f"[{timestamp_str}] {phase_color}[{phase.upper():>9}]{reset_color} {level_color}{message}{reset_color}\n"
            
            return log_text
        else:
            return f"Error fetching logs: Status {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def clear_cache(job_id: str = None) -> str:
    """Clear log cache"""
    global log_cursors, log_cache
    
    if job_id:
        if job_id in log_cursors:
            del log_cursors[job_id]
        if job_id in log_cache:
            del log_cache[job_id]
        return f"Cache cleared for job {job_id}"
    else:
        log_cursors.clear()
        log_cache.clear()
        return "All cache cleared"

def get_log_stats() -> str:
    """Get log statistics"""
    try:
        client = get_client()
        response = client.get("/v1/logs/stats")
        
        if response.status_code == 200:
            stats = response.json()
            return f"""
📊 Log Service Statistics:
- Active Streams: {stats.get('active_streams', 0)}
- Total Log Entries: {stats.get('total_log_entries', 0):,}
- Retention Period: {stats.get('retention_hours', 0)} hours
- Active Jobs: {len(stats.get('active_jobs', []))}

Cached Data:
- Jobs with cursors: {len(log_cursors)}
- Jobs in cache: {len(log_cache)}
- Total cached entries: {sum(len(logs) for logs in log_cache.values())}
"""
        return "Failed to fetch statistics"
    except Exception as e:
        return f"Error: {str(e)}"

def select_job_from_table(evt: gr.SelectData, table_data):
    """Handle job selection from table"""
    if evt.index[0] < len(table_data):
        job_id = table_data[evt.index[0]][0]
        return job_id
    return ""

def auto_refresh_logs(job_id: str, phase_filter: str, auto_refresh: bool):
    """Auto-refresh logs if enabled"""
    if auto_refresh and job_id:
        return get_job_logs(job_id, phase_filter, limit=50)
    return None

# Create Gradio interface
with gr.Blocks(title="Log Stream Dashboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🚀 Log Streaming Dashboard")
    gr.Markdown("Real-time log viewer for sandbox runner jobs")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Active jobs table
            gr.Markdown("### Active Jobs")
            jobs_table = gr.Dataframe(
                headers=["Job ID", "Status", "Phase", "Progress", "Weight Class", "Miner"],
                value=get_active_jobs(),
                interactive=False,
                row_count=(5, 'fixed'),
                col_count=(6, 'fixed')
            )
            
            with gr.Row():
                refresh_jobs_btn = gr.Button("🔄 Refresh Jobs", variant="primary")
                stats_btn = gr.Button("📊 Statistics", variant="secondary")
        
        with gr.Column(scale=1):
            # Controls
            gr.Markdown("### Controls")
            job_id_input = gr.Textbox(
                label="Job ID",
                placeholder="Enter job ID or select from table",
                lines=1
            )
            
            phase_filter = gr.Dropdown(
                label="Phase Filter",
                choices=["all", "build", "prep", "inference", "vllm"],
                value="all"
            )
            
            auto_refresh = gr.Checkbox(
                label="Auto Refresh (every 2s)",
                value=False
            )
            
            with gr.Row():
                get_logs_btn = gr.Button("📜 Get Logs", variant="primary")
                clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="stop")
    
    # Log viewer
    with gr.Row():
        log_output = gr.Textbox(
            label="Log Output",
            lines=25,
            max_lines=50,
            value="Select a job to view logs...",
            show_copy_button=True,
            elem_classes="log-terminal"
        )
    
    # Statistics output
    stats_output = gr.Textbox(
        label="Statistics",
        lines=10,
        visible=False
    )
    
    # Event handlers
    refresh_jobs_btn.click(
        fn=lambda: get_active_jobs(),
        outputs=jobs_table
    )
    
    jobs_table.select(
        fn=select_job_from_table,
        inputs=jobs_table,
        outputs=job_id_input
    )
    
    get_logs_btn.click(
        fn=get_job_logs,
        inputs=[job_id_input, phase_filter],
        outputs=log_output
    )
    
    clear_cache_btn.click(
        fn=lambda job_id: (clear_cache(job_id), get_job_logs(job_id, phase_filter.value) if job_id else ""),
        inputs=job_id_input,
        outputs=[stats_output, log_output]
    )
    
    stats_btn.click(
        fn=lambda: (get_log_stats(), gr.update(visible=True)),
        outputs=[stats_output, stats_output]
    )
    
    # Auto-refresh timer
    timer = gr.Timer(value=2, active=False)
    auto_refresh.change(
        fn=lambda x: gr.update(active=x),
        inputs=auto_refresh,
        outputs=timer
    )
    
    timer.tick(
        fn=lambda job_id, phase: get_job_logs(job_id, phase, limit=50) if job_id else None,
        inputs=[job_id_input, phase_filter],
        outputs=log_output
    )
    
    # CSS for terminal styling
    app.css = """
    .log-terminal {
        font-family: 'Courier New', monospace !important;
        background-color: #1e1e1e !important;
        color: #d4d4d4 !important;
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