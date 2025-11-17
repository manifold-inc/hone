"""
Advanced Log Streaming Dashboard using Panel

Features:
- Real-time HTTP polling (can be upgraded to WebSocket)
- Resizable panes with splitters
- Terminal-style log display
- Interactive job cards
- Search and filtering
- Export logs functionality
"""

import panel as pn
import param
import httpx
from datetime import datetime
from typing import Dict, List

# Configuration
API_BASE_URL = "http://localhost:8080"
API_KEY = "dev-key-12345"

CSS = """
.job-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    padding: 15px;
    margin: 10px;
    color: white;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    min-height: 150px;
}

.job-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}

.job-card-active {
    border: 2px solid #4CAF50;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
    100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
}

.log-terminal {
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    padding: 10px;
    border-radius: 5px;
    overflow-y: auto;
    max-height: 600px;
}

.log-phase-build { color: #64B5F6; }
.log-phase-prep { color: #81C784; }
.log-phase-inference { color: #FFD54F; }
.log-phase-vllm { color: #BA68C8; }
.log-error { color: #FF6B6B; }
.log-warning { color: #FFA726; }

.status-badge {
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    display: inline-block;
}

.status-running { background: #4CAF50; }
.status-building { background: #2196F3; }
.status-completed { background: #8BC34A; }
.status-failed { background: #F44336; }
"""

# Panel extensions
pn.extension("tabulator", sizing_mode="stretch_width", raw_css=[CSS])


class LogStreamDashboard(param.Parameterized):
    """Interactive log streaming dashboard"""

    # Parameters
    selected_job = param.String(default="", doc="Currently selected job")
    auto_refresh = param.Boolean(default=True, doc="Enable auto-refresh")
    refresh_interval = param.Integer(
        default=2, bounds=(1, 10), doc="Refresh interval (seconds)"
    )
    phase_filter = param.Selector(
        default="all",
        objects=["all", "build", "prep", "inference", "vllm"],
        doc="Filter by phase",
    )
    search_term = param.String(default="", doc="Search in logs")

    def __init__(self, **params):
        super().__init__(**params)

        # API client
        self.client = httpx.Client(
            base_url=API_BASE_URL,
            headers={"X-API-Key": API_KEY},
            timeout=30.0,
        )

        # State
        self.log_cursors: Dict[str, str] = {}
        self.log_cache: Dict[str, List[dict]] = {}
        self.job_summaries: Dict[str, dict] = {}

        # UI Components
        self._build_ui()

        # Periodic refresh
        if self.auto_refresh:
            pn.state.add_periodic_callback(
                self.refresh_logs, period=self.refresh_interval * 1000
            )

    # ---------- API HELPERS ----------

    def get_active_jobs(self) -> List[str]:
        """Fetch active jobs from API"""
        try:
            response = self.client.get("/v1/logs/active")
            if response.status_code == 200:
                return response.json().get("active_jobs", [])
        except Exception as e:
            print(f"Error fetching jobs: {e}")
        return []

    def get_job_logs(self, job_id: str) -> Dict:
        """Fetch logs for a specific job"""
        try:
            params = {"limit": 100}

            # Use cursor if available
            if job_id in self.log_cursors:
                params["cursor_id"] = self.log_cursors[job_id]

            # Apply phase filter
            if self.phase_filter != "all":
                params["phase"] = self.phase_filter

            response = self.client.get(f"/v1/logs/{job_id}", params=params)

            if response.status_code == 200:
                data = response.json()

                # Update cursor
                if data.get("cursor_id"):
                    self.log_cursors[job_id] = data["cursor_id"]

                # Cache logs
                if job_id not in self.log_cache:
                    self.log_cache[job_id] = []

                for entry in data.get("entries", []):
                    self.log_cache[job_id].append(entry)

                return data
        except Exception as e:
            print(f"Error fetching logs: {e}")

        return {"entries": []}

    def get_job_status(self, job_id: str) -> Dict:
        """Fetch job status"""
        try:
            response = self.client.get(f"/v1/jobs/{job_id}")
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}

    # ---------- UI CONSTRUCTION ----------

    def _build_ui(self):
        """Build static UI components (header, controls, containers)"""

        # Header
        self.header = pn.pane.Markdown(
            """
# 🚀 Live Log Streaming Dashboard  
Real-time log viewer for all running jobs
            """,
            height=80,
        )

        # Controls
        self.auto_refresh_toggle = pn.widgets.Toggle(
            name="Auto Refresh",
            value=self.auto_refresh,
            button_type="success",
            width=120,
        )
        self.refresh_slider = pn.widgets.IntSlider(
            name="Refresh Interval (s)",
            value=self.refresh_interval,
            start=1,
            end=10,
            step=1,
            width=220,
        )
        self.manual_refresh_btn = pn.widgets.Button(
            name="🔄 Refresh Now", button_type="primary", width=130
        )
        # NOTE: button_type 'info' is not supported in this Panel version → use 'primary'
        self.stats_btn = pn.widgets.Button(
            name="📊 Stats", button_type="primary", width=100
        )

        # Wire up simple callbacks
        self.manual_refresh_btn.on_click(lambda e: self.refresh_logs())
        self.auto_refresh_toggle.param.watch(self._toggle_auto_refresh, "value")
        self.refresh_slider.param.watch(self._update_refresh_interval, "value")

        self.controls = pn.Row(
            self.auto_refresh_toggle,
            self.refresh_slider,
            self.manual_refresh_btn,
            self.stats_btn,
        )

        # Job grid and log viewer containers
        self.job_grid = pn.GridBox(ncols=3, sizing_mode="stretch_both")
        self.log_viewer = pn.Column(sizing_mode="stretch_both")

        # Initial content
        self.refresh_logs()

    # ---------- CALLBACK HELPERS ----------

    def _toggle_auto_refresh(self, event):
        self.auto_refresh = event.new
        # To fully support this, you'd add/remove periodic callbacks dynamically.

    def _update_refresh_interval(self, event):
        self.refresh_interval = event.new
        # To fully support this, you'd recreate the periodic callback.

    # ---------- CARD & LOG RENDERING ----------

    def create_job_card(self, job_id: str) -> pn.pane.HTML:
        """Create an interactive job card"""

        # Get job summary
        if job_id not in self.job_summaries:
            status = self.get_job_status(job_id)
            self.job_summaries[job_id] = {
                "status": status.get("status", "unknown"),
                "phase": status.get("current_phase", "unknown"),
                "progress": status.get("progress_percentage", 0) or 0,
                "miner": (status.get("miner_hotkey", "")[:12] + "...")
                if status.get("miner_hotkey")
                else "unknown",
                "weight_class": status.get("weight_class", "unknown"),
            }

        summary = self.job_summaries[job_id]
        status_str = (summary["status"] or "unknown").lower()
        is_active = status_str in ["running", "building", "prep", "inference"]

        # Get recent logs
        if job_id not in self.log_cache:
            self.get_job_logs(job_id)

        logs = self.log_cache.get(job_id, [])
        recent_logs = logs[-3:] if logs else []

        # Format recent logs
        log_html = ""
        for entry in recent_logs:
            msg = entry.get("message", "")
            if len(msg) > 80:
                msg = msg[:77] + "..."
            log_html += (
                f'<div style="font-size: 11px; color: #ccc; margin: 2px 0;">{msg}</div>'
            )

        card_html = f"""
        <div class="job-card {'job-card-active' if is_active else ''}">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <strong>{job_id}</strong>
                <span class="status-badge status-{status_str}">{status_str.upper()}</span>
            </div>
            <div style="font-size: 12px; margin: 5px 0;">
                📍 {summary['phase']} | ⚡ {summary['weight_class']} | 👤 {summary['miner']}
            </div>
            <div style="margin-top: 10px;">
                <div style="background: rgba(0,0,0,0.3); padding: 5px; border-radius: 3px;">
                    {log_html if log_html else '<div style="color: #bbb;">No logs yet...</div>'}
                </div>
            </div>
            <div style="margin-top: 10px;">
                <div style="background: rgba(255,255,255,0.1); height: 4px; border-radius: 2px;">
                    <div style="background: #4CAF50; height: 100%; width: {summary['progress']}%; border-radius: 2px;"></div>
                </div>
            </div>
        </div>
        """

        return pn.pane.HTML(card_html, min_width=320, height=200)

    def create_log_display(self, job_id: str) -> pn.Column:
        """Create full log display for selected job"""

        if not job_id:
            return pn.Column(
                pn.pane.Markdown("## Select a job to view logs (not wired yet)"),
                height=600,
            )

        # Get all cached logs
        logs = self.log_cache.get(job_id, [])

        # Apply search filter
        if self.search_term:
            logs = [
                log
                for log in logs
                if self.search_term.lower() in log.get("message", "").lower()
            ]

        # Format logs as HTML
        log_html = '<div class="log-terminal">'

        for entry in logs:
            timestamp = entry.get("timestamp", "")
            phase = entry.get("phase", "") or ""
            level = entry.get("level", "info") or "info"
            message = entry.get("message", "")

            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp_str = dt.strftime("%H:%M:%S")
            except Exception:
                timestamp_str = timestamp[:19]

            phase_class = f"log-phase-{phase}" if phase else ""
            level_class = f"log-{level}" if level in ["error", "warning"] else ""

            log_html += f"""
            <div>
                <span style="color: #666;">[{timestamp_str}]</span>
                <span class="{phase_class}">[{phase.upper():>9}]</span>
                <span class="{level_class}">{message}</span>
            </div>
            """

        log_html += "</div>"

        # Summary info
        summary = self.job_summaries.get(job_id, {})
        progress = summary.get("progress", 0)
        status = summary.get("status", "unknown")
        phase = summary.get("phase", "unknown")
        miner = summary.get("miner", "unknown")

        info = pn.Row(
            pn.indicators.Number(
                value=progress,
                name="Progress %",
                format="{value}%",
                font_size="20pt",
                title_size="10pt",
            ),
            pn.pane.Markdown(
                f"""
**Status:** {status}  
**Phase:** {phase}  
**Miner:** {miner}
                """
            ),
        )

        # Top controls (visual; not fully wired)
        controls = pn.Row(
            pn.widgets.Select(
                name="Phase Filter",
                value=self.phase_filter,
                options=["all", "build", "prep", "inference", "vllm"],
                width=150,
            ),
            pn.widgets.TextInput(
                name="Search",
                value=self.search_term,
                placeholder="Search logs...",
                width=200,
            ),
            pn.widgets.Button(name="📥 Export", button_type="success", width=90),
            pn.widgets.Button(name="🔙 Back", button_type="warning", width=90),
        )

        return pn.Column(
            f"## 📜 Logs for {job_id}",
            controls,
            info,
            pn.pane.HTML(log_html, height=500, sizing_mode="stretch_width"),
            f"**Total Entries:** {len(logs)}",
            sizing_mode="stretch_both",
        )

    # ---------- REFRESH & LAYOUT ----------

    def refresh_logs(self):
        """Refresh all logs and update UI"""
        if self.selected_job:
            # Update selected job logs
            self.get_job_logs(self.selected_job)
            self.log_viewer.objects = [self.create_log_display(self.selected_job)]
        else:
            # Update job grid
            active_jobs = self.get_active_jobs()

            self.job_grid.objects = []

            if not active_jobs:
                self.job_grid.objects = [pn.pane.Markdown("### No active jobs with logs")]
            else:
                cards = []
                for job_id in active_jobs[:12]:  # Limit to 12 for performance
                    cards.append(self.create_job_card(job_id))
                self.job_grid.objects = cards

    def render_main(self):
        """Main content area (grid or logs)"""
        if self.selected_job:
            return self.log_viewer
        return self.job_grid

    def get_layout(self):
        """Get the main template layout"""
        template = pn.template.FastListTemplate(
            title="Log Stream Dashboard",
            header=self.header,
            sidebar=[
                "## Controls",
                self.controls,
                pn.Spacer(height=20),
                "## Filters",
                pn.widgets.Select(
                    name="Phase",
                    value=self.phase_filter,
                    options=["all", "build", "prep", "inference", "vllm"],
                ),
                pn.widgets.TextInput(
                    name="Search",
                    value=self.search_term,
                    placeholder="Search...",
                ),
            ],
            main=[pn.bind(self.render_main)],
        )
        return template


_dashboard = LogStreamDashboard()
_dashboard_template = _dashboard.get_layout()
_dashboard_template.servable()
