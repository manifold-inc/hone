from typing import Any, Dict

def handle_query(app_state, payload: dict | None) -> Dict[str, Any]:
    app_state.queries_handled += 1
    app_state.last_payload = payload or {}
    message = (app_state.last_payload or {}).get("message", "ping")
    
    return {
        "response": app_state.cfg.default_response_text,
        "echo": message,
        "hotkey": app_state.cfg.hotkey,
    }

def handle_health() -> Dict[str, Any]:
    return {"ok": True}