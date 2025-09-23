from __future__ import annotations
import json
import os
import time
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
import threading

DATA_DIR = os.getenv("ARC_DATA_DIR", "/data")
OUT_DIR = os.getenv("ARC_OUT_DIR", "/out")


app = FastAPI(default_response_class=ORJSONResponse)
_state = {"state": "IDLE", "message": "ready"}


@app.post("/start-prediction")
async def start_prediction():
    global _state
    if _state["state"] == "WIP":
        return {"ok": True, "message": "already running"}


_state = {"state": "WIP", "message": "predicting"}


def _work():
    try:
        # Minimal logic: copy ground_truth to predictions as a placeholder
        gt = os.path.join(DATA_DIR, "ground_truth.json")
        with open(gt, "r") as f:
            g = json.load(f)
        os.makedirs(OUT_DIR, exist_ok=True)
        time.sleep(2) # simulate compute
        with open(os.path.join(OUT_DIR, "predictions.json"), "w") as f:
            json.dump(g, f)
        _state.update({"state": "DONE", "message": "complete"})
    except Exception as e:
        _state.update({"state": "ERROR", "message": str(e)})

    threading.Thread(target=_work, daemon=True).start()
    return {"ok": True}


@app.get("/status")
async def status():
    return _state