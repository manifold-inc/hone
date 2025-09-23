from __future__ import annotations
import asyncio
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse


from .config import settings
from .models import CheckMinerRequest, CheckMinerResponse, TaskState
from .task_manager import TaskManager
from .logging_conf import configure_logging


configure_logging(settings.log_level)
log = logging.getLogger(__name__)


app = FastAPI(title="ARC‑AGI‑2 Orchestrator", default_response_class=ORJSONResponse)
manager = TaskManager()

@app.post("/check-miner-solution", response_model=CheckMinerResponse)
async def check_miner_solution(req: CheckMinerRequest):
    # task_id provided → look up
    if req.task_id:
        rec = manager.get(req.task_id)
        if not rec:
            raise HTTPException(status_code=404, detail="task not found")
        return CheckMinerResponse(
            task_id=rec.task_id,
            state=rec.state,
            message=rec.message,
            metrics=rec.metrics,
            outputs_s3_uri=rec.outputs_s3_uri,
            )
    rec = await manager.submit(miner_uid=req.miner_uid, s3_dataset_url=str(req.s3_dataset_url), docker_image=req.docker_image)
    log.info({"event": "task_submitted", "task_id": rec.task_id, "miner_uid": req.miner_uid})
    return CheckMinerResponse(task_id=rec.task_id, state=rec.state, message=rec.message)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


