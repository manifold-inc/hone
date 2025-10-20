from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException, Path
from fastapi.responses import JSONResponse
import json
import os
from loguru import logger

from common.epistula import Epistula
from common.constants import CHECK_TASK_ENDPOINT
from miner.handlers import handle_check_task

router = APIRouter()

@router.get(f"{CHECK_TASK_ENDPOINT}/{{task_id}}")
async def check_task(request: Request, task_id: str = Path(..., description="Task ID to check")) -> JSONResponse:
    """Check the status of a submitted task"""
    
    if os.getenv("SKIP_EPISTULA_VERIFY", "false").lower() == "true":
        logger.warning("⚠️ EPISTULA VERIFICATION SKIPPED (TEST MODE)")
        
        task_status = handle_check_task(task_id)
        
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return JSONResponse(
            content={"data": task_status},
            headers={"Content-Type": "application/json"}
        )
    
    else:
        signature = request.headers.get("Body-Signature")
        
        if not signature:
            raise HTTPException(status_code=401, detail="Missing Body-Signature header")
        
        body = await request.body()

        is_valid, error, parsed_body = Epistula.verify_request(body, signature)
        
        if not is_valid:
            raise HTTPException(status_code=401, detail=f"Invalid signature: {error}")
        
        task_status = handle_check_task(task_id)
        
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        response_body, response_headers = Epistula.create_request(
            keypair=request.app.state.keypair,
            receiver_hotkey=parsed_body['signed_by'],
            data=task_status,
            version=1
        )
        
        return JSONResponse(
            content=response_body,
            headers=response_headers
        )