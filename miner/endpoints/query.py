from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional
import json
import os
from datetime import datetime, timezone
from loguru import logger

from common.epistula import Epistula
from common.constants import QUERY_ENDPOINT
from miner.handlers import handle_query

router = APIRouter()

def log_query_request(query_data: Dict[str, Any], signed_for: Optional[str] = None) -> None:
    """
    Log query request data to .logs/query_req.log file

    Args:
        query_data: The parsed query data
        signed_for: The hotkey this request was signed for (if available)
    """
    log_dir = ".logs"
    os.makedirs(log_dir, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_data": query_data,
        "signed_for": signed_for
    }

    log_file = os.path.join(log_dir, "query_req.log")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry, indent=2))
        f.write("\n---\n")

@router.post(QUERY_ENDPOINT)
async def query(request: Request, body: Dict[str, Any]) -> JSONResponse:
    """
    Accept a query and return a task ID for async processing
    """
    signature = request.headers.get("Body-Signature")
    
    if os.getenv("SKIP_EPISTULA_VERIFY", "false").lower() == "true":
        logger.warning("⚠️ EPISTULA VERIFICATION SKIPPED (TEST MODE)")
        parsed_body = body.copy()
        if 'data' in body:
            query_data = body['data']
        else:
            query_data = body

        # Log the query request (test mode - no signed_for available)
        log_query_request(query_data, signed_for=None)

        # test mode - create task and return task ID
        response_data = handle_query(request.app.state, query_data)

        return JSONResponse(
            content={"data": response_data},
            headers={"Content-Type": "application/json"}
        )
    else:
        if not signature:
            raise HTTPException(status_code=401, detail="Missing Body-Signature header")
        
        body_bytes = json.dumps(body, sort_keys=True).encode('utf-8')
        is_valid, error, parsed_body = Epistula.verify_request(body_bytes, signature)
        
        if not is_valid:
            raise HTTPException(status_code=401, detail=f"Invalid signature: {error}")
        
        if parsed_body['signed_for'] != request.app.state.cfg.hotkey:
            raise HTTPException(status_code=403, detail="Request not intended for this miner")

        query_data = Epistula.extract_data(parsed_body)

        # Log the query request with signed_for field
        log_query_request(query_data, signed_for=parsed_body['signed_for'])

        response_data = handle_query(request.app.state, query_data)
        
        response_body, response_headers = Epistula.create_request(
            keypair=request.app.state.keypair,
            receiver_hotkey=parsed_body['signed_by'],
            data=response_data,
            version=1
        )
        
        return JSONResponse(
            content=response_body,
            headers=response_headers
        )