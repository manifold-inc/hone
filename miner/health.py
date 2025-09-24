from __future__ import annotations
from fastapi import APIRouter, Request
import os
from loguru import logger

from common.epistula import Epistula
from common.constants import HEALTH_ENDPOINT
from miner.handlers import handle_health

router = APIRouter()

@router.get(HEALTH_ENDPOINT)
@router.post(HEALTH_ENDPOINT)
async def health(request: Request):
    if request.method == "GET" or os.getenv("SKIP_EPISTULA_VERIFY", "false").lower() == "true":
        return {"status": "healthy"}
    
    body = await request.body()
    signature = request.headers.get("Body-Signature")
    
    if not signature:
        return {"status": "healthy"}
    
    is_valid, error, parsed_body = Epistula.verify_request(body, signature)
    
    if not is_valid:
        return {"status": "healthy", "signature_valid": False}
    
    health_data = handle_health()
    
    response_body, response_headers = Epistula.create_request(
        keypair=request.app.state.keypair,
        receiver_hotkey=parsed_body['signed_by'],
        data=health_data,
        version=1
    )
    
    return response_body