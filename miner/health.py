from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException

from common.epistula import Epistula
from common.constants import HEALTH_ENDPOINT
from miner.handlers import handle_health

router = APIRouter()

@router.post(HEALTH_ENDPOINT)
async def health(request: Request):
    body = await request.body()
    signature = request.headers.get("Body-Signature")
    
    if not signature:
        raise HTTPException(status_code=401, detail="Missing Body-Signature header")
    
    is_valid, error, parsed_body = Epistula.verify_request(body, signature)
    
    if not is_valid:
        raise HTTPException(status_code=401, detail=f"Invalid signature: {error}")
    
    health_data = handle_health()
    
    response_body, response_headers = Epistula.create_request(
        keypair=request.app.state.keypair,
        receiver_hotkey=parsed_body['signed_by'],
        data=health_data,
        version=1
    )
    
    return response_body