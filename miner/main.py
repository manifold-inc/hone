from __future__ import annotations
import uvicorn
from fastapi import FastAPI
from loguru import logger

from miner.config import MinerConfig
from miner.keypair import load_keypair
from miner.endpoints.info import router as info_router

def create_app(cfg: MinerConfig) -> FastAPI:
    app = FastAPI(title="Hone Miner", version="0.1.0")
    
    keypair = load_keypair(cfg)
    cfg.hotkey = keypair.ss58_address
    
    app.state.cfg = cfg
    app.state.keypair = keypair
    app.state.queries_handled = 0
    app.state.last_payload = None
    
    app.include_router(info_router)
    
    logger.info(f"Miner initialized with hotkey: {cfg.hotkey}")
    return app

def run():
    cfg = MinerConfig()
    app = create_app(cfg)
    
    logger.info(
        f"Starting miner hotkey={cfg.hotkey} on {cfg.host}:{cfg.port}"
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port)

if __name__ == "__main__":
    run()