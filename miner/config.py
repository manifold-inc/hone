from __future__ import annotations
import os
from dataclasses import dataclass
from common.constants import MINER_PORT

@dataclass
class MinerConfig:
    host: str = os.getenv("MINER_HOST", "0.0.0.0")
    port: int = int(os.getenv("MINER_PORT", MINER_PORT))
    
    wallet_name: str = os.getenv("WALLET_NAME", "default")
    wallet_hotkey: str = os.getenv("WALLET_HOTKEY", "default")
    wallet_path: str = os.getenv("WALLET_PATH", "~/.bittensor/wallets")
    
    hotkey: str = ""
    
    default_response_text: str = os.getenv("DEFAULT_RESPONSE_TEXT", "pong")