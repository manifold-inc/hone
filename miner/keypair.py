import os
import json
import hashlib
from pathlib import Path
from substrateinterface import Keypair
from loguru import logger

def load_keypair(config) -> Keypair:
    """Load keypair from wallet or create mock keypair in test mode"""
    
    # local test mode
    if os.getenv("SKIP_EPISTULA_VERIFY", "false").lower() == "true":
        logger.warning("⚠️ TEST MODE: Creating mock keypair")
        seed_str = f"mock_{config.wallet_name}_{config.wallet_hotkey}_seed"
        seed_bytes = hashlib.sha256(seed_str.encode()).digest()
        keypair = Keypair.create_from_seed(seed_bytes.hex())
        logger.info(f"Mock keypair created: {keypair.ss58_address[:8]}...")
        return keypair
    
    # normal mode - load from wallet file
    wallet_path = Path(config.wallet_path).expanduser()
    file_path = wallet_path / config.wallet_name / "hotkeys" / config.wallet_hotkey
    
    try:
        with open(file_path, "r") as file:
            keypair_data = json.load(file)
        
        if "secretSeed" in keypair_data:
            keypair = Keypair.create_from_seed(keypair_data["secretSeed"])
        elif "secretKey" in keypair_data:
            keypair = Keypair.create_from_seed(keypair_data["secretKey"])
        else:
            raise ValueError("Could not find secret key in hotkey file")
        
        logger.info(f"Loaded keypair from {file_path}")
        return keypair
    except FileNotFoundError:
        logger.error(f"Failed to load keypair: {file_path} not found")
        raise
    except Exception as e:
        logger.error(f"Failed to load keypair: {e}")
        raise