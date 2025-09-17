from pathlib import Path
import json
from substrateinterface import Keypair
from loguru import logger

def load_keypair(cfg) -> Keypair:
    base_path = Path(cfg.wallet_path).expanduser()
    file_path = base_path / cfg.wallet_name / "hotkeys" / cfg.wallet_hotkey
    
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
    except Exception as e:
        logger.error(f"Failed to load keypair: {e}")
        raise