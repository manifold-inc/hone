from typing import Dict
from loguru import logger

async def discover_miners(chain) -> Dict[int, Dict]:
    try:
        chain.connect()
        miners = chain.get_miners()
        logger.info(f"Discovered {len(miners)} miners from chain")
        return miners
    except Exception as e:
        logger.error(f"Chain discovery failed: {e}")
        return {}