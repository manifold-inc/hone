from typing import Dict, List
from loguru import logger
from substrateinterface.exceptions import SubstrateRequestException
from common.chain import can_set_weights

async def calculate_scores(db, config) -> Dict[int, float]:
    current_block = config.current_block_provider()
    window_blocks = config.score_window_blocks
    min_responses = config.min_responses

    rows = await db.get_recent_results(window_blocks=window_blocks, current_block=current_block)
    counts: Dict[int, int] = {}
    success: Dict[int, int] = {}

    for r in rows:
        uid = int(r['uid'])
        counts[uid] = counts.get(uid, 0) + 1
        if r['success']:
            success[uid] = success.get(uid, 0) + 1

    scores: Dict[int, float] = {}
    for uid, n in counts.items():
        if n >= min_responses:
            s = success.get(uid, 0) / n
            scores[uid] = s
        else:
            logger.debug(f"UID {uid}: only {n} responses < min_responses={min_responses}; skipping")

    await db.save_scores(scores)
    return scores

def _normalize_scores(scores: Dict[int, float], weight_max: int = 65535) -> Dict[int, float]:
    if not scores:
        return {}
    
    total = sum(scores.values())
    if total <= 0:
        return {uid: 0.0 for uid in scores.keys()}
    
    normalized = {uid: (s / total) * weight_max for uid, s in scores.items()}
    return normalized

def _validate_scores(scores: Dict[int, float]) -> bool:
    if not scores:
        logger.warning("No scores provided")
        return False
    
    if any(s < 0 for s in scores.values()):
        logger.error("Negative scores found")
        return False
    
    if sum(scores.values()) <= 0:
        logger.error("Total score is zero or negative")
        return False
    
    return True

async def set_weights(chain, config, scores: Dict[int, float], version: int = 0) -> bool:
    if not _validate_scores(scores):
        return False

    weights = _normalize_scores(scores)
    if not weights:
        logger.warning("No weights to set after normalization")
        return False

    uids: List[int] = sorted(weights.keys())
    weight_values: List[float] = [weights[u] for u in uids]

    logger.info(f"Setting weights for {len(uids)} UIDs")
    logger.debug(f"UIDs: {uids[:10]}..." if len(uids) > 10 else f"UIDs: {uids}")
    logger.debug(f"Weights (normalized): {weight_values[:10]}..." if len(weight_values) > 10 else f"Weights: {weight_values}")

    if not chain.substrate:
        chain.connect()

    try:
        result = chain.set_weights(
            uids=uids,
            weights=weight_values,
            version=version,
            wait_for_inclusion=config.__dict__.get('wait_for_inclusion', False),
            wait_for_finalization=config.__dict__.get('wait_for_finalization', True)
        )
        
        if result == "success":
            logger.info("✅ Successfully set weights on chain")
            return True
        else:
            logger.error(f"Unexpected result from set_weights: {result}")
            return False
            
    except SubstrateRequestException as e:
        logger.error(f"Failed to set weights - Substrate error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to set weights - Unexpected error: {e}", exc_info=True)
        return False

async def check_can_set_weights(chain, config) -> bool:
    if not chain.substrate:
        chain.connect()
    
    if chain.validator_uid is None:
        logger.error("Validator UID not found - cannot check weight setting capability")
        return False
    
    try:
        can_set = can_set_weights(
            chain.substrate, 
            chain.netuid, 
            chain.validator_uid
        )
        
        if not can_set:
            logger.warning("Cannot set weights yet - rate limit not reached")
        
        return can_set
    except Exception as e:
        logger.error(f"Error checking if weights can be set: {e}")
        return False