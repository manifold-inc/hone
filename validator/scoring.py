from typing import Dict, List
from loguru import logger
from substrateinterface.exceptions import SubstrateRequestException
from common.chain import can_set_weights
import os


async def calculate_scores(db, config) -> Dict[int, Dict[str, float]]:
    """
    Calculate comprehensive scores for miners based on:
    - Exact match rate (40% weight)
    - Partial correctness (30% weight) 
    - Grid similarity (20% weight)
    - Efficiency (10% weight)
    """
    current_block = config.current_block_provider()
    window_blocks = config.score_window_blocks
    min_responses = config.min_responses

    rows = await db.get_recent_results(window_blocks=window_blocks, current_block=current_block)
    
    # agg metrics per miner
    miner_stats: Dict[int, Dict] = {}
    
    for r in rows:
        uid = int(r['uid'])
        if uid not in miner_stats:
            miner_stats[uid] = {
                'count': 0,
                'exact_matches': 0,
                'partial_sum': 0.0,
                'similarity_sum': 0.0,
                'efficiency_sum': 0.0,
                'successful_responses': 0
            }
        
        stats = miner_stats[uid]
        stats['count'] += 1
        
        if r['success']:
            stats['successful_responses'] += 1
            stats['exact_matches'] += 1 if r.get('exact_match', False) else 0
            stats['partial_sum'] += float(r.get('partial_correctness', 0.0))
            stats['similarity_sum'] += float(r.get('grid_similarity', 0.0))
            stats['efficiency_sum'] += float(r.get('efficiency_score', 0.0))
    
    # weighted scores
    scores: Dict[int, Dict[str, float]] = {}
    weights = {
        'exact_match': 0.4,
        'partial': 0.3,
        'similarity': 0.2,
        'efficiency': 0.1
    }
    
    for uid, stats in miner_stats.items():
        if stats['count'] < min_responses:
            logger.debug(f"UID {uid}: only {stats['count']} responses < min_responses={min_responses}")
            continue
        
        if stats['successful_responses'] == 0:
            scores[uid] = {
                "score": 0.0,
                "exact_match_rate": 0.0,
                "partial_correctness_avg": 0.0,
                "efficiency_avg": 0.0
            }
            continue
        
        exact_rate = stats['exact_matches'] / stats['count']
        partial_avg = stats['partial_sum'] / stats['successful_responses']
        similarity_avg = stats['similarity_sum'] / stats['successful_responses']
        efficiency_avg = stats['efficiency_sum'] / stats['successful_responses']
        
        final_score = (
            weights['exact_match'] * exact_rate +
            weights['partial'] * partial_avg +
            weights['similarity'] * similarity_avg +
            weights['efficiency'] * efficiency_avg
        )
        
        scores[uid] = {
            "score": final_score,
            "exact_match_rate": exact_rate,
            "partial_correctness_avg": partial_avg,
            "efficiency_avg": efficiency_avg
        }
        
        logger.info(f"UID {uid} | Score: {final_score:.3f} | "
                   f"Exact: {exact_rate:.2f} | Partial: {partial_avg:.2f} | "
                   f"Efficiency: {efficiency_avg:.2f}")
    
    await db.save_scores(scores)
    
    return {uid: metrics["score"] for uid, metrics in scores.items()}


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


async def set_weights(chain, config, scores: Dict[int, float]) -> bool:
    
    BURN_UID = int(os.getenv("BURN_UID", "251"))
    BURN_WEIGHT_PERCENT = float(os.getenv("BURN_WEIGHT_PERCENT", "0.99"))
    
    use_burn = BURN_WEIGHT_PERCENT > 0
    
    if use_burn:
        logger.info(f"🔥 Burn protection enabled: {BURN_WEIGHT_PERCENT*100:.0f}% to UID {BURN_UID}")
    
    if not chain.substrate:
        chain.connect()
    
    nodes = chain.get_nodes()
    total_uids = len(nodes)
    logger.info(f"Total UIDs in subnet: {total_uids}")
    
    all_uids = list(range(total_uids))
    all_weights = [0.0] * total_uids
    
    if not _validate_scores(scores):
        if use_burn:
            all_weights[BURN_UID] = 1.0
            logger.info("No valid scores, setting 100% weight to burn UID")
        else:
            logger.warning("No valid scores and burn protection disabled - cannot set weights")
            return False
    else:
        if use_burn:
            remaining_weight_percent = 1.0 - BURN_WEIGHT_PERCENT
            burn_weight_percent = BURN_WEIGHT_PERCENT
            
            total_score = sum(scores.values())
            if total_score > 0:
                miner_percentages = {uid: (score / total_score) * remaining_weight_percent 
                                    for uid, score in scores.items()}
                
                all_weights[BURN_UID] = burn_weight_percent
                
                for uid, weight_pct in miner_percentages.items():
                    all_weights[uid] = weight_pct
                
                logger.info(f"Setting weights: {BURN_WEIGHT_PERCENT*100:.0f}% to burn UID {BURN_UID}, "
                           f"{remaining_weight_percent*100:.0f}% split among {len(scores)} miners")
            else:
                logger.warning("Total score is zero, setting 100% to burn UID")
                all_weights[BURN_UID] = 1.0
        else:
            total_score = sum(scores.values())
            if total_score > 0:
                for uid, score in scores.items():
                    all_weights[uid] = score / total_score
            else:
                logger.warning("Total score is zero, cannot set weights")
                return False
    
    weight_sum = sum(all_weights)
    logger.info(f"Total weight sum: {weight_sum:.10f}")
    
    if abs(weight_sum - 1.0) > 1e-6:
        logger.warning(f"Weight sum {weight_sum} != 1.0, normalizing...")
        all_weights = [w / weight_sum for w in all_weights]
        weight_sum = sum(all_weights)
        logger.info(f"Normalized weight sum: {weight_sum:.10f}")
    
    logger.info("=" * 60)
    logger.info("Non-zero weights being set:")
    for uid, weight in enumerate(all_weights):
        if weight > 0:
            percentage = weight * 100
            logger.info(f"  UID {uid:>3} - Weight: {percentage:>6.2f}%")
    logger.info("=" * 60)

    try:
        result = chain.set_weights(
            uids=all_uids,
            weights=all_weights
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
    """Check if the validator can set weights based on rate limiting"""
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