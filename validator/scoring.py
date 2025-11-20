from typing import Dict, List, Tuple
from loguru import logger
from substrateinterface.exceptions import SubstrateRequestException
from common.chain import can_set_weights
import os
import math


async def calculate_scores_and_update_leaderboard(db, config, miners: Dict[int, Dict]) -> Tuple[Dict[int, float], Dict[int, str]]:
    """
    NEW INCENTIVE MECHANISM:
    
    1. Calculate aggregate metrics for all REGISTERED miners from recent query results
    2. Filter miners meeting minimum accuracy floor (default 20%)
    3. Update leaderboard with current top performers
    4. Return exponentially distributed rewards for top 5 miners
    
    Returns:
        (scores: Dict[uid -> weight], hotkey_map: Dict[uid -> hotkey])
    """
    current_block = config.current_block_provider()
    window_blocks = config.score_window_blocks
    min_responses = config.min_responses
    min_accuracy_floor = config.min_accuracy_floor
    top_miners_count = config.top_miners_count

    logger.info(f"=" * 80)
    logger.info(f"Calculating scores for leaderboard")
    logger.info(f"  Min accuracy floor: {min_accuracy_floor * 100:.1f}%")
    logger.info(f"  Top miners to reward: {top_miners_count}")
    logger.info(f"  Window: {window_blocks} blocks")
    logger.info(f"=" * 80)

    # get recent results
    rows = await db.get_recent_results(window_blocks=window_blocks, current_block=current_block)
    
    # aggregate metrics per hotkey (only for registered miners)
    registered_hotkeys = {miner.get("hotkey") for miner in miners.values() if miner.get("hotkey")}
    
    miner_stats: Dict[str, Dict] = {}
    
    for r in rows:
        hotkey = r.get('hotkey')
        if not hotkey or hotkey not in registered_hotkeys:
            continue
            
        if hotkey not in miner_stats:
            miner_stats[hotkey] = {
                'count': 0,
                'exact_matches': 0,
                'partial_sum': 0.0,
                'similarity_sum': 0.0,
                'efficiency_sum': 0.0,
                'successful_responses': 0,
                'uid': r.get('uid')
            }
        
        stats = miner_stats[hotkey]
        stats['count'] += 1
        
        if r['success']:
            stats['successful_responses'] += 1
            stats['exact_matches'] += 1 if r.get('exact_match', False) else 0
            stats['partial_sum'] += float(r.get('partial_correctness', 0.0))
            stats['similarity_sum'] += float(r.get('grid_similarity', 0.0))
            stats['efficiency_sum'] += float(r.get('efficiency_score', 0.0))
    
    # calculate overall scores and filter by floor
    qualifying_miners: List[Tuple[str, int, Dict]] = []
    
    for hotkey, stats in miner_stats.items():
        if stats['count'] < min_responses:
            logger.debug(f"Hotkey {hotkey[:12]}...: only {stats['count']} responses < min_responses={min_responses}")
            continue
        
        if stats['successful_responses'] == 0:
            continue
        
        exact_rate = stats['exact_matches'] / stats['count']
        partial_avg = stats['partial_sum'] / stats['successful_responses']
        similarity_avg = stats['similarity_sum'] / stats['successful_responses']
        efficiency_avg = stats['efficiency_sum'] / stats['successful_responses']
        
        # check floor requirement - exact match rate must be >= floor
        if exact_rate < min_accuracy_floor:
            logger.debug(f"Hotkey {hotkey[:12]}...: Below floor ({exact_rate*100:.1f}% < {min_accuracy_floor*100:.1f}%)")
            continue
        
        # calculate overall score using same weights as before
        weights = {
            'exact_match': 0.4,
            'partial': 0.3,
            'similarity': 0.2,
            'efficiency': 0.1
        }
        
        overall_score = (
            weights['exact_match'] * exact_rate +
            weights['partial'] * partial_avg +
            weights['similarity'] * similarity_avg +
            weights['efficiency'] * efficiency_avg
        )
        
        metrics = {
            "overall_score": overall_score,
            "exact_match_rate": exact_rate,
            "partial_correctness_avg": partial_avg,
            "grid_similarity_avg": similarity_avg,
            "efficiency_avg": efficiency_avg
        }
        
        qualifying_miners.append((hotkey, stats['uid'], metrics))
        
        logger.info(
            f"Hotkey {hotkey[:12]}... | Score: {overall_score:.3f} | "
            f"Exact: {exact_rate*100:.1f}% | Partial: {partial_avg:.2f} | "
            f"Similarity: {similarity_avg:.2f} | Efficiency: {efficiency_avg:.2f}"
        )
    
    logger.info(f"Found {len(qualifying_miners)} miners meeting floor requirement")
    
    if not qualifying_miners:
        logger.warning("No miners meet minimum accuracy floor")
        return {}, {}
    
    # sort by overall score (descending)
    qualifying_miners.sort(key=lambda x: x[2]["overall_score"], reverse=True)
    
    top_miners = qualifying_miners[:top_miners_count]
    
    for hotkey, uid, metrics in top_miners:
        # get repo info from most recent submission
        miner = await db.get_miner_by_hotkey(hotkey)
        if miner:
            # find most recent query result for this hotkey to get repo info
            recent_results = await db.get_recent_results(window_blocks=window_blocks, current_block=current_block)
            repo_info = None
            for r in recent_results:
                if r.get('hotkey') == hotkey and r.get('repo_url'):
                    repo_info = r
                    break
            
            if repo_info:
                await db.update_leaderboard(
                    hotkey=hotkey,
                    uid=uid,
                    overall_score=metrics["overall_score"],
                    exact_match_rate=metrics["exact_match_rate"],
                    partial_correctness_avg=metrics["partial_correctness_avg"],
                    grid_similarity_avg=metrics["grid_similarity_avg"],
                    efficiency_avg=metrics["efficiency_avg"],
                    repo_url=repo_info.get('repo_url'),
                    repo_branch=repo_info.get('repo_branch', 'main'),
                    repo_commit=repo_info.get('repo_commit'),
                    repo_path=repo_info.get('repo_path', '')
                )
    
    # remove miners from leaderboard who are no longer in top N
    current_leaderboard = await db.get_leaderboard(limit=100)  # get all
    leaderboard_hotkeys = {entry['hotkey'] for entry in current_leaderboard}
    top_hotkeys = {hotkey for hotkey, _, _ in top_miners}
    
    for hotkey in (leaderboard_hotkeys - top_hotkeys):
        logger.info(f"Removing {hotkey[:12]}... from leaderboard (no longer in top {top_miners_count})")
        await db.remove_from_leaderboard(hotkey)
        
    decay_factor = 0.8  # (higher = more concentrated)
    
    exponential_weights = []
    for rank in range(len(top_miners)):
        exponential_weights.append(math.exp(-decay_factor * rank))
    
    total_weight = sum(exponential_weights)
    normalized_weights = [w / total_weight for w in exponential_weights]
    
    # create final weight dictionary
    final_scores = {}
    hotkey_map = {}
    
    for i, (hotkey, uid, metrics) in enumerate(top_miners):
        weight = normalized_weights[i]
        final_scores[uid] = weight
        hotkey_map[uid] = hotkey
        
        logger.info(
            f"#{i+1} | UID {uid} | Hotkey {hotkey[:12]}... | "
            f"Score: {metrics['overall_score']:.3f} | "
            f"Weight: {weight*100:.2f}%"
        )
    
    # save detailed scores to database
    scores_with_metrics = {}
    for hotkey, uid, metrics in qualifying_miners:
        scores_with_metrics[uid] = metrics
    
    await db.save_scores(scores_with_metrics, {uid: hk for hk, uid, _ in qualifying_miners})
    
    logger.info(f"=" * 80)
    logger.info(f"Leaderboard updated: {len(top_miners)} miners in top {top_miners_count}")
    logger.info(f"Exponential distribution with decay factor: {decay_factor}")
    logger.info(f"=" * 80)
    
    return final_scores, hotkey_map


async def set_weights(chain, config, scores: Dict[int, float]) -> bool:
    """
    Set weights on chain    
    """
    
    BURN_UID = int(os.getenv("BURN_UID", "251"))
    BURN_PERCENTAGE = float(os.getenv("BURN_PERCENTAGE", "0.95"))
    MINER_PERCENTAGE = 1.0 - BURN_PERCENTAGE
    
    if not chain.substrate:
        chain.connect()
    
    nodes = chain.get_nodes()
    total_uids = len(nodes)
    all_uids = list(range(total_uids))
    all_weights = [0.0] * total_uids
    
    if not scores or sum(scores.values()) <= 0:
        all_weights[BURN_UID] = 1.0
        logger.warning("No qualifying miners - burning 100%")
    else:
        all_weights[BURN_UID] = BURN_PERCENTAGE
        
        for uid, weight in scores.items():
            if uid < total_uids:
                all_weights[uid] = weight * MINER_PERCENTAGE
        
        weight_sum = sum(all_weights)
        if weight_sum > 0:
            all_weights = [w / weight_sum for w in all_weights]
        
        logger.info(f"Weights: {BURN_PERCENTAGE*100:.0f}% burn + {MINER_PERCENTAGE*100:.0f}% to top miners")
    
    if not scores or sum(scores.values()) <= 0:
        # no qualifying miners - burn everything
        logger.warning("No qualifying miners found - burning 100% to burn UID")
        all_weights[BURN_UID] = 1.0
        
        logger.info("=" * 60)
        logger.info("BURN MODE - No miners meet minimum requirements")
        logger.info(f"  UID {BURN_UID:>3} - Weight: 100.00%")
        logger.info("=" * 60)
    else:
        # distribute according to exponential weights
        for uid, weight in scores.items():
            if uid < total_uids:
                all_weights[uid] = weight
            else:
                logger.warning(f"UID {uid} out of range (max: {total_uids-1})")
        
        # normalize to ensure sum = 1.0
        weight_sum = sum(all_weights)
        if weight_sum > 0:
            all_weights = [w / weight_sum for w in all_weights]
        
        logger.info("=" * 60)
        logger.info("Exponential distribution for top miners:")
        for uid, weight in enumerate(all_weights):
            if weight > 0:
                percentage = weight * 100
                logger.info(f"  UID {uid:>3} - Weight: {percentage:>6.2f}%")
        logger.info("=" * 60)
    
    weight_sum = sum(all_weights)
    logger.info(f"Total weight sum: {weight_sum:.10f}")
    
    if abs(weight_sum - 1.0) > 1e-6:
        logger.warning(f"Weight sum {weight_sum} != 1.0, normalizing...")
        all_weights = [w / weight_sum for w in all_weights]
        weight_sum = sum(all_weights)
        logger.info(f"Normalized weight sum: {weight_sum:.10f}")

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