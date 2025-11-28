from typing import Dict, List, Tuple
from loguru import logger
from substrateinterface.exceptions import SubstrateRequestException
from common.chain import can_set_weights
import os
import math


async def calculate_scores_and_update_leaderboard(db, config, miners: Dict[int, Dict]) -> Tuple[Dict[int, float], Dict[int, str]]:
    """
    Calculate scores based on exact_match_rate only
    
    1. Get aggregate exact_match_rate for all REGISTERED miners from recent query results
    2. Filter miners meeting minimum accuracy floor
    3. Update leaderboard with current top performers
    4. Return exponentially distributed rewards for top N miners
    
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

    rows = await db.get_recent_results(window_blocks=window_blocks, current_block=current_block)
    
    registered_hotkeys = {miner.get("hotkey") for miner in miners.values() if miner.get("hotkey")}
    
    miner_stats: Dict[str, Dict] = {}
    
    for r in rows:
        hotkey = r.get('hotkey') if hasattr(r, 'get') else r['hotkey']
        if not hotkey or hotkey not in registered_hotkeys:
            continue
            
        if hotkey not in miner_stats:
            miner_stats[hotkey] = {
                'count': 0,
                'exact_match_rate_sum': 0.0,
                'successful_responses': 0,
                'uid': r.get('uid') if hasattr(r, 'get') else r['uid']
            }
        
        stats = miner_stats[hotkey]
        stats['count'] += 1
        
        success = r.get('success') if hasattr(r, 'get') else r['success']
        if success:
            stats['successful_responses'] += 1
            
            exact_match_rate = r.get('exact_match_rate') if hasattr(r, 'get') else r['exact_match_rate']
            if exact_match_rate is not None:
                stats['exact_match_rate_sum'] += float(exact_match_rate)
            
        logger.debug(f"Hotkey {hotkey[:12]}... - Stats: {stats}")
    
    qualifying_miners: List[Tuple[str, int, float]] = []
    
    for hotkey, stats in miner_stats.items():
        if stats['count'] < min_responses:
            logger.debug(f"Hotkey {hotkey[:12]}...: only {stats['count']} responses < min_responses={min_responses}")
            continue
        
        if stats['successful_responses'] == 0:
            logger.debug(f"Hotkey {hotkey[:12]}...: no successful responses")
            continue
        
        avg_exact_match_rate = stats['exact_match_rate_sum'] / stats['successful_responses']
        
        if avg_exact_match_rate < min_accuracy_floor:
            logger.debug(f"Hotkey {hotkey[:12]}...: Below floor ({avg_exact_match_rate*100:.1f}% < {min_accuracy_floor*100:.1f}%)")
            continue
        
        qualifying_miners.append((hotkey, stats['uid'], avg_exact_match_rate))
        
        logger.info(
            f"Hotkey {hotkey[:12]}... | Exact Match Rate: {avg_exact_match_rate*100:.2f}% | "
            f"Responses: {stats['successful_responses']}"
        )
    
    logger.info(f"Found {len(qualifying_miners)} miners meeting floor requirement")
    
    if not qualifying_miners:
        logger.warning("No miners meet minimum accuracy floor")
        return {}, {}
    
    # sort by exact_match_rate (descending)
    qualifying_miners.sort(key=lambda x: x[2], reverse=True)
    
    top_miners = qualifying_miners[:top_miners_count]
    
    for hotkey, uid, exact_match_rate in top_miners:
        miner = await db.get_miner_by_hotkey(hotkey)
        if miner:
            recent_results = await db.get_recent_results(window_blocks=window_blocks, current_block=current_block)
            repo_info = None
            for r in recent_results:
                r_hotkey = r.get('hotkey') if hasattr(r, 'get') else r['hotkey']
                r_repo_url = r.get('repo_url') if hasattr(r, 'get') else r['repo_url']
                if r_hotkey == hotkey and r_repo_url:
                    repo_info = r
                    break
            
            if repo_info:
                await db.update_leaderboard(
                    hotkey=hotkey,
                    uid=uid,
                    exact_match_rate=exact_match_rate,
                    repo_url=repo_info.get('repo_url') if hasattr(repo_info, 'get') else repo_info['repo_url'],
                    repo_branch=repo_info.get('repo_branch', 'main') if hasattr(repo_info, 'get') else repo_info['repo_branch'] or 'main',
                    repo_commit=repo_info.get('repo_commit') if hasattr(repo_info, 'get') else repo_info['repo_commit'],
                    repo_path=repo_info.get('repo_path', '') if hasattr(repo_info, 'get') else repo_info['repo_path'] or ''
                )
    
    # remove miners from leaderboard who are no longer in top N
    current_leaderboard = await db.get_leaderboard(limit=100)
    leaderboard_hotkeys = {entry['hotkey'] for entry in current_leaderboard}
    top_hotkeys = {hotkey for hotkey, _, _ in top_miners}
    
    for hotkey in (leaderboard_hotkeys - top_hotkeys):
        logger.info(f"Removing {hotkey[:12]}... from leaderboard (no longer in top {top_miners_count})")
        await db.remove_from_leaderboard(hotkey)
        
    decay_factor = 0.8
    
    exponential_weights = []
    for rank in range(len(top_miners)):
        exponential_weights.append(math.exp(-decay_factor * rank))
    
    total_weight = sum(exponential_weights)
    normalized_weights = [w / total_weight for w in exponential_weights]
    
    final_scores = {}
    hotkey_map = {}
    
    for i, (hotkey, uid, exact_match_rate) in enumerate(top_miners):
        weight = normalized_weights[i]
        final_scores[uid] = weight
        hotkey_map[uid] = hotkey
        
        logger.info(
            f"#{i+1} | UID {uid} | Hotkey {hotkey[:12]}... | "
            f"Exact Match Rate: {exact_match_rate*100:.2f}% | "
            f"Weight: {weight*100:.2f}%"
        )
    
    # save scores to database
    scores_for_db = {uid: exact_match_rate for hotkey, uid, exact_match_rate in qualifying_miners}
    hotkey_map_for_db = {uid: hotkey for hotkey, uid, _ in qualifying_miners}
    
    await db.save_scores(scores_for_db, hotkey_map_for_db)
    
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
    array_size = max(len(nodes), BURN_UID + 1)
    all_weights = [0.0] * array_size
    
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
        logger.warning("No qualifying miners found - burning 100% to burn UID")
        all_weights[BURN_UID] = 1.0
        
        logger.info("=" * 60)
        logger.info("BURN MODE - No miners meet minimum requirements")
        logger.info(f"  UID {BURN_UID:>3} - Weight: 100.00%")
        logger.info("=" * 60)
    else:
        for uid, weight in scores.items():
            if uid < total_uids:
                all_weights[uid] = weight
            else:
                logger.warning(f"UID {uid} out of range (max: {total_uids-1})")
        
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