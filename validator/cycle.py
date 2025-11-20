import asyncio
from loguru import logger
from validator import discovery, query, scoring
from datetime import datetime, timezone

async def run_query_cycle(validator, state):
    """
    Run query cycle via sandbox runner with submission history and daily limits
    
    New flow:
    1. Discover miners from chain
    2. Check daily submission limits
    3. Fetch /info from eligible miners
    4. Check submission history to avoid re-evaluation
    5. Submit new jobs to sandbox runner OR use cached metrics
    6. Poll for completion
    7. Save metrics to database and update submission history
    """
    
    # publish validator heartbeat
    try:
        with open("validator/.version", "r") as f:
            validator_version = f.read().strip()
    except Exception as e:
        logger.warning(f"Could not read .version file: {e}")
        validator_version = "unknown"

    try:
        logger.info(f"Publishing validator heartbeat")
        validator.telemetry_client.publish(
            "/validator/heartbeat",
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "version": validator_version,
                "cycle_count": validator.state.get("cycle_count"),
                "wallet_hotkey": validator.config.hotkey
            },
        )
    except Exception as e:
        logger.warning(f"Failed to publish telemetry heartbeat: {e}")

    current_block = validator.get_current_block()
    
    # check if enough time has passed since last query
    if state['last_query_block'] and (current_block - state['last_query_block']) < validator.config.query_interval_blocks:
        logger.debug(f"Skipping query cycle - only {current_block - state['last_query_block']} blocks since last query")
        return
    
    logger.info(f"=" * 80)
    logger.info(f"Starting query cycle at block {current_block}")
    logger.info(f"=" * 80)
    
    # discover miners from chain
    miners = await discovery.discover_miners(validator.chain)
    
    # persist miner info to database
    for uid, miner_node in miners.items():
        await validator.db.upsert_miner(
            uid=uid,
            hotkey=miner_node.get('hotkey'),
            ip=miner_node.get('ip'),
            port=miner_node.get('port'),
            stake=miner_node.get('stake'),
            last_update_block=current_block
        )
    
    if not miners:
        logger.warning("No miners discovered from chain, skipping query cycle")
        state['last_query_block'] = current_block
        state['cycle_count'] += 1
        return
    
    logger.info(f"Discovered and persisted {len(miners)} miners")
    
    # query miners via sandbox runner (with submission history and daily limits)
    await query.query_miners_via_sandbox(
        validator.chain,
        validator.db,
        validator.config,
        miners,
        current_block,
        validator.telemetry_client
    )
    
    # cleanup old database records
    await validator.maybe_cleanup_database()
    
    # update state
    state['last_query_block'] = current_block
    state['cycle_count'] += 1
    
    logger.info(f"Query cycle {state['cycle_count']} complete")


async def run_weights_cycle(validator, state):
    """
    Set weights based on top 5 miners from leaderboard with exponential distribution
    
    NEW MECHANISM:
    1. Calculate aggregate metrics for all registered miners
    2. Filter by minimum accuracy floor (default 20%)
    3. Update leaderboard with top performers
    4. Apply exponential distribution to top 5 miners
    5. Burn 100% if no miners meet floor
    """
    current_block = validator.get_current_block()
    
    # check if enough time has passed since last weight setting
    if state['last_weights_block'] and (current_block - state['last_weights_block']) < validator.config.weights_interval_blocks:
        logger.debug(f"Skipping weights cycle - only {current_block - state['last_weights_block']} blocks since last weights")
        return
    
    logger.info(f"=" * 80)
    logger.info(f"Starting weights cycle at block {current_block}")
    logger.info(f"=" * 80)
    
    # get current registered miners
    miners = await discovery.discover_miners(validator.chain)
    
    # calculate scores and update leaderboard
    scores, hotkey_map = await scoring.calculate_scores_and_update_leaderboard(
        validator.db, 
        validator.config,
        miners
    )

    if scores:
        logger.info(f"Calculated weights for {len(scores)} top miners")
        await scoring.set_weights(validator.chain, validator.config, scores)
    else:
        logger.warning("No miners meet minimum requirements - burning 100%")
        await scoring.set_weights(validator.chain, validator.config, {})
    
    state['last_weights_block'] = current_block
    logger.info(f"Weights cycle complete")


async def run_continuous(validator, stop_event: asyncio.Event = None):
    """
    Main loop that runs query and weights cycles continuously
    """
    logger.info("Starting continuous validator loop")
    logger.info(f"Config:")
    logger.info(f"  Max submissions per day: {validator.config.max_submissions_per_day}")
    logger.info(f"  Min accuracy floor: {validator.config.min_accuracy_floor * 100:.1f}%")
    logger.info(f"  Top miners count: {validator.config.top_miners_count}")
    
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stop event set, exiting cycle runner")
            break
        
        try:
            # run query cycle (miners via sandbox runner with history/limits)
            await run_query_cycle(validator, validator.state)
            
            # run weights cycle (exponential distribution for top 5)
            await run_weights_cycle(validator, validator.state)
            
            logger.info(f"Completed cycle {validator.state['cycle_count']}, waiting before next cycle...")
            
            # wait between cycles
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in validator cycle: {e}", exc_info=True)
            logger.exception(e)
            await asyncio.sleep(5)