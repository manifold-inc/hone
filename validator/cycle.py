import asyncio
from loguru import logger
from validator import discovery, query, scoring
from datetime import datetime, timezone

async def run_query_cycle(validator, state):
    """
    Run query cycle via sandbox runner
    
    New flow:
    1. Discover miners from chain
    2. Fetch /info from each miner
    3. Submit jobs to sandbox runner
    4. Poll for completion (up to 3 hours)
    5. Save metrics to database
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
    
    # query miners via sandbox runner
    # this will:
    # 1. fetch /info from each miner
    # 2. submit jobs to sandbox runner
    # 3. poll until completion
    # 4. save metrics to database
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
    Set weights based on accumulated scores from database
    
    This remains unchanged - scoring logic stays the same,
    it just reads from metrics stored by sandbox runner jobs
    """
    current_block = validator.get_current_block()
    
    # check if enough time has passed since last weight setting
    if state['last_weights_block'] and (current_block - state['last_weights_block']) < validator.config.weights_interval_blocks:
        logger.debug(f"Skipping weights cycle - only {current_block - state['last_weights_block']} blocks since last weights")
        return
    
    logger.info(f"=" * 80)
    logger.info(f"Starting weights cycle at block {current_block}")
    logger.info(f"=" * 80)
    
    # calculate scores from recent query results
    scores = await scoring.calculate_scores(validator.db, validator.config)

    if scores:
        logger.info(f"Calculated scores for {len(scores)} miners")
        await scoring.set_weights(validator.chain, validator.config, scores)
    else:
        logger.warning("No scores calculated - insufficient data or no miners")
    
    state['last_weights_block'] = current_block
    logger.info(f"Weights cycle complete")


async def run_continuous(validator, stop_event: asyncio.Event = None):
    """
    Main loop that runs query and weights cycles continuously
    
    This remains largely unchanged - just runs the cycles
    """
    logger.info("Starting continuous validator loop")
    
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stop event set, exiting cycle runner")
            break
        
        try:
            # run query cycle (miners via sandbox runner)
            await run_query_cycle(validator, validator.state)
            
            # run weights cycle (calculate and set weights)
            await run_weights_cycle(validator, validator.state)
            
            logger.info(f"Completed cycle {validator.state['cycle_count']}, waiting before next cycle...")
            
            # wait between cycles
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in validator cycle: {e}", exc_info=True)
            await asyncio.sleep(5)