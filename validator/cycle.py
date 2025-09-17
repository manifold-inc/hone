import asyncio
from loguru import logger
from validator import discovery, query, scoring

async def run_query_cycle(validator, state):
    current_block = validator.get_current_block()
    if (state['last_query_block'] is None) or ((current_block - state['last_query_block']) >= validator.config.query_interval_blocks):
        logger.info(f"Starting query cycle at block {current_block}")
        
        miners = await discovery.discover_miners(validator.chain)
        for uid, miner_node in miners.items():
            await validator.db.upsert_miner(
                uid=uid,
                hotkey=miner_node.get('hotkey'),
                ip=miner_node.get('ip'),
                port=miner_node.get('port'),
                stake=miner_node.get('stake'),
                last_update_block=current_block
            )
        
        if miners:
            logger.info(f"Persisted {len(miners)} miners to database")
        
        await query.query_miners(validator.chain, validator.db, validator.config, miners, current_block)
        
        state['last_query_block'] = current_block
        state['cycle_count'] += 1

async def run_weights_cycle(validator, state):
    current_block = validator.get_current_block()
    if (state['last_weights_block'] is None) or ((current_block - state['last_weights_block']) >= validator.config.weights_interval_blocks):
        logger.info(f"Starting weights cycle at block {current_block}")
        scores = await scoring.calculate_scores(validator.db, validator.config)
        await scoring.set_weights(validator.chain, validator.config, scores)
        state['last_weights_block'] = current_block

async def run_continuous(validator, stop_event: asyncio.Event = None):
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Cycle runner stopping...")
            break
        
        try:
            await run_query_cycle(validator, validator.state)
            await run_weights_cycle(validator, validator.state)
        except Exception as e:
            logger.error(f"Error in cycle: {e}")
        
        try:
            await asyncio.wait_for(
                stop_event.wait() if stop_event else asyncio.sleep(60),
                timeout=60
            )
            break
        except asyncio.TimeoutError:
            continue