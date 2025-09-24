import asyncio
import hashlib
from loguru import logger
from validator import discovery, query, scoring

async def run_query_cycle(validator, state):
    """Run continuous queries for CYCLE_DURATION blocks"""
    current_block = validator.get_current_block()
    
    if state['last_query_block'] and (current_block - state['last_query_block']) < validator.config.query_interval_blocks:
        return
    
    logger.info(f"Starting query cycle at block {current_block}")
    cycle_start_block = current_block
    
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
    
    queries_in_cycle = 0
    while True:
        current_block = validator.get_current_block()
        
        if (current_block - cycle_start_block) >= validator.config.cycle_duration:
            logger.info(f"Query cycle complete after {queries_in_cycle} query rounds")
            break
        
        problems_batch = []
        num_problems = min(5, len(miners)) or 1
        
        for i in range(num_problems):
            try:
                difficulty = validator.sample_difficulty()
                problem = validator.synthetic_generator.generate_problem(
                    difficulty=difficulty,
                    return_metadata=True
                )
                
                problem_str = str(problem['input']) + str(problem['metadata']['transformation_chain'])
                problem_id = hashlib.sha256(problem_str.encode()).hexdigest()[:16]
                
                problems_batch.append({
                    'id': problem_id,
                    'problem': problem,
                    'difficulty': difficulty
                })
                
                logger.info(f'Generated problem {problem_id} | difficulty={difficulty} | steps={len(problem["metadata"]["transformation_chain"])}')
            except Exception as e:
                logger.error(f"Failed to generate problem: {e}")
        
        if problems_batch:
            await query.query_miners_with_problems(
                validator.chain, 
                validator.db, 
                validator.config, 
                miners, 
                problems_batch,
                current_block
            )
            queries_in_cycle += 1
        
        await asyncio.sleep(15)
    
    state['last_query_block'] = cycle_start_block
    state['cycle_count'] += 1

async def run_weights_cycle(validator, state):
    """Set weights based on accumulated scores"""
    current_block = validator.get_current_block()
    
    if state['last_weights_block'] and (current_block - state['last_weights_block']) < validator.config.weights_interval_blocks:
        return
    
    logger.info(f"Starting weights cycle at block {current_block}")
    scores = await scoring.calculate_scores(validator.db, validator.config)
    
    if scores:
        await scoring.set_weights(validator.chain, validator.config, scores)
    else:
        logger.warning("No scores to set weights")
    
    state['last_weights_block'] = current_block

async def run_continuous(validator, stop_event: asyncio.Event = None):
    """Main loop that runs cycles continuously"""
    
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Cycle runner stopping...")
            break
        
        try:
            await run_query_cycle(validator, validator.state)
            
            await run_weights_cycle(validator, validator.state)
            
            logger.info(f"Completed cycle {validator.state['cycle_count']}, waiting before next cycle...")
            
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in cycle: {e}")
            await asyncio.sleep(5)