import asyncio
import hashlib
from loguru import logger
from validator import discovery, query, scoring
import random

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
                num_train = random.randint(
                    validator.config.min_train_examples, 
                    validator.config.max_train_examples
                )
                
                chain_length = random.randint(3, 5)
                
                problem_set = validator.synthetic_generator.generate_problem_set(
                    num_train=num_train,
                    num_test=1,
                    chain_length=chain_length,
                )
                
                actual_train_count = len(problem_set.get('train_examples', []))
                if actual_train_count == 0:
                    logger.warning(f"Generated problem set has no training examples (requested {num_train}), skipping")
                    continue
                
                if not problem_set.get('test_input') or not problem_set.get('test_output'):
                    logger.warning(f"Generated problem set missing test input/output, skipping")
                    continue
                
                problem_str = str(problem_set['test_input']) + str(problem_set['metadata']['transformation_chain'])
                problem_id = hashlib.sha256(problem_str.encode()).hexdigest()[:16]
                
                problems_batch.append({
                    'id': problem_id,
                    'problem_set': problem_set,
                    'num_train_examples': actual_train_count,
                    'metadata': {
                        'base_task_num': problem_set['metadata']['base_task'],
                        'chain_length': problem_set['metadata']['chain_length'],
                        'transformation_chain': problem_set['metadata']['transformation_chain']
                    }
                })
                
                chain_length_actual = len(problem_set['metadata']['transformation_chain'])
                logger.info(f'Generated problem {problem_id} '
                           f'train_examples={actual_train_count} | chain_length={chain_length_actual}')
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
            await validator.maybe_cleanup_database()

        else:
            logger.warning("No valid problems generated in this round, will retry")
        
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