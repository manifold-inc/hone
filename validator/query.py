import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from loguru import logger
import json

from common.epistula import Epistula
from common.constants import QUERY_ENDPOINT, CHECK_TASK_ENDPOINT, MAX_POLL_ATTEMPTS, POLL_INTERVAL 


def calculate_grid_similarity(grid1: List[List[int]], grid2: List[List[int]]) -> float:
    """Calculate pixel-wise similarity between two grids"""
    if not grid1 or not grid2:
        return 0.0
    
    # size mismatch
    if len(grid1) != len(grid2) or (grid1 and len(grid1[0]) != len(grid2[0])):
        return 0.0
    
    total_cells = len(grid1) * len(grid1[0])
    if total_cells == 0:
        return 0.0
    
    matching_cells = sum(
        1 for i in range(len(grid1))
        for j in range(len(grid1[0]))
        if grid1[i][j] == grid2[i][j]
    )
    
    return matching_cells / total_cells

def calculate_partial_correctness(predicted: List[List[int]], expected: List[List[int]]) -> float:
    """
    Calculate partial correctness score considering:
    - Shape matching
    - Color distribution
    - Pattern similarity
    """
    if not predicted or not expected:
        return 0.0
    
    score = 0.0
    weights = {'shape': 0.3, 'grid': 0.5, 'colors': 0.2}
    
    # shape score
    shape_match = (len(predicted) == len(expected) and 
                   len(predicted[0]) == len(expected[0]) if predicted else False)
    score += weights['shape'] if shape_match else 0
    
    # grid similarity
    if shape_match:
        score += weights['grid'] * calculate_grid_similarity(predicted, expected)
    
    # color distribution similarity
    pred_colors = set()
    exp_colors = set()
    for row in predicted:
        pred_colors.update(row)
    for row in expected:
        exp_colors.update(row)
    
    if exp_colors:
        color_overlap = len(pred_colors & exp_colors) / len(exp_colors)
        score += weights['colors'] * color_overlap
    
    return min(1.0, score)

def calculate_efficiency_score(response_time: float, max_time: float = 30.0) -> float:
    """Calculate efficiency score based on response time"""
    if response_time >= max_time:
        return 0.0
    return 1.0 - (response_time / max_time)

def _deep_validate_data(data: Any, path: str = "root") -> Tuple[bool, str]:
    """Deep validation of data structure for serialization"""
    try:
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)        
        if isinstance(data, dict) and 'train_examples' in data:
            original_len = len(data['train_examples'])
            deserialized_len = len(deserialized['train_examples'])
            if original_len != deserialized_len:
                return False, f"train_examples length mismatch: {original_len} -> {deserialized_len}"
            
            for i, (orig_ex, deser_ex) in enumerate(zip(data['train_examples'], deserialized['train_examples'])):
                if 'input' not in deser_ex or 'output' not in deser_ex:
                    return False, f"Example {i} missing input/output after serialization"
        
        return True, "OK"
    except Exception as e:
        return False, f"Serialization error: {e}"

async def _submit_task_to_miner(
    session: aiohttp.ClientSession,
    chain,
    config,
    uid: int,
    miner: Dict,
    problem_data: Dict
) -> Optional[str]:
    """Submit a task to a miner and get back a task ID"""
    ip = miner.get("ip")
    port = miner.get("port") or config.default_miner_port
    url = f"http://{ip}:{port}{QUERY_ENDPOINT}"
    
    query_data = {
        "problem_id": problem_data['id'],
        "train_examples": problem_data['problem_set']['train_examples'],
        "test_input": problem_data['problem_set']['test_input'],
        "num_train": problem_data['num_train_examples']
    }

        
    is_valid, msg = _deep_validate_data(query_data, "query_data")
    if not is_valid:
        logger.error(f"⚠️  Data validation failed for UID {uid}: {msg}")
        return None
    
    body, headers = Epistula.create_request(
        keypair=chain.keypair,
        receiver_hotkey=miner.get("hotkey"),
        data=query_data,
        version=1
    )
        
    try:
        async with session.post(url, json=body, headers=headers, timeout=5) as resp:
            if resp.status != 200:
                response_text = await resp.text()
                return None
            
            response_text = await resp.text()
            response_json = json.loads(response_text)
            task_id = response_json.get('data', {}).get('task_id')
            
            if not task_id:
                logger.error(f"No task_id in response from UID {uid}")
                logger.debug(f"Full response: {response_json}")
                return None
            
            logger.debug(f"UID {uid} accepted task with ID: {task_id}")
            return task_id
            
    except (asyncio.TimeoutError, aiohttp.ClientError, json.JSONDecodeError) as e:
        return None

async def _poll_task_result(
    session: aiohttp.ClientSession,
    chain,
    config,
    uid: int,
    miner: Dict,
    task_id: str,
    problem_data: Dict,
    max_attempts: int = MAX_POLL_ATTEMPTS,
    poll_interval: float = POLL_INTERVAL
) -> Dict:
    """Poll for task result from miner"""
    ip = miner.get("ip")
    port = miner.get("port") or config.default_miner_port
    url = f"http://{ip}:{port}{CHECK_TASK_ENDPOINT}/{task_id}"
    
    t0 = datetime.now(timezone.utc).replace(tzinfo=None)
    
    for attempt in range(max_attempts):
        try:
            check_data = {"task_id": task_id}
            body, headers = Epistula.create_request(
                keypair=chain.keypair,
                receiver_hotkey=miner.get("hotkey"),
                data=check_data,
                version=1
            )
            body_json = json.dumps(body, sort_keys=True)

            async with session.get(url, data=body_json, headers=headers, timeout=5) as resp:
                if resp.status != 200:
                    await asyncio.sleep(poll_interval)
                    continue
                
                response_text = await resp.text()
                response_json = json.loads(response_text)
                task_data = response_json.get('data', {})
                
                status = task_data.get('status')
                
                if status == 'completed':
                    dt = (datetime.now(timezone.utc).replace(tzinfo=None) - t0).total_seconds()
                    predicted_output = task_data.get('result', {}).get('output')
                    
                    if not predicted_output or not isinstance(predicted_output, list):
                        logger.error(f"Invalid output format from UID {uid}")
                        return {
                            "uid": uid,
                            "problem_id": problem_data['id'],
                            "success": False,
                            "response": None,
                            "error": "Invalid output format",
                            "rt": dt,
                            "metrics": {
                                "exact_match": False,
                                "partial_correctness": 0.0,
                                "grid_similarity": 0.0,
                                "efficiency_score": 0.0
                            },
                            "base_task_num": problem_data.get('metadata', {}).get('base_task_num'),
                            "chain_length": problem_data.get('metadata', {}).get('chain_length'),
                            "transformation_chain": problem_data.get('metadata', {}).get('transformation_chain'),
                            "num_train_examples": problem_data.get('num_train_examples')
                        }
                    
                    expected_output = problem_data['problem_set']['test_output']
                    exact_match = predicted_output == expected_output
                    partial_correctness = calculate_partial_correctness(predicted_output, expected_output)
                    grid_similarity = calculate_grid_similarity(predicted_output, expected_output)
                    efficiency_score = calculate_efficiency_score(dt)
                    
                    logger.info(f"UID {uid} | Problem {problem_data['id']} | Task {task_id} | "
                              f"Exact: {exact_match} | Partial: {partial_correctness:.2f} | "
                              f"Similarity: {grid_similarity:.2f} | Time: {dt:.2f}s")
                    
                    return {
                        "uid": uid,
                        "problem_id": problem_data['id'],
                        "success": True,
                        "response": task_data.get('result', {}),
                        "error": None,
                        "rt": dt,
                        "metrics": {
                            "exact_match": exact_match,
                            "partial_correctness": partial_correctness,
                            "grid_similarity": grid_similarity,
                            "efficiency_score": efficiency_score
                        },
                        "base_task_num": problem_data.get('metadata', {}).get('base_task_num'),
                        "chain_length": problem_data.get('metadata', {}).get('chain_length'),
                        "transformation_chain": problem_data.get('metadata', {}).get('transformation_chain'),
                        "num_train_examples": problem_data.get('num_train_examples')
                    }
                
                elif status == 'failed':
                    dt = (datetime.now(timezone.utc).replace(tzinfo=None) - t0).total_seconds()
                    error_msg = task_data.get('error', 'Unknown error')
                    logger.error(f"Task {task_id} failed for UID {uid}: {error_msg}")
                    return {
                        "uid": uid,
                        "problem_id": problem_data['id'],
                        "success": False,
                        "response": None,
                        "error": error_msg,
                        "rt": dt,
                        "metrics": {
                            "exact_match": False,
                            "partial_correctness": 0.0,
                            "grid_similarity": 0.0,
                            "efficiency_score": 0.0
                        },
                        "base_task_num": problem_data.get('metadata', {}).get('base_task_num'),
                        "chain_length": problem_data.get('metadata', {}).get('chain_length'),
                        "transformation_chain": problem_data.get('metadata', {}).get('transformation_chain'),
                        "num_train_examples": problem_data.get('num_train_examples')
                    }
                
                elif status in ['pending', 'processing']:
                    await asyncio.sleep(poll_interval)
                    continue
                
                else:
                    logger.warning(f"Unknown status '{status}' for task {task_id} from UID {uid}")
                    await asyncio.sleep(poll_interval)
                    continue
                    
        except (asyncio.TimeoutError, aiohttp.ClientError, json.JSONDecodeError) as e:
            logger.error(f"Error checking task {task_id} for UID {uid}: {e}")
            await asyncio.sleep(poll_interval)
            continue
    
    dt = (datetime.now(timezone.utc).replace(tzinfo=None) - t0).total_seconds()
    logger.error(f"Timeout waiting for task {task_id} from UID {uid} after {max_attempts} attempts")
    return {
        "uid": uid,
        "problem_id": problem_data['id'],
        "success": False,
        "response": None,
        "error": "Timeout waiting for result",
        "rt": dt,
        "metrics": {
            "exact_match": False,
            "partial_correctness": 0.0,
            "grid_similarity": 0.0,
            "efficiency_score": 0.0
        },
        "base_task_num": problem_data.get('metadata', {}).get('base_task_num'),
        "chain_length": problem_data.get('metadata', {}).get('chain_length'),
        "transformation_chain": problem_data.get('metadata', {}).get('transformation_chain'),
        "num_train_examples": problem_data.get('num_train_examples')
    }

async def _query_one_with_problem(
    session: aiohttp.ClientSession,
    chain,
    config,
    uid: int,
    miner: Dict,
    problem_data: Dict
) -> Dict:
    """Query a single miner with an ARC problem using task-based approach"""
    
    task_id = await _submit_task_to_miner(session, chain, config, uid, miner, problem_data)
    
    if not task_id:
        return {
            "uid": uid,
            "problem_id": problem_data['id'],
            "success": False,
            "response": None,
            "error": "Failed to submit task",
            "rt": 0.0,
            "metrics": {
                "exact_match": False,
                "partial_correctness": 0.0,
                "grid_similarity": 0.0,
                "efficiency_score": 0.0
            }
        }
    
    result = await _poll_task_result(
        session, chain, config, uid, miner, task_id, problem_data,
        max_attempts=50,
        poll_interval=10
    )
    
    return result

async def query_miners_with_problems(
    chain,
    db,
    config,
    miners: Dict[int, Dict],
    problems_batch: List[Dict],
    current_block: int
) -> Dict[int, List[Dict]]:
    """Query all miners with multiple problems using task-based approach"""
    results: Dict[int, List[Dict]] = {uid: [] for uid in miners.keys()}
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for problem_data in problems_batch:
            for uid, miner in miners.items():
                tasks.append(_query_one_with_problem(
                    session, chain, config, uid, miner, problem_data
                ))
        
        for fut in asyncio.as_completed(tasks):
            res = await fut
            uid = res["uid"]
            results[uid].append(res)
            
            await db.record_query_result(
                block=current_block,
                uid=uid,
                success=res["success"],
                response=res["response"],
                error=res["error"],
                response_time=res["rt"],
                ts=datetime.now(timezone.utc).replace(tzinfo=None),
                exact_match=res["metrics"]["exact_match"],
                partial_correctness=res["metrics"]["partial_correctness"],
                grid_similarity=res["metrics"]["grid_similarity"],
                efficiency_score=res["metrics"]["efficiency_score"],
                problem_id=res["problem_id"],
                base_task_num=res.get("base_task_num"),
                chain_length=res.get("chain_length"),
                transformation_chain=res.get("transformation_chain"),
                num_train_examples=res.get("num_train_examples")
            )
    
    total_queries = sum(len(r) for r in results.values())
    successful = sum(1 for uid_results in results.values() for r in uid_results if r["success"])
    exact_matches = sum(1 for uid_results in results.values() for r in uid_results if r["metrics"]["exact_match"])
    
    logger.info(f"Queried {len(miners)} miners with {len(problems_batch)} problems")
    logger.info(f"Total: {total_queries} | Success: {successful} | Exact: {exact_matches}")
    
    return results