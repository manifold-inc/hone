import asyncio
import aiohttp
from typing import Dict, List
from datetime import datetime, timezone
from loguru import logger
import json

from common.epistula import Epistula
from common.constants import QUERY_ENDPOINT, DEFAULT_TIMEOUT

def calculate_grid_similarity(grid1: List[List[int]], grid2: List[List[int]]) -> float:
    """Calculate pixel-wise similarity between two grids"""
    if not grid1 or not grid2:
        return 0.0
    
    # Handle size mismatch
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
    
    # Shape score
    shape_match = (len(predicted) == len(expected) and 
                   len(predicted[0]) == len(expected[0]) if predicted else False)
    score += weights['shape'] if shape_match else 0
    
    # Grid similarity
    if shape_match:
        score += weights['grid'] * calculate_grid_similarity(predicted, expected)
    
    # Color distribution similarity
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

async def _query_one_with_problem(
    session: aiohttp.ClientSession,
    chain,
    config,
    uid: int,
    miner: Dict,
    problem_data: Dict
) -> Dict:
    """Query a single miner with an ARC problem"""
    ip = miner.get("ip")
    port = miner.get("port") or config.default_miner_port
    url = f"http://{ip}:{port}{QUERY_ENDPOINT}"
    
    # Prepare the query with just input (miner should predict output)
    query_data = {
        "problem_id": problem_data['id'],
        "input": problem_data['problem']['input'],
        "difficulty": problem_data['difficulty']
    }
    
    logger.info(f"Querying UID {uid} with problem {problem_data['id']} (difficulty: {problem_data['difficulty']})")
    
    body, headers = Epistula.create_request(
        keypair=chain.keypair,
        receiver_hotkey=miner.get("hotkey"),
        data=query_data,
        version=1
    )

    t0 = datetime.now(timezone.utc)
    try:
        async with session.post(url, json=body, headers=headers, timeout=DEFAULT_TIMEOUT) as resp:
            response_text = await resp.text()
            dt = (datetime.now(timezone.utc) - t0).total_seconds()
            
            if resp.status != 200:
                logger.error(f"Failed for UID {uid}: HTTP {resp.status}")
                return {
                    "uid": uid,
                    "problem_id": problem_data['id'],
                    "success": False,
                    "response": None,
                    "error": f"HTTP {resp.status}",
                    "rt": dt,
                    "metrics": {
                        "exact_match": False,
                        "partial_correctness": 0.0,
                        "grid_similarity": 0.0,
                        "efficiency_score": 0.0
                    }
                }
            
            # Parse and validate response
            try:
                response_json = json.loads(response_text)
                payload = response_json.get('data', {})
                predicted_output = payload.get('output')
                
                if not predicted_output or not isinstance(predicted_output, list):
                    raise ValueError("Invalid output format")
                
                # Calculate metrics
                expected_output = problem_data['problem']['output']
                exact_match = predicted_output == expected_output
                partial_correctness = calculate_partial_correctness(predicted_output, expected_output)
                grid_similarity = calculate_grid_similarity(predicted_output, expected_output)
                efficiency_score = calculate_efficiency_score(dt)
                
                logger.info(f"UID {uid} | Problem {problem_data['id']} | "
                          f"Exact: {exact_match} | Partial: {partial_correctness:.2f} | "
                          f"Similarity: {grid_similarity:.2f} | Time: {dt:.2f}s")
                
                return {
                    "uid": uid,
                    "problem_id": problem_data['id'],
                    "success": True,
                    "response": payload,
                    "error": None,
                    "rt": dt,
                    "metrics": {
                        "exact_match": exact_match,
                        "partial_correctness": partial_correctness,
                        "grid_similarity": grid_similarity,
                        "efficiency_score": efficiency_score
                    }
                }
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Invalid response from UID {uid}: {e}")
                return {
                    "uid": uid,
                    "problem_id": problem_data['id'],
                    "success": False,
                    "response": None,
                    "error": str(e),
                    "rt": dt,
                    "metrics": {
                        "exact_match": False,
                        "partial_correctness": 0.0,
                        "grid_similarity": 0.0,
                        "efficiency_score": 0.0
                    }
                }
            
    except asyncio.TimeoutError:
        dt = (datetime.now(timezone.utc) - t0).total_seconds()
        logger.error(f"Timeout for UID {uid}")
        return {
            "uid": uid,
            "problem_id": problem_data['id'],
            "success": False,
            "response": None,
            "error": "Timeout",
            "rt": dt,
            "metrics": {
                "exact_match": False,
                "partial_correctness": 0.0,
                "grid_similarity": 0.0,
                "efficiency_score": 0.0
            }
        }
    except Exception as e:
        dt = (datetime.now(timezone.utc) - t0).total_seconds()
        logger.error(f"Failed for UID {uid}: {e}")
        return {
            "uid": uid,
            "problem_id": problem_data['id'],
            "success": False,
            "response": None,
            "error": str(e),
            "rt": dt,
            "metrics": {
                "exact_match": False,
                "partial_correctness": 0.0,
                "grid_similarity": 0.0,
                "efficiency_score": 0.0
            }
        }

async def query_miners_with_problems(
    chain,
    db,
    config,
    miners: Dict[int, Dict],
    problems_batch: List[Dict],
    current_block: int
) -> Dict[int, List[Dict]]:
    """Query all miners with multiple problems"""
    results: Dict[int, List[Dict]] = {uid: [] for uid in miners.keys()}
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Create all query tasks
        tasks = []
        for problem_data in problems_batch:
            for uid, miner in miners.items():
                tasks.append(_query_one_with_problem(
                    session, chain, config, uid, miner, problem_data
                ))
        
        # Execute and collect results
        for fut in asyncio.as_completed(tasks):
            res = await fut
            uid = res["uid"]
            results[uid].append(res)
            
            # Store in database with metrics
            await db.record_query_result(
                block=current_block,
                uid=uid,
                success=res["success"],
                response=res["response"],
                error=res["error"],
                response_time=res["rt"],
                ts=datetime.utcnow(),
                exact_match=res["metrics"]["exact_match"],
                partial_correctness=res["metrics"]["partial_correctness"],
                grid_similarity=res["metrics"]["grid_similarity"],
                efficiency_score=res["metrics"]["efficiency_score"],
                problem_difficulty=problem_data['difficulty'],
                problem_id=res["problem_id"]
            )
    
    # Summary logging
    total_queries = sum(len(r) for r in results.values())
    successful = sum(1 for uid_results in results.values() for r in uid_results if r["success"])
    exact_matches = sum(1 for uid_results in results.values() for r in uid_results if r["metrics"]["exact_match"])
    
    logger.info(f"Queried {len(miners)} miners with {len(problems_batch)} problems")
    logger.info(f"Total: {total_queries} | Success: {successful} | Exact: {exact_matches}")
    
    return results