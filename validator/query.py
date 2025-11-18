import asyncio
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timezone
from loguru import logger

from validator.sandbox_client import SandboxRunnerClient
from validator.telemetry import TelemetryClient


async def fetch_miner_info(
    session: aiohttp.ClientSession,
    uid: int,
    miner: Dict,
    default_port: int,
    timeout: int
) -> Optional[Dict]:
    """
    Fetch miner info from /info endpoint
    
    Returns:
        {
            "repo_url": str,
            "repo_branch": str,
            "repo_commit": str (optional),
            "repo_path": str,
            "weight_class": str,
            "use_vllm": bool,
            "vllm_config": dict (optional),
            "custom_env_vars": dict,
            "version": str,
            "hotkey": str
        }
    """
    ip = miner.get("ip")
    port = miner.get("port") or default_port
    url = f"http://{ip}:{port}/info"
    
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning(f"UID {uid}: /info returned {resp.status}")
                return None
            
            info = await resp.json()
            
            # validate required fields
            if not info.get("repo_url"):
                logger.error(f"UID {uid}: Missing repo_url in /info response")
                return None
            
            if not info.get("weight_class"):
                logger.warning(f"UID {uid}: Missing weight_class, defaulting to 1xH200")
                info["weight_class"] = "1xH200"
            
            logger.debug(f"UID {uid}: Fetched info - repo: {info.get('repo_url')}, weight: {info.get('weight_class')}")
            return info
            
    except asyncio.TimeoutError:
        logger.warning(f"UID {uid}: Timeout fetching /info")
        return None
    except aiohttp.ClientError as e:
        logger.warning(f"UID {uid}: Network error fetching /info: {e}")
        return None
    except Exception as e:
        logger.error(f"UID {uid}: Unexpected error fetching /info: {e}")
        return None


async def query_miners_via_sandbox(
    chain,
    db,
    config,
    miners: Dict[int, Dict],
    current_block: int,
    telemetry_client: TelemetryClient
) -> Dict[int, Dict]:
    """
    Query miners by submitting jobs to sandbox runner
    
    Flow:
    1. Fetch /info from each miner to get repo URL and requirements
    2. Submit job to sandbox runner for each miner
    3. Poll sandbox runner for job completion (up to 3 hours)
    4. Retrieve metrics from completed jobs
    5. Save results to database
    
    Args:
        chain: Chain interface
        db: Database
        config: Validator config
        miners: Dict of {uid: miner_info}
        current_block: Current block number
        telemetry_client: Telemetry client
    
    Returns:
        Dict of {uid: result_dict}
    """
    
    sandbox_client = SandboxRunnerClient(
        endpoint=config.sandbox_runner_endpoint,
        api_key=config.sandbox_runner_api_key
    )
    
    # step 1: fetch miner info concurrently
    logger.info(f"Fetching /info from {len(miners)} miners...")
    
    miner_infos = {}
    timeout = aiohttp.ClientTimeout(total=config.miner_info_timeout_seconds)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            fetch_miner_info(session, uid, miner, config.default_miner_port, config.miner_info_timeout_seconds)
            for uid, miner in miners.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for uid, result in zip(miners.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"UID {uid}: Exception fetching info: {result}")
            elif result:
                miner_infos[uid] = result
    
    logger.info(f"Successfully fetched info from {len(miner_infos)}/{len(miners)} miners")
    
    # step 2: submit jobs to sandbox runner
    job_submissions = {}
    
    for uid, info in miner_infos.items():
        try:
            miner = miners[uid]
            
            logger.info(f"UID {uid}: Submitting job to sandbox runner")
            logger.debug(f"  Repo: {info['repo_url']}")
            logger.debug(f"  Branch: {info.get('repo_branch', 'main')}")
            logger.debug(f"  Weight: {info['weight_class']}")
            
            response = await sandbox_client.submit_job(
                repo_url=info["repo_url"],
                repo_branch=info.get("repo_branch", "main"),
                repo_commit=info.get("repo_commit"),
                repo_path=info.get("repo_path", ""),
                weight_class=info["weight_class"],
                miner_hotkey=miner.get("hotkey"),
                validator_hotkey=config.hotkey,
                priority=5,
                use_vllm=info.get("use_vllm", False),
                vllm_config=info.get("vllm_config"),
                custom_env_vars=info.get("custom_env_vars", {})
            )
            
            job_id = response.get("job_id")
            queue_position = response.get("queue_position")
            
            job_submissions[uid] = {
                "job_id": job_id,
                "miner": miner,
                "info": info,
                "submitted_at": datetime.now(timezone.utc)
            }
            
            logger.info(f"UID {uid}: Job {job_id} submitted (queue position: {queue_position})")
            
        except Exception as e:
            logger.error(f"UID {uid}: Failed to submit job to sandbox: {e}")
            
            # record submission failure
            await db.record_query_result(
                block=current_block,
                uid=uid,
                success=False,
                response=None,
                error=f"Sandbox job submission failed: {e}",
                response_time=0.0,
                ts=datetime.now(timezone.utc).replace(tzinfo=None),
                exact_match=False,
                partial_correctness=0.0,
                grid_similarity=0.0,
                efficiency_score=0.0,
                problem_id=None
            )
    
    if not job_submissions:
        logger.warning("No jobs were successfully submitted to sandbox runner")
        return {}
    
    logger.info(f"Submitted {len(job_submissions)} jobs to sandbox runner")
    
    # step 3: poll all jobs until completion
    results = {}
    
    async def poll_and_record(uid: int, job_info: Dict):
        """Poll a single job and record results to database"""
        job_id = job_info["job_id"]
        miner = job_info["miner"]
        
        start_time = datetime.now(timezone.utc)
        
        def on_status_change(job_id, status, full_status):
            """Log status changes"""
            progress = full_status.get("progress_percentage", 0)
            phase = full_status.get("current_phase", "unknown")
            logger.info(f"UID {uid} | Job {job_id} | {status} | {phase} | {progress:.1f}%")
        
        # poll until complete with 3h timeout
        final_status = await sandbox_client.poll_until_complete(
            job_id=job_id,
            poll_interval=config.sandbox_poll_interval_seconds,
            max_attempts=config.sandbox_max_poll_attempts,
            on_status_change=on_status_change
        )
        
        end_time = datetime.now(timezone.utc)
        execution_time = (end_time - start_time).total_seconds()
        
        # step 4: get metrics if job completed successfully
        metrics_data = None
        if final_status.get("status") == "completed":
            metrics_data = await sandbox_client.get_job_metrics(job_id)
        
        # step 5: record to database
        if metrics_data and metrics_data.get("metrics"):
            metrics = metrics_data["metrics"]
            
            await db.record_query_result(
                block=current_block,
                uid=uid,
                success=True,
                response={"job_id": job_id, "sandbox_status": "completed"},
                error=None,
                response_time=execution_time,
                ts=datetime.now(timezone.utc).replace(tzinfo=None),
                exact_match=metrics.get("exact_match", False),
                partial_correctness=metrics.get("partial_correctness", 0.0),
                grid_similarity=metrics.get("grid_similarity", 0.0),
                efficiency_score=metrics.get("efficiency_score", 0.0),
                problem_id=metrics.get("problem_id"),
                base_task_num=metrics.get("base_task_num"),
                chain_length=metrics.get("chain_length"),
                transformation_chain=metrics.get("transformation_chain"),
                num_train_examples=metrics.get("num_train_examples")
            )
            
            logger.info(
                f"UID {uid} | Job {job_id} completed | "
                f"Exact: {metrics.get('exact_match')} | "
                f"Partial: {metrics.get('partial_correctness', 0):.2f} | "
                f"Similarity: {metrics.get('grid_similarity', 0):.2f} | "
                f"Time: {execution_time:.1f}s"
            )
            
            results[uid] = {
                "uid": uid,
                "success": True,
                "metrics": metrics,
                "execution_time": execution_time,
                "job_id": job_id
            }
            
            # publish to telemetry
            try:
                telemetry_client.publish(
                    "/validator/sandbox_job_completed",
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "block": current_block,
                        "uid": uid,
                        "job_id": job_id,
                        "execution_time": execution_time,
                        "metrics": {
                            "exact_match": metrics.get("exact_match"),
                            "partial_correctness": metrics.get("partial_correctness"),
                            "grid_similarity": metrics.get("grid_similarity"),
                            "efficiency_score": metrics.get("efficiency_score"),
                        }
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to publish telemetry: {e}")
            
        else:
            # job failed or timed out
            error_msg = final_status.get("error_message", f"Job status: {final_status.get('status')}")
            
            await db.record_query_result(
                block=current_block,
                uid=uid,
                success=False,
                response=None,
                error=error_msg,
                response_time=execution_time,
                ts=datetime.now(timezone.utc).replace(tzinfo=None),
                exact_match=False,
                partial_correctness=0.0,
                grid_similarity=0.0,
                efficiency_score=0.0,
                problem_id=None
            )
            
            logger.error(f"UID {uid} | Job {job_id} failed: {error_msg}")
            
            results[uid] = {
                "uid": uid,
                "success": False,
                "error": error_msg,
                "job_id": job_id
            }
    
    # poll all jobs concurrently
    logger.info(f"Polling {len(job_submissions)} jobs (timeout: {config.sandbox_runner_timeout_hours}h)...")
    
    tasks = [
        poll_and_record(uid, job_info)
        for uid, job_info in job_submissions.items()
    ]
    
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # log summary
    successful = sum(1 for r in results.values() if r.get("success"))
    failed = len(results) - successful
    
    logger.info(f"Sandbox runner cycle complete: {successful} successful, {failed} failed")
    
    return results