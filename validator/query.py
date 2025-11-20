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
    Query miners by submitting jobs to sandbox runner with submission history and daily limits
        
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
    
    logger.info(f"=" * 80)
    logger.info(f"Checking daily submission limits for {len(miners)} miners...")
    logger.info(f"Max submissions per day: {config.max_submissions_per_day}")
    logger.info(f"=" * 80)
    
    # filter out miners who have hit daily limit
    eligible_miners = {}
    for uid, miner in miners.items():
        hotkey = miner.get("hotkey")
        if not hotkey:
            logger.warning(f"UID {uid}: No hotkey found, skipping")
            continue
        
        can_submit, current_count = await db.check_daily_submission_limit(
            hotkey, config.max_submissions_per_day
        )
        
        if not can_submit:
            logger.info(f"UID {uid} ({hotkey[:12]}...): Daily limit reached ({current_count}/{config.max_submissions_per_day})")
            continue
        
        eligible_miners[uid] = miner
        logger.debug(f"UID {uid} ({hotkey[:12]}...): Eligible ({current_count}/{config.max_submissions_per_day} used)")
    
    if not eligible_miners:
        logger.warning("No miners eligible for submission (all hit daily limits)")
        return {}
    
    logger.info(f"Fetching /info from {len(eligible_miners)} eligible miners...")
    
    # fetch miner info concurrently
    miner_infos = {}
    timeout = aiohttp.ClientTimeout(total=config.miner_info_timeout_seconds)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            fetch_miner_info(session, uid, miner, config.default_miner_port, config.miner_info_timeout_seconds)
            for uid, miner in eligible_miners.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for uid, result in zip(eligible_miners.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"UID {uid}: Exception fetching info: {result}")
            elif result:
                miner_infos[uid] = result
    
    logger.info(f"Successfully fetched info from {len(miner_infos)}/{len(eligible_miners)} miners")
    
    # check submission history and decide what to evaluate
    jobs_to_submit = {}
    cached_results = {}
    
    for uid, info in miner_infos.items():
        miner = eligible_miners[uid]
        hotkey = miner.get("hotkey")
        
        # check if identical solution already evaluated
        history = await db.get_submission_history(
            hotkey=hotkey,
            repo_url=info["repo_url"],
            repo_branch=info.get("repo_branch", "main"),
            repo_commit=info.get("repo_commit"),
            repo_path=info.get("repo_path", ""),
            weight_class=info["weight_class"]
        )
        
        if history:
            # use cached metrics
            logger.info(f"UID {uid} ({hotkey[:12]}...): Using cached metrics (evaluated {history['evaluation_count']} times)")
            
            cached_results[uid] = {
                "uid": uid,
                "hotkey": hotkey,
                "info": info,
                "from_cache": True,
                "exact_match_rate": float(history["exact_match_rate"]),
                "partial_correctness_avg": float(history["partial_correctness_avg"]),
                "grid_similarity_avg": float(history["grid_similarity_avg"]),
                "efficiency_avg": float(history["efficiency_avg"]),
                "overall_score": float(history["overall_score"]),
                "last_evaluated_at": history["last_evaluated_at"]
            }
            
            # record to query_results with from_cache=True
            await db.record_query_result(
                block=current_block,
                uid=uid,
                hotkey=hotkey,
                success=True,
                response={"cached": True, "history_id": history["id"]},
                error=None,
                response_time=0.0,
                ts=datetime.now(timezone.utc).replace(tzinfo=None),
                exact_match=history["exact_match_rate"] == 1.0,
                partial_correctness=float(history["partial_correctness_avg"]),
                grid_similarity=float(history["grid_similarity_avg"]),
                efficiency_score=float(history["efficiency_avg"]),
                repo_url=info["repo_url"],
                repo_branch=info.get("repo_branch", "main"),
                repo_commit=info.get("repo_commit"),
                repo_path=info.get("repo_path", ""),
                weight_class=info["weight_class"],
                from_cache=True
            )
            
            # increment daily submission count
            await db.increment_daily_submissions(hotkey)
            
        else:
            # new solution - needs evaluation
            jobs_to_submit[uid] = {
                "miner": miner,
                "info": info
            }
    
    logger.info(f"Results: {len(cached_results)} from cache, {len(jobs_to_submit)} need evaluation")
    
    if not jobs_to_submit:
        logger.info("All submissions were cached - no sandbox jobs needed")
        return cached_results
    
    # submit new jobs to sandbox runner
    job_submissions = {}
    
    for uid, job_data in jobs_to_submit.items():
        try:
            miner = job_data["miner"]
            info = job_data["info"]
            hotkey = miner.get("hotkey")
            
            logger.info(f"UID {uid}: Submitting NEW solution to sandbox runner")
            logger.debug(f"  Repo: {info['repo_url']}")
            logger.debug(f"  Branch: {info.get('repo_branch', 'main')}")
            logger.debug(f"  Weight: {info['weight_class']}")
            
            response = await sandbox_client.submit_job(
                repo_url=info["repo_url"],
                repo_branch=info.get("repo_branch", "main"),
                repo_commit=info.get("repo_commit"),
                repo_path=info.get("repo_path", ""),
                weight_class=info["weight_class"],
                miner_hotkey=hotkey,
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
                hotkey=miner.get("hotkey"),
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
        logger.info("No new jobs submitted (all failed)")
        return cached_results
    
    logger.info(f"Submitted {len(job_submissions)} new jobs to sandbox runner")
    
    # poll all jobs until completion
    fresh_results = {}
    
    async def poll_and_record(uid: int, job_info: Dict):
        """Poll a single job and record results to database"""
        job_id = job_info["job_id"]
        miner = job_info["miner"]
        info = job_info["info"]
        hotkey = miner.get("hotkey")
        
        start_time = datetime.now(timezone.utc)
        
        def on_status_change(job_id, status, full_status):
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
        
        # get metrics if job completed successfully
        metrics_data = None
        if final_status.get("status") == "completed":
            metrics_data = await sandbox_client.get_job_metrics(job_id)
        
        # record to database
        if metrics_data and metrics_data.get("metrics"):
            metrics = metrics_data["metrics"]
            
            # calculate aggregate metrics for this job
            exact_match = metrics.get("exact_match", False)
            partial_correctness = metrics.get("partial_correctness", 0.0)
            grid_similarity = metrics.get("grid_similarity", 0.0)
            efficiency_score = metrics.get("efficiency_score", 0.0)
            
            await db.record_query_result(
                block=current_block,
                uid=uid,
                hotkey=hotkey,
                success=True,
                response={"job_id": job_id, "sandbox_status": "completed"},
                error=None,
                response_time=execution_time,
                ts=datetime.now(timezone.utc).replace(tzinfo=None),
                exact_match=exact_match,
                partial_correctness=partial_correctness,
                grid_similarity=grid_similarity,
                efficiency_score=efficiency_score,
                problem_id=metrics.get("problem_id"),
                base_task_num=metrics.get("base_task_num"),
                chain_length=metrics.get("chain_length"),
                transformation_chain=metrics.get("transformation_chain"),
                num_train_examples=metrics.get("num_train_examples"),
                repo_url=info["repo_url"],
                repo_branch=info.get("repo_branch", "main"),
                repo_commit=info.get("repo_commit"),
                repo_path=info.get("repo_path", ""),
                weight_class=info["weight_class"],
                from_cache=False
            )
            
            # save to submission history for future caching
            await db.save_submission_history(
                hotkey=hotkey,
                repo_url=info["repo_url"],
                repo_branch=info.get("repo_branch", "main"),
                repo_commit=info.get("repo_commit"),
                repo_path=info.get("repo_path", ""),
                weight_class=info["weight_class"],
                use_vllm=info.get("use_vllm", False),
                vllm_config=info.get("vllm_config"),
                exact_match_rate=1.0 if exact_match else 0.0,
                partial_correctness_avg=partial_correctness,
                grid_similarity_avg=grid_similarity,
                efficiency_avg=efficiency_score,
                overall_score=partial_correctness  # will be recalculated in scoring
            )
            
            # increment daily submission count
            await db.increment_daily_submissions(hotkey)
            
            logger.info(
                f"UID {uid} | Job {job_id} completed | "
                f"Exact: {exact_match} | "
                f"Partial: {partial_correctness:.2f} | "
                f"Similarity: {grid_similarity:.2f} | "
                f"Time: {execution_time:.1f}s"
            )
            
            fresh_results[uid] = {
                "uid": uid,
                "hotkey": hotkey,
                "success": True,
                "from_cache": False,
                "exact_match_rate": 1.0 if exact_match else 0.0,
                "partial_correctness_avg": partial_correctness,
                "grid_similarity_avg": grid_similarity,
                "efficiency_avg": efficiency_score,
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
                            "exact_match": exact_match,
                            "partial_correctness": partial_correctness,
                            "grid_similarity": grid_similarity,
                            "efficiency_score": efficiency_score,
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
                hotkey=hotkey,
                success=False,
                response=None,
                error=error_msg,
                response_time=execution_time,
                ts=datetime.now(timezone.utc).replace(tzinfo=None),
                exact_match=False,
                partial_correctness=0.0,
                grid_similarity=0.0,
                efficiency_score=0.0,
                problem_id=None,
                repo_url=info["repo_url"],
                repo_branch=info.get("repo_branch", "main"),
                repo_commit=info.get("repo_commit"),
                repo_path=info.get("repo_path", ""),
                weight_class=info["weight_class"],
                from_cache=False
            )
            
            logger.error(f"UID {uid} | Job {job_id} failed: {error_msg}")
            
            fresh_results[uid] = {
                "uid": uid,
                "hotkey": hotkey,
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
    
    # combine cached and fresh results
    all_results = {**cached_results, **fresh_results}
    
    # log summary
    successful_fresh = sum(1 for r in fresh_results.values() if r.get("success"))
    failed_fresh = len(fresh_results) - successful_fresh
    
    logger.info(f"Sandbox runner cycle complete:")
    logger.info(f"  Cached: {len(cached_results)}")
    logger.info(f"  Fresh successful: {successful_fresh}")
    logger.info(f"  Fresh failed: {failed_fresh}")
    
    return all_results