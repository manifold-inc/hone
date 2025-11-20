import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from loguru import logger


class SandboxRunnerClient:
    """Client for interacting with Sandbox Runner API"""
    
    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    async def submit_job(
        self,
        repo_url: str,
        repo_branch: str,
        repo_commit: Optional[str],
        repo_path: str,
        weight_class: str,
        miner_hotkey: str,
        validator_hotkey: str,
        priority: int = 5,
        use_vllm: bool = False,
        vllm_config: Optional[Dict] = None,
        custom_env_vars: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Submit a job to sandbox runner
        
        Returns:
            {
                "job_id": str,
                "status": str,
                "queue_position": int,
                "estimated_start_time": str (ISO)
            }
        """
        url = f"{self.endpoint}/v1/jobs/submit"
        
        payload = {
            "repo_url": repo_url,
            "repo_branch": repo_branch,
            "repo_commit": repo_commit,
            "repo_path": repo_path,
            "weight_class": weight_class,
            "miner_hotkey": miner_hotkey,
            "validator_hotkey": validator_hotkey,
            "priority": priority,
            "use_vllm": use_vllm,
            "vllm_config": vllm_config,
            "custom_env_vars": custom_env_vars or {}
        }
        
        # remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 201:
                        text = await resp.text()
                        logger.error(f"Failed to submit job: {resp.status} - {text}")
                        raise Exception(f"Job submission failed: {resp.status}")
                    
                    return await resp.json()
        except asyncio.TimeoutError:
            logger.error("Timeout submitting job to sandbox runner")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Network error submitting job: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status
        
        Returns:
            {
                "job_id": str,
                "status": str,  # pending, cloning, building, prep, inference, completed, failed, timeout
                "current_phase": str,
                "progress_percentage": float,
                "started_at": str,
                "completed_at": str,
                "error_message": str
            }
        """
        url = f"{self.endpoint}/v1/jobs/{job_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 404:
                        return None
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Failed to get job status: {resp.status} - {text}")
                        return None
                    
                    return await resp.json()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting status for job {job_id}")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Network error getting job status: {e}")
            return None
    
    async def get_job_metrics(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job metrics (only available after completion)
        
        Returns:
            {
                "job_id": str,
                "status": str,
                "metrics": {
                    "exact_match": bool,
                    "partial_correctness": float,
                    "grid_similarity": float,
                    "efficiency_score": float,
                    "problem_id": str,
                    "base_task_num": int,
                    "chain_length": int,
                    "transformation_chain": [...],
                    "num_train_examples": int
                },
                "completed_at": str
            }
        """
        url = f"{self.endpoint}/v1/jobs/{job_id}/metrics"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 404:
                        logger.debug(f"Metrics not found for job {job_id} (may not be ready yet)")
                        return None
                    if resp.status == 400:
                        text = await resp.text()
                        logger.debug(f"Metrics not available for job {job_id}: {text}")
                        return None
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(f"Failed to get metrics: {resp.status} - {text}")
                        return None
                    
                    return await resp.json()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting metrics for job {job_id}")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Network error getting metrics: {e}")
            return None
    
    async def poll_until_complete(
        self,
        job_id: str,
        poll_interval: int = 30,
        max_attempts: int = 360,
        on_status_change: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Poll job until completion with timeout
        
        Args:
            job_id: Job identifier
            poll_interval: Seconds between polls
            max_attempts: Maximum poll attempts (default 360 * 30s = 3 hours)
            on_status_change: Optional callback when status changes: fn(job_id, status, full_status_dict)
        
        Returns:
            Final job status dict with 'status' field indicating outcome
        """
        last_status = None
        last_phase = None
        
        for attempt in range(max_attempts):
            status_data = await self.get_job_status(job_id)
            
            if not status_data:
                logger.error(f"Job {job_id} not found during polling (attempt {attempt + 1}/{max_attempts})")
                await asyncio.sleep(poll_interval)
                continue
            
            current_status = status_data.get("status")
            current_phase = status_data.get("current_phase")
            
            # log status or phase changes
            if current_status != last_status or current_phase != last_phase:
                progress = status_data.get("progress_percentage", 0)
                logger.info(
                    f"Job {job_id} | Status: {current_status} | Phase: {current_phase} | Progress: {progress:.1f}%"
                )
                
                if on_status_change:
                    try:
                        on_status_change(job_id, current_status, status_data)
                    except Exception as e:
                        logger.warning(f"Error in status change callback: {e}")
                
                last_status = current_status
                last_phase = current_phase
            
            # terminal states
            if current_status in ["completed", "failed", "timeout", "cancelled"]:
                logger.info(f"Job {job_id} finished with status: {current_status}")
                return status_data
            
            await asyncio.sleep(poll_interval)
        
        # polling timeout
        logger.error(f"Job {job_id} polling timed out after {max_attempts} attempts ({max_attempts * poll_interval}s)")
        return {
            "job_id": job_id,
            "status": "timeout",
            "error_message": f"Polling timeout after {max_attempts * poll_interval}s"
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job"""
        url = f"{self.endpoint}/v1/jobs/{job_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 404:
                        logger.warning(f"Job {job_id} not found for cancellation")
                        return False
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Failed to cancel job: {resp.status} - {text}")
                        return False
                    
                    logger.info(f"Job {job_id} cancelled successfully")
                    return True
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False