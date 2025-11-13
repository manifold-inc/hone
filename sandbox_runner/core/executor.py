"""
Executor Module

1. Repository cloning
2. Docker image building
3. Input data download
4. Prep phase execution (with internet)
5. Inference phase execution (without internet)
6. Output data upload
7. Cleanup

multiple execution modes with fallback chain:
- docker+gvisor (most secure)
- docker (secure)
- direct (fallback)
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import tempfile

from core.job_queue import Job, JobStatus
from core.gpu_pool import GPUPoolManager
from config import Config
from utils.s3 import S3Manager, S3TransferError
from utils.validation import RepositoryValidator, ValidationError
from utils.metrics import calculate_detailed_metrics
from security.network import NetworkPolicy, IptablesNetworkPolicy
from execution.docker_gvisor import DockerGVisorExecutor
from execution.docker_only import DockerOnlyExecutor
from execution.direct import DirectExecutor

logger = logging.getLogger(__name__)


class ExecutorError(Exception):
    """Raised when job execution fails."""
    pass


class Executor:
    """
    Main executor that orchestrates job execution
    
    lifecycle:
    CLONING → BUILDING → PREP → INFERENCE → COMPLETED/FAILED    
    """
    
    def __init__(self, config: Config, gpu_pool: GPUPoolManager):
        """
        Initialize executor with configuration and dependencies
        
        Args:
            config: Application configuration
            gpu_pool: GPU pool manager for resource tracking
        """
        self.config = config
        self.gpu_pool = gpu_pool
        
        self.s3_manager = S3Manager(config.storage)
        self.validator = RepositoryValidator(config.execution.allowed_repo_hosts)
        self.network_policy = NetworkPolicy(config.security.network_policy)
        
        self.execution_mode = config.execution.mode
        self.fallback_enabled = config.execution.fallback_on_error
        
        self._init_executors()
        
        self._active_jobs: Dict[str, Job] = {}
        self._job_tasks: Dict[str, asyncio.Task] = {}  # job_id -> execution task

        if config.security.network_policy and self._check_iptables():
            self.network_policy = IptablesNetworkPolicy(config.security.network_policy)
            logger.info("Using IptablesNetworkPolicy for enhanced network control")
        else:
            self.network_policy = NetworkPolicy(config.security.network_policy)
            logger.info("Using basic NetworkPolicy (Docker network modes only)")

        
        logger.info(
            f"Executor initialized (mode={self.execution_mode}, "
            f"fallback={self.fallback_enabled})"
        )
    
    def _init_executors(self):
        """Initialize execution mode handlers."""
        self.docker_gvisor_executor = None
        self.docker_executor = None
        self.direct_executor = None
        
        if "gvisor" in self.execution_mode:
            try:
                self.docker_gvisor_executor = DockerGVisorExecutor(self.config)
                if self.docker_gvisor_executor.is_available():
                    logger.info("Docker+gVisor executor available")
                else:
                    logger.warning("Docker+gVisor executor not available")
                    self.docker_gvisor_executor = None
            except Exception as e:
                logger.warning(f"Failed to initialize gVisor executor: {e}")
        
        try:
            self.docker_executor = DockerOnlyExecutor(self.config)
            if self.docker_executor.is_available():
                logger.info("Docker-only executor available")
            else:
                logger.warning("Docker executor not available")
                self.docker_executor = None
        except Exception as e:
            logger.warning(f"Failed to initialize Docker executor: {e}")
        
        try:
            self.direct_executor = DirectExecutor(self.config)
            logger.info("Direct executor available")
        except Exception as e:
            logger.warning(f"Failed to initialize direct executor: {e}")

    def _check_iptables(self) -> bool:
        """Check if iptables is available"""
        import subprocess
        try:
            result = subprocess.run(
                ['which', 'iptables'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def execute_job(self, job: Job) -> None:
        """
        Execute a complete job with all phases.
        
        This is the main entry point for job execution.
        
        Args:
            job: Job to execute
            
        Raises:
            ExecutorError: If job execution fails
        """
        job_id = job.job_id
        logger.info(f"Starting job execution with vLLM: {job_id}")
        
        self._active_jobs[job_id] = job        
        work_dir = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))
        
        try:
            # Clone repository
            job.status = JobStatus.CLONING
            job.current_phase = "cloning"
            job.progress_percentage = 10.0
            repo_path = await self._clone_repository(job, work_dir)

            if job.repo_path:
                work_dir_path = repo_path / job.repo_path
                logger.info(f"Using subdirectory: {work_dir_path}")
                logger.info(f"Subdirectory exists: {work_dir_path.exists()}")
                if work_dir_path.exists():
                    logger.info(f"Files in subdirectory: {list(work_dir_path.iterdir())}")
                
                if not work_dir_path.exists():
                    raise ExecutorError(f"Specified repo path does not exist: {job.repo_path}")
            else:
                work_dir_path = repo_path
                logger.info(f"Using repo root: {work_dir_path}")

            work_dir_path = repo_path / job.repo_path if job.repo_path else repo_path
            if job.repo_path and not work_dir_path.exists():
                raise ExecutorError(f"Specified repo path does not exist: {job.repo_path}")

            # Validate repository
            self.validator.validate_all(work_dir_path, job.repo_url)
            
            # Build Docker image
            job.status = JobStatus.BUILDING
            job.current_phase = "building"
            job.progress_percentage = 30.0
            image_id = await self._build_docker_image(job, work_dir_path)

            if isinstance(self.network_policy, IptablesNetworkPolicy):
                # start monitoring in background
                monitor_task = asyncio.create_task(
                    self.network_policy.monitor_connections(job.job_id, 60)
                )
            
            # Download input data
            job.current_phase = "downloading_input"
            job.progress_percentage = 45.0
            await self._download_input_data(job, work_dir)
            
            # Run complete vLLM pipeline (prep -> vllm -> inference)
            job.status = JobStatus.PREP
            job.current_phase = "vllm_pipeline"
            job.progress_percentage = 50.0
            
            executor = self._get_executor()
            
            # Check if executor supports vLLM pipeline
            if hasattr(executor, 'run_job_with_vllm'):
                exit_code, stdout, stderr = await executor.run_job_with_vllm(
                    image_id=image_id,
                    job=job,
                    work_dir=work_dir,
                    prep_timeout=self.config.execution.prep_timeout_seconds,
                    inference_timeout=self.config.execution.inference_timeout_seconds
                )
                
                if exit_code != 0:
                    raise ExecutorError(f"vLLM pipeline failed with exit code {exit_code}")
            else:
                # Fallback to old method
                logger.warning("Executor doesn't support vLLM pipeline, using legacy method")
                
                job.status = JobStatus.PREP
                prep_success = await self._run_prep_phase(job, image_id, work_dir)
                if not prep_success:
                    raise ExecutorError("Prep phase failed")
                
                job.status = JobStatus.INFERENCE
                inference_success = await self._run_inference_phase(job, image_id, work_dir)
                if not inference_success:
                    raise ExecutorError("Inference phase failed")
                
            if isinstance(self.network_policy, IptablesNetworkPolicy):
                try:
                    connections = await asyncio.wait_for(monitor_task, timeout=5)
                    if connections:
                        logger.info(f"Detected {len(connections)} connection attempts during job execution")
                        # optionally save to job metadata
                        job.network_activity = connections
                except asyncio.TimeoutError:
                    monitor_task.cancel()
            
            job.current_phase = "calculating_metrics"
            job.progress_percentage = 90.0
            try:
                logger.info(f"Calculating metrics for job {job_id}")
                metrics = await self._calculate_job_metrics(job, work_dir)
                
                job.metrics = metrics
                
                metrics_path = work_dir / "output" / "metrics.json"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                logger.info(f"Metrics calculated: {metrics['aggregate']}")
                
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for job {job_id}: {e}")

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress_percentage = 100.0
            
            logger.info(f"Job completed successfully: {job_id}")
            
        except ValidationError as e:
            logger.error(f"Job validation failed: {job_id}: {e}")
            job.status = JobStatus.FAILED
            job.error_message = f"Validation failed: {str(e)}"
            job.completed_at = datetime.utcnow()
        
        except ExecutorError as e:
            logger.error(f"Job execution failed: {job_id}: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
        
        except asyncio.TimeoutError:
            logger.error(f"Job timeout: {job_id}")
            job.status = JobStatus.TIMEOUT
            job.error_message = "Job execution timeout"
            job.completed_at = datetime.utcnow()
        
        except Exception as e:
            logger.exception(f"Unexpected error during job execution: {job_id}: {e}")
            job.status = JobStatus.FAILED
            job.error_message = f"Unexpected error: {str(e)}"
            job.completed_at = datetime.utcnow()
        
        finally:
            # Cleanup
            await self._cleanup(job, work_dir, image_id if 'image_id' in locals() else None)
            
            # Remove from active jobs and task tracking
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
            if job_id in self._job_tasks:
                del self._job_tasks[job_id]
    
    async def _clone_repository(self, job: Job, work_dir: Path) -> Path:
        """
        Clone Git repository
        
        Args:
            job: Job object with repository information
            work_dir: Working directory for job
            
        Returns:
            Path to cloned repository
            
        Raises:
            ExecutorError: If clone fails
        """
        repo_path = work_dir / "repo"
        
        logger.info(f"Cloning repository: {job.repo_url}")
        
        clone_cmd = [
            "git", "clone",
            "--depth", "1",
            "--branch", job.repo_branch,
        ]
        
        if job.repo_commit:
            clone_cmd = ["git", "clone", job.repo_url, str(repo_path)]
        else:
            clone_cmd.extend([job.repo_url, str(repo_path)])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *clone_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.execution.repo_clone_timeout_seconds
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise ExecutorError(f"Git clone failed: {error_msg}")
            
            if job.repo_commit:
                checkout_cmd = ["git", "checkout", job.repo_commit]
                process = await asyncio.create_subprocess_exec(
                    *checkout_cmd,
                    cwd=repo_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30
                )
                
                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    raise ExecutorError(f"Git checkout failed: {error_msg}")
            
            logger.info(f"Repository cloned successfully to {repo_path}")
            return repo_path
            
        except asyncio.TimeoutError:
            raise ExecutorError(
                f"Repository clone timeout after "
                f"{self.config.execution.repo_clone_timeout_seconds} seconds"
            )
        except Exception as e:
            raise ExecutorError(f"Failed to clone repository: {e}")
    
    async def _build_docker_image(self, job: Job, repo_path: Path) -> str:
        """
        Build Docker image from repository
        
        Args:
            job: Job object
            repo_path: Path to cloned repository
            
        Returns:
            Docker image ID
            
        Raises:
            ExecutorError: If build fails
        """
        logger.info(f"Building Docker image for job {job.job_id}")
        
        executor = self._get_executor()
        
        try:
            image_id = await executor.build_image(
                job_id=job.job_id,
                repo_path=repo_path,
                timeout_seconds=self.config.execution.repo_build_timeout_seconds
            )
            
            logger.info(f"Docker image built successfully: {image_id[:12]}")
            return image_id
            
        except Exception as e:
            raise ExecutorError(f"Failed to build Docker image: {e}")
    
    async def _download_input_data(self, job: Job, work_dir: Path):
        """
        Download input data from S3
        
        Args:
            job: Job object with S3 paths
            work_dir: Working directory
            
        Raises:
            ExecutorError: If download fails
        """
        if not job.input_s3_path:
            logger.debug("No input data path specified, skipping download")
            return
        
        input_dir = work_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading input data from {job.input_s3_path}")
        
        try:
            success = await self.s3_manager.download_directory(
                s3_prefix=job.input_s3_path,
                local_dir=input_dir
            )
            
            if not success:
                success = await self.s3_manager.download_input_data(
                    s3_path=job.input_s3_path,
                    local_path=input_dir / "input_data"
                )
            
            if not success:
                raise ExecutorError("Failed to download input data")
            
            logger.info("Input data downloaded successfully")
            
        except S3TransferError as e:
            raise ExecutorError(f"S3 download failed: {e}")
    
    async def _run_prep_phase(
        self,
        job: Job,
        image_id: str,
        work_dir: Path
    ) -> bool:
        """
        Run prep phase with internet access
        
        Args:
            job: Job object
            image_id: Docker image ID
            work_dir: Working directory
            
        Returns:
            True if prep phase succeeded
        """
        logger.info(f"Running prep phase for job {job.job_id}")
        
        executor = self._get_executor()
        
        try:
            exit_code, stdout, stderr = await executor.run_container(
                image_id=image_id,
                job=job,
                phase="prep",
                network_enabled=True,  # Internet access for prep
                work_dir=work_dir,
                timeout_seconds=self.config.execution.prep_timeout_seconds
            )
            
            # Log output
            if stdout:
                logger.debug(f"Prep stdout:\n{stdout[:1000]}")
            if stderr:
                logger.debug(f"Prep stderr:\n{stderr[:1000]}")
            
            if exit_code != 0:
                logger.error(f"Prep phase failed with exit code {exit_code}")
                job.error_message = f"Prep phase exit code: {exit_code}"
                return False
            
            logger.info("Prep phase completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Prep phase error: {e}")
            logger.exception(e)
            job.error_message = f"Prep phase error: {str(e)}"
            return False
    
    async def _run_inference_phase(
        self,
        job: Job,
        image_id: str,
        work_dir: Path
    ) -> bool:
        """
        Run inference phase without internet access.
        
        Args:
            job: Job object
            image_id: Docker image ID
            work_dir: Working directory
            
        Returns:
            True if inference phase succeeded
        """
        logger.info(f"Running inference phase for job {job.job_id}")
        
        executor = self._get_executor()
        
        try:
            exit_code, stdout, stderr = await executor.run_container(
                image_id=image_id,
                job=job,
                phase="inference",
                network_enabled=False,  # No internet for inference
                work_dir=work_dir,
                timeout_seconds=self.config.execution.inference_timeout_seconds
            )
            
            # Log output
            if stdout:
                logger.debug(f"Inference stdout:\n{stdout[:1000]}")
            if stderr:
                logger.debug(f"Inference stderr:\n{stderr[:1000]}")
            
            if exit_code != 0:
                logger.error(f"Inference phase failed with exit code {exit_code}")
                job.error_message = f"Inference phase exit code: {exit_code}"
                return False
            
            logger.info("Inference phase completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Inference phase error: {e}")
            job.error_message = f"Inference phase error: {str(e)}"
            return False
    
    async def _upload_output_data(self, job: Job, work_dir: Path):
        """
        Upload output data to S3.
        
        Args:
            job: Job object with S3 paths
            work_dir: Working directory
            
        Raises:
            ExecutorError: If upload fails
        """
        output_dir = work_dir / "output"
        
        if not output_dir.exists() or not list(output_dir.iterdir()):
            logger.warning("No output data found, skipping upload")
            return
        
        logger.info(f"Uploading output data to {job.output_s3_path}")
        
        try:
            success = await self.s3_manager.upload_directory(
                local_dir=output_dir,
                s3_prefix=job.output_s3_path
            )
            
            if not success:
                raise ExecutorError("Failed to upload output data")
            
            logger.info("Output data uploaded successfully")
            
        except S3TransferError as e:
            raise ExecutorError(f"S3 upload failed: {e}")
    
    async def _cleanup(
        self,
        job: Job,
        work_dir: Path,
        image_id: Optional[str] = None
    ):
        """
        Clean up job resources
        
        Args:
            job: Job object
            work_dir: Working directory to remove
            image_id: Docker image ID to remove (optional)
        """
        logger.info(f"Cleaning up job {job.job_id}")
        
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
                logger.debug(f"Removed work directory: {work_dir}")
            
            if image_id:
                executor = self._get_executor()
                await executor.remove_image(image_id)
            
        except Exception as e:
            logger.warning(f"Cleanup error for job {job.job_id}: {e}")
    
    def _get_executor(self):
        """
        Get appropriate executor based on configuration and availability
        
        Returns:
            Executor instance
            
        Raises:
            ExecutorError: If no executor is available
        """
        if self.execution_mode == "docker+gvisor" and self.docker_gvisor_executor:
            return self.docker_gvisor_executor
        
        if self.fallback_enabled and self.docker_executor:
            logger.warning("Falling back to docker-only mode")
            return self.docker_executor
        
        if self.fallback_enabled and self.direct_executor:
            logger.warning("Falling back to direct execution mode")
            return self.direct_executor
        
        raise ExecutorError("No execution mode available")
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get status of an active job
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status or None if not found
        """
        job = self._active_jobs.get(job_id)
        return job.status if job else None
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get active job object
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job object or None if not found
        """
        return self._active_jobs.get(job_id)
        
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel an active job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled
        """
        job = self._active_jobs.get(job_id)
        if not job:
            return False
        
        logger.info(f"Cancelling job: {job_id}")
        
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        
        task = self._job_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Cancelled execution task for job {job_id}")
            except Exception as e:
                logger.warning(f"Error while cancelling job {job_id}: {e}")
        
        return True
    
    def get_active_jobs(self) -> Dict[str, Job]:
        """
        Get all active jobs
        
        Returns:
            Dictionary of job_id to Job object
        """
        return self._active_jobs.copy()
    
    async def _calculate_job_metrics(self, job: Job, work_dir: Path) -> Dict:
        """
        Calculate metrics for a completed inference job
        
        Args:
            job: Job object
            work_dir: Working directory with results
            
        Returns:
            Dictionary with calculated metrics
        """
        results_file = work_dir / "output" / "results.json"
        
        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            return {"error": "Results file not found"}
        
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            
            if results_data.get("phase") != "inference":
                return {"error": "Not an inference job"}
            
            if results_data.get("status") != "success":
                return {"error": f"Inference failed: {results_data.get('error')}"}
            
            metrics = calculate_detailed_metrics(results_data)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {"error": str(e)}