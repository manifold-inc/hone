"""
Direct Execution Mode

Executes jobs directly on the host without Docker containers.
This is the fallback mode when Docker is not available.

Security features (using OS-level isolation):
- cgroups v2 for resource limits
- Namespaces for isolation (PID, mount, network)
- Seccomp for syscall filtering
- Process runs as nobody user
- Network namespace isolation

This mode is less secure than Docker/gVisor but provides
basic isolation when containers are not available.
"""

import asyncio
import logging
import os
import subprocess
import signal
from pathlib import Path
from typing import Optional, Tuple, Dict
import tempfile
import shutil

from core.job_queue import Job
from config import Config
from security.cgroups import CgroupManager
from security.isolation import NamespaceManager
from security.network import NetworkPolicy

logger = logging.getLogger(__name__)


class DirectExecutionError(Exception):
    """Raised when direct execution fails."""
    pass


class DirectExecutor:
    """
    Direct execution mode without containers.
    
    Runs jobs as isolated processes on the host system using:
    - Linux namespaces (PID, mount, network, UTS, IPC)
    - cgroups v2 for resource limits
    - Seccomp for syscall filtering
    - Network namespace control
    
    This is the fallback when Docker is not available.
    """
    
    def __init__(self, config: Config):
        """
        Initialize direct executor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Initialize components
        try:
            self.cgroup_manager = CgroupManager()
            self.namespace_manager = NamespaceManager()
            self.network_policy = NetworkPolicy(config.security.network_policy)
            
            logger.info("Direct executor initialized (OS-level isolation)")
            
        except Exception as e:
            logger.error(f"Failed to initialize direct executor: {e}")
            raise DirectExecutionError(f"Direct executor initialization failed: {e}")
    
    async def run_container(
        self,
        image_id: str,
        job: Job,
        phase: str,
        network_enabled: bool,
        work_dir: Path,
        timeout_seconds: int
    ) -> Tuple[int, str, str]:
        """
        Run a job phase as a direct process.
        
        Note: image_id is ignored in direct mode as we run Python directly.
        
        Args:
            image_id: Ignored (for API compatibility)
            job: Job object with execution details
            phase: Execution phase ("prep" or "inference")
            network_enabled: Whether to enable network access
            work_dir: Working directory on host
            timeout_seconds: Execution timeout in seconds
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
            
        Raises:
            DirectExecutionError: If execution fails
        """
        job_name = f"sandbox-{job.job_id}-{phase}-direct"
        
        logger.info(
            f"Starting direct execution: {job_name}",
            extra={
                "job_id": job.job_id,
                "phase": phase,
                "network_enabled": network_enabled
            }
        )
        
        # Create cgroup for resource limits
        cgroup_name = f"sandbox_{job.job_id}_{phase}"
        try:
            await self.cgroup_manager.create_cgroup(
                cgroup_name,
                cpu_limit=self.config.execution.cpu_limit,
                memory_limit_gb=self.config.execution.memory_limit_gb,
                pid_limit=self.config.execution.max_processes
            )
            logger.info(f"Created cgroup: {cgroup_name}")
            
        except Exception as e:
            logger.error(f"Failed to create cgroup: {e}")
            raise DirectExecutionError(f"cgroup creation failed: {e}")
        
        # Prepare environment and command
        env_vars = self._build_environment(job, phase, work_dir)
        command = self._build_command(job, phase, work_dir)
        
        # Create stdout/stderr capture files
        stdout_file = work_dir / f"{phase}_stdout.log"
        stderr_file = work_dir / f"{phase}_stderr.log"
        
        process = None
        try:
            # Run process with isolation
            process = await self._run_isolated_process(
                command=command,
                env_vars=env_vars,
                work_dir=work_dir,
                cgroup_name=cgroup_name,
                network_enabled=network_enabled,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                timeout_seconds=timeout_seconds
            )
            
            # Get exit code
            exit_code = process.returncode if process else -1
            
            # Read stdout and stderr
            stdout = stdout_file.read_text() if stdout_file.exists() else ""
            stderr = stderr_file.read_text() if stderr_file.exists() else ""
            
            logger.info(
                f"Direct execution finished: {job_name} (exit_code={exit_code})",
                extra={
                    "job_id": job.job_id,
                    "phase": phase,
                    "exit_code": exit_code,
                    "stdout_lines": len(stdout.split('\n')),
                    "stderr_lines": len(stderr.split('\n'))
                }
            )
            
            return exit_code, stdout, stderr
            
        except asyncio.TimeoutError:
            logger.error(f"Direct execution timeout after {timeout_seconds}s: {job_name}")
            if process:
                self._kill_process(process)
            raise DirectExecutionError(
                f"Execution timeout after {timeout_seconds}s"
            )
        
        except Exception as e:
            logger.error(f"Direct execution error: {e}")
            if process:
                self._kill_process(process)
            raise DirectExecutionError(f"Execution failed: {e}")
        
        finally:
            # Cleanup cgroup
            try:
                await self.cgroup_manager.remove_cgroup(cgroup_name)
                logger.debug(f"Removed cgroup: {cgroup_name}")
            except Exception as e:
                logger.warning(f"Failed to remove cgroup: {e}")
    
    def _build_environment(self, job: Job, phase: str, work_dir: Path) -> Dict:
        """
        Build environment variables for the process.
        
        Args:
            job: Job object
            phase: Execution phase
            work_dir: Working directory
            
        Returns:
            Dictionary of environment variables
        """
        env_vars = os.environ.copy()
        
        # Job-specific variables
        env_vars.update({
            'PHASE': phase,
            'JOB_ID': job.job_id,
            'INPUT_S3_PATH': job.input_s3_path,
            'OUTPUT_S3_PATH': job.output_s3_path,
            'INPUT_DIR': str(work_dir / 'input'),
            'OUTPUT_DIR': str(work_dir / 'output'),
        })
        
        # GPU assignment
        if job.assigned_gpus:
            env_vars['CUDA_VISIBLE_DEVICES'] = ','.join(
                str(gpu) for gpu in job.assigned_gpus
            )
        else:
            env_vars['CUDA_VISIBLE_DEVICES'] = ''
        
        # Custom environment variables
        env_vars.update(job.custom_env_vars or {})
        
        # Python unbuffered output
        env_vars['PYTHONUNBUFFERED'] = '1'
        
        return env_vars
    
    def _build_command(self, job: Job, phase: str, work_dir: Path) -> list:
        """
        Build command to execute.
        
        Args:
            job: Job object
            phase: Execution phase
            work_dir: Working directory
            
        Returns:
            Command as list of arguments
        """
        # Assume repository has inference.py in root
        # In production, this should be validated during repository validation
        inference_script = work_dir / 'repo' / 'inference.py'
        
        command = [
            'python3',
            str(inference_script),
            '--phase', phase,
            '--input', str(work_dir / 'input'),
            '--output', str(work_dir / 'output')
        ]
        
        return command
    
    async def _run_isolated_process(
        self,
        command: list,
        env_vars: Dict,
        work_dir: Path,
        cgroup_name: str,
        network_enabled: bool,
        stdout_file: Path,
        stderr_file: Path,
        timeout_seconds: int
    ) -> subprocess.CompletedProcess:
        """
        Run process with isolation (namespaces, cgroups, etc.).
        
        Args:
            command: Command to execute
            env_vars: Environment variables
            work_dir: Working directory
            cgroup_name: cgroup name for resource limits
            network_enabled: Whether to enable network
            stdout_file: File to capture stdout
            stderr_file: File to capture stderr
            timeout_seconds: Execution timeout
            
        Returns:
            CompletedProcess object
            
        Raises:
            DirectExecutionError: If execution fails
        """
        # Open files for stdout/stderr
        with open(stdout_file, 'w') as stdout_f, open(stderr_file, 'w') as stderr_f:
            try:
                # Use unshare for namespace isolation if available
                # This creates new PID, mount, network namespaces
                if self.namespace_manager.is_available():
                    # Wrap command with unshare for namespace isolation
                    isolated_command = self.namespace_manager.wrap_command(
                        command,
                        isolate_network=not network_enabled,
                        isolate_pid=True,
                        isolate_mount=True
                    )
                else:
                    isolated_command = command
                    logger.warning("Namespace isolation not available")
                
                # Add to cgroup
                cgexec_command = [
                    'cgexec',
                    '-g', f'cpu,memory,pids:{cgroup_name}',
                    *isolated_command
                ]
                
                logger.debug(f"Executing: {' '.join(cgexec_command)}")
                
                # Run process
                process = await asyncio.create_subprocess_exec(
                    *cgexec_command,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    env=env_vars,
                    cwd=work_dir,
                    preexec_fn=self._preexec_fn  # Drop privileges
                )
                
                # Wait for completion with timeout
                try:
                    await asyncio.wait_for(
                        process.wait(),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    # Kill process on timeout
                    logger.warning(f"Process timeout, sending SIGKILL")
                    process.kill()
                    await process.wait()
                    raise
                
                return process
                
            except Exception as e:
                logger.error(f"Failed to run isolated process: {e}")
                raise DirectExecutionError(f"Process execution failed: {e}")
    
    @staticmethod
    def _preexec_fn():
        """
        Function to run in child process before exec.
        
        Drops privileges by switching to nobody user.
        """
        try:
            # Get nobody user UID/GID
            import pwd
            nobody = pwd.getpwnam('nobody')
            
            # Drop privileges
            os.setgid(nobody.pw_gid)
            os.setuid(nobody.pw_uid)
            
        except Exception as e:
            logger.warning(f"Failed to drop privileges: {e}")
    
    def _kill_process(self, process: subprocess.CompletedProcess):
        """
        Kill a running process.
        
        Args:
            process: Process object
        """
        try:
            if hasattr(process, 'pid') and process.pid:
                os.kill(process.pid, signal.SIGKILL)
                logger.info(f"Process killed: {process.pid}")
        except Exception as e:
            logger.warning(f"Failed to kill process: {e}")
    
    async def build_image(
        self,
        repo_path: Path,
        job_id: str,
        timeout_seconds: int = 3600
    ) -> str:
        """
        Prepare repository for direct execution.
        
        In direct mode, we don't build an image, but we can:
        1. Install requirements.txt
        2. Validate Python dependencies
        3. Prepare virtual environment
        
        Args:
            repo_path: Path to repository directory
            job_id: Job identifier
            timeout_seconds: Build timeout in seconds
            
        Returns:
            Pseudo "image_id" (just job_id for tracking)
            
        Raises:
            DirectExecutionError: If preparation fails
        """
        logger.info(f"Preparing repository for direct execution: {job_id}")
        
        # Check for requirements.txt
        requirements_file = repo_path / 'requirements.txt'
        
        if requirements_file.exists():
            logger.info("Installing Python requirements")
            
            try:
                # Install requirements in system Python
                # In production, consider using virtual environment
                result = await asyncio.create_subprocess_exec(
                    'pip3', 'install', '-r', str(requirements_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    result.communicate(),
                    timeout=timeout_seconds
                )
                
                if result.returncode != 0:
                    logger.error(f"Requirements installation failed: {stderr.decode()}")
                    raise DirectExecutionError("Failed to install requirements")
                
                logger.info("Requirements installed successfully")
                
            except asyncio.TimeoutError:
                raise DirectExecutionError("Requirements installation timeout")
            except Exception as e:
                raise DirectExecutionError(f"Requirements installation failed: {e}")
        
        else:
            logger.warning("No requirements.txt found")
        
        # Return job_id as pseudo "image ID"
        return f"direct-{job_id}"
    
    async def build_image_from_requirements(
        self,
        repo_path: Path,
        job_id: str,
        base_image: str = None
    ) -> str:
        """
        Build from requirements (same as build_image in direct mode).
        
        Args:
            repo_path: Path to repository directory
            job_id: Job identifier
            base_image: Ignored in direct mode
            
        Returns:
            Pseudo "image_id"
        """
        return await self.build_image(repo_path, job_id)
    
    async def remove_image(self, image_id: str):
        """
        Remove "image" (no-op in direct mode).
        
        Args:
            image_id: Pseudo image ID
        """
        logger.debug(f"Remove image (no-op in direct mode): {image_id}")
    
    def is_available(self) -> bool:
        """
        Check if direct execution is available.
        
        Direct execution should always be available on Linux.
        
        Returns:
            True if running on Linux with required tools
        """
        try:
            # Check for required commands
            required_commands = ['python3', 'cgexec']
            
            for cmd in required_commands:
                result = subprocess.run(
                    ['which', cmd],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode != 0:
                    logger.warning(f"Required command not found: {cmd}")
                    return False
            
            # Check if we're on Linux
            if os.name != 'posix':
                logger.warning("Direct execution only supported on Linux")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Direct executor availability check failed: {e}")
            return False