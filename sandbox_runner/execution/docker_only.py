"""
Docker-Only Execution Mode

Executes jobs in Docker containers without gVisor.
This is the fallback mode when gVisor is not available.

Security features:
- Standard Docker isolation
- Resource limits (CPU, memory)
- Network control (host for prep, none for inference)
- GPU assignment via CUDA_VISIBLE_DEVICES
- Read-only root filesystem where possible
- Dropped capabilities
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict
import time

import docker
from docker.errors import DockerException, ImageNotFound, ContainerError

from core.job_queue import Job
from config import Config

logger = logging.getLogger(__name__)


class DockerExecutionError(Exception):
    """Raised when Docker execution fails."""
    pass


class DockerOnlyExecutor:
    """
    Docker-only container executor.
    
    Runs jobs in Docker containers with standard Docker isolation.
    Falls back to this mode if gVisor is not available.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Docker-only executor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized (docker-only mode)")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise DockerExecutionError(f"Docker not available: {e}")
    
    async def run_container(
        self,
        image_id: str,
        job: Job,
        phase: str,
        network_enabled: bool,
        work_dir: Path,
        timeout_seconds: int
    ) -> tuple[int, str, str]:
        """
        Run a container for a specific job phase.
        
        Args:
            image_id: Docker image ID or tag
            job: Job object with execution details
            phase: Execution phase ("prep" or "inference")
            network_enabled: Whether to enable network access
            work_dir: Working directory on host
            timeout_seconds: Execution timeout in seconds
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
            
        Raises:
            DockerExecutionError: If container execution fails
        """
        container_name = f"sandbox-{job.job_id}-{phase}"
        
        logger.info(
            f"Starting Docker container: {container_name}",
            extra={
                "job_id": job.job_id,
                "phase": phase,
                "image": image_id,
                "network_enabled": network_enabled
            }
        )
        
        # Prepare container configuration
        container_config = self._build_container_config(
            image_id=image_id,
            job=job,
            phase=phase,
            network_enabled=network_enabled,
            work_dir=work_dir,
            container_name=container_name
        )
        
        container = None
        try:
            # Create container
            container = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.containers.create(**container_config)
            )
            
            logger.info(f"Container created: {container.id[:12]}")
            
            # Start container
            await asyncio.get_event_loop().run_in_executor(
                None,
                container.start
            )
            
            logger.info(f"Container started: {container.id[:12]}")
            
            # Wait for container with timeout
            exit_code = await self._wait_for_container(container, timeout_seconds)
            
            # Get logs
            stdout, stderr = await self._get_container_logs(container)
            
            logger.info(
                f"Container finished: {container.id[:12]} (exit_code={exit_code})",
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
            logger.error(f"Container timeout after {timeout_seconds}s: {container_name}")
            if container:
                await self._kill_container(container)
            raise DockerExecutionError(f"Container timeout after {timeout_seconds}s")
        
        except ContainerError as e:
            logger.error(f"Container execution error: {e}")
            raise DockerExecutionError(f"Container execution failed: {e}")
        
        except DockerException as e:
            logger.error(f"Docker error: {e}")
            raise DockerExecutionError(f"Docker error: {e}")
        
        finally:
            # Clean up container
            if container:
                await self._cleanup_container(container)
    
    def _build_container_config(
        self,
        image_id: str,
        job: Job,
        phase: str,
        network_enabled: bool,
        work_dir: Path,
        container_name: str
    ) -> Dict:
        """
        Build Docker container configuration.
        
        Args:
            image_id: Docker image ID
            job: Job object
            phase: Execution phase
            network_enabled: Whether to enable network
            work_dir: Working directory on host
            container_name: Container name
            
        Returns:
            Dictionary with container configuration
        """
        # Prepare environment variables
        env_vars = {
            'PHASE': phase,
            'JOB_ID': job.job_id,
            'INPUT_S3_PATH': job.input_s3_path,
            'OUTPUT_S3_PATH': job.output_s3_path,
        }
        
        # Add GPU assignment
        if job.assigned_gpus:
            env_vars['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in job.assigned_gpus)
        else:
            env_vars['CUDA_VISIBLE_DEVICES'] = ''
        
        # Add custom environment variables
        env_vars.update(job.custom_env_vars or {})
        
        # Prepare command
        command = [
            'python', 'inference.py',
            '--phase', phase,
            '--input', '/input',
            '--output', '/output'
        ]
        
        # Network mode
        network_mode = 'host' if network_enabled else 'none'
        
        # Resource limits
        mem_limit = f"{self.config.execution.memory_limit_gb}g"
        nano_cpus = int(self.config.execution.cpu_limit * 1e9)
        
        # Volumes
        volumes = {
            str(work_dir / 'input'): {'bind': '/input', 'mode': 'ro'},
            str(work_dir / 'output'): {'bind': '/output', 'mode': 'rw'},
        }
        
        # Security options
        security_opt = ['no-new-privileges']
        if self.config.security.readonly_rootfs:
            security_opt.append('readonly')
        
        # Capabilities to drop
        cap_drop = self.config.security.drop_capabilities or [
            'CAP_SYS_ADMIN',
            'CAP_NET_ADMIN',
            'CAP_SYS_MODULE',
        ]
        
        config = {
            'image': image_id,
            'name': container_name,
            'command': command,
            'environment': env_vars,
            'network_mode': network_mode,
            'mem_limit': mem_limit,
            'nano_cpus': nano_cpus,
            'volumes': volumes,
            'working_dir': '/workspace',
            'user': 'nobody',  # Run as non-root user
            'detach': True,
            'remove': False,  # Don't auto-remove, we need logs
            'security_opt': security_opt,
            'cap_drop': cap_drop,
        }
        
        # Add GPU support if GPUs are assigned
        if job.assigned_gpus:
            config['device_requests'] = [
                docker.types.DeviceRequest(
                    device_ids=[str(gpu) for gpu in job.assigned_gpus],
                    capabilities=[['gpu']]
                )
            ]
        
        return config
    
    async def _wait_for_container(
        self,
        container,
        timeout_seconds: int
    ) -> int:
        """
        Wait for container to finish with timeout.
        
        Args:
            container: Docker container object
            timeout_seconds: Timeout in seconds
            
        Returns:
            Container exit code
            
        Raises:
            asyncio.TimeoutError: If container exceeds timeout
        """
        start_time = time.time()
        
        while True:
            # Check if timeout exceeded
            if time.time() - start_time > timeout_seconds:
                raise asyncio.TimeoutError()
            
            # Reload container status
            await asyncio.get_event_loop().run_in_executor(
                None,
                container.reload
            )
            
            # Check if container is still running
            if container.status != 'running':
                return container.attrs['State']['ExitCode']
            
            # Sleep before next check
            await asyncio.sleep(1)
    
    async def _get_container_logs(self, container) -> tuple[str, str]:
        """
        Get container stdout and stderr logs.
        
        Args:
            container: Docker container object
            
        Returns:
            Tuple of (stdout, stderr)
        """
        try:
            logs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: container.logs(stdout=True, stderr=True).decode('utf-8', errors='replace')
            )
            
            # Docker doesn't separate stdout/stderr by default
            # We'll return all logs as stdout for simplicity
            return logs, ""
            
        except Exception as e:
            logger.warning(f"Failed to get container logs: {e}")
            return "", ""
    
    async def _kill_container(self, container):
        """
        Kill a running container.
        
        Args:
            container: Docker container object
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                container.kill
            )
            logger.info(f"Container killed: {container.id[:12]}")
        except Exception as e:
            logger.warning(f"Failed to kill container: {e}")
    
    async def _cleanup_container(self, container):
        """
        Remove a container.
        
        Args:
            container: Docker container object
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                container.remove,
                True  # force=True
            )
            logger.debug(f"Container removed: {container.id[:12]}")
        except Exception as e:
            logger.warning(f"Failed to remove container: {e}")
    
    async def build_image(
        self,
        repo_path: Path,
        job_id: str,
        timeout_seconds: int = 3600
    ) -> str:
        """
        Build Docker image from repository.
        
        Args:
            repo_path: Path to repository directory
            job_id: Job identifier for tagging
            timeout_seconds: Build timeout in seconds
            
        Returns:
            Docker image ID
            
        Raises:
            DockerExecutionError: If build fails
        """
        image_tag = f"sandbox-job-{job_id}"
        
        logger.info(f"Building Docker image: {image_tag}")
        
        try:
            # Check if Dockerfile exists
            dockerfile_path = repo_path / 'Dockerfile'
            if not dockerfile_path.exists():
                raise DockerExecutionError("Dockerfile not found")
            
            # Build image
            image, build_logs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.images.build(
                    path=str(repo_path),
                    tag=image_tag,
                    rm=True,  # Remove intermediate containers
                    forcerm=True,  # Always remove intermediate containers
                    timeout=timeout_seconds
                )
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    logger.debug(f"Build: {log['stream'].strip()}")
            
            logger.info(f"Docker image built: {image.id[:12]} ({image_tag})")
            
            return image.id
            
        except DockerException as e:
            logger.error(f"Docker build failed: {e}")
            raise DockerExecutionError(f"Docker build failed: {e}")
    
    async def build_image_from_requirements(
        self,
        repo_path: Path,
        job_id: str,
        base_image: str = "python:3.11-slim"
    ) -> str:
        """
        Build Docker image from requirements.txt (no Dockerfile).
        
        Creates a temporary Dockerfile that:
        1. Uses base Python image
        2. Copies repository contents
        3. Installs requirements
        
        Args:
            repo_path: Path to repository directory
            job_id: Job identifier for tagging
            base_image: Base Docker image to use
            
        Returns:
            Docker image ID
            
        Raises:
            DockerExecutionError: If build fails
        """
        image_tag = f"sandbox-job-{job_id}"
        
        logger.info(f"Building Docker image from requirements: {image_tag}")
        
        # Create temporary Dockerfile
        dockerfile_content = f"""
FROM {base_image}

# Set working directory
WORKDIR /workspace

# Copy repository contents
COPY . /workspace/

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Set user
USER nobody

# Entry point
CMD ["python", "inference.py"]
"""
        
        temp_dockerfile = repo_path / 'Dockerfile.generated'
        try:
            temp_dockerfile.write_text(dockerfile_content)
            
            # Build image using temporary Dockerfile
            image, build_logs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.images.build(
                    path=str(repo_path),
                    dockerfile='Dockerfile.generated',
                    tag=image_tag,
                    rm=True,
                    forcerm=True
                )
            )
            
            logger.info(f"Docker image built from requirements: {image.id[:12]}")
            
            return image.id
            
        except DockerException as e:
            logger.error(f"Docker build from requirements failed: {e}")
            raise DockerExecutionError(f"Docker build failed: {e}")
        
        finally:
            # Clean up temporary Dockerfile
            if temp_dockerfile.exists():
                temp_dockerfile.unlink()
    
    async def remove_image(self, image_id: str):
        """
        Remove a Docker image.
        
        Args:
            image_id: Docker image ID
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.docker_client.images.remove,
                image_id,
                True  # force=True
            )
            logger.info(f"Docker image removed: {image_id[:12]}")
        except ImageNotFound:
            logger.debug(f"Image not found (already removed?): {image_id[:12]}")
        except Exception as e:
            logger.warning(f"Failed to remove image {image_id[:12]}: {e}")
    
    def is_available(self) -> bool:
        """
        Check if Docker is available.
        
        Returns:
            True if Docker daemon is accessible
        """
        try:
            self.docker_client.ping()
            return True
        except Exception:
            return False