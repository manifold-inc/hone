"""
Docker + gVisor Execution Mode

Executes jobs in Docker containers with gVisor user-space kernel.
This is the most secure execution mode.

gVisor provides:
- User-space kernel implementation
- System call interception and filtering
- Reduced attack surface
- Better isolation than standard Docker

Security features:
- gVisor runsc runtime
- Network control (host for prep, none for inference)
- GPU assignment via CUDA_VISIBLE_DEVICES
- Resource limits (CPU, memory)
- Read-only root filesystem
- Dropped capabilities
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import time
import json

import docker
from docker.errors import DockerException, ImageNotFound, ContainerError

from core.job_queue import Job
from config import Config

logger = logging.getLogger(__name__)


class DockerExecutionError(Exception):
    """Raised when Docker+gVisor execution fails."""
    pass


class DockerGVisorExecutor:
    """
    Docker + gVisor container executor.
    
    Runs jobs in Docker containers with gVisor runtime for enhanced security.
    gVisor provides a user-space kernel that intercepts system calls,
    providing stronger isolation than standard Docker.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Docker + gVisor executor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Check gVisor availability
        self._check_gvisor_available()
        
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized (gVisor mode)")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise DockerExecutionError(f"Docker not available: {e}")
        
        # gVisor configuration
        self.gvisor_runtime = "runsc"  # gVisor runtime name
        self.gvisor_platform = config.security.gvisor.platform  # ptrace or kvm
        
        logger.info(
            f"gVisor executor initialized (platform={self.gvisor_platform})"
        )
    
    def _check_gvisor_available(self):
        """
        Check if gVisor (runsc) is available on the system.
        
        Raises:
            DockerExecutionError: If gVisor is not available
        """
        try:
            result = subprocess.run(
                ['runsc', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                logger.info(f"gVisor available: {result.stdout.strip()}")
            else:
                raise DockerExecutionError("gVisor not available")
                
        except FileNotFoundError:
            raise DockerExecutionError("gVisor (runsc) not found in PATH")
        except subprocess.TimeoutExpired:
            raise DockerExecutionError("gVisor check timeout")
        except Exception as e:
            raise DockerExecutionError(f"gVisor check failed: {e}")
    
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
        Run a container for a specific job phase with gVisor.
        
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
        container_name = f"sandbox-{job.job_id}-{phase}-gvisor"
        
        logger.info(
            f"Starting gVisor container: {container_name}",
            extra={
                "job_id": job.job_id,
                "phase": phase,
                "image": image_id,
                "network_enabled": network_enabled,
                "gvisor_platform": self.gvisor_platform
            }
        )
        
        # Prepare container configuration with gVisor runtime
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
            # Create container with gVisor runtime
            container = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.containers.create(**container_config)
            )
            
            logger.info(
                f"gVisor container created: {container.id[:12]}",
                extra={"container_id": container.id, "runtime": self.gvisor_runtime}
            )
            
            # Start container
            await asyncio.get_event_loop().run_in_executor(
                None,
                container.start
            )
            
            logger.info(f"gVisor container started: {container.id[:12]}")
            
            # Wait for container with timeout
            exit_code = await self._wait_for_container(container, timeout_seconds)
            
            # Get logs
            stdout, stderr = await self._get_container_logs(container)
            
            logger.info(
                f"gVisor container finished: {container.id[:12]} (exit_code={exit_code})",
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
            logger.error(
                f"gVisor container timeout after {timeout_seconds}s: {container_name}"
            )
            if container:
                await self._kill_container(container)
            raise DockerExecutionError(
                f"Container timeout after {timeout_seconds}s"
            )
        
        except ContainerError as e:
            logger.error(f"gVisor container execution error: {e}")
            raise DockerExecutionError(f"Container execution failed: {e}")
        
        except DockerException as e:
            logger.error(f"Docker error with gVisor: {e}")
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
        Build Docker container configuration with gVisor runtime.
        
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
            env_vars['CUDA_VISIBLE_DEVICES'] = ','.join(
                str(gpu) for gpu in job.assigned_gpus
            )
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
        
        # Network mode based on gVisor config and phase
        if self.config.security.gvisor.network_mode == "none" or not network_enabled:
            network_mode = 'none'
        elif self.config.security.gvisor.network_mode == "host":
            network_mode = 'host'
        else:
            # Sandbox network (default for gVisor)
            network_mode = 'bridge'
        
        # For inference, always block network
        if phase == "inference":
            network_mode = 'none'
        
        # Resource limits
        mem_limit = f"{self.config.execution.memory_limit_gb}g"
        nano_cpus = int(self.config.execution.cpu_limit * 1e9)
        
        # Volumes
        volumes = {
            str(work_dir / 'input'): {'bind': '/input', 'mode': 'ro'},
            str(work_dir / 'output'): {'bind': '/output', 'mode': 'rw'},
        }
        
        # Security options for gVisor
        security_opt = ['no-new-privileges']
        
        # Capabilities to drop
        cap_drop = self.config.security.drop_capabilities or [
            'CAP_SYS_ADMIN',
            'CAP_NET_ADMIN',
            'CAP_SYS_MODULE',
            'CAP_SYS_PTRACE',
            'CAP_SYS_RAWIO',
        ]
        
        # Build configuration
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
            'runtime': self.gvisor_runtime,  # Use gVisor runtime
        }
        
        # Add GPU support if GPUs are assigned
        if job.assigned_gpus:
            config['device_requests'] = [
                docker.types.DeviceRequest(
                    device_ids=[str(gpu) for gpu in job.assigned_gpus],
                    capabilities=[['gpu']]
                )
            ]
        
        # Add gVisor-specific configuration via labels
        config['labels'] = {
            'gvisor.platform': self.gvisor_platform,
            'gvisor.file-access': self.config.security.gvisor.file_access,
            'gvisor.overlay': str(self.config.security.gvisor.overlay).lower(),
        }
        
        # PID limit (prevent fork bombs)
        config['pids_limit'] = self.config.execution.max_processes
        
        # Read-only root filesystem (if enabled)
        if self.config.security.readonly_rootfs:
            config['read_only'] = True
            # Add tmpfs for directories that need write access
            config['tmpfs'] = {
                '/tmp': 'rw,noexec,nosuid,size=1g',
                '/var/tmp': 'rw,noexec,nosuid,size=1g',
            }
        
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
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    f"Container timeout after {elapsed:.1f}s "
                    f"(limit: {timeout_seconds}s)"
                )
                raise asyncio.TimeoutError()
            
            # Reload container status
            await asyncio.get_event_loop().run_in_executor(
                None,
                container.reload
            )
            
            # Check if container is still running
            if container.status != 'running':
                exit_code = container.attrs['State']['ExitCode']
                logger.debug(
                    f"Container stopped with exit code {exit_code} "
                    f"after {elapsed:.1f}s"
                )
                return exit_code
            
            # Sleep before next check
            await asyncio.sleep(1)
    
    async def _get_container_logs(self, container) -> Tuple[str, str]:
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
                lambda: container.logs(
                    stdout=True,
                    stderr=True
                ).decode('utf-8', errors='replace')
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
            logger.info(f"gVisor container killed: {container.id[:12]}")
        except Exception as e:
            logger.warning(f"Failed to kill gVisor container: {e}")
    
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
            logger.debug(f"gVisor container removed: {container.id[:12]}")
        except Exception as e:
            logger.warning(f"Failed to remove gVisor container: {e}")
    
    async def build_image(
        self,
        repo_path: Path,
        job_id: str,
        timeout_seconds: int = 3600
    ) -> str:
        """
        Build Docker image with real-time monitoring and early failure detection.
        
        Args:
            repo_path: Path to repository directory
            job_id: Job identifier for tagging
            timeout_seconds: Build timeout in seconds
            
        Returns:
            Docker image ID
            
        Raises:
            DockerExecutionError: If build fails or times out
        """
        image_tag = f"sandbox-job-{job_id}"
        
        logger.info(f"Building Docker image: {image_tag} (timeout={timeout_seconds}s)")
        
        # Check if Dockerfile exists
        dockerfile_path = repo_path / 'Dockerfile'
        if not dockerfile_path.exists():
            raise DockerExecutionError("Dockerfile not found")
        
        try:
            # Build with streaming logs for early error detection
            image_id = await self._build_with_streaming(
                repo_path, image_tag, timeout_seconds
            )
            
            logger.info(f"Docker image built: {image_id[:12]} ({image_tag})")
            return image_id
            
        except DockerExecutionError:
            # Already logged, just re-raise
            raise
        
        except Exception as e:
            logger.exception(f"Unexpected build error: {e}")
            raise DockerExecutionError(f"Build error: {e}")

    async def _build_with_streaming(
        self,
        repo_path: Path,
        image_tag: str,
        timeout_seconds: int
    ) -> str:
        """
        Build image with streaming logs to detect failures immediately.
        
        This monitors the build output in real-time and terminates
        immediately if an error is detected.
        
        Args:
            repo_path: Repository path
            image_tag: Image tag
            timeout_seconds: Timeout in seconds
            
        Returns:
            Image ID
            
        Raises:
            DockerExecutionError: On failure or timeout
        """
        build_started = asyncio.Event()
        build_failed = asyncio.Event()
        error_message = None
        image_id = None
        
        def build_worker():
            """Worker thread that builds the image."""
            nonlocal image_id, error_message
            
            try:
                # Get low-level API client for streaming
                api_client = self.docker_client.api
                
                # Start build with streaming
                build_logs = api_client.build(
                    path=str(repo_path),
                    tag=image_tag,
                    rm=True,
                    forcerm=True,
                    decode=True  # Decode JSON responses
                )
                
                build_started.set()
                
                # Process streaming logs
                for log_entry in build_logs:
                    if 'stream' in log_entry:
                        line = log_entry['stream'].strip()
                        if line:
                            logger.info(f"Build: {line}")
                    
                    # Check for errors
                    if 'error' in log_entry:
                        error_message = log_entry['error']
                        logger.error(f"Build error detected: {error_message}")
                        build_failed.set()
                        return None
                    
                    if 'errorDetail' in log_entry:
                        error_message = log_entry['errorDetail'].get('message', 'Unknown error')
                        logger.error(f"Build error detail: {error_message}")
                        build_failed.set()
                        return None
                
                # Build succeeded, get image
                images = self.docker_client.images.list(name=image_tag)
                if images:
                    image_id = images[0].id
                    return image_id
                else:
                    error_message = "Image not found after build"
                    build_failed.set()
                    return None
                    
            except Exception as e:
                error_message = str(e)
                logger.error(f"Build worker exception: {e}")
                build_failed.set()
                return None
        
        # Start build in thread
        build_task = asyncio.get_event_loop().run_in_executor(None, build_worker)
        
        # Wait for build to start
        try:
            await asyncio.wait_for(build_started.wait(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("Build failed to start within 30 seconds")
            raise DockerExecutionError("Build failed to start")
        
        # Monitor build with timeout
        start_time = asyncio.get_event_loop().time()
        check_interval = 1.0  # Check every second
        
        while True:
            # Check if build failed
            if build_failed.is_set():
                # Kill immediately
                logger.error(f"Build failed: {error_message}")
                await self._kill_build_process(image_tag)
                await self._cleanup_partial_image(image_tag)
                raise DockerExecutionError(f"Build failed: {error_message}")
            
            # Check if build completed
            if build_task.done():
                try:
                    result = build_task.result()
                    if result:
                        return result
                    else:
                        # Build failed
                        await self._kill_build_process(image_tag)
                        await self._cleanup_partial_image(image_tag)
                        raise DockerExecutionError(f"Build failed: {error_message or 'Unknown error'}")
                except Exception as e:
                    await self._kill_build_process(image_tag)
                    await self._cleanup_partial_image(image_tag)
                    raise DockerExecutionError(f"Build error: {e}")
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                logger.error(f"Build timeout after {elapsed:.1f}s")
                build_task.cancel()
                await self._kill_build_process(image_tag)
                await self._cleanup_partial_image(image_tag)
                raise DockerExecutionError(f"Build timeout after {timeout_seconds}s")
            
            # Wait before next check
            await asyncio.sleep(check_interval)

    async def _kill_build_process(self, image_tag: str):
        """
        Immediately kill Docker build processes.
        
        Args:
            image_tag: Image tag being built
        """
        try:
            logger.warning(f"Killing Docker build for {image_tag}")
            
            # Kill all containers with this tag
            def kill_containers():
                try:
                    # List all containers (including stopped ones)
                    all_containers = self.docker_client.containers.list(all=True)
                    
                    for container in all_containers:
                        # Check if container is related to this build
                        if image_tag in str(container.image) or image_tag in container.name:
                            try:
                                logger.info(f"Killing container: {container.id[:12]}")
                                container.kill()
                                container.remove(force=True)
                            except Exception as e:
                                logger.warning(f"Failed to kill container {container.id[:12]}: {e}")
                except Exception as e:
                    logger.warning(f"Error listing containers: {e}")
            
            await asyncio.get_event_loop().run_in_executor(None, kill_containers)
            
            # Prune build cache
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.docker_client.api.prune_builds
            )
            
            logger.info("Build processes killed")
            
        except Exception as e:
            logger.warning(f"Failed to kill build process: {e}")

    async def _cleanup_partial_image(self, image_tag: str):
        """
        Remove partial/failed image immediately.
        
        Args:
            image_tag: Image tag to remove
        """
        try:
            logger.info(f"Cleaning up partial image: {image_tag}")
            
            def remove_images():
                try:
                    images = self.docker_client.images.list(name=image_tag)
                    for image in images:
                        try:
                            logger.info(f"Removing image: {image.id[:12]}")
                            self.docker_client.images.remove(image.id, force=True)
                        except Exception as e:
                            logger.warning(f"Failed to remove image: {e}")
                except Exception as e:
                    logger.warning(f"Error listing images: {e}")
            
            await asyncio.get_event_loop().run_in_executor(None, remove_images)
            
        except Exception as e:
            logger.warning(f"Failed to cleanup image: {e}")
    
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
        Check if Docker and gVisor are available.
        
        Returns:
            True if both Docker daemon and gVisor runtime are accessible
        """
        try:
            # Check Docker
            self.docker_client.ping()
            
            # Check gVisor runtime
            result = subprocess.run(
                ['runsc', '--version'],
                capture_output=True,
                timeout=5
            )
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    async def get_gvisor_info(self) -> Dict:
        """
        Get gVisor runtime information.
        
        Returns:
            Dictionary with gVisor configuration and status
        """
        try:
            result = subprocess.run(
                ['runsc', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return {
                "available": result.returncode == 0,
                "version": result.stdout.strip() if result.returncode == 0 else None,
                "platform": self.gvisor_platform,
                "network_mode": self.config.security.gvisor.network_mode,
                "file_access": self.config.security.gvisor.file_access,
                "overlay": self.config.security.gvisor.overlay,
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }