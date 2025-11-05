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
import sys
import shutil
from collections import deque

import docker
from docker.errors import DockerException, ImageNotFound, ContainerError

from core.job_queue import Job
from config import Config

logger = logging.getLogger(__name__)


class DockerExecutionError(Exception):
    """Raised when Docker+gVisor execution fails."""
    pass


class BuildLogDisplay:
    """
    Real-time build log display with fixed scrollable box.
    Uses a simple approach: fixed box that shows last N lines.
    """
    
    def __init__(self, max_lines: int = 15):
        """
        Initialize build log display.
        
        Args:
            max_lines: Maximum number of lines to display in the scrolling box
        """
        self.max_lines = max_lines
        self.log_buffer = deque(maxlen=max_lines)
        self.terminal_width = min(shutil.get_terminal_size((120, 20)).columns, 150)
        self.box_active = False
        self.line_count = 0
        
    def start(self):
        """Start the build log display box."""
        if self.box_active:
            return
            
        self.box_active = True
        
        # Print header
        print("\n" + "=" * self.terminal_width)
        print(f"{'🔨 DOCKER BUILD LOGS (streaming...)':^{self.terminal_width}}")
        print("=" * self.terminal_width)
        print()
        
    def _clean_line(self, line: str) -> str:
        """
        Clean a log line for display.
        
        Args:
            line: Raw log line
            
        Returns:
            Cleaned line without ANSI codes
        """
        # Remove ANSI escape sequences
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned = ansi_escape.sub('', line)
        
        # Truncate if too long
        max_width = self.terminal_width - 4
        if len(cleaned) > max_width:
            cleaned = cleaned[:max_width-3] + "..."
        
        return cleaned
    
    def update(self, line: str):
        """
        Add a new line to the build log display.
        Simply prints it to stdout with a prefix.
        
        Args:
            line: New log line to display
        """
        if not self.box_active:
            self.start()
        
        # Clean the line
        cleaned = self._clean_line(line.rstrip())
        
        if not cleaned:  # Skip empty lines
            return
        
        # Add to buffer
        self.log_buffer.append(cleaned)
        self.line_count += 1
        
        # Print with a simple prefix
        print(f"│ {cleaned}")
        sys.stdout.flush()
    
    def end(self):
        """End the build log display box."""
        if not self.box_active:
            return
        
        self.box_active = False
        
        # Print footer
        print()
        print("=" * self.terminal_width)
        print(f"{'✅ BUILD COMPLETE ({self.line_count} lines)':^{self.terminal_width}}")
        print("=" * self.terminal_width)
        print()


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
            'auto_remove': False,  # Don't auto-remove, we need logs
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
    
    def _should_show_log_line(self, line: str) -> bool:
        """
        Determine if a log line should be displayed.
        Filters out progress bars, repetitive updates, etc.
        
        Args:
            line: Log line to check
            
        Returns:
            True if line should be shown
        """
        # Skip empty lines
        if not line.strip():
            return False
        
        # Skip lines that are just progress indicators
        progress_indicators = [
            'Reading package lists...',
            '(Reading database',
            'Fetched ',
            'Get:',
        ]
        
        for indicator in progress_indicators:
            if indicator in line and line.count('%') > 0:
                return False
        
        return True
    
    async def _stream_build_logs(self, build_generator, display: BuildLogDisplay):
        """
        Stream build logs in real-time to the display.
        
        Args:
            build_generator: Docker build log generator
            display: BuildLogDisplay instance for rendering logs
        """
        last_step = None
        
        for log_entry in build_generator:
            try:
                if 'stream' in log_entry:
                    line = log_entry['stream'].rstrip()
                    
                    if not line:
                        continue
                    
                    # Track build steps
                    if line.startswith('Step '):
                        last_step = line
                        display.update(line)
                    elif self._should_show_log_line(line):
                        display.update(line)
                    
                    # Also log important lines to file
                    if 'Step ' in line or 'ERROR' in line or 'Successfully' in line:
                        logger.info(f"Build: {line}")
                
                elif 'error' in log_entry:
                    error_line = f"❌ ERROR: {log_entry['error']}"
                    display.update(error_line)
                    logger.error(f"Build error: {log_entry['error']}")
                
                elif 'status' in log_entry:
                    # Handle pull/push status messages
                    status = log_entry['status']
                    if 'id' in log_entry:
                        status = f"[{log_entry['id']}] {status}"
                    if self._should_show_log_line(status):
                        display.update(status)
                
                # Small delay to prevent overwhelming output
                await asyncio.sleep(0.005)
                
            except Exception as e:
                logger.warning(f"Error processing build log entry: {e}")
                continue
    
    async def build_image(
        self,
        repo_path: Path,
        job_id: str,
        timeout_seconds: int = 3600
    ) -> str:
        """
        Build Docker image from repository with real-time log streaming.
        
        Note: Image building is the same for gVisor and regular Docker.
        gVisor runtime is only used when running containers.
        
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
        
        logger.info(f"Building Docker image for gVisor: {image_tag}")
        
        # Initialize build log display
        display = BuildLogDisplay(max_lines=15)
        
        try:
            # Check if Dockerfile exists
            dockerfile_path = repo_path / 'Dockerfile'
            if not dockerfile_path.exists():
                raise DockerExecutionError("Dockerfile not found")
            
            # Start the display
            display.start()
            
            # Build image with streaming logs
            def build_with_logs():
                """Execute build and return generator."""
                return self.docker_client.api.build(
                    path=str(repo_path),
                    tag=image_tag,
                    rm=True,
                    forcerm=True,
                    decode=True,  # Decode JSON logs
                    timeout=timeout_seconds
                )
            
            # Get build generator
            build_generator = await asyncio.get_event_loop().run_in_executor(
                None,
                build_with_logs
            )
            
            # Stream logs to display
            await self._stream_build_logs(build_generator, display)
            
            # End display
            display.end()
            
            # Get the built image
            image = self.docker_client.images.get(image_tag)
            
            logger.info(
                f"Docker image built for gVisor: {image.id[:12]} ({image_tag})"
            )
            
            return image.id
            
        except DockerException as e:
            display.end()
            logger.error(f"Docker build failed: {e}")
            raise DockerExecutionError(f"Docker build failed: {e}")
        except Exception as e:
            display.end()
            logger.error(f"Unexpected build error: {e}")
            raise DockerExecutionError(f"Build failed: {e}")
    
    async def build_image_from_requirements(
        self,
        repo_path: Path,
        job_id: str,
        base_image: str = "python:3.11-slim"
    ) -> str:
        """
        Build Docker image from requirements.txt (no Dockerfile) with streaming logs.
        
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
        
        logger.info(f"Building Docker image from requirements for gVisor: {image_tag}")
        
        # Initialize build log display
        display = BuildLogDisplay(max_lines=15)
        
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
            
            # Start the display
            display.start()
            
            # Build image with streaming logs
            def build_with_logs():
                """Execute build and return generator."""
                return self.docker_client.api.build(
                    path=str(repo_path),
                    dockerfile='Dockerfile.generated',
                    tag=image_tag,
                    rm=True,
                    forcerm=True,
                    decode=True
                )
            
            # Get build generator
            build_generator = await asyncio.get_event_loop().run_in_executor(
                None,
                build_with_logs
            )
            
            # Stream logs to display
            await self._stream_build_logs(build_generator, display)
            
            # End display
            display.end()
            
            # Get the built image
            image = self.docker_client.images.get(image_tag)
            
            logger.info(
                f"Docker image built from requirements for gVisor: {image.id[:12]}"
            )
            
            return image.id
            
        except DockerException as e:
            display.end()
            logger.error(f"Docker build from requirements failed: {e}")
            raise DockerExecutionError(f"Docker build failed: {e}")
        except Exception as e:
            display.end()
            logger.error(f"Unexpected build error: {e}")
            raise DockerExecutionError(f"Build failed: {e}")
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