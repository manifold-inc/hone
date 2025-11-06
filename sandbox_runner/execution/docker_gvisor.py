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
import re

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
    Fixed-position scrolling build log box that always shows the latest N lines.
    When active, normal prints should go through write_below_box().
    Falls back to plain prints if stdout is not a TTY.
    """
    def __init__(self, box_lines: int = 10, title: str = "🔨 BUILD LOGS"):
        import shutil, sys, re
        from collections import deque

        self.box_lines = max(3, box_lines)
        self.title = title
        self.log_buffer = deque(maxlen=self.box_lines)
        self.terminal_width = min(shutil.get_terminal_size((120, 20)).columns, 140)
        self.box_active = False
        self.total_lines = 0
        self.is_tty = sys.stdout.isatty()

        # ANSI codes
        self.SAVE = "\033[s"
        self.RESTORE = "\033[u"
        self.CLEAR_LINE = "\033[2K"
        self.MOVE_UP_FMT = "\033[{}A"
        self.MOVE_DOWN_FMT = "\033[{}B"
        self.CARRIAGE = "\r"

        self._ansi_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        # Offsets (relative; we compute via moves, not multiple saved cursors)
        # Layout we print in start():
        #   ─ border (1)
        #   title (1)
        #   ─ border (1)
        #   content box lines (box_lines)
        #   ─ border (1)
        #   (cursor ends AFTER the bottom border on a new empty line)
        #
        # From "after" to "top of content" = move up (box_lines + 1)
        self._move_up_to_content = self.box_lines + 1

    def _strip_ansi(self, s: str) -> str:
        return self._ansi_re.sub("", s)

    def _clean_line(self, line: str) -> str:
        cleaned = self._strip_ansi(line.rstrip("\r\n"))
        cleaned = ''.join(ch for ch in cleaned if ch.isprintable() or ch in '\t ')
        max_width = self.terminal_width - 4
        if len(cleaned) > max_width:
            cleaned = cleaned[:max_width - 3] + "..."
        return cleaned

    def _should_display(self, line: str) -> bool:
        if not line.strip():
            return False
        skip_patterns = [
            r'^\(Reading database \.\.\. \d+%',
            r'^Get:\d+ http',
            r'^Fetched \d+',
            r'^Reading package lists\.\.\.',
            r'^Building dependency tree\.\.\.',
            r'^Reading state information\.\.\.',
        ]
        import re
        return not any(re.search(p, line) for p in skip_patterns)

    def start(self):
        if self.box_active:
            return
        self.box_active = True

        if not self.is_tty:
            print(f"\n{'='*self.terminal_width}\n{self.title:^{self.terminal_width}}\n{'='*self.terminal_width}")
            print(f"[TTY not detected] Streaming logs plainly below...")
            print('-'*self.terminal_width)
            return

        border = "─" * self.terminal_width
        header = f"{self.title:^{self.terminal_width}}"

        # Header
        sys.stdout.write("\n" + border + "\n")
        sys.stdout.write(header + "\n")
        sys.stdout.write(border + "\n")

        # Content box (empty lines)
        for _ in range(self.box_lines):
            sys.stdout.write(f"│{' ' * (self.terminal_width - 2)}│\n")

        # Footer border and final empty line where normal logs will continue
        sys.stdout.write(border + "\n")
        sys.stdout.write(self.SAVE)  # Save "anchor_after" (below the box)
        sys.stdout.flush()

    def update(self, line: str):
        if not self.box_active:
            self.start()

        cleaned = self._clean_line(line)
        if not self._should_display(cleaned):
            return

        self.log_buffer.append(cleaned)
        self.total_lines += 1

        if not self.is_tty:
            # Plain print fallback
            print(cleaned)
            return

        # Restore to "after box", move up into the content area
        sys.stdout.write(self.RESTORE)
        sys.stdout.write(self.MOVE_UP_FMT.format(self._move_up_to_content))

        # Redraw content box
        buf = list(self.log_buffer)
        # Ensure we always draw exactly box_lines rows (right-aligned to latest)
        start_idx = max(0, len(buf) - self.box_lines)
        view = buf[start_idx:]

        # Draw lines
        for i in range(self.box_lines):
            sys.stdout.write(self.CLEAR_LINE + self.CARRIAGE)
            if i < len(view):
                content = view[i]
                padding = self.terminal_width - 2 - 1 - len(content)
                if padding < 0:
                    padding = 0
                sys.stdout.write(f"│ {content}{' ' * padding}│\n")
            else:
                sys.stdout.write(f"│{' ' * (self.terminal_width - 2)}│\n")

        # Move cursor back down to "after box" and re-save anchor
        sys.stdout.write(self.MOVE_DOWN_FMT.format(self._move_up_to_content))
        sys.stdout.write(self.SAVE)
        sys.stdout.flush()

    def write_below_box(self, text: str):
        """
        Safely print a normal log line below the box while it is active.
        """
        if not self.box_active or not self.is_tty:
            print(text)
            return
        sys.stdout.write(self.RESTORE)  # go to "after box"
        sys.stdout.write(self.CLEAR_LINE + self.CARRIAGE)
        sys.stdout.write(text.rstrip("\n") + "\n")
        sys.stdout.write(self.SAVE)     # keep "after box" current
        sys.stdout.flush()

    def end(self, status: str = "✅ BUILD COMPLETE"):
        if not self.box_active:
            return
        self.box_active = False

        if not self.is_tty:
            print('-'*self.terminal_width)
            print(f"{status} ({self.total_lines} lines processed)")
            print('='*self.terminal_width + "\n")
            return

        # Go to after-box and print completion banner cleanly below it
        sys.stdout.write(self.RESTORE)
        sys.stdout.write(self.CLEAR_LINE + self.CARRIAGE)
        border = "─" * self.terminal_width
        msg = f"{status} ({self.total_lines} lines processed)"
        sys.stdout.write(border + "\n")
        sys.stdout.write(f"{msg:^{self.terminal_width}}\n")
        sys.stdout.write(border + "\n\n")
        sys.stdout.flush()


async def _stream_docker_logs(build_generator, display: BuildLogDisplay):
    loop = asyncio.get_event_loop()
    try:
        for entry in build_generator:
            # Typical keys: 'stream', 'status', 'error', 'id', 'aux'
            if 'error' in entry:
                display.update(f"ERROR: {entry['error']}")
            elif 'stream' in entry:
                for ln in entry['stream'].splitlines():
                    display.update(ln)
            elif 'status' in entry:
                msg = f"{entry.get('id', '')} {entry['status']}".strip()
                display.update(msg)
            await asyncio.sleep(0.01)
    except Exception as ex:
        display.update(f"ERROR: {ex}")
        await asyncio.sleep(0)


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
            
            # Inspect container configuration for debugging
            try:
                container_details = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.docker_client.api.inspect_container(container.id)
                )
                
                logger.info(
                    f"Container details - Image: {container_details.get('Image', 'unknown')[:12]}, "
                    f"Cmd: {container_details['Config'].get('Cmd')}, "
                    f"Entrypoint: {container_details['Config'].get('Entrypoint')}, "
                    f"WorkingDir: {container_details['Config'].get('WorkingDir')}, "
                    f"User: {container_details['Config'].get('User')}"
                )
            except Exception as e:
                logger.warning(f"Failed to inspect container: {e}")
            
            # Verify image exists and inspect it before starting
            try:
                image_details = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.docker_client.api.inspect_image(image_id)
                )
                logger.info(
                    f"Image inspection before start - "
                    f"Cmd: {image_details['Config'].get('Cmd')}, "
                    f"Entrypoint: {image_details['Config'].get('Entrypoint')}, "
                    f"WorkingDir: {image_details['Config'].get('WorkingDir')}, "
                    f"User: {image_details['Config'].get('User')}"
                )
            except Exception as e:
                logger.error(f"Failed to inspect image {image_id}: {e}")
            
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
            
            # Get detailed container state
            container_state = container.attrs['State']
            
            logger.info(
                f"gVisor container finished: {container.id[:12]} (exit_code={exit_code})",
                extra={
                    "job_id": job.job_id,
                    "phase": phase,
                    "exit_code": exit_code,
                    "started_at": container_state.get('StartedAt'),
                    "finished_at": container_state.get('FinishedAt'),
                    "error": container_state.get('Error', ''),
                    "oom_killed": container_state.get('OOMKilled', False),
                    "stdout_lines": len(stdout.split('\n')) if stdout else 0,
                    "stderr_lines": len(stderr.split('\n')) if stderr else 0
                }
            )
            
            # Log container output if exit code is non-zero
            if exit_code != 0:
                logger.error(
                    f"Container failed with exit code {exit_code}. "
                    f"Exit code 127 typically means 'command not found'.",
                    extra={
                        "exit_code": exit_code,
                        "container_error": container_state.get('Error', ''),
                        "stdout_preview": stdout[:500] if stdout else "(empty)",
                        "stderr_preview": stderr[:500] if stderr else "(empty)"
                    }
                )
                
                # Log full output at debug level
                if stdout:
                    logger.info(f"Container stdout:\n{stdout}")
                if stderr:
                    logger.error(f"Container stderr:\n{stderr}")
            
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
            'python3', 'inference.py',  # Changed from 'python' to 'python3'
            '--phase', phase,
            '--input', '/input',
            '--output', '/output'
        ]
        
        logger.info(
            f"Container command for {phase} phase: {' '.join(command)}",
            extra={
                "job_id": job.job_id,
                "phase": phase,
                "working_dir": '/app',
                "user": 'nobody',
                "network_mode": "TBD"  # Will be determined below
            }
        )
        
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
            #'mem_limit': mem_limit,
            #'nano_cpus': nano_cpus,
            'volumes': volumes,
            'working_dir': '/app',
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
            config['devices'] = [f'/dev/nvidia{gpu}:/dev/nvidia{gpu}' for gpu in job.assigned_gpus]
            config['devices'].append('/dev/nvidiactl:/dev/nvidiactl')
            config['devices'].append('/dev/nvidia-uvm:/dev/nvidia-uvm')

        
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
        Remove a container
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: container.remove(force=True))
            logger.debug(f"gVisor container removed: {container.id[:12]}")
        except Exception as e:
            logger.warning(f"Failed to remove gVisor container: {e}", exc_info=True)

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
        
        # Initialize build log display with 10 lines
        display = BuildLogDisplay(box_lines=10)
        
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
                    decode=True,
                    timeout=timeout_seconds
                )
            
            # Get build generator
            build_generator = await asyncio.get_event_loop().run_in_executor(None, build_with_logs)

            await _stream_docker_logs(build_generator, display)

            display.end("✅ BUILD COMPLETE")
            
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
        display = BuildLogDisplay(box_lines=10)
        
        # Create temporary Dockerfile
        dockerfile_content = f"""
FROM {base_image}

# Set working directory
WORKDIR /app

# Copy repository contents
COPY . /app/

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
    
    async def inspect_image_for_command(self, image_id: str) -> Dict:
        """
        Inspect an image to verify the command/entrypoint configuration.
        Useful for debugging command execution issues.
        
        Args:
            image_id: Docker image ID
            
        Returns:
            Dictionary with image command configuration
        """
        try:
            image_details = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.api.inspect_image(image_id)
            )
            
            config = image_details.get('Config', {})
            
            info = {
                "cmd": config.get('Cmd'),
                "entrypoint": config.get('Entrypoint'),
                "working_dir": config.get('WorkingDir'),
                "user": config.get('User'),
                "env": [e for e in config.get('Env', []) if 'PATH' in e],
            }
            
            logger.info(
                f"Image {image_id[:12]} configuration: "
                f"Cmd={info['cmd']}, "
                f"Entrypoint={info['entrypoint']}, "
                f"WorkingDir={info['working_dir']}, "
                f"User={info['user']}"
            )
            
            if not info.get('cmd') and not info.get('entrypoint'):
                logger.warning(
                    f"Image {image_id[:12]} has no CMD or ENTRYPOINT defined! "
                    "This will cause the container command to fail."
                )
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to inspect image {image_id}: {e}")
            return {}
    
    async def verify_container_environment(
        self, 
        image_id: str,
    ) -> Dict:
        """
        Run diagnostic commands in a container to verify the environment.
        Useful for debugging 'command not found' errors (exit code 127).
        
        Args:
            image_id: Docker image ID
            
        Returns:
            Dictionary with diagnostic results
        """
        results = {}
        
        diagnostic_commands = [
            ('which_python', ['which', 'python']),
            ('which_python3', ['which', 'python3']),
            ('ls_app', ['ls', '-la', '/app']),
            ('ls_root', ['ls', '-la', '/']),
            ('pwd', ['pwd']),
            ('whoami', ['whoami']),
            ('env', ['env']),
        ]
        
        logger.info(f"Running diagnostics on image {image_id[:12]}...")
        
        for test_name, cmd in diagnostic_commands:
            try:
                container = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.docker_client.containers.run(
                        image=image_id,
                        command=cmd,
                        detach=False,
                        remove=True,
                        runtime=self.gvisor_runtime,
                    )
                )
                results[test_name] = container.decode('utf-8').strip()
                logger.info(f"✓ Diagnostic {test_name}: {results[test_name][:100]}")
            except Exception as e:
                results[test_name] = f"ERROR: {e}"
                logger.warning(f"✗ Diagnostic {test_name} failed: {e}")
        
        return results
    
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