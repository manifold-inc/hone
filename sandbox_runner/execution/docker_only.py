import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import time
import sys
import httpx

import docker
from docker.errors import DockerException, ImageNotFound, ContainerError

from core.job_queue import Job
from config import Config

logger = logging.getLogger(__name__)


class DockerExecutionError(Exception):
    """Raised when Docker execution fails"""
    pass


class BuildLogDisplay:
    """
    Fixed-position scrolling build log box that always shows the latest N lines
    When active, normal prints should go through write_below_box()
    Falls back to plain prints if stdout is not a TTY
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

        self.SAVE = "\033[s"
        self.RESTORE = "\033[u"
        self.CLEAR_LINE = "\033[2K"
        self.MOVE_UP_FMT = "\033[{}A"
        self.MOVE_DOWN_FMT = "\033[{}B"
        self.CARRIAGE = "\r"

        self._ansi_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

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

        sys.stdout.write("\n" + border + "\n")
        sys.stdout.write(header + "\n")
        sys.stdout.write(border + "\n")

        for _ in range(self.box_lines):
            sys.stdout.write(f"│{' ' * (self.terminal_width - 2)}│\n")

        sys.stdout.write(border + "\n")
        sys.stdout.write(self.SAVE) 
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
            print(cleaned)
            return

        sys.stdout.write(self.RESTORE)
        sys.stdout.write(self.MOVE_UP_FMT.format(self._move_up_to_content))

        buf = list(self.log_buffer)
        start_idx = max(0, len(buf) - self.box_lines)
        view = buf[start_idx:]

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

        sys.stdout.write(self.MOVE_DOWN_FMT.format(self._move_up_to_content))
        sys.stdout.write(self.SAVE)
        sys.stdout.flush()

    def write_below_box(self, text: str):
        """
        Safely print a normal log line below the box while it is active
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

async def _stream_container_logs(container, display: BuildLogDisplay) -> str:
    """
    Stream container logs in real time into the BuildLogDisplay
    """
    loop = asyncio.get_event_loop()
    collected: List[str] = []

    def _read_logs_blocking():
        try:
            for chunk in container.logs(stdout=True, stderr=True, stream=True, follow=True):
                try:
                    text = chunk.decode("utf-8", errors="replace")
                except Exception:
                    text = str(chunk)
                for ln in text.splitlines():
                    display.update(ln)
                    collected.append(ln + "\n")
        except Exception as e:
            display.update(f"[log-stream] ERROR: {e}")

    task = loop.run_in_executor(None, _read_logs_blocking)
    await task

    return "".join(collected)


class DockerOnlyExecutor:
    """
    Docker-only container executor    
    """
    
    def __init__(self, config: Config):
        """
        Initialize Docker-only executor
        
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
        timeout_seconds: int,
        network_name: str = None
    ) -> Tuple[int, str, str]:
        """
        Run a container for a specific job phase
        
        Args:
            image_id: Docker image ID or tag
            job: Job object with execution details
            phase: Execution phase ("prep" or "inference")
            network_enabled: Whether to enable network access
            work_dir: Working directory on host
            timeout_seconds: Execution timeout in seconds
            network_name: Docker network to attach to (optional)
            
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
                "network_enabled": network_enabled,
                "network_name": network_name
            }
        )
        
        container_config = self._build_container_config(
            image_id=image_id,
            job=job,
            phase=phase,
            network_enabled=network_enabled,
            work_dir=work_dir,
            container_name=container_name,
            network_name=network_name 
        )
        
        container = None
        display = BuildLogDisplay(box_lines=50, title=f"🧪 {phase.upper()} LOGS")
        
        try:
            container = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.docker_client.containers.create(**container_config)
            )
            logger.info(
                f"Docker container created: {container.id[:12]}",
                extra={"container_id": container.id}
            )
            
            display.start()
            
            await asyncio.get_event_loop().run_in_executor(None, container.start)
            logger.info(f"Docker container started: {container.id[:12]}")
            
            logs_task = asyncio.create_task(_stream_container_logs(container, display))
            
            exit_code = await self._wait_for_container(container, timeout_seconds)
            
            try:
                combined_logs = await asyncio.wait_for(logs_task, timeout=5)
            except asyncio.TimeoutError:
                logs_task.cancel()
                try:
                    _ = await logs_task
                except Exception:
                    pass
                final = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
                )
                combined_logs = final
            
            status_txt = "✅ EXECUTION COMPLETE" if exit_code == 0 else f"❌ EXECUTION FAILED (code {exit_code})"
            display.end(status_txt)
            
            container_state = container.attrs['State']
            logger.info(
                f"Docker container finished: {container.id[:12]} (exit_code={exit_code})",
                extra={
                    "job_id": job.job_id,
                    "phase": phase,
                    "exit_code": exit_code,
                    "started_at": container_state.get('StartedAt'),
                    "finished_at": container_state.get('FinishedAt'),
                    "error": container_state.get('Error', ''),
                    "oom_killed": container_state.get('OOMKilled', False),
                    "stdout_lines": len(combined_logs.splitlines()) if combined_logs else 0,
                }
            )
            
            if exit_code != 0:
                logger.error(
                    f"Container failed with exit code {exit_code}.",
                    extra={
                        "exit_code": exit_code,
                        "container_error": container_state.get('Error', ''),
                        "stdout_preview": (combined_logs[:500] if combined_logs else "(empty)"),
                    }
                )
                if combined_logs:
                    logger.info(f"Container combined logs:\n{combined_logs}")
            
            return exit_code, combined_logs or "", ""
            
        except asyncio.TimeoutError:
            display.update(f"[timeout] Container exceeded {timeout_seconds}s → killing…")
            if container:
                await self._kill_container(container)
            display.end(f"⏱️ EXECUTION TIMEOUT ({timeout_seconds}s)")
            raise DockerExecutionError(f"Container timeout after {timeout_seconds}s")
        
        except ContainerError as e:
            display.end("❌ EXECUTION ERROR")
            logger.error(f"Docker container execution error: {e}")
            raise DockerExecutionError(f"Container execution failed: {e}")
        
        except DockerException as e:
            display.end("❌ DOCKER ERROR")
            logger.error(f"Docker error: {e}")
            raise DockerExecutionError(f"Docker error: {e}")
        
        finally:
            if container:
                await self._cleanup_container(container)
    
    def _build_container_config(
        self,
        image_id: str,
        job: Job,
        phase: str,
        network_enabled: bool,
        work_dir: Path,
        container_name: str,
        network_name: str = None
    ) -> Dict:
        """
        Build Docker container configuration
        
        Args:
            image_id: Docker image ID
            job: Job object
            phase: Execution phase
            network_enabled: Whether to enable network
            work_dir: Working directory on host
            container_name: Container name
            network_name: Docker network to attach to (optional)
            
        Returns:
            Dictionary with container configuration
        """
        env_vars = {
            'PHASE': phase,
            'JOB_ID': job.job_id,
            'INPUT_S3_PATH': job.input_s3_path,
            'OUTPUT_S3_PATH': job.output_s3_path,
        }
        
        if job.assigned_gpus:
            env_vars['CUDA_VISIBLE_DEVICES'] = ','.join(
                str(gpu) for gpu in job.assigned_gpus
            )
        else:
            env_vars['CUDA_VISIBLE_DEVICES'] = ''
        
        env_vars.update(job.custom_env_vars or {})
        
        command = [
            'python3', 'inference.py', 
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
                "user": 'root',
                "network_name": network_name
            }
        )
        
        if network_name:
            network_mode = network_name
        else:
            network_mode = 'host' if network_enabled else 'none'
            if phase == "inference":
                network_mode = 'none'
        
        mem_limit = f"{self.config.execution.memory_limit_gb}g"
        nano_cpus = int(self.config.execution.cpu_limit * 1e9)
        
        model_cache_dir = work_dir / 'models'
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        volumes = {
            str(work_dir / 'input'): {'bind': '/input', 'mode': 'ro'},
            str(work_dir / 'output'): {'bind': '/output', 'mode': 'rw'},
            str(model_cache_dir): {'bind': '/app/models', 'mode': 'rw'},
        }
        
        security_opt = ['no-new-privileges']
        
        cap_drop = self.config.security.drop_capabilities or [
            'CAP_SYS_ADMIN',
            'CAP_NET_ADMIN',
            'CAP_SYS_MODULE',
            'CAP_SYS_PTRACE',
            'CAP_SYS_RAWIO',
        ]
        
        config = {
            'image': image_id,
            'name': container_name,
            'command': command,
            'environment': env_vars,
            'volumes': volumes,
            'working_dir': '/app',
            'user': 'root',
            'detach': True,
            'auto_remove': False,
            'security_opt': security_opt,
            'cap_drop': cap_drop,
        }
        
        if network_name:
            config['network'] = network_name
        else:
            config['network_mode'] = network_mode
        
        if job.assigned_gpus:
            config['device_requests'] = [
                docker.types.DeviceRequest(
                    device_ids=[str(gpu) for gpu in job.assigned_gpus],
                    capabilities=[['gpu', 'compute', 'utility']]
                )
            ]
            config['runtime'] = 'nvidia'
        
        # this causes issues with docker / hf parallel download - need to adjust dynamically
        #config['pids_limit'] = self.config.execution.max_processes
        
        if self.config.security.readonly_rootfs:
            config['read_only'] = True
            config['tmpfs'] = {
                '/tmp': 'rw,noexec,nosuid,size=1g',
                '/var/tmp': 'rw,noexec,nosuid,size=1g'
            }
        
        return config
    
    async def _wait_for_container(
        self,
        container,
        timeout_seconds: int
    ) -> int:
        """
        Wait for container to finish with timeout
        
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
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    f"Container timeout after {elapsed:.1f}s "
                    f"(limit: {timeout_seconds}s)"
                )
                raise asyncio.TimeoutError()
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                container.reload
            )
            
            if container.status != 'running':
                exit_code = container.attrs['State']['ExitCode']
                logger.debug(
                    f"Container stopped with exit code {exit_code} "
                    f"after {elapsed:.1f}s"
                )
                return exit_code
            
            await asyncio.sleep(1)
    
    async def _kill_container(self, container):
        """
        Kill a running container
        
        Args:
            container: Docker container object
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                container.kill
            )
            logger.info(f"Docker container killed: {container.id[:12]}")
        except Exception as e:
            logger.warning(f"Failed to kill Docker container: {e}")
    
    async def _cleanup_container(self, container):
        """
        Remove a container
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: container.remove(force=True))
            logger.debug(f"Docker container removed: {container.id[:12]}")
        except Exception as e:
            logger.warning(f"Failed to remove Docker container: {e}", exc_info=True)
    
    async def build_image(
        self,
        repo_path: Path,
        job_id: str,
        timeout_seconds: int = 3600
    ) -> str:
        """
        Build Docker image from repository with real-time log streaming
        
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
        
        display = BuildLogDisplay(box_lines=50)
        
        try:
            dockerfile_path = repo_path / 'Dockerfile'
            if not dockerfile_path.exists():
                raise DockerExecutionError("Dockerfile not found")
            
            display.start()
            
            def build_with_logs():
                """Execute build and return generator"""
                return self.docker_client.api.build(
                    path=str(repo_path),
                    tag=image_tag,
                    rm=True,
                    forcerm=True,
                    decode=True,
                    timeout=timeout_seconds
                )
            
            build_generator = await asyncio.get_event_loop().run_in_executor(None, build_with_logs)
            
            await _stream_docker_logs(build_generator, display)
            
            display.end("✅ BUILD COMPLETE")
            
            image = self.docker_client.images.get(image_tag)
            
            logger.info(
                f"Docker image built: {image.id[:12]} ({image_tag})"
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
        
    async def remove_image(self, image_id: str):
        """
        Remove a Docker image
        
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
        Check if Docker is available
        
        Returns:
            True if Docker daemon is accessible
        """
        try:
            self.docker_client.ping()
            return True
        except Exception:
            return False
            
    async def run_job_with_vllm(
        self,
        image_id: str,
        job: Job,
        work_dir: Path,
        prep_timeout: int,
        inference_timeout: int
    ) -> Tuple[int, str, str]:
        """
        Run complete job with vLLM pipeline using shared Docker network.
        """
        network_name = f"sandbox-job-{job.job_id}"
        vllm_container = None
        vllm_port = 8000
        models_dir = work_dir / 'models'
        
        try:
            logger.info("=" * 60)
            logger.info(f"STEP 0: Creating shared network {network_name}")
            logger.info("=" * 60)
            
            network = await self._create_network(network_name)
            logger.info(f"✓ Created network: {network_name}")
            
            logger.info("=" * 60)
            logger.info("STEP 1: PREP - Downloading models")
            logger.info("=" * 60)
            
            prep_exit_code, prep_stdout, prep_stderr = await self.run_container(
                image_id=image_id,
                job=job,
                phase="prep",
                network_enabled=True,
                work_dir=work_dir,
                timeout_seconds=prep_timeout,
                network_name=None
            )
            
            if prep_exit_code != 0:
                logger.error(f"Prep phase failed with exit code {prep_exit_code}")
                return prep_exit_code, prep_stdout, prep_stderr
            
            logger.info("✓ Prep phase completed successfully")
            
            logger.info("=" * 60)
            logger.info("STEP 2: Starting vLLM server")
            logger.info("=" * 60)
            
            vllm_container, vllm_id = await self.start_vllm_container(
                job=job,
                models_dir=models_dir,
                network_name=network_name,
                port=vllm_port
            )
            
            vllm_ready = await self.wait_for_vllm_ready_from_logs(
                container=vllm_container,
                timeout_seconds=300
            )
            
            if not vllm_ready:
                raise DockerExecutionError("vLLM failed to start")
            
            logger.info("=" * 60)
            logger.info("STEP 3: INFERENCE - Using vLLM API")
            logger.info("=" * 60)
            
            vllm_container_name = f"vllm-{job.job_id}"
            job.custom_env_vars['VLLM_API_BASE'] = f'http://{vllm_container_name}:{vllm_port}'
            
            logger.info(f"Inference will connect to vLLM at: {job.custom_env_vars['VLLM_API_BASE']}")
            
            inf_exit_code, inf_stdout, inf_stderr = await self.run_container(
                image_id=image_id,
                job=job,
                phase="inference",
                network_enabled=False,
                work_dir=work_dir,
                timeout_seconds=inference_timeout,
                network_name=network_name
            )
            
            if inf_exit_code != 0:
                logger.error(f"Inference phase failed with exit code {inf_exit_code}")
                return inf_exit_code, inf_stdout, inf_stderr
            
            logger.info("✓ Inference phase completed successfully")
            
            combined_stdout = f"=== PREP PHASE ===\n{prep_stdout}\n\n=== INFERENCE PHASE ===\n{inf_stdout}"
            combined_stderr = f"=== PREP PHASE ===\n{prep_stderr}\n\n=== INFERENCE PHASE ===\n{inf_stderr}"
            
            return 0, combined_stdout, combined_stderr
            
        finally:
            # Cleanup vLLM container
            if vllm_container:
                await self.stop_vllm_container(vllm_container)
            
            # Cleanup network
            try:
                await self._remove_network(network_name)
            except Exception as e:
                logger.warning(f"Failed to remove network {network_name}: {e}")
    
    async def _create_network(self, network_name: str):
        """Create an isolated Docker network without internet access."""
        try:
            ipam_pool = docker.types.IPAMPool(
                subnet='172.28.0.0/16',
                gateway='172.28.0.1'  # Gateway for internal routing only
            )
            ipam_config = docker.types.IPAMConfig(
                pool_configs=[ipam_pool]
            )
            
            network = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.networks.create(
                    network_name,
                    driver="bridge",
                    check_duplicate=True,
                    ipam=ipam_config,
                    options={
                        "com.docker.network.bridge.enable_ip_masquerade": "false"  # Disable NAT
                    }
                )
            )
            logger.info(f"Created isolated Docker network: {network_name}")
            return network
        except Exception as e:
            logger.error(f"Failed to create network {network_name}: {e}")
            raise DockerExecutionError(f"Network creation failed: {e}")
    
    async def _remove_network(self, network_name: str):
        """Remove a Docker network."""
        try:
            network = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.docker_client.networks.get(network_name)
            )
            await asyncio.get_event_loop().run_in_executor(
                None,
                network.remove
            )
            logger.info(f"Removed Docker network: {network_name}")
        except Exception as e:
            logger.warning(f"Failed to remove network {network_name}: {e}")
    
    async def start_vllm_container(
        self,
        job: Job,
        models_dir: Path,
        network_name: str,
        port: int = 8000
    ) -> Tuple[Any, str]:
        """
        Start vLLM container with models mounted on shared network.
        
        Args:
            job: Job object
            models_dir: Path to models directory
            network_name: Docker network name to attach to
            port: vLLM API port (internal to network)
        
        Returns:
            Tuple of (container, container_id)
        """
        container_name = f"vllm-{job.job_id}"
        vllm_image = "vllm/vllm-openai:latest"
        
        logger.info(f"Starting vLLM container: {container_name} on network {network_name}")
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.docker_client.images.get(vllm_image)
            )
            logger.info(f"vLLM image already available: {vllm_image}")
        except ImageNotFound:
            logger.info(f"Pulling vLLM image: {vllm_image}...")
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.docker_client.images.pull(vllm_image)
                )
                logger.info(f"✓ vLLM image pulled successfully")
            except Exception as e:
                logger.error(f"Failed to pull vLLM image: {e}")
                raise DockerExecutionError(f"Failed to pull vLLM image: {e}")
        
        model_info_path = models_dir.parent / "output" / "model_info.json"
        if model_info_path.exists():
            import json
            with open(model_info_path) as f:
                model_info = json.load(f)
                model_name = model_info.get("model_name", "unsloth/Meta-Llama-3.1-8B-Instruct")
        else:
            model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
        
        model_dir_name = model_name.replace("/", "--")
        model_path_in_container = f"/app/models/{model_dir_name}"
        
        config = {
            'image': vllm_image,
            'name': container_name,
            'command': [
                '--model', model_path_in_container,
                '--host', '0.0.0.0',
                '--port', str(port),
                '--dtype', 'half',
                '--gpu-memory-utilization', '0.8',
            ],
            'volumes': {
                str(models_dir): {'bind': '/app/models', 'mode': 'ro'}
            },
            'detach': True,
            'auto_remove': False,
            'network': network_name,  
        }
        
        if job.assigned_gpus:
            config['device_requests'] = [
                docker.types.DeviceRequest(
                    device_ids=[str(gpu) for gpu in job.assigned_gpus],
                    capabilities=[['gpu', 'compute', 'utility']]
                )
            ]
            config['runtime'] = 'nvidia'
            config['environment'] = {
                'CUDA_VISIBLE_DEVICES': ','.join(str(gpu) for gpu in job.assigned_gpus)
            }
        
        try:
            container = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.docker_client.containers.create(**config)
            )
            
            await asyncio.get_event_loop().run_in_executor(None, container.start)
            
            logger.info(f"vLLM container started: {container.id[:12]}")
            
            return container, container.id
            
        except Exception as e:
            logger.error(f"Failed to start vLLM container: {e}")
            raise DockerExecutionError(f"vLLM container start failed: {e}")

    async def wait_for_vllm_ready_from_logs(
        self,
        container,
        timeout_seconds: int = 300,
        check_interval: float = 0.5
    ) -> bool:
        """
        Wait for vLLM API to be ready by monitoring logs for "application startup complete."
        
        Args:
            container: Docker container object
            timeout_seconds: Maximum wait time
            check_interval: Seconds between log checks
            
        Returns:
            True if vLLM is ready
        """
        logger.info(f"Waiting for vLLM startup (checking logs for 'application startup complete.')...")
        
        display = BuildLogDisplay(box_lines=30, title="🚀 vLLM STARTUP LOGS")
        display.start()
        
        start_time = time.time()
        
        found_startup_complete = asyncio.Event()
        
        async def stream_logs():
            try:
                for chunk in container.logs(stdout=True, stderr=True, stream=True, follow=True):
                    try:
                        text = chunk.decode("utf-8", errors="replace")
                    except Exception:
                        text = str(chunk)
                    
                    for ln in text.splitlines():
                        display.update(ln)
                        
                        if "application startup complete" in ln.lower():
                            found_startup_complete.set()
                            return
                    
                    await asyncio.sleep(0.01)
            except Exception as e:
                display.update(f"[log-stream] ERROR: {e}")
        
        log_task = asyncio.create_task(stream_logs())
        
        try:
            while time.time() - start_time < timeout_seconds:
                if found_startup_complete.is_set():
                    log_task.cancel()
                    try:
                        await log_task
                    except asyncio.CancelledError:
                        pass
                    
                    display.end("✅ vLLM API READY")
                    return True
                
                await asyncio.get_event_loop().run_in_executor(
                    None, container.reload
                )
                if container.status not in ['running', 'created']:
                    log_task.cancel()
                    try:
                        await log_task
                    except asyncio.CancelledError:
                        pass
                    
                    display.end(f"❌ vLLM CONTAINER DIED (status: {container.status})")
                    logger.error(f"vLLM container died with status: {container.status}")
                    return False
                
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0:
                    display.write_below_box(
                        f"⏳ Waiting for vLLM startup... ({elapsed:.0f}s/{timeout_seconds}s)"
                    )
                
                await asyncio.sleep(check_interval)
            
            log_task.cancel()
            try:
                await log_task
            except asyncio.CancelledError:
                pass
            
            display.end(f"⏱️ vLLM STARTUP TIMEOUT ({timeout_seconds}s)")
            logger.error(f"vLLM API failed to start within {timeout_seconds}s")
            return False
            
        except Exception as e:
            log_task.cancel()
            try:
                await log_task
            except asyncio.CancelledError:
                pass
            
            display.end("❌ vLLM STARTUP ERROR")
            logger.error(f"Error waiting for vLLM: {e}")
            raise

    async def stop_vllm_container(self, container) -> None:
        """Stop and remove vLLM container."""
        try:
            if container:
                
                logger.info(f"Stopping vLLM container: {container.id[:12]}")
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: container.stop(timeout=10)
                )
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: container.remove(force=True)
                )
                logger.info("✓ vLLM container stopped")
        except Exception as e:
            logger.warning(f"Failed to stop vLLM container: {e}")