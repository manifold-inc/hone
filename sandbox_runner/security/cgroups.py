"""
cgroups v2 Resource Management

Provides control over system resources for containers and processes:
- CPU limits (cores, CPU time)
- Memory limits (RAM usage)
- PID limits (prevent fork bombs)
- I/O limits (disk bandwidth)
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CgroupError(Exception):
    """Raised when cgroup operation fails"""
    pass


class CgroupManager:
    """
    Manager for cgroups v2 resource control
    
    Provides high-level interface for:
    - Creating cgroups for jobs
    - Setting resource limits
    - Moving processes to cgroups
    - Cleaning up cgroups
    """
    
    CGROUP_ROOT = Path("/sys/fs/cgroup")
    SANDBOX_PREFIX = "sandbox"
    
    def __init__(self):
        """
        Initialize cgroup manager
        
        Raises:
            CgroupError: If cgroups v2 is not available
        """
        self.cgroup_root = self.CGROUP_ROOT
        self.sandbox_root = self.cgroup_root / self.SANDBOX_PREFIX
        
        if not self._is_cgroup_v2_available():
            logger.warning("cgroups v2 not available")
            raise CgroupError("cgroups v2 not available")
        
        try:
            self._ensure_sandbox_root()
        except Exception as e:
            logger.error(f"Failed to create sandbox root cgroup: {e}")
            raise CgroupError(f"Failed to initialize sandbox cgroup: {e}")
        
        logger.info(f"cgroup manager initialized (root: {self.sandbox_root})")
    
    def _is_cgroup_v2_available(self) -> bool:
        """
        Check if cgroups v2 (unified hierarchy) is available
        
        Returns:
            True if cgroups v2 is available
        """
        if not self.cgroup_root.exists():
            logger.warning(f"cgroup root not found: {self.cgroup_root}")
            return False
        
        controllers_file = self.cgroup_root / "cgroup.controllers"
        if not controllers_file.exists():
            logger.warning("cgroup.controllers not found (v2 required)")
            return False
        
        try:
            controllers = controllers_file.read_text().strip().split()
            required = ['cpu', 'memory', 'pids']
            
            missing = [c for c in required if c not in controllers]
            if missing:
                logger.warning(f"Missing cgroup controllers: {missing}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to read cgroup controllers: {e}")
            return False
    
    def _ensure_sandbox_root(self):
        """
        Ensure sandbox root cgroup exists
        
        Creates /sys/fs/cgroup/sandbox if it doesn't exist.
        """
        if not self.sandbox_root.exists():
            try:
                self.sandbox_root.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created sandbox cgroup root: {self.sandbox_root}")
            except Exception as e:
                raise CgroupError(f"Failed to create sandbox root: {e}")
        
        try:
            subtree_control = self.sandbox_root / "cgroup.subtree_control"
            controllers = "+cpu +memory +pids +io"
            subtree_control.write_text(controllers)
            logger.debug(f"Enabled controllers: {controllers}")
        except Exception as e:
            logger.warning(f"Failed to enable controllers: {e}")
    
    async def create_cgroup(
        self,
        name: str,
        cpu_limit: Optional[int] = None,
        memory_limit_gb: Optional[int] = None,
        pid_limit: Optional[int] = None
    ) -> Path:
        """
        Create a new cgroup with resource limits
        
        Args:
            name: Cgroup name (should be unique per job)
            cpu_limit: CPU limit in cores (optional)
            memory_limit_gb: Memory limit in GB (optional)
            pid_limit: Maximum number of PIDs (optional)
            
        Returns:
            Path to created cgroup
            
        Raises:
            CgroupError: If creation fails
        """
        cgroup_path = self.sandbox_root / name
        
        try:
            if cgroup_path.exists():
                logger.warning(f"cgroup already exists: {name}, removing it first")
                await self.remove_cgroup(name)
            
            cgroup_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cgroup: {name}")
            
            if cpu_limit:
                await self._set_cpu_limit(cgroup_path, cpu_limit)
            
            if memory_limit_gb:
                await self._set_memory_limit(cgroup_path, memory_limit_gb)
            
            if pid_limit:
                await self._set_pid_limit(cgroup_path, pid_limit)
            
            logger.info(
                f"cgroup configured: {name} "
                f"(cpu={cpu_limit}, mem={memory_limit_gb}GB, pids={pid_limit})"
            )
            
            return cgroup_path
            
        except Exception as e:
            logger.error(f"Failed to create cgroup {name}: {e}")
            raise CgroupError(f"cgroup creation failed: {e}")
    
    async def _set_cpu_limit(self, cgroup_path: Path, cpu_cores: int):
        """
        Set CPU limit for cgroup
        
        Args:
            cgroup_path: Path to cgroup
            cpu_cores: Number of CPU cores
        """
        try:
            quota = cpu_cores * 100000
            period = 100000
            
            cpu_max_file = cgroup_path / "cpu.max"
            cpu_max_file.write_text(f"{quota} {period}\n")
            
            logger.debug(f"Set CPU limit: {cpu_cores} cores")
            
        except Exception as e:
            logger.error(f"Failed to set CPU limit: {e}")
            raise CgroupError(f"CPU limit failed: {e}")
    
    async def _set_memory_limit(self, cgroup_path: Path, memory_gb: int):
        """
        Set memory limit for cgroup
        
        Args:
            cgroup_path: Path to cgroup
            memory_gb: Memory limit in GB
        """
        try:
            memory_bytes = memory_gb * 1024 * 1024 * 1024
            
            memory_max_file = cgroup_path / "memory.max"
            memory_max_file.write_text(f"{memory_bytes}\n")
            
            memory_swap_file = cgroup_path / "memory.swap.max"
            if memory_swap_file.exists():
                memory_swap_file.write_text("0\n")
            
            logger.debug(f"Set memory limit: {memory_gb}GB")
            
        except Exception as e:
            logger.error(f"Failed to set memory limit: {e}")
            raise CgroupError(f"Memory limit failed: {e}")
    
    async def _set_pid_limit(self, cgroup_path: Path, max_pids: int):
        """
        Set PID limit for cgroup (prevent fork bombs)
        
        Args:
            cgroup_path: Path to cgroup
            max_pids: Maximum number of PIDs
        """
        try:
            pids_max_file = cgroup_path / "pids.max"
            pids_max_file.write_text(f"{max_pids}\n")
            
            logger.debug(f"Set PID limit: {max_pids}")
            
        except Exception as e:
            logger.error(f"Failed to set PID limit: {e}")
            raise CgroupError(f"PID limit failed: {e}")
    
    async def add_process_to_cgroup(self, cgroup_name: str, pid: int):
        """
        Add a process to a cgroup
        
        Args:
            cgroup_name: Name of cgroup
            pid: Process ID to add
            
        Raises:
            CgroupError: If operation fails
        """
        cgroup_path = self.sandbox_root / cgroup_name
        
        if not cgroup_path.exists():
            raise CgroupError(f"cgroup not found: {cgroup_name}")
        
        try:
            cgroup_procs = cgroup_path / "cgroup.procs"
            cgroup_procs.write_text(f"{pid}\n")
            
            logger.debug(f"Added process {pid} to cgroup {cgroup_name}")
            
        except Exception as e:
            logger.error(f"Failed to add process to cgroup: {e}")
            raise CgroupError(f"Failed to add process: {e}")
    
    async def get_cgroup_stats(self, cgroup_name: str) -> dict:
        """
        Get current resource usage statistics for a cgroup
        
        Args:
            cgroup_name: Name of cgroup
            
        Returns:
            Dictionary with usage statistics
        """
        cgroup_path = self.sandbox_root / cgroup_name
        
        if not cgroup_path.exists():
            raise CgroupError(f"cgroup not found: {cgroup_name}")
        
        stats = {}
        
        try:
            cpu_stat = cgroup_path / "cpu.stat"
            if cpu_stat.exists():
                cpu_data = {}
                for line in cpu_stat.read_text().strip().split('\n'):
                    if line:
                        key, value = line.split()
                        cpu_data[key] = int(value)
                stats['cpu'] = cpu_data
            
            memory_current = cgroup_path / "memory.current"
            if memory_current.exists():
                stats['memory_bytes'] = int(memory_current.read_text().strip())
            
            pids_current = cgroup_path / "pids.current"
            if pids_current.exists():
                stats['pids'] = int(pids_current.read_text().strip())
            
            cgroup_procs = cgroup_path / "cgroup.procs"
            if cgroup_procs.exists():
                pids = cgroup_procs.read_text().strip().split('\n')
                stats['process_count'] = len([p for p in pids if p])
            
        except Exception as e:
            logger.warning(f"Failed to read cgroup stats: {e}")
        
        return stats
    
    async def remove_cgroup(self, cgroup_name: str):
        """
        Remove a cgroup
        
        Args:
            cgroup_name: Name of cgroup to remove
        """
        cgroup_path = self.sandbox_root / cgroup_name
        
        if not cgroup_path.exists():
            logger.debug(f"cgroup does not exist: {cgroup_name}")
            return
        
        try:
            cgroup_procs = cgroup_path / "cgroup.procs"
            if cgroup_procs.exists():
                pids = cgroup_procs.read_text().strip().split('\n')
                for pid_str in pids:
                    if pid_str:
                        try:
                            pid = int(pid_str)
                            os.kill(pid, 9)  # SIGKILL
                            logger.debug(f"Killed process {pid} in cgroup {cgroup_name}")
                        except ProcessLookupError:
                            pass  # Process already dead
                        except Exception as e:
                            logger.warning(f"Failed to kill process {pid_str}: {e}")
            
            cgroup_path.rmdir()
            logger.info(f"Removed cgroup: {cgroup_name}")
            
        except Exception as e:
            logger.error(f"Failed to remove cgroup {cgroup_name}: {e}")
            raise CgroupError(f"cgroup removal failed: {e}")
    
    async def cleanup_all_sandbox_cgroups(self):
        """
        Remove all sandbox cgroups        
        """
        if not self.sandbox_root.exists():
            return
        
        logger.warning("Cleaning up all sandbox cgroups")
        
        try:
            for cgroup_dir in self.sandbox_root.iterdir():
                if cgroup_dir.is_dir():
                    try:
                        await self.remove_cgroup(cgroup_dir.name)
                    except Exception as e:
                        logger.error(f"Failed to remove cgroup {cgroup_dir.name}: {e}")
            
            logger.info("All sandbox cgroups cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup sandbox cgroups: {e}")
    
    def is_available(self) -> bool:
        """
        Check if cgroups v2 is available
        
        Returns:
            True if cgroups v2 is available
        """
        return self._is_cgroup_v2_available()