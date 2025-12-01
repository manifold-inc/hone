"""
Linux Namespace Isolation

Provides process isolation using Linux namespaces:
- PID namespace: Isolate process tree
- Network namespace: Isolate network stack
- Mount namespace: Isolate filesystem mounts
- UTS namespace: Isolate hostname
- IPC namespace: Isolate inter-process communication
- User namespace: Isolate user/group IDs (when available)
"""

import logging
import os
import subprocess
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)


class NamespaceError(Exception):
    """Raised when namespace operation fails"""
    pass


class NamespaceManager:
    """
    Manager for Linux namespace isolation
    
    Provides high-level interface for creating isolated execution environments
    using Linux namespaces, similar to containers but without Docker
    """
    
    def __init__(self):
        """
        Initialize namespace manager
        """
        self.unshare_available = self._check_unshare_available()
        
        if self.unshare_available:
            logger.info("Namespace isolation available (unshare)")
        else:
            logger.warning("Namespace isolation not available")
    
    def _check_unshare_available(self) -> bool:
        """
        Check if unshare command is available
        
        Returns:
            True if unshare is available
        """
        try:
            result = subprocess.run(
                ['which', 'unshare'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def wrap_command(
        self,
        command: List[str],
        isolate_network: bool = True,
        isolate_pid: bool = True,
        isolate_mount: bool = True,
        isolate_uts: bool = True,
        isolate_ipc: bool = True
    ) -> List[str]:
        """
        Wrap a command with namespace isolation
        
        Args:
            command: Command to wrap
            isolate_network: Create new network namespace
            isolate_pid: Create new PID namespace
            isolate_mount: Create new mount namespace
            isolate_uts: Create new UTS namespace (hostname)
            isolate_ipc: Create new IPC namespace
            
        Returns:
            Wrapped command with unshare
            
        Example:
            wrap_command(['python', 'script.py'], isolate_network=True)
            Returns: ['unshare', '--net', '--pid', '--fork', 'python', 'script.py']
        """
        if not self.unshare_available:
            logger.warning("unshare not available, returning unwrapped command")
            return command
        
        unshare_cmd = ['unshare']
        
        if isolate_network:
            unshare_cmd.append('--net')
        
        if isolate_pid:
            unshare_cmd.extend(['--pid', '--fork'])
        
        if isolate_mount:
            unshare_cmd.append('--mount')
        
        if isolate_uts:
            unshare_cmd.append('--uts')
        
        if isolate_ipc:
            unshare_cmd.append('--ipc')
        
        full_command = unshare_cmd + ['--'] + command
        
        logger.debug(f"Wrapped command: {' '.join(full_command)}")
        
        return full_command
    
    def create_network_namespace(self, namespace_name: str) -> bool:
        """
        Create a named network namespace
        
        Args:
            namespace_name: Name for the namespace
            
        Returns:
            True if namespace was created
        """
        try:
            result = subprocess.run(
                ['ip', 'netns', 'add', namespace_name],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"Created network namespace: {namespace_name}")
                return True
            else:
                logger.error(
                    f"Failed to create network namespace: {result.stderr.decode()}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Failed to create network namespace: {e}")
            return False
    
    def delete_network_namespace(self, namespace_name: str) -> bool:
        """
        Delete a named network namespace
        
        Args:
            namespace_name: Name of the namespace to delete
            
        Returns:
            True if namespace was deleted
        """
        try:
            result = subprocess.run(
                ['ip', 'netns', 'delete', namespace_name],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"Deleted network namespace: {namespace_name}")
                return True
            else:
                logger.warning(
                    f"Failed to delete network namespace: {result.stderr.decode()}"
                )
                return False
                
        except Exception as e:
            logger.warning(f"Failed to delete network namespace: {e}")
            return False
    
    def exec_in_network_namespace(
        self,
        namespace_name: str,
        command: List[str]
    ) -> subprocess.CompletedProcess:
        """
        Execute a command in a named network namespace
        
        Args:
            namespace_name: Name of the namespace
            command: Command to execute
            
        Returns:
            CompletedProcess object
        """
        try:
            netns_command = ['ip', 'netns', 'exec', namespace_name] + command
            
            result = subprocess.run(
                netns_command,
                capture_output=True,
                timeout=30
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to exec in network namespace: {e}")
            raise NamespaceError(f"Network namespace exec failed: {e}")
    
    def isolate_process(
        self,
        pid: int,
        network: bool = True,
        pid_ns: bool = False  
    ) -> bool:
        """
        Attempt to isolate an existing process
                
        Args:
            pid: Process ID to isolate
            network: Isolate network namespace
            pid_ns: Isolate PID namespace (not supported for existing processes)
            
        Returns:
            True if isolation was successful
        """
        if pid_ns:
            logger.warning(
                "Cannot change PID namespace of existing process"
            )
        
        if network:
            try:
                ns_name = f"sandbox_{pid}"
                
                if not self.create_network_namespace(ns_name):
                    return False
                
                logger.warning(
                    "Moving existing process to network namespace not implemented"
                )
                return False
                
            except Exception as e:
                logger.error(f"Failed to isolate process {pid}: {e}")
                return False
        
        return True
    
    def get_namespace_info(self, pid: int) -> dict:
        """
        Get namespace information for a process
        
        Args:
            pid: Process ID
            
        Returns:
            Dictionary with namespace information
        """
        info = {}
        
        try:
            ns_path = Path(f"/proc/{pid}/ns")
            
            if not ns_path.exists():
                logger.warning(f"Namespace info not found for PID {pid}")
                return info
            
            for ns_type in ['net', 'pid', 'mnt', 'uts', 'ipc', 'user']:
                ns_link = ns_path / ns_type
                if ns_link.exists():
                    target = os.readlink(str(ns_link))
                    info[ns_type] = target
            
        except Exception as e:
            logger.warning(f"Failed to get namespace info: {e}")
        
        return info
    
    def check_namespace_isolation(
        self,
        pid1: int,
        pid2: int,
        namespace_type: str = 'net'
    ) -> bool:
        """
        Check if two processes are in different namespaces
        
        Args:
            pid1: First process ID
            pid2: Second process ID
            namespace_type: Type of namespace to check ('net', 'pid', etc.)
            
        Returns:
            True if processes are in different namespaces
        """
        try:
            info1 = self.get_namespace_info(pid1)
            info2 = self.get_namespace_info(pid2)
            
            if namespace_type not in info1 or namespace_type not in info2:
                return False
            
            return info1[namespace_type] != info2[namespace_type]
            
        except Exception as e:
            logger.error(f"Failed to check namespace isolation: {e}")
            return False
    
    def is_available(self) -> bool:
        """
        Check if namespace isolation is available
        
        Returns:
            True if namespace isolation is available
        """
        return self.unshare_available
    
    def get_required_capabilities(self) -> List[str]:
        """
        Get list of Linux capabilities required for namespace operations
        
        Returns:
            List of capability names
        """
        return [
            'CAP_SYS_ADMIN', 
            'CAP_NET_ADMIN', 
            'CAP_SYS_PTRACE', 
        ]
    
    def create_pid_namespace_wrapper(self, command: List[str]) -> List[str]:
        """
        Create a PID namespace wrapper for a command
        
        This wraps the command to run in a new PID namespace where:
        - The command becomes PID 1 in the namespace
        - Child processes can't see processes outside the namespace
        - Provides better isolation than just process groups
        
        Args:
            command: Command to wrap
            
        Returns:
            Wrapped command
        """
        return self.wrap_command(
            command,
            isolate_network=False,
            isolate_pid=True,
            isolate_mount=False,
            isolate_uts=False,
            isolate_ipc=False
        )
    
    def create_network_namespace_wrapper(self, command: List[str]) -> List[str]:
        """
        Create a network namespace wrapper for a command
        
        This wraps the command to run in a new network namespace where:
        - The command has no network access by default
        - Network interfaces must be explicitly configured
        - Provides network isolation
        
        Args:
            command: Command to wrap
            
        Returns:
            Wrapped command
        """
        return self.wrap_command(
            command,
            isolate_network=True,
            isolate_pid=False,
            isolate_mount=False,
            isolate_uts=False,
            isolate_ipc=False
        )
    
    def create_full_isolation_wrapper(self, command: List[str]) -> List[str]:
        """
        Create a full isolation wrapper with all namespaces
        
        Args:
            command: Command to wrap
            
        Returns:
            Wrapped command with all isolation
        """
        return self.wrap_command(
            command,
            isolate_network=True,
            isolate_pid=True,
            isolate_mount=True,
            isolate_uts=True,
            isolate_ipc=True
        )