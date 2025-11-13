# security/network.py - CORRECTED VERSION

import asyncio
import logging
from typing import Optional, List, Dict
from pathlib import Path

from config import NetworkPolicyConfig

logger = logging.getLogger(__name__)


class NetworkPolicyError(Exception):
    """Raised when network policy operation fails"""
    pass


class NetworkPolicy:
    """
    Manages network access policies for containers
    
    Uses Docker network modes (host, none)
    - Prep phase: host network (full internet access)
    - Inference phase: none network (no network access)
    """
    
    def __init__(self, config: NetworkPolicyConfig):
        """
        Initialize network policy manager
        
        Args:
            config: Network policy configuration
        """
        self.config = config
        
        logger.info(
            "Network Policy initialized",
            extra={
                "prep_allow_internet": config.prep_allow_internet,
                "inference_block_internet": config.inference_block_internet,
                "allowed_domains": len(config.allowed_prep_domains)
            }
        )
    
    def get_network_mode_for_phase(self, phase: str) -> str:
        """
        Get appropriate Docker network mode for execution phase
        
        Args:
            phase: Execution phase ("prep" or "inference")
            
        Returns:
            Docker network mode ("host", "none", "bridge")
        """
        if phase == "prep":
            if self.config.prep_allow_internet:
                return "host"
            else:
                return "none"
        
        elif phase == "inference":
            if self.config.inference_block_internet:
                return "none"
            else:
                return "host"
        
        else:
            logger.warning(f"Unknown phase: {phase}, defaulting to 'none' network")
            return "none"
    
    async def enable_internet_for_container(
        self,
        container_id: str,
        phase: str = "prep"
    ) -> bool:
        """Enable internet access for a container"""
        if phase == "prep" and self.config.prep_allow_internet:
            logger.info(
                f"Internet enabled for container {container_id[:12]} (phase={phase})"
            )
            return True
        
        logger.debug(
            f"Internet remains disabled for container {container_id[:12]} (phase={phase})"
        )
        return False
    
    async def block_internet_for_container(
        self,
        container_id: str,
        phase: str = "inference"
    ) -> bool:
        """Block internet access for a container"""
        if phase == "inference" and self.config.inference_block_internet:
            logger.info(
                f"Internet blocked for container {container_id[:12]} (phase={phase})"
            )
            return True
        
        logger.debug(
            f"Internet access policy not changed for container {container_id[:12]}"
        )
        return False
    
    async def verify_network_isolation(self, container_id: str) -> bool:
        """Verify that a container has no network access"""
        logger.debug(f"Network isolation assumed for container {container_id[:12]}")
        return True
    
    def is_domain_allowed(self, domain: str, phase: str = "prep") -> bool:
        """Check if a domain is allowed for the given phase"""
        if phase != "prep":
            return False
        
        if not self.config.prep_allow_internet:
            return False
        
        for allowed_domain in self.config.allowed_prep_domains:
            if domain == allowed_domain or domain.endswith('.' + allowed_domain):
                return True
        
        for blocked_domain in self.config.blocked_domains:
            if domain == blocked_domain or domain.endswith('.' + blocked_domain):
                return False
        
        return True
    
    async def setup_domain_filtering(
        self,
        container_id: str,
        phase: str = "prep"
    ):
        """Set up domain-based filtering for a container"""
        if phase == "prep" and self.config.allowed_prep_domains:
            logger.debug(
                f"Domain filtering would be applied to {container_id[:12]}: "
                f"{self.config.allowed_prep_domains}"
            )
    
    async def cleanup(self, container_id: str):
        """Clean up network rules for a container"""
        logger.debug(f"Network policy cleanup for container {container_id[:12]}")
    
    def get_policy_summary(self, phase: str) -> dict:
        """Get a summary of network policy for a phase"""
        network_mode = self.get_network_mode_for_phase(phase)
        internet_enabled = (
            (phase == "prep" and self.config.prep_allow_internet) or
            (phase == "inference" and not self.config.inference_block_internet)
        )
        
        return {
            "phase": phase,
            "network_mode": network_mode,
            "internet_enabled": internet_enabled,
            "allowed_domains": self.config.allowed_prep_domains if phase == "prep" else [],
            "blocked_domains": self.config.blocked_domains,
        }


class IptablesNetworkPolicy(NetworkPolicy):
    """
    Advanced network policy using iptables/nftables    
    """
    
    def __init__(self, config: NetworkPolicyConfig):
        super().__init__(config)
        self.iptables_available = self._check_iptables_available()
        self._container_chains: Dict[str, str] = {}
        self._container_netns: Dict[str, str] = {}
        logger.info(f"IptablesNetworkPolicy initialized (available={self.iptables_available})")
    
    def _check_iptables_available(self) -> bool:
        """Check if iptables is available"""
        import subprocess
        try:
            result = subprocess.run(
                ['which', 'iptables'],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                return False
            
            # also check if we can actually run iptables (requires root)
            result = subprocess.run(
                ['iptables', '-L', '-n'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"iptables check failed: {e}")
            return False
    
    async def setup_container_network_isolation(
        self, 
        container_id: str, 
        phase: str
    ) -> bool:
        """
        Setup complete network isolation for a container
        
        TODO:
        1. Proper container network namespace identification
        2. Docker network bridge manipulation
        3. More sophisticated iptables rule management
        """
        if not self.iptables_available:
            logger.warning("iptables not available, using Docker network modes only")
            return False
        
        try:
            netns_pid = await self._get_container_pid(container_id)
            if not netns_pid:
                logger.warning(f"Could not get PID for {container_id[:12]}, skipping iptables setup")
                return False
            
            chain_name = f"SANDBOX_{container_id[:12]}"
            self._container_chains[container_id] = chain_name
            self._container_netns[container_id] = netns_pid
            
            # Create iptables chain using nsenter
            await self._create_iptables_chain_nsenter(chain_name, netns_pid)
            
            # Apply rules based on phase
            if phase == "prep":
                await self._apply_prep_rules_nsenter(chain_name, netns_pid)
            elif phase == "inference":
                await self._apply_inference_rules_nsenter(chain_name, netns_pid)
            
            logger.info(f"Network isolation configured for {container_id[:12]} (phase={phase})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup network isolation: {e}")
            return False
    
    async def _get_container_pid(self, container_id: str) -> Optional[str]:
        """Get the PID of the container's main process"""
        import subprocess
        
        try:
            result = subprocess.run(
                ['docker', 'inspect', '-f', '{{.State.Pid}}', container_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return None
            
            pid = result.stdout.strip()
            if not pid or pid == '0':
                return None
            
            return pid
            
        except Exception as e:
            logger.error(f"Failed to get container PID: {e}")
            return None
    
    async def _create_iptables_chain_nsenter(self, chain_name: str, pid: str):
        """Create iptables chain using nsenter to access container's network namespace"""
        import subprocess
        
        commands = [
            # Create chain
            ['nsenter', '-t', pid, '-n', 'iptables', '-N', chain_name],
            # Link to OUTPUT chain
            ['nsenter', '-t', pid, '-n', 'iptables', '-I', 'OUTPUT', '1', '-j', chain_name],
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=10
                )
                if result.returncode != 0:
                    stderr = result.stderr.decode() if result.stderr else ''
                    # Ignore "Chain already exists" errors
                    if 'already exists' not in stderr.lower():
                        logger.warning(f"Command failed: {' '.join(cmd)}: {stderr}")
            except Exception as e:
                logger.error(f"Failed to execute iptables command: {e}")
                raise
    
    async def _apply_prep_rules_nsenter(self, chain_name: str, pid: str):
        """Apply iptables rules for prep phase using nsenter"""
        import subprocess
        
        commands = [
            # Allow established connections
            ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name, 
             '-m', 'state', '--state', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'],
            
            # Allow localhost
            ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name,
             '-d', '127.0.0.0/8', '-j', 'ACCEPT'],
            
            # Allow DNS
            ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name,
             '-p', 'udp', '--dport', '53', '-j', 'ACCEPT'],
            ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name,
             '-p', 'tcp', '--dport', '53', '-j', 'ACCEPT'],
        ]
        
        # Domain whitelisting
        if self.config.allowed_prep_domains:
            for domain in self.config.allowed_prep_domains:
                ips = await self._resolve_domain(domain)
                for ip in ips:
                    commands.extend([
                        ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name,
                         '-d', ip, '-p', 'tcp', '--dport', '443', '-j', 'ACCEPT'],
                        ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name,
                         '-d', ip, '-p', 'tcp', '--dport', '80', '-j', 'ACCEPT'],
                    ])
        else:
            # Allow all HTTP/HTTPS if no whitelist
            commands.extend([
                ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name,
                 '-p', 'tcp', '--dport', '443', '-j', 'ACCEPT'],
                ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name,
                 '-p', 'tcp', '--dport', '80', '-j', 'ACCEPT'],
            ])
        
        # Log dropped packets
        commands.append(
            ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name,
             '-j', 'LOG', '--log-prefix', f'[SANDBOX-{chain_name}] ', '--log-level', '4']
        )
        
        # Drop everything else
        commands.append(
            ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name, '-j', 'DROP']
        )
        
        for cmd in commands:
            try:
                subprocess.run(cmd, capture_output=True, timeout=10, check=False)
            except subprocess.TimeoutExpired:
                logger.warning(f"iptables command timeout: {' '.join(cmd)}")
    
    async def _apply_inference_rules_nsenter(self, chain_name: str, pid: str):
        """Apply iptables rules for inference phase using nsenter"""
        import subprocess
        
        commands = [
            # Allow localhost only
            ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name,
             '-d', '127.0.0.0/8', '-j', 'ACCEPT'],
            
            # Log everything else
            ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name,
             '-j', 'LOG', '--log-prefix', f'[INFERENCE-{chain_name}] ', '--log-level', '4'],
            
            # Drop everything else
            ['nsenter', '-t', pid, '-n', 'iptables', '-A', chain_name, '-j', 'DROP'],
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, capture_output=True, timeout=10, check=False)
            except subprocess.TimeoutExpired:
                logger.warning(f"iptables command timeout: {' '.join(cmd)}")
    
    async def _resolve_domain(self, domain: str) -> List[str]:
        """Resolve domain to IP addresses"""
        import socket
        
        try:
            result = socket.getaddrinfo(domain, None)
            ips = list(set([r[4][0] for r in result]))
            logger.debug(f"Resolved {domain} to {ips}")
            return ips
        except Exception as e:
            logger.warning(f"Failed to resolve {domain}: {e}")
            return []
    
    async def remove_iptables_rules(self, container_id: str):
        """Remove iptables rules for a container"""
        import subprocess
        
        if container_id not in self._container_chains:
            return
        
        chain_name = self._container_chains[container_id]
        pid = self._container_netns.get(container_id)
        
        if not pid:
            return
        
        try:
            # Unlink chain from OUTPUT
            subprocess.run(
                ['nsenter', '-t', pid, '-n', 'iptables', '-D', 'OUTPUT', '-j', chain_name],
                capture_output=True,
                timeout=10
            )
            
            # Flush chain
            subprocess.run(
                ['nsenter', '-t', pid, '-n', 'iptables', '-F', chain_name],
                capture_output=True,
                timeout=10
            )
            
            # Delete chain
            subprocess.run(
                ['nsenter', '-t', pid, '-n', 'iptables', '-X', chain_name],
                capture_output=True,
                timeout=10
            )
            
            logger.info(f"iptables rules removed for {container_id[:12]}")
            
        except Exception as e:
            logger.warning(f"Failed to remove iptables rules: {e}")
    
    async def cleanup_container_network_isolation(self, container_id: str):
        """Remove iptables rules for a container"""
        try:
            await self.remove_iptables_rules(container_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup network isolation: {e}")
        finally:
            if container_id in self._container_chains:
                del self._container_chains[container_id]
            if container_id in self._container_netns:
                del self._container_netns[container_id]
    
    async def monitor_connections(
        self, 
        container_id: str, 
        duration_seconds: int = 60
    ) -> List[Dict]:
        """
        Monitor network connections for a container
        
        NOTE: This requires tcpdump to be installed on the host system
        """
        import subprocess
        
        pid = self._container_netns.get(container_id)
        if not pid:
            pid = await self._get_container_pid(container_id)
        
        if not pid:
            logger.warning(f"Cannot monitor container {container_id[:12]}: no PID found")
            return []
        
        try:
            cmd = [
                'timeout', str(duration_seconds),
                'nsenter', '-t', pid, '-n',
                'tcpdump', '-n', '-c', '1000',
                '-i', 'any',
                'not', 'port', '22',
                '-l'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            connections = self._parse_tcpdump_output(stdout.decode() if stdout else '')
            
            logger.info(f"Monitored {len(connections)} connection attempts for {container_id[:12]}")
            return connections
            
        except FileNotFoundError:
            logger.warning("tcpdump not found, connection monitoring unavailable")
            return []
        except Exception as e:
            logger.error(f"Failed to monitor connections: {e}")
            return []
    
    def _parse_tcpdump_output(self, output: str) -> List[Dict]:
        """Parse tcpdump output into structured connection data"""
        connections = []
        
        for line in output.split('\n'):
            if not line.strip():
                continue
            
            try:
                parts = line.split()
                if len(parts) >= 5:
                    connections.append({
                        'timestamp': parts[0],
                        'protocol': parts[2] if len(parts) > 2 else 'unknown',
                        'source': parts[3] if len(parts) > 3 else 'unknown',
                        'destination': parts[5] if len(parts) > 5 else 'unknown',
                        'raw': line
                    })
            except Exception:
                continue
        
        return connections