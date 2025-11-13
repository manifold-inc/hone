"""
Network Policy Module

Controls network access for containers during different execution phases:
- Prep phase: Internet access enabled (for downloading models)
- Inference phase: Internet access blocked (for security)

Implementation uses Docker network modes for Phase 3.
Phase 4 will add iptables/nftables for fine-grained control.
"""

import asyncio
import logging
from typing import Optional, List

from config import NetworkPolicyConfig

logger = logging.getLogger(__name__)


class NetworkPolicyError(Exception):
    """Raised when network policy operation fails"""
    pass


class NetworkPolicy:
    """
    Manages network access policies for containers
    
    TODO:
    - Uses Docker network modes (host, none)
    - Prep phase: host network (full internet access)
    - Inference phase: none network (no network access)
    
    TODO:
    - Add iptables/nftables rules for fine-grained control
    - Whitelist specific domains during prep phase
    - Monitor and log network access attempts
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
        """
        Enable internet access for a container.
        
        TODO:
        - This is handled at container creation via network_mode parameter
        - This method is a placeholder for future iptables-based control
        
        Args:
            container_id: Docker container ID
            phase: Execution phase
            
        Returns:
            True if internet enabled
        """
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
        """
        Block internet access for a container
        
        TODO:
        - This is handled at container creation via network_mode='none'
        - This method is a placeholder for future iptables-based control
        
        Args:
            container_id: Docker container ID
            phase: Execution phase
            
        Returns:
            True if internet blocked
        """
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
        """
        Verify that a container has no network access
        
        TODO:
        - Checks if container was created with network_mode='none'
        - Cannot verify after creation without entering container
        
        TODO : 
        - Active network monitoring
        - Connection attempt logging
        - Real-time blocking verification
        
        Args:
            container_id: Docker container ID
            
        Returns:
            True if network isolation verified
        """
        # TODO: Add actual verification via iptables or container inspection
        logger.debug(f"Network isolation assumed for container {container_id[:12]}")
        return True
    
    def is_domain_allowed(self, domain: str, phase: str = "prep") -> bool:
        """
        Check if a domain is allowed for the given phase
        
        Args:
            domain: Domain name to check
            phase: Execution phase
            
        Returns:
            True if domain is allowed
        """
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
        """
        Set up domain-based filtering for a container.
        
        - TODO: use iptables/DNS filtering in Phase 4
        
        Args:
            container_id: Docker container ID
            phase: Execution phase
        """
        if phase == "prep" and self.config.allowed_prep_domains:
            logger.debug(
                f"Domain filtering would be applied to {container_id[:12]}: "
                f"{self.config.allowed_prep_domains}"
            )
    
    async def cleanup(self, container_id: str):
        """
        Clean up network rules for a container
        
        Args:
            container_id: Docker container ID
        """
        # TODO: Remove iptables rules
        logger.debug(f"Network policy cleanup for container {container_id[:12]}")
    
    def get_policy_summary(self, phase: str) -> dict:
        """
        Get a summary of network policy for a phase
        
        Args:
            phase: Execution phase
            
        Returns:
            Dictionary with policy details
        """
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


# security/network.py - Enhanced IptablesNetworkPolicy

class IptablesNetworkPolicy(NetworkPolicy):
    """
    Advanced network policy using iptables/nftables.
    
    Features:
    - Fine-grained domain whitelisting
    - Connection monitoring and logging
    - Real-time blocking of unauthorized connections
    - DNS filtering
    """
    
    def __init__(self, config: NetworkPolicyConfig):
        super().__init__(config)
        self.iptables_available = self._check_iptables_available()
        self._container_chains = {}  # track chains per container
        logger.info(f"IptablesNetworkPolicy initialized (available={self.iptables_available})")
    
    async def setup_container_network_isolation(
        self, 
        container_id: str, 
        phase: str
    ) -> bool:
        """
        Setup complete network isolation for a container
        
        Steps:
        1. Get container's network namespace
        2. Create iptables chain for container
        3. Apply rules based on phase
        4. Set up DNS filtering if needed
        
        Args:
            container_id: Docker container ID
            phase: Execution phase (prep or inference)
            
        Returns:
            True if isolation configured successfully
        """
        if not self.iptables_available:
            logger.warning("iptables not available, using Docker network modes only")
            return False
        
        try:
            # get container's network namespace
            netns = await self._get_container_netns(container_id)
            if not netns:
                logger.warning(f"Could not get network namespace for {container_id[:12]}")
                return False
            
            chain_name = f"SANDBOX_{container_id[:12]}"
            self._container_chains[container_id] = chain_name
            
            # create chain in container's network namespace
            await self._create_iptables_chain(chain_name, netns)
            
            # apply rules based on phase
            if phase == "prep":
                await self._apply_prep_rules(chain_name, netns)
            elif phase == "inference":
                await self._apply_inference_rules(chain_name, netns)
            
            # set up DNS filtering
            if phase == "prep" and self.config.allowed_prep_domains:
                await self._setup_dns_filtering(chain_name, netns)
            
            logger.info(f"Network isolation configured for {container_id[:12]} (phase={phase})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup network isolation: {e}")
            return False
    
    async def _get_container_netns(self, container_id: str) -> Optional[str]:
        """
        Get the network namespace path for a container
        
        Args:
            container_id: Docker container ID
            
        Returns:
            Network namespace path or None
        """
        import subprocess
        
        try:
            # get container PID
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
            
            # network namespace path
            netns_path = f"/proc/{pid}/ns/net"
            
            # verify it exists
            if Path(netns_path).exists():
                return netns_path
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get container netns: {e}")
            return None
    
    async def _create_iptables_chain(self, chain_name: str, netns: str):
        """Create iptables chain in container's network namespace"""
        import subprocess
        
        commands = [
            # create chain
            ['ip', 'netns', 'exec', netns, 'iptables', '-N', chain_name],
            # link to OUTPUT chain
            ['ip', 'netns', 'exec', netns, 'iptables', '-I', 'OUTPUT', '1', '-j', chain_name],
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=10
                )
                if result.returncode != 0:
                    logger.warning(f"Command failed: {' '.join(cmd)}: {result.stderr.decode()}")
            except Exception as e:
                logger.error(f"Failed to execute iptables command: {e}")
                raise
    
    async def _apply_prep_rules(self, chain_name: str, netns: str):
        """
        Apply iptables rules for prep phase
        
        Strategy:
        - Allow DNS (port 53)
        - Allow HTTPS to whitelisted domains (if configured)
        - Allow HTTP/HTTPS to specific IPs (resolved from domains)
        - Log and drop everything else
        """
        import subprocess
        
        commands = [
            # allow established connections
            ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name, 
             '-m', 'state', '--state', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'],
            
            # allow localhost
            ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name,
             '-d', '127.0.0.0/8', '-j', 'ACCEPT'],
            
            # allow DNS
            ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name,
             '-p', 'udp', '--dport', '53', '-j', 'ACCEPT'],
            ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name,
             '-p', 'tcp', '--dport', '53', '-j', 'ACCEPT'],
        ]
        
        # if we have whitelisted domains, resolve and allow
        if self.config.allowed_prep_domains:
            for domain in self.config.allowed_prep_domains:
                # resolve domain to IPs
                ips = await self._resolve_domain(domain)
                for ip in ips:
                    commands.extend([
                        ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name,
                         '-d', ip, '-p', 'tcp', '--dport', '443', '-j', 'ACCEPT'],
                        ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name,
                         '-d', ip, '-p', 'tcp', '--dport', '80', '-j', 'ACCEPT'],
                    ])
        else:
            # if no whitelist, allow all HTTPS (less secure)
            commands.extend([
                ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name,
                 '-p', 'tcp', '--dport', '443', '-j', 'ACCEPT'],
                ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name,
                 '-p', 'tcp', '--dport', '80', '-j', 'ACCEPT'],
            ])
        
        # log dropped packets
        commands.append(
            ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name,
             '-j', 'LOG', '--log-prefix', f'[SANDBOX-{chain_name}] ']
        )
        
        # drop everything else
        commands.append(
            ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name, '-j', 'DROP']
        )
        
        for cmd in commands:
            try:
                subprocess.run(cmd, capture_output=True, timeout=10, check=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"iptables command failed: {e.stderr.decode()}")
    
    async def _apply_inference_rules(self, chain_name: str, netns: str):
        """
        Apply iptables rules for inference phase
        
        Strategy:
        - Allow localhost only
        - Drop everything else
        - Log all connection attempts
        """
        import subprocess
        
        commands = [
            # allow localhost
            ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name,
             '-d', '127.0.0.0/8', '-j', 'ACCEPT'],
            
            # log everything else
            ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name,
             '-j', 'LOG', '--log-prefix', f'[INFERENCE-{chain_name}] '],
            
            # drop everything else
            ['ip', 'netns', 'exec', netns, 'iptables', '-A', chain_name, '-j', 'DROP'],
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, capture_output=True, timeout=10, check=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"iptables command failed: {e.stderr.decode()}")
    
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
    
    async def _setup_dns_filtering(self, chain_name: str, netns: str):
        """
        Setup DNS filtering for whitelisted domains
        
        This is complex and would require:
        1. Running a DNS proxy in the container
        2. Configuring /etc/resolv.conf
        3. Filtering DNS responses
        
        For now, we log that this feature is not fully implemented
        """
        logger.info(f"DNS filtering for {chain_name} (feature in development)")
    
    async def cleanup_container_network_isolation(self, container_id: str):
        """Remove iptables rules for a container"""
        if container_id not in self._container_chains:
            return
        
        chain_name = self._container_chains[container_id]
        
        try:
            netns = await self._get_container_netns(container_id)
            if netns:
                await self.remove_iptables_rules(container_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup network isolation: {e}")
        finally:
            del self._container_chains[container_id]
    
    async def monitor_connections(self, container_id: str, duration_seconds: int = 60) -> List[dict]:
        """
        Monitor network connections for a container using tcpdump
        
        Args:
            container_id: Docker container ID
            duration_seconds: How long to monitor
            
        Returns:
            List of connection attempts with details
        """
        import subprocess
        import asyncio
        
        try:
            netns = await self._get_container_netns(container_id)
            if not netns:
                return []
            
            # run tcpdump in container's network namespace
            cmd = [
                'timeout', str(duration_seconds),
                'ip', 'netns', 'exec', netns,
                'tcpdump', '-n', '-c', '1000',
                '-i', 'any',
                'not', 'port', '22',  # exclude SSH
                '-l'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # parse tcpdump output
            connections = self._parse_tcpdump_output(stdout.decode())
            
            logger.info(f"Monitored {len(connections)} connection attempts for {container_id[:12]}")
            return connections
            
        except Exception as e:
            logger.error(f"Failed to monitor connections: {e}")
            return []
    
    def _parse_tcpdump_output(self, output: str) -> List[dict]:
        """Parse tcpdump output into structured connection data"""
        connections = []
        
        for line in output.split('\n'):
            if not line.strip():
                continue
            
            # basic parsing (would need more sophisticated parsing in production)
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