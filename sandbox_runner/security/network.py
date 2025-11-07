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


class IptablesNetworkPolicy(NetworkPolicy):
    """
    Advanced network policy using iptables/nftables.
    
    ---
    TODO: 
    - Fine-grained domain whitelisting
    - Connection monitoring and logging
    - Real-time blocking of unauthorized connections
    - DNS filtering
    ----    
    """
    
    def __init__(self, config: NetworkPolicyConfig):
        super().__init__(config)
        logger.info("IptablesNetworkPolicy initialized (Phase 4 feature - coming soon)")
    
    async def setup_iptables_rules(self, container_id: str, phase: str):
        """
        Setup iptables rules for container        
        """
        # TODO: Get container network namespace
        # TODO: Create iptables chain for container
        # TODO: Add rules to allow/block based on config
        # TODO: Log connection attempts
        pass
    
    async def remove_iptables_rules(self, container_id: str):
        """
        Remove iptables rules for container     
        """
        # TODO: Remove iptables chain
        pass