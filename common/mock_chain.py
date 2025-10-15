import time
from typing import Dict, List, Optional, Any
from loguru import logger
from substrateinterface import Keypair
import random
import hashlib

from .chain import Node, ChainInterface


class MockChainInterface(ChainInterface):
    """
    Mock implementation of ChainInterface for local testing without Bittensor
    """
    
    def __init__(
        self,
        endpoint: str = "mock://localhost",
        netuid: int = 5,
        wallet_name: Optional[str] = None,
        wallet_hotkey: Optional[str] = None,
        wallet_path: Optional[str] = None,
        ss58_format: int = 42,
        num_mock_miners: int = 3
    ):
        self.endpoint = endpoint
        self.netuid = netuid
        self.ss58_format = ss58_format
        self.substrate = None
        self.validator_uid = 0
        
        if wallet_name == "validator":
            seed_bytes = hashlib.sha256(b"validator_test_seed").digest()
            self.keypair = Keypair.create_from_seed(seed_bytes.hex())
        else:
            seed_bytes = hashlib.sha256(b"generic_test_seed").digest()
            self.keypair = Keypair.create_from_seed(seed_bytes.hex())
        
        self.mock_block = 1000
        self.last_weight_set_block = 0
        self.num_mock_miners = num_mock_miners
        self.mock_nodes = self._generate_mock_nodes()
        self.weight_history = []
        
        logger.info(f"🔧 MockChainInterface initialized with {num_mock_miners} mock miners")
    
    def _generate_mock_nodes(self) -> List[Node]:
        """Generate mock nodes for testing"""
        nodes = []
        
        # Add validator (UID 0)
        validator_node = Node(
            hotkey=self.keypair.ss58_address if self.keypair else "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            coldkey="5GNJqTPyNqANBkUVMN1LPPrxXnFouWXoe2wNSmmEoLctxiZY",
            uid=0,
            node_id=0,
            incentive=1.0,
            netuid=self.netuid,
            stake=1000.0,
            trust=1.0,
            vtrust=1.0,
            consensus=1.0,
            last_updated=self.mock_block - 10,
            last_update=self.mock_block - 10,
            ip="127.0.0.1",
            ip_type=4,
            port=8092,
            protocol=4,
            is_validator=True
        )
        nodes.append(validator_node)
        
        for i in range(1, self.num_mock_miners + 1):
            seed_str = f"mock_miner_seed_{i}_test"
            seed_bytes = hashlib.sha256(seed_str.encode()).digest()
            miner_keypair = Keypair.create_from_seed(seed_bytes.hex())
            
            miner_node = Node(
                hotkey=miner_keypair.ss58_address,
                coldkey=f"5MockColdkey{i}{'x' * (47 - len(str(i)) - 12)}",
                uid=i,
                node_id=i,
                incentive=random.uniform(0.1, 0.3),
                netuid=self.netuid,
                stake=random.uniform(10, 100),
                trust=random.uniform(0.5, 1.0),
                vtrust=random.uniform(0.5, 1.0),
                consensus=random.uniform(0.5, 1.0),
                last_updated=self.mock_block - random.randint(5, 50),
                last_update=self.mock_block - random.randint(5, 50),
                ip=f"miner{i}", 
                ip_type=4,
                port=8091 if i == 1 else (8092 if i == 2 else 8093), # for now only support 3 test mock miners
                protocol=4,
                is_validator=False
            )
            nodes.append(miner_node)
        
        logger.info(f"Generated {len(nodes)} mock nodes (1 validator, {self.num_mock_miners} miners)")
        return nodes
    
    def connect(self):
        """Mock connection - just log"""
        logger.info(f"🔧 Mock chain connected to {self.endpoint}")
        self.substrate = "MockSubstrate"
        
        if self.keypair:
            for node in self.mock_nodes:
                if node.hotkey == self.keypair.ss58_address:
                    self.validator_uid = node.uid
                    logger.info(f"🔧 Found validator UID: {self.validator_uid}")
                    break
    
    def get_current_block(self) -> int:
        """Simulate block progression"""
        self.mock_block += 1
        logger.debug(f"🔧 Current mock block: {self.mock_block}")
        return self.mock_block
    
    def get_nodes(self, block: Optional[int] = None) -> List[Node]:
        """Return mock nodes"""
        logger.debug(f"🔧 Returning {len(self.mock_nodes)} mock nodes")
        return self.mock_nodes
    
    def get_miners(self) -> Dict[int, Dict]:
        """Return mock miners"""
        miners = {}
        for node in self.mock_nodes:
            if not node.is_validator:
                miners[node.uid] = node.to_dict()
        logger.debug(f"🔧 Returning {len(miners)} mock miners")
        return miners
    
    def set_weights(
        self,
        uids: List[int],
        weights: List[float],
        version: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True
    ) -> Optional[str]:
        """Mock weight setting - just log and store"""
        
        blocks_since_last = self.mock_block - self.last_weight_set_block
        min_interval = 10
        
        if blocks_since_last < min_interval:
            logger.warning(f"🔧 Mock rate limit: {blocks_since_last} blocks since last update, {min_interval} required")
            return None
        
        logger.info("=" * 60)
        logger.info("🔧 MOCK WEIGHT SETTING")
        logger.info(f"Block: {self.mock_block} | Version: {version}")
        logger.info("-" * 60)
        
        total_weight = sum(weights)
        if total_weight > 0:
            normalized = {uid: w/total_weight for uid, w in zip(uids, weights)}
        else:
            normalized = {uid: 0 for uid in uids}
        
        for uid, weight in zip(uids, weights):
            norm_weight = normalized[uid]
            miner_info = next((n for n in self.mock_nodes if n.uid == uid), None)
            if miner_info:
                logger.info(f"  UID {uid} (port {miner_info.port}): {norm_weight:.4f} ({weight:.0f} raw)")
            else:
                logger.info(f"  UID {uid}: {norm_weight:.4f} ({weight:.0f} raw)")
        
        logger.info("=" * 60)
        
        self.weight_history.append({
            "block": self.mock_block,
            "uids": uids,
            "weights": weights,
            "normalized": normalized,
            "version": version,
            "timestamp": time.time()
        })
        
        self.last_weight_set_block = self.mock_block
        
        if wait_for_finalization:
            logger.info("✅ Mock weights 'finalized' successfully")
        elif wait_for_inclusion:
            logger.info("✅ Mock weights 'included' successfully")
        else:
            logger.info("✅ Mock weights 'submitted' successfully")
        
        return "success"
    
    def _get_validator_uid(self):
        """Mock validator UID lookup"""
        if self.keypair:
            for node in self.mock_nodes:
                if node.hotkey == self.keypair.ss58_address:
                    self.validator_uid = node.uid
                    logger.info(f"🔧 Found mock validator UID: {self.validator_uid}")
                    break
    
    def get_ss58_address(self) -> Optional[str]:
        """Return mock SS58 address"""
        return self.keypair.ss58_address if self.keypair else None
    
    def query_substrate(
        self,
        module: str,
        method: str,
        params: Optional[List[Any]] = None,
        block: Optional[int] = None
    ) -> Any:
        """Mock substrate queries"""
        logger.debug(f"🔧 Mock query: {module}.{method} with params {params}")
        
        if module == "System" and method == "Number":
            return self.get_current_block()
        elif module == "SubtensorModule":
            if method == "Tempo":
                return 360
            elif method == "WeightsSetRateLimit":
                return 10 
            elif method == "LastUpdate":
                return {i: self.mock_block - 20 for i in range(len(self.mock_nodes))}
        
        return 0
    
    def get_weight_history(self) -> List[Dict]:
        """Get history of weight settings for analysis"""
        return self.weight_history


def create_mock_miner_keypairs(count: int = 3) -> List[Keypair]:
    """Create deterministic keypairs for mock miners"""
    keypairs = []
    for i in range(1, count + 1):
        seed_str = f"mock_miner_seed_{i}_test"
        seed_bytes = hashlib.sha256(seed_str.encode()).digest()
        kp = Keypair.create_from_seed(seed_bytes.hex())
        keypairs.append(kp)
        logger.info(f"Mock Miner {i} keypair: {kp.ss58_address}")
    return keypairs


def create_mock_validator_keypair() -> Keypair:
    """Create deterministic keypair for mock validator"""
    seed_bytes = hashlib.sha256(b"validator_test_seed").digest()
    kp = Keypair.create_from_seed(seed_bytes.hex())
    logger.info(f"Mock Validator keypair: {kp.ss58_address}")
    return kp