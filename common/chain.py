import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import os
from async_substrate_interface import SubstrateInterface
from substrateinterface import Keypair
from substrateinterface.exceptions import SubstrateRequestException
from scalecodec.utils.ss58 import ss58_encode
from tenacity import retry, stop_after_attempt, wait_exponential
import bittensor as bt


U16_MAX = 65535
SS58_FORMAT = 42


class Node:
    def __init__(self, hotkey: str, coldkey: str = "", uid: int = 0, **kwargs):
        self.hotkey = hotkey
        self.coldkey = coldkey
        self.uid = uid
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def to_dict(self):
        return vars(self)


def format_error_message(error_message: dict | None) -> str:
    """Format substrate error messages"""
    err_type, err_name, err_description = (
        "UnknownType",
        "UnknownError",
        "Unknown Description",
    )
    if isinstance(error_message, dict):
        err_type = error_message.get("type", err_type)
        err_name = error_message.get("name", err_name)
        err_description = error_message.get("docs", [err_description])[0]
    return f"substrate returned `{err_name} ({err_type})` error. Description: `{err_description}`"


def get_substrate(subtensor_address: str) -> SubstrateInterface:
    """Create a new substrate connection"""
    substrate = SubstrateInterface(
        ss58_format=SS58_FORMAT,
        use_remote_preset=True,
        url=subtensor_address,
    )
    logger.info(f"Connected to {subtensor_address}")
    return substrate


def load_hotkey_keypair(wallet_name: str, hotkey_name: str, wallet_path: Optional[str] = None) -> Keypair:
    """Load hotkey keypair from Bittensor wallet files"""
    base_path = Path(wallet_path) if wallet_path else Path.home() / ".bittensor" / "wallets"
    file_path = base_path / wallet_name / "hotkeys" / hotkey_name
    
    try:
        with open(file_path, "r") as file:
            keypair_data = json.load(file)
        
        if "secretSeed" in keypair_data:
            keypair = Keypair.create_from_seed(keypair_data["secretSeed"])
        elif "secretKey" in keypair_data:
            keypair = Keypair.create_from_seed(keypair_data["secretKey"])
        else:
            raise ValueError("Could not find secret key in hotkey file")
        
        logger.info(f"Loaded keypair from {file_path}")
        return keypair
    except Exception as e:
        raise ValueError(f"Failed to load keypair: {str(e)}")


def query_substrate(
    substrate: SubstrateInterface,
    module: str,
    method: str,
    params: list[Any],
    return_value: bool = True,
    block: int | None = None,
) -> tuple[SubstrateInterface, Any]:
    """Query substrate with reconnection"""
    try:
        block_hash = substrate.get_block_hash(block) if block is not None else None
        query_result = substrate.query(module, method, params, block_hash=block_hash)
        return_val = query_result.value if return_value else query_result
        return substrate, return_val
    except Exception as e:
        logger.debug(f"Substrate query failed with error: {e}. Reconnecting and retrying.")
        substrate = SubstrateInterface(url=substrate.url, ss58_format=SS58_FORMAT, use_remote_preset=True)
        block_hash = substrate.get_block_hash(block) if block is not None else None
        query_result = substrate.query(module, method, params, block_hash=block_hash)
        return_val = query_result.value if return_value else query_result
        return substrate, return_val


def _ss58_encode(address: list[int] | list[list[int]], ss58_format: int = SS58_FORMAT) -> str:
    """Encode SS58 address"""
    if not isinstance(address[0], int):
        address = address[0]
    return ss58_encode(bytes(address).hex(), ss58_format)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def _get_nodes_for_uid(substrate: SubstrateInterface, netuid: int, block: int | None = None):
    """Internal function to get nodes using runtime API"""
    if block is not None:
        block_hash = substrate.get_block_hash(block)
    else:
        block_hash = None

    response = substrate.runtime_call(
        api="SubnetInfoRuntimeApi",
        method="get_metagraph",
        params=[netuid],
        block_hash=block_hash,
    )
    metagraph = response.value

    nodes = []

    for uid in range(len(metagraph["hotkeys"])):
        axon = metagraph["axons"][uid]

        node = Node(
            hotkey=_ss58_encode(metagraph["hotkeys"][uid], SS58_FORMAT),
            coldkey=_ss58_encode(metagraph["coldkeys"][uid], SS58_FORMAT),
            node_id=uid,
            uid=uid,
            incentive=metagraph["incentives"][uid],
            netuid=metagraph["netuid"],
            alpha_stake=metagraph.get("alpha_stake", [0]*256)[uid] * 10**-9,
            tao_stake=metagraph.get("tao_stake", [0]*256)[uid] * 10**-9,
            stake=metagraph["total_stake"][uid] * 10**-9,
            trust=metagraph["trust"][uid],
            vtrust=metagraph["consensus"][uid],
            consensus=metagraph["consensus"][uid],
            last_updated=float(metagraph["last_update"][uid]),
            last_update=float(metagraph["last_update"][uid]),
            ip=str(axon["ip"]),
            ip_type=axon["ip_type"],
            port=axon["port"],
            protocol=axon.get("protocol", 4),
            is_validator=metagraph['validator_permit'][uid]
        )
        nodes.append(node)
    logger.info(f"Got {len(nodes)} nodes for Subnet {netuid}")
    return nodes


def get_nodes_for_netuid(substrate: SubstrateInterface, netuid: int, block: int | None = None) -> list[Node]:
    """Get nodes for a specific netuid"""
    substrate = get_substrate(subtensor_address=substrate.url)
    return _get_nodes_for_uid(substrate, netuid, block)


def _normalize_and_quantize_weights(node_ids: list[int], node_weights: list[float]) -> tuple[list[int], list[int]]:
    """
    Normalize and quantize weights for chain submission.
    
    CRITICAL: Returns ALL UIDs with their weights (including zeros) to prevent 
    commit-reveal and chain normalization issues.
    """
    if len(node_ids) != len(node_weights) or any(uid < 0 for uid in node_ids) or any(weight < 0 for weight in node_weights):
        raise ValueError("Invalid input: length mismatch or negative values")
    if not any(node_weights):
        return [], []
    
    total_weight = sum(node_weights)
    scaling_factor = U16_MAX / total_weight
    
    node_weights_formatted = []
    node_ids_formatted = []
    
    for node_id, node_weight in zip(node_ids, node_weights):
        node_ids_formatted.append(node_id)
        quantized_weight = round(node_weight * scaling_factor) if node_weight > 0 else 0
        node_weights_formatted.append(quantized_weight)
    
    actual_sum = sum(node_weights_formatted)
    drift = U16_MAX - actual_sum
    if drift != 0:
        max_idx = max(range(len(node_weights_formatted)), key=lambda i: node_weights_formatted[i])
        node_weights_formatted[max_idx] += drift
    
    return node_ids_formatted, node_weights_formatted


def blocks_since_last_update(substrate: SubstrateInterface, netuid: int, node_id: int) -> int:
    """Get blocks since last update for a node"""
    substrate, current_block = query_substrate(substrate, "System", "Number", [], return_value=True)
    substrate, last_updated_value = query_substrate(substrate, "SubtensorModule", "LastUpdate", [netuid], return_value=False)
    
    try:
        updated: int = current_block - last_updated_value[node_id]
    except TypeError:
        updated: int = current_block - last_updated_value[node_id].value
    return updated


def min_interval_to_set_weights(substrate: SubstrateInterface, netuid: int) -> int:
    """Get minimum interval required to set weights"""
    substrate, weights_set_rate_limit = query_substrate(
        substrate, "SubtensorModule", "WeightsSetRateLimit", [netuid], return_value=True
    )
    assert isinstance(weights_set_rate_limit, int), "WeightsSetRateLimit should be an int"
    return weights_set_rate_limit


def can_set_weights(substrate: SubstrateInterface, netuid: int, validator_node_id: int) -> bool:
    """Check if weights can be set"""
    blocks_since_update = blocks_since_last_update(substrate, netuid, validator_node_id)
    min_interval = min_interval_to_set_weights(substrate, netuid)
    if min_interval is None:
        return True

    can_set_weights = blocks_since_update is not None and blocks_since_update >= min_interval
    if not can_set_weights:
        logger.error(
            f"It is too soon to set weights! {blocks_since_update} blocks since last update, {min_interval} blocks required."
        )
    return can_set_weights


def set_node_weights(
    substrate: SubstrateInterface,
    keypair: Keypair,
    node_ids: list[int],
    node_weights: list[float],
    netuid: int,
    validator_node_id: int,
    version_key: int = 0,
    wait_for_inclusion: bool = True, 
    wait_for_finalization: bool = True,
    wallet_name: Optional[str] = None,
    wallet_hotkey: Optional[str] = None,
    wallet_path: Optional[str] = None,
) -> bool:
    """Set node weights with all checks"""
    
    logger.info("=" * 60)
    logger.info("Setting weights for the following UIDs:")
    total_weight = sum(node_weights)
    for uid, weight in zip(node_ids, node_weights):
        if weight > 0:
            percentage = (weight / total_weight * 100) if total_weight > 0 else 0
            logger.info(f"  UID {uid:>3} - Weight: {percentage:>6.2f}%")
    logger.info("=" * 60)
    
    node_ids_formatted, node_weights_formatted = _normalize_and_quantize_weights(node_ids, node_weights)

    substrate = get_substrate(subtensor_address=substrate.url)

    if not can_set_weights(substrate, netuid, validator_node_id):
        substrate.close()
        return False

    substrate, commit_reveal_enabled = query_substrate(
        substrate,
        "SubtensorModule",
        "CommitRevealWeightsEnabled",
        [netuid],
        return_value=True,
    )

    logger.info(f"Commit reveal enabled: {commit_reveal_enabled}")

    if commit_reveal_enabled:
        substrate.close()
        
        if not wallet_name or not wallet_hotkey:
            logger.error("Commit-reveal requires wallet_name and wallet_hotkey")
            return False
        
        try:
            config = bt.subtensor.config()
            config.subtensor.chain_endpoint = substrate.url
            config.subtensor.network = 'finney'
            
            subtensor = bt.subtensor(config=config)
            
            total_weight_formatted = sum(node_weights_formatted)
            if total_weight_formatted > 0:
                node_weights_float = [w / total_weight_formatted for w in node_weights_formatted]
            else:
                node_weights_float = node_weights_formatted
            
            wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey, path=wallet_path)
            
            logger.info("Quantized weights (ticks) being sent to chain:")
            for uid, weight_tick, weight_float in zip(node_ids_formatted, node_weights_formatted, node_weights_float):
                if weight_tick > 0:
                    logger.info(f"  UID {uid:>3} - Ticks: {weight_tick:>5} - Normalized: {weight_float:.8f}")
            
            result, msg = subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=node_ids_formatted,
                weights=node_weights_float,
                version_key=version_key,
                wait_for_finalization=wait_for_finalization,
                wait_for_inclusion=wait_for_inclusion,
            )
            
            if result:
                logger.info("✅ Successfully set weights using commit-reveal")
                return True
            else:
                logger.error(f"❌ Failed to set weights: {msg}")
                return False
                
        except Exception as e:
            logger.error(f"Exception during commit-reveal weight setting: {e}")
            return False

class ChainInterface:
    
    def __init__(
        self,
        endpoint: str,
        netuid: int,
        wallet_name: Optional[str] = None,
        wallet_hotkey: Optional[str] = None,
        wallet_path: Optional[str] = None,
        ss58_format: int = SS58_FORMAT
    ):
        self.endpoint = endpoint
        self.netuid = netuid
        self.ss58_format = ss58_format
        self.substrate: Optional[SubstrateInterface] = None
        self.keypair: Optional[Keypair] = None
        self.validator_uid: Optional[int] = None
        self.wallet_name = wallet_name
        self.wallet_hotkey = wallet_hotkey
        self.wallet_path = wallet_path
        
        if wallet_name and wallet_hotkey:
            self.keypair = load_hotkey_keypair(wallet_name, wallet_hotkey, wallet_path)
    
    def connect(self):
        """Connect to the chain"""
        if self.substrate:
            return
        
        self.substrate = get_substrate(self.endpoint)
        
        if self.keypair:
            self._get_validator_uid()
    
    def _get_validator_uid(self):
        """Get the UID for our validator hotkey"""
        if not self.keypair or not self.substrate:
            return
        
        nodes = self.get_nodes()
        for node in nodes:
            if node.hotkey == self.keypair.ss58_address:
                self.validator_uid = node.uid
                logger.info(f"Found validator UID: {self.validator_uid}")
                break
    
    def get_ss58_address(self) -> Optional[str]:
        """Returns the SS58 address of the loaded keypair, if any"""
        return self.keypair.ss58_address if self.keypair else None
    
    def get_current_block(self) -> int:
        """Get current block number"""
        assert self.substrate, "Call connect() first"
        header = self.substrate.get_block_header()
        return int(header['header']['number'])
    
    def get_nodes(self, block: Optional[int] = None) -> List[Node]:
        """Get all nodes in the subnet"""
        assert self.substrate, "Call connect() first"
        return get_nodes_for_netuid(self.substrate, self.netuid, block)
    
    def get_miners(self) -> Dict[int, Dict]:
        """Get miners"""
        nodes = self.get_nodes()
        miners = {}
        for node in nodes:
            if not node.is_validator:
                miners[node.uid] = node.to_dict()
        return miners
    
    def set_weights(
        self,
        uids: List[int],
        weights: List[float],
        version: int = 803, # from subnet hyperparameters
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True
    ) -> Optional[str]:
        """
        Set weights for specified UIDs.
        
        IMPORTANT: Pass ALL UIDs in the subnet with their weights (including 0.0 for unused UIDs)
        to prevent commit-reveal issues.
        
        Args:
            uids: List of ALL UIDs in subnet (e.g., [0, 1, 2, ..., 255])
            weights: List of weights corresponding to UIDs (e.g., [0.0, 0.5, 0.0, ..., 0.5])
        """
        
        assert self.substrate, "Call connect() first"
        assert self.keypair, "Keypair required to sign transactions"
        
        if self.validator_uid is None:
            self._get_validator_uid()
            if self.validator_uid is None:
                raise ValueError("Could not find validator UID for keypair")
        
        success = set_node_weights(
            substrate=self.substrate,
            keypair=self.keypair,
            node_ids=uids,
            node_weights=weights,
            netuid=self.netuid,
            validator_node_id=self.validator_uid,
            version_key=version,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wallet_name=self.wallet_name,
            wallet_hotkey=self.wallet_hotkey,
            wallet_path=self.wallet_path,
        )
        
        if success:
            return "success"
        else:
            raise SubstrateRequestException("Failed to set weights")
    
    def query_substrate(
        self,
        module: str,
        method: str,
        params: Optional[List[Any]] = None,
        block: Optional[int] = None
    ) -> Any:
        """Generic substrate query method"""
        assert self.substrate, "Call connect() first"
        params = params or []
        substrate, value = query_substrate(self.substrate, module, method, params, return_value=True, block=block)
        self.substrate = substrate
        return value