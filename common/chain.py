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
    """Normalize and quantize weights for chain submission"""
    if len(node_ids) != len(node_weights) or any(uid < 0 for uid in node_ids) or any(weight < 0 for weight in node_weights):
        raise ValueError("Invalid input: length mismatch or negative values")
    if not any(node_weights):
        return [], []
    
    scaling_factor = U16_MAX / max(node_weights)
    
    node_weights_formatted = []
    node_ids_formatted = []
    for node_id, node_weight in zip(node_ids, node_weights):
        if node_weight > 0:
            node_ids_formatted.append(node_id)
            node_weights_formatted.append(round(node_weight * scaling_factor))
    
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


def _send_weights_to_chain(
    substrate: SubstrateInterface,
    keypair: Keypair,
    node_ids: list[int],
    node_weights: list[int],
    netuid: int,
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str | None]:
    """Send weights to chain with retries"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=5),
        reraise=True,
    )
    def _send_weights():
        try:
            with substrate as si:
                rpc_call = si.compose_call(
                    call_module="SubtensorModule",
                    call_function="set_weights",
                    call_params={
                        "dests": node_ids,
                        "weights": node_weights,
                        "netuid": netuid,
                        "version_key": version_key,
                    },
                )
                extrinsic_to_send = si.create_signed_extrinsic(
                    call=rpc_call, 
                    keypair=keypair, 
                    era={"period": 5}
                )
                
                response = si.submit_extrinsic(
                    extrinsic_to_send,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                
                if not wait_for_finalization and not wait_for_inclusion:
                    return True, "Not waiting for finalization or inclusion."
                
                response.process_events()
                
                if response.is_success:
                    return True, "Successfully set weights."
                
                return False, format_error_message(response.error_message)
                
        except Exception as e:
            logger.exception(f"Exception in _send_weights: {str(e)}")
            raise
    
    return _send_weights()


def set_node_weights(
    substrate: SubstrateInterface,
    keypair: Keypair,
    node_ids: list[int],
    node_weights: list[float],
    netuid: int,
    validator_node_id: int,
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    wallet_name: Optional[str] = None,
    wallet_hotkey: Optional[str] = None,
    wallet_path: Optional[str] = None,
) -> bool:
    """Set node weights with all checks"""
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
            config.subtensor.network = None
            
            subtensor = bt.subtensor(config=config)
            
            max_weight = max(node_weights_formatted) if node_weights_formatted else 1
            node_weights_float = [w / max_weight for w in node_weights_formatted]
            
            wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey, path=wallet_path)
            
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

    logger.info(f"Setting weights for subnet {netuid} with version key {version_key}...")
    
    try:
        success, error_message = _send_weights_to_chain(
            substrate,
            keypair,
            node_ids_formatted,
            node_weights_formatted,
            netuid,
            version_key,
            wait_for_inclusion,
            wait_for_finalization,
        )
    except Exception as e:
        logger.error(f"Exception during weight setting: {e}")
        substrate.close()
        return False

    if success:
        if wait_for_finalization:
            logger.info("✅ Successfully set weights and finalized")
        elif wait_for_inclusion:
            logger.info("✅ Successfully set weights and included")
        else:
            logger.info("✅ Successfully set weights")
    else:
        logger.error(f"❌ Failed to set weights: {error_message}")

    substrate.close()
    return success


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
        version: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True
    ) -> Optional[str]:
        
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