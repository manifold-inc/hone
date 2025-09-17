from typing import Union
import time
from datetime import timedelta
import hashlib
from substrateinterface import Keypair

from .constants import BLOCK_TIME

def ss58_to_account_id(ss58_address: str) -> bytes:
    """
    Convert SS58 address to AccountId bytes
    
    Args:
        ss58_address: SS58 encoded address
        
    Returns:
        32-byte AccountId
    """
    keypair = Keypair(ss58_address=ss58_address)
    return keypair.public_key


def account_id_to_ss58(account_id: Union[bytes, str], ss58_format: int = 42) -> str:
    """
    Convert AccountId to SS58 address
    
    Args:
        account_id: 32-byte AccountId or hex string
        ss58_format: SS58 format (42 for Substrate)
        
    Returns:
        SS58 encoded address
    """
    if isinstance(account_id, str):
        if account_id.startswith('0x'):
            account_id = account_id[2:]
        account_id = bytes.fromhex(account_id)
    
    keypair = Keypair(public_key=account_id, ss58_format=ss58_format)
    return keypair.ss58_address


def calculate_next_epoch_block(current_block: int, epoch_length: int = 360) -> int:
    """
    Calculate the next epoch block number
    
    Args:
        current_block: Current block number
        epoch_length: Length of epoch in blocks
        
    Returns:
        Next epoch block number
    """
    return ((current_block // epoch_length) + 1) * epoch_length


def calculate_time_to_next_epoch(current_block: int, epoch_length: int = 360) -> timedelta:
    """
    Calculate time until next epoch
    
    Args:
        current_block: Current block number
        epoch_length: Length of epoch in blocks
        
    Returns:
        Time until next epoch
    """
    next_epoch_block = calculate_next_epoch_block(current_block, epoch_length)
    blocks_until_epoch = next_epoch_block - current_block
    seconds_until_epoch = blocks_until_epoch * BLOCK_TIME
    return timedelta(seconds=seconds_until_epoch)


def normalize_scores(scores: dict) -> dict:
    """
    Normalize scores to sum to 1.0
    
    Args:
        scores: Dict of UID -> score
        
    Returns:
        Normalized scores
    """
    total = sum(scores.values())
    if total > 0:
        return {uid: score / total for uid, score in scores.items()}
    else:
        # if all scores are 0, distribute equally
        num_miners = len(scores)
        if num_miners > 0:
            equal_score = 1.0 / num_miners
            return {uid: equal_score for uid in scores}
        else:
            return {}


def scores_to_weights(scores: dict, max_weight: int = 65535) -> dict:
    """
    Convert normalized scores to integer weights
    
    Args:
        scores: Dict of UID -> normalized score (0-1)
        max_weight: Maximum weight value
        
    Returns:
        Dict of UID -> weight
    """
    weights = {}
    remaining = max_weight
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (uid, score) in enumerate(sorted_scores):
        if i == len(sorted_scores) - 1:
            weights[uid] = remaining
        else:
            weight = int(score * max_weight)
            weights[uid] = weight
            remaining -= weight
    
    return weights


def create_nonce() -> int:
    """Create a nonce for Epistula requests"""
    return time.time_ns()


def is_valid_ss58_address(address: str) -> bool:
    """
    Check if a string is a valid SS58 address
    
    Args:
        address: Address to check
        
    Returns:
        True if valid SS58 address
    """
    try:
        Keypair(ss58_address=address)
        return True
    except:
        return False


def hash_data(data: Union[str, bytes]) -> str:
    """
    Create SHA256 hash of data
    
    Args:
        data: Data to hash
        
    Returns:
        Hex string of hash
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def format_stake(stake_rao: int) -> str:
    """
    Format stake from rao to TAO with proper units
    
    Args:
        stake_rao: Stake in rao (1e-9 TAO)
        
    Returns:
        Formatted string
    """
    stake_tao = stake_rao / 1e9
    
    if stake_tao >= 1000:
        return f"{stake_tao/1000:.2f}k τ"
    elif stake_tao >= 1:
        return f"{stake_tao:.2f} τ"
    else:
        return f"{stake_tao:.4f} τ"


def retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    Retry a function with exponential backoff
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise last_exception