import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from common.constants import (
    MAINNET_ENDPOINT,
    VALIDATOR_PORT,
    NETUID_MAINNET,
)

import bittensor as bt

def resolve_hotkey(
    wallet_name: str | None, 
    wallet_hotkey: str | None,
    wallet_path: str | None
) -> Optional[str]:
    """
    Returns the SS58 hotkey address for this wallet, or None if it can't be loaded.
    """
    if not wallet_name:
        return None

    try:
        expanded_path = os.path.expanduser(wallet_path) if wallet_path else None
        
        wallet = bt.wallet(
            name=wallet_name,
            hotkey=wallet_hotkey,
            path=expanded_path,
        )
        return wallet.hotkey.ss58_address
    except Exception as e:
        print(f"Could not resolve wallet hotkey ss58: {e}")
        return None


@dataclass
class ValidatorConfig:
    netuid: int = int(os.getenv("NETUID", str(NETUID_MAINNET)))
    chain_endpoint: str = os.getenv("CHAIN_ENDPOINT", MAINNET_ENDPOINT)
    validator_port: int = int(os.getenv("VALIDATOR_PORT", str(VALIDATOR_PORT)))
    
    wallet_name: Optional[str] = os.getenv("WALLET_NAME")
    wallet_hotkey: Optional[str] = os.getenv("WALLET_HOTKEY")
    wallet_path: Optional[str] = os.getenv("WALLET_PATH", "~/.bittensor/wallets")

    hotkey: Optional[str] = resolve_hotkey(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey, wallet_path=wallet_path)

    default_miner_port: int = int(os.getenv("MINER_PORT", "8091"))

    db_url: str = os.getenv("DB_URL")
    
    use_mock_chain: bool = os.getenv("USE_MOCK_CHAIN", "false").lower() == "true"

    _cycle_duration: int = field(default_factory=lambda: int(os.getenv("CYCLE_DURATION", "30")))
    
    current_block_provider: Callable[[], int] = field(default=lambda: 0)

    retention_days: int = int(os.getenv("RETENTION_DAYS", "30"))
    cleanup_interval_hours: int = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))
    
    # sandbox runner configuration
    sandbox_runner_endpoint: str = os.getenv("SANDBOX_RUNNER_ENDPOINT")
    sandbox_runner_api_key: Optional[str] = os.getenv("SANDBOX_RUNNER_API_KEY")
    sandbox_runner_timeout_hours: int = int(os.getenv("SANDBOX_RUNNER_TIMEOUT_HOURS", "3"))
    
    # job polling configuration
    sandbox_poll_interval_seconds: int = int(os.getenv("SANDBOX_POLL_INTERVAL", "30"))
    sandbox_max_poll_attempts: int = int(os.getenv("SANDBOX_MAX_POLL_ATTEMPTS", "360"))  # 3h with 30s intervals
    
    # miner info fetching
    miner_info_timeout_seconds: int = int(os.getenv("MINER_INFO_TIMEOUT", "5"))
    
    @property
    def cycle_duration(self) -> int:
        """Duration of each query cycle in blocks"""
        return self._cycle_duration
    
    @property
    def query_interval_blocks(self) -> int:
        """Minimum blocks between query cycles"""
        return self.cycle_duration + 5
    
    @property
    def weights_interval_blocks(self) -> int:
        """Minimum blocks between weight settings"""
        return self.cycle_duration + 5
    
    @property
    def score_window_blocks(self) -> int:
        """Look back window for scoring (e.g., last 4 cycles)"""
        return self.cycle_duration * 4
    
    @property
    def min_responses(self) -> int:
        return 1
    
    @property
    def idle_sleep_seconds(self) -> int:
        return 2