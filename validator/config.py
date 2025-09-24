# validator/config.py
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from common.constants import (
    MAINNET_ENDPOINT,
    VALIDATOR_PORT,
    NETUID_MAINNET,
)

@dataclass
class ValidatorConfig:
    netuid: int = int(os.getenv("NETUID", str(NETUID_MAINNET)))
    chain_endpoint: str = os.getenv("CHAIN_ENDPOINT", MAINNET_ENDPOINT)
    validator_port: int = int(os.getenv("VALIDATOR_PORT", str(VALIDATOR_PORT)))
    
    wallet_name: Optional[str] = os.getenv("WALLET_NAME")
    wallet_hotkey: Optional[str] = os.getenv("WALLET_HOTKEY")
    wallet_path: Optional[str] = os.getenv("WALLET_PATH", "~/.bittensor/wallets")

    default_miner_port: int = int(os.getenv("MINER_PORT", "8091"))

    db_url: str = os.getenv("DB_URL")
    
    # Add the mock chain flag
    use_mock_chain: bool = os.getenv("USE_MOCK_CHAIN", "false").lower() == "true"

    # Use private field for cycle_duration to avoid property/field conflict
    _cycle_duration: int = field(default_factory=lambda: int(os.getenv("CYCLE_DURATION", "30")))
    
    current_block_provider: Callable[[], int] = field(default=lambda: 0)
    
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