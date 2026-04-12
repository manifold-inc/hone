"""Single source of truth for all hone configuration."""

from __future__ import annotations

import os
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class HoneConfig(BaseSettings):
    """Merged config from CLI args, env vars, and on-chain hparams."""

    model_config = {"env_prefix": "HONE_", "extra": "ignore", "env_file": ".env", "env_file_encoding": "utf-8"}

    # --- Subnet ---
    netuid: int = 5
    subtensor_network: str = "finney"
    subtensor_address: str | None = None
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    wallet_path: str = "~/.bittensor/wallets"

    # --- Model ---
    model_type: str = "llama"
    model_name_or_path: str | None = None
    model_args: dict[str, Any] = Field(default_factory=lambda: {
        "hidden_size": 1024,
        "num_hidden_layers": 9,
        "num_attention_heads": 8,
        "intermediate_size": 2688,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
    })

    # --- SparseLoCo training ---
    blocks_per_window: int = 3
    inner_steps: int = 30
    inner_lr: float = 1.2e-4
    outer_lr: float = 1.0
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0
    batch_size: int = 4
    sequence_length: int = 2048

    # --- Compression ---
    topk_k: int = 64
    chunk_size: int = 4096
    quant_bits: int = 2
    ef_momentum: float = 0.95
    ef_freeze_pct: float = 0.05

    # --- FSDP ---
    fsdp_enabled: bool = False

    # --- Validator ---
    validator_offset: int = 1
    windows_per_weights: int = 10
    gather_peer_count: int = 15
    reserve_peer_count: int = 5
    gather_share: float = 0.75
    reserve_decay_ratio: float = 0.1
    burn_rate: float = 0.0
    openskill_beta: float = 25.0 / 6.0
    openskill_tau: float = 25.0 / 300.0
    bma_warmup_windows: int = 50

    # --- R2 / S3 ---
    r2_gradients_account_id: str = ""
    r2_gradients_bucket_name: str = ""
    r2_gradients_read_access_key_id: str = ""
    r2_gradients_read_secret_access_key: str = ""
    r2_gradients_write_access_key_id: str = ""
    r2_gradients_write_secret_access_key: str = ""
    r2_dataset_account_id: str = ""
    r2_dataset_bucket_name: str = ""
    r2_dataset_read_access_key_id: str = ""
    r2_dataset_read_secret_access_key: str = ""
    dataset_bins_path: str = "/tmp/hone/dataset"

    # --- Hone API ---
    hone_api_url: str = Field(
        default="http://localhost:3001",
        validation_alias=AliasChoices("hone_api_url", "HONE_API_URL"),
    )
    hone_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("hone_api_key", "HONE_API_KEY"),
    )

    @property
    def r2_gradients_endpoint(self) -> str:
        return f"https://{self.r2_gradients_account_id}.r2.cloudflarestorage.com"

    @property
    def r2_dataset_endpoint(self) -> str:
        return f"https://{self.r2_dataset_account_id}.r2.cloudflarestorage.com"

    @property
    def is_rank_zero(self) -> bool:
        return int(os.environ.get("RANK", "0")) == 0

    @property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", "0"))

    @property
    def world_size(self) -> int:
        return int(os.environ.get("WORLD_SIZE", "1"))

    @property
    def rank(self) -> int:
        return int(os.environ.get("RANK", "0"))


def load_config(**overrides: Any) -> HoneConfig:
    """Load config merging env vars with explicit overrides."""
    return HoneConfig(**overrides)
