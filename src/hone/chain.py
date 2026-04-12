from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import boto3
import numpy as np
import torch
from botocore.config import Config as BotoConfig
from bittensor import Subtensor, Wallet

from hone.config import HoneConfig

logger = logging.getLogger(__name__)


def _retry(fn: Callable[[], Any], attempts: int = 3, delay: float = 2.0) -> Any:
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            logger.debug("retry %d/%d after %s", i + 1, attempts, e)
            time.sleep(delay)


def _s3(endpoint: str, key: str, secret: str) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name="auto",
        config=BotoConfig(signature_version="s3v4"),
    )


def _has_r2_creds(config: HoneConfig) -> bool:
    return bool(
        config.r2_gradients_account_id
        and config.r2_gradients_bucket_name
        and config.r2_gradients_write_access_key_id
        and config.r2_gradients_write_secret_access_key
    )


def get_highest_stake_uid(metagraph: Any) -> int:
    return int(np.argmax(np.asarray(metagraph.S)))


class ChainManager:
    def __init__(self, config: HoneConfig) -> None:
        self.config = config
        self.subtensor: Subtensor | None = None
        self.wallet: Wallet | None = None
        self.metagraph: Any | None = None

    def connect(self) -> None:
        net = self.config.subtensor_address or self.config.subtensor_network
        self.subtensor = Subtensor(network=net)
        self.wallet = Wallet(
            name=self.config.wallet_name,
            hotkey=self.config.wallet_hotkey,
            path=os.path.expanduser(self.config.wallet_path),
        )
        self.sync_metagraph()

    def sync_metagraph(self) -> None:
        if not self.subtensor:
            raise RuntimeError("not connected")
        self.metagraph = self.subtensor.metagraph(self.config.netuid, lite=True)

    def get_uid(self) -> int | None:
        if not self.wallet or not self.metagraph:
            return None
        try:
            return self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            return None

    def set_weights(self, uids: list[int], weights: list[float]) -> None:
        if not self.subtensor or not self.wallet:
            raise RuntimeError("not connected")
        uid_tensor = torch.tensor(uids, dtype=torch.int64)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        _retry(lambda: self.subtensor.set_weights(
            wallet=self.wallet, netuid=self.config.netuid,
            uids=uid_tensor, weights=weight_tensor,
        ))

    def commit_bucket(self, bucket_info: dict[str, str]) -> None:
        if not self.subtensor or not self.wallet:
            raise RuntimeError("not connected")
        vals = [bucket_info.get(k, "") for k in ("account_id", "bucket_name", "access_key_id", "secret_access_key")]
        if not all(vals):
            logger.warning("Skipping commit_bucket: missing credentials")
            return
        data = ":".join(vals)
        commit_fn = getattr(self.subtensor, "commit", getattr(self.subtensor, "set_commitment", None))
        if commit_fn is None:
            logger.warning("subtensor has no commit/set_commitment method")
            return
        _retry(lambda: commit_fn(self.wallet, self.config.netuid, data))

    def get_commitments(self) -> dict[int, dict[str, str]]:
        if not self.subtensor or not self.metagraph:
            raise RuntimeError("not connected")
        out: dict[int, dict[str, str]] = {}
        keys = ("account_id", "bucket_name", "access_key_id", "secret_access_key")
        n = int(np.asarray(self.metagraph.n).item())
        for uid in range(n):
            try:
                raw = self.subtensor.get_commitment(self.config.netuid, uid)
            except Exception:
                continue
            if not raw:
                continue
            parts = raw.split(":", 3)
            if len(parts) == 4 and all(parts):
                out[uid] = dict(zip(keys, parts))
        return out

    def commit_hparams(self, hparams: dict[str, Any]) -> None:
        if not self.config.is_rank_zero:
            return
        if not _has_r2_creds(self.config):
            logger.warning("Skipping commit_hparams: no R2 credentials configured")
            return
        client = _s3(
            self.config.r2_gradients_endpoint,
            self.config.r2_gradients_write_access_key_id,
            self.config.r2_gradients_write_secret_access_key,
        )
        body = json.dumps(hparams).encode()
        _retry(lambda: client.put_object(
            Bucket=self.config.r2_gradients_bucket_name,
            Key="hparams.json", Body=body, ContentType="application/json",
        ))

    def get_hparams(self) -> dict[str, Any] | None:
        if not self.subtensor or not self.metagraph:
            return None
        try:
            top_uid = get_highest_stake_uid(self.metagraph)
            raw = self.subtensor.get_commitment(self.config.netuid, top_uid)
        except Exception:
            return None
        if not raw:
            return None
        parts = raw.split(":", 3)
        if len(parts) != 4 or not all(parts):
            return None
        try:
            client = _s3(f"https://{parts[0]}.r2.cloudflarestorage.com", parts[2], parts[3])
            blob = client.get_object(Bucket=parts[1], Key="hparams.json")["Body"].read()
            return json.loads(blob)
        except Exception:
            logger.debug("Failed to load hparams from top validator", exc_info=True)
            return None

    async def block_listener(self) -> AsyncIterator[tuple[int, int]]:
        if not self.subtensor:
            raise RuntimeError("not connected")
        bpw = self.config.blocks_per_window
        block = self.subtensor.get_current_block()
        cur_window = block // bpw
        while True:
            await asyncio.sleep(2)
            try:
                block = self.subtensor.get_current_block()
            except Exception:
                logger.debug("block poll failed", exc_info=True)
                continue
            new_window = block // bpw
            if new_window != cur_window:
                cur_window = new_window
                yield block, new_window
