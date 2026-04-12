from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import boto3
import numpy as np
from botocore.config import Config as BotoConfig
from bittensor import Subtensor, Wallet

from hone.config import HoneConfig


def _retry(fn: Callable[[], Any], attempts: int = 4, delay: float = 1.0) -> Any:
    err: BaseException | None = None
    for _ in range(attempts):
        try:
            return fn()
        except BaseException as e:
            err = e
            time.sleep(delay)
    assert err is not None
    raise err


def _s3(endpoint: str, key: str, secret: str) -> Any:
    cfg = BotoConfig(signature_version="s3v4")
    return boto3.client(
        "s3", endpoint_url=endpoint, aws_access_key_id=key, aws_secret_access_key=secret,
        region_name="auto", config=cfg,
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
            name=self.config.wallet_name, hotkey=self.config.wallet_hotkey,
            path=os.path.expanduser(self.config.wallet_path),
        )
        st, nu = self.subtensor, self.config.netuid
        self.metagraph = _retry(lambda: st.metagraph(nu, lite=True))  # type: ignore[union-attr]

    def sync_metagraph(self) -> None:
        if not self.subtensor:
            raise RuntimeError("not connected")
        st, nu = self.subtensor, self.config.netuid
        self.metagraph = _retry(lambda: st.metagraph(nu, lite=True))  # type: ignore[union-attr]

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
        w, st, nu = self.wallet, self.subtensor, self.config.netuid
        _retry(lambda: st.set_weights(wallet=w, netuid=nu, uids=uids, weights=weights))  # type: ignore[union-attr]

    def commit_bucket(self, bucket_info: dict[str, str]) -> None:
        if not self.subtensor or not self.wallet:
            raise RuntimeError("not connected")
        s = ":".join(
            (bucket_info["account_id"], bucket_info["bucket_name"],
             bucket_info["access_key_id"], bucket_info["secret_access_key"])
        )
        pub = getattr(self.subtensor, "commit", self.subtensor.set_commitment)
        w, st, nu = self.wallet, self.subtensor, self.config.netuid
        _retry(lambda: pub(w, nu, s))  # type: ignore[misc]

    def get_commitments(self) -> dict[int, dict[str, str]]:
        if not self.subtensor or not self.metagraph:
            raise RuntimeError("not connected")
        out: dict[int, dict[str, str]] = {}
        st, mg, nu = self.subtensor, self.metagraph, self.config.netuid
        for uid in range(int(np.asarray(mg.n).item())):
            raw = _retry(lambda u=uid: st.get_commitment(nu, u))  # type: ignore[union-attr]
            if not raw:
                continue
            p = raw.split(":", 3)
            if len(p) != 4:
                continue
            out[uid] = dict(zip(("account_id", "bucket_name", "access_key_id", "secret_access_key"), p))
        return out

    def commit_hparams(self, hparams: dict[str, Any]) -> None:
        if not self.config.is_rank_zero:
            return
        b = _s3(
            self.config.r2_gradients_endpoint,
            self.config.r2_gradients_write_access_key_id,
            self.config.r2_gradients_write_secret_access_key,
        )
        bn, body = self.config.r2_gradients_bucket_name, json.dumps(hparams).encode()
        _retry(lambda: b.put_object(Bucket=bn, Key="hparams.json", Body=body, ContentType="application/json"))

    def get_hparams(self) -> dict[str, Any] | None:
        if not self.subtensor:
            return None
        st, nu = self.subtensor, self.config.netuid
        mg = self.metagraph or _retry(lambda: st.metagraph(nu, lite=True))  # type: ignore[union-attr]
        raw = _retry(lambda: st.get_commitment(nu, get_highest_stake_uid(mg)))  # type: ignore[union-attr]
        if not raw:
            return None
        p = raw.split(":", 3)
        if len(p) != 4:
            return None
        c = _s3(f"https://{p[0]}.r2.cloudflarestorage.com", p[2], p[3])
        try:
            blob = _retry(lambda: c.get_object(Bucket=p[1], Key="hparams.json")["Body"].read())
        except BaseException:
            return None
        return json.loads(blob)

    async def block_listener(self) -> AsyncIterator[tuple[int, int]]:
        if not self.subtensor:
            raise RuntimeError("not connected")
        bpw = self.config.blocks_per_window
        st = self.subtensor
        block = _retry(st.get_current_block)  # type: ignore[union-attr]
        cur_w = block // bpw
        while True:
            await asyncio.sleep(2)
            nb = _retry(st.get_current_block)  # type: ignore[union-attr]
            nw = nb // bpw
            if nw != cur_w:
                cur_w = nw
                yield nb, nw
