from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

import aioboto3
import torch
import zstandard

from hone.compress import CompressedTensor
from hone.config import HoneConfig

_zstd_compress = zstandard.ZstdCompressor()
_zstd_decompress = zstandard.ZstdDecompressor()


def _serialize(state_dict: dict[str, CompressedTensor]) -> bytes:
    payload: dict[str, dict[str, Any]] = {}
    for name, ct in state_dict.items():
        cs = getattr(ct, "chunk_size", None)
        if cs is None:
            raise ValueError(f"CompressedTensor {name!r} missing chunk_size")
        payload[name] = {
            "indices": ct.indices.detach().cpu(),
            "values": ct.values.detach().cpu(),
            "quant_params": ct.quant_params.detach().cpu(),
            "shape": list(ct.shape),
            "numel": int(ct.numel),
            "chunk_size": int(cs),
        }
    buf = BytesIO()
    torch.save(payload, buf)
    return _zstd_compress.compress(buf.getvalue())


def _deserialize(data: bytes) -> dict[str, CompressedTensor]:
    raw = _zstd_decompress.decompress(data)
    loaded = torch.load(BytesIO(raw), map_location="cpu", weights_only=False)
    out: dict[str, CompressedTensor] = {}
    for name, d in loaded.items():
        ct = CompressedTensor(
            indices=d["indices"],
            values=d["values"],
            quant_params=d["quant_params"],
            shape=tuple(int(x) for x in d["shape"]),
            numel=int(d["numel"]),
        )
        ct.chunk_size = int(d["chunk_size"])  # type: ignore[attr-defined]
        out[name] = ct
    return out


def _validate(state_dict: dict[str, CompressedTensor], chunk_size: int) -> bool:
    if not state_dict:
        return False
    expected_cs: int | None = None
    for ct in state_dict.values():
        if not isinstance(ct, CompressedTensor):
            return False
        cs_o = getattr(ct, "chunk_size", None)
        if cs_o is None:
            return False
        cs_i = int(cs_o)
        if cs_i != chunk_size:
            return False
        if expected_cs is None:
            expected_cs = cs_i
        elif cs_i != expected_cs:
            return False
        if not torch.isfinite(ct.quant_params).all():
            return False
        idx = ct.indices
        if idx.numel() and ((idx.long() >= chunk_size).any() or (idx.long() < 0).any()):
            return False
        if int(ct.numel) != int(torch.tensor(ct.shape).prod().item()):
            return False
    return True


@asynccontextmanager
async def _s3_client(
    endpoint: str, access_key_id: str, secret_access_key: str,
) -> AsyncGenerator[Any, None]:
    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
    ) as client:
        yield client


class GradientStore:
    def __init__(self, config: HoneConfig) -> None:
        self._cfg = config

    def _key(self, window: int, uid: int) -> str:
        return f"gradients/{window}/{uid}.pt.zst"

    async def put(self, uid: int, window: int, state_dict: dict[str, CompressedTensor]) -> None:
        body = _serialize(state_dict)
        ep = self._cfg.r2_gradients_endpoint
        async with _s3_client(
            ep,
            self._cfg.r2_gradients_write_access_key_id,
            self._cfg.r2_gradients_write_secret_access_key,
        ) as s3:
            await s3.put_object(
                Bucket=self._cfg.r2_gradients_bucket_name,
                Key=self._key(window, uid),
                Body=body,
            )

    async def get(
        self, uid: int, window: int, bucket_info: dict[str, str],
    ) -> dict[str, CompressedTensor] | None:
        try:
            aid = bucket_info["account_id"]
            bucket = bucket_info["bucket_name"]
            ak = bucket_info["access_key_id"]
            sk = bucket_info["secret_access_key"]
            ep = f"https://{aid}.r2.cloudflarestorage.com"
            async with _s3_client(ep, ak, sk) as s3:
                resp = await s3.get_object(Bucket=bucket, Key=self._key(window, uid))
                body = await resp["Body"].read()
            return _deserialize(body)
        except Exception:
            return None

    async def gather(
        self,
        uids: list[int],
        window: int,
        buckets: dict[int, dict[str, str]],
        timeout: float = 30.0,
    ) -> tuple[dict[int, dict[str, CompressedTensor]], list[int]]:
        async def _one(uid: int) -> dict[str, CompressedTensor] | BaseException | None:
            if uid not in buckets:
                return None
            return await asyncio.wait_for(self.get(uid, window, buckets[uid]), timeout=timeout)

        results = await asyncio.gather(*(_one(uid) for uid in uids), return_exceptions=True)
        valid: dict[int, dict[str, CompressedTensor]] = {}
        skipped: list[int] = []
        canonical: set[str] | None = None
        cs = int(self._cfg.chunk_size)
        for uid, res in zip(uids, results, strict=True):
            if isinstance(res, asyncio.CancelledError):
                raise res
            if isinstance(res, Exception) or res is None:
                skipped.append(uid)
                continue
            if not _validate(res, cs):
                skipped.append(uid)
                continue
            keys = set(res.keys())
            if canonical is None:
                canonical = keys
            elif keys != canonical:
                skipped.append(uid)
                continue
            valid[uid] = res
        return valid, skipped
