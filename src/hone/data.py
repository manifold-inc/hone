"""Memory-mapped sharded pretokenized data for LM training."""

from __future__ import annotations

import bisect
import logging
import os
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

if TYPE_CHECKING:
    from hone.config import HoneConfig

logger = logging.getLogger(__name__)


class ShardedDataset(Dataset[dict[str, torch.Tensor]]):
    """Contiguous int32 token streams split into fixed-length LM windows."""

    def __init__(self, shard_paths: list[str], sequence_length: int) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        self._sequence_length = sequence_length
        self._window = sequence_length + 1

        self._mmaps: list[np.memmap] = []
        self._prefix: list[int] = [0]
        total = 0
        for path in shard_paths:
            mm = np.memmap(path, dtype=np.int32, mode="r")
            self._mmaps.append(mm)
            total += int(mm.shape[0])
            self._prefix.append(total)

        self._num_sequences = total // self._window
        if self._num_sequences < 1:
            raise ValueError(
                f"Need at least {self._window} tokens across shards; got {total}"
            )

    def __len__(self) -> int:
        return self._num_sequences

    def _read_window(self, start: int) -> np.ndarray:
        end = start + self._window
        out = np.empty(self._window, dtype=np.int32)
        offset = 0
        pos = start
        while pos < end:
            shard_idx = bisect.bisect_right(self._prefix, pos) - 1
            shard = self._mmaps[shard_idx]
            base = self._prefix[shard_idx]
            local = pos - base
            chunk = min(len(shard) - local, end - pos)
            out[offset : offset + chunk] = shard[local : local + chunk]
            offset += chunk
            pos += chunk
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._num_sequences:
            raise IndexError(idx)
        start = idx * self._window
        window = self._read_window(start)
        tokens = torch.from_numpy(window.astype(np.int64, copy=False))
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def load_shards(data_dir: str) -> list[str]:
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(data_dir)
    names = sorted(f for f in os.listdir(data_dir) if f.endswith(".bin"))
    paths = [os.path.join(data_dir, f) for f in names]
    return [p for p in paths if os.path.isfile(p)]


def build_dataloader(
    config: HoneConfig,
    shard_paths: list[str],
    seed: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    import torch.distributed as dist

    dataset = ShardedDataset(shard_paths, config.sequence_length)
    sampler = None
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=seed,
        )
        shuffle = False

    num_workers = min(4, os.cpu_count() or 1)
    worker_init = partial(_worker_init_fn, base_seed=seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init,
        generator=gen,
    )


def _worker_init_fn(worker_id: int, base_seed: int) -> None:
    wseed = base_seed + worker_id
    np.random.seed(wseed)
    torch.manual_seed(wseed)


def deterministic_sample_indices(
    uid: int,
    window: int,
    dataset_len: int,
    num_samples: int,
    seed: int = 42,
) -> list[int]:
    """Deterministic indices for provenance checks (miner / validator alignment)."""
    if dataset_len <= 0 or num_samples <= 0:
        return []
    entropy = [int(uid), int(window), int(seed), int(dataset_len), int(num_samples)]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    replace = num_samples > dataset_len
    idx = rng.choice(dataset_len, size=num_samples, replace=replace)
    return [int(x) for x in idx.tolist()]
