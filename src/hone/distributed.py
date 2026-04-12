"""Distributed process group helpers for multi-GPU training."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def init_distributed() -> bool:
    """Initialize the process group when launched with torchrun.

    Returns True when multi-rank training is active (world size > 1).
    Returns False for single-process / single-GPU runs.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size() > 1

    rank_env = os.environ.get("RANK")
    world_env = os.environ.get("WORLD_SIZE")
    local_env = os.environ.get("LOCAL_RANK")
    if rank_env is None or world_env is None or local_env is None:
        return False

    world_size = int(world_env)
    if world_size <= 1:
        return False

    local_rank = int(local_env)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend)
    logger.info("Initialized distributed: rank %s world_size %s", get_rank(), world_size)
    return True


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return get_rank() == 0


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK") is not None:
        return int(os.environ["LOCAL_RANK"])
    return 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized() and get_world_size() > 1:
        dist.barrier()


def gather_object(obj: Any, dst: int = 0) -> list[Any] | None:
    """Gather picklable Python objects to ``dst``. Returns the list only on ``dst``."""
    if not dist.is_available() or not dist.is_initialized():
        if get_rank() == dst:
            return [obj]
        return None

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == dst:
        gathered: list[Any] = [None] * world_size
        dist.gather_object(obj, object_gather_list=gathered, dst=dst)
        return gathered
    dist.gather_object(obj, object_gather_list=None, dst=dst)
    return None
