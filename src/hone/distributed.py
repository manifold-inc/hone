"""Distributed training utilities for multi-GPU training."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor as DT

import hone


class DistributedHelper:
    """Helper class for distributed training operations."""

    def __init__(self):
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.is_master = self.rank == 0
        self.device = None

    def init_process_group(
        self, backend: str = "nccl", timeout_minutes: int = 45
    ) -> None:
        if not dist.is_initialized() and self.world_size > 1:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(minutes=timeout_minutes),
                rank=self.rank,
                world_size=self.world_size,
            )
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cpu")
            hone.logger.info(
                f"[Distributed] rank={self.rank}, world_size={self.world_size}, "
                f"local_rank={self.local_rank}, device={self.device}"
            )

    def destroy_process_group(self) -> None:
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def is_distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized() and self.world_size > 1

    def should_continue(self, local_has_batch: bool, device: torch.device) -> bool:
        if not self.is_distributed():
            return local_has_batch
        flag = torch.tensor([int(local_has_batch)], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def ddp_reduce(
        self,
        value: int | float | torch.Tensor,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
        device: torch.device | None = None,
    ) -> float:
        if not self.is_distributed():
            return float(value.item() if isinstance(value, torch.Tensor) else value)
        if device is None:
            device = self.device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        if not isinstance(value, torch.Tensor):
            tensor = torch.tensor(float(value), device=device)
        else:
            tensor = value.to(device)
        dist.all_reduce(tensor, op=op)
        return float(tensor.item())

    def barrier(self, device_ids: list | None = None) -> None:
        if not self.is_distributed():
            return
        if device_ids is not None and dist.get_backend() == "nccl":
            dist.barrier(device_ids=device_ids)
        else:
            dist.barrier()

    def safe_barrier(self, tag: str = "", local_rank: int | None = None) -> bool:
        if not self.is_distributed():
            return True
        try:
            if torch.cuda.is_available() and local_rank is not None:
                try:
                    dist.barrier(device_ids=[local_rank])
                    return True
                except TypeError:
                    pass
            dist.barrier()
            return True
        except Exception as e:
            if tag:
                hone.logger.error(f"[barrier:{tag}] failed: {e}")
            return False

    def all_ok(self, ok: bool, device: torch.device, tag: str = "") -> bool:
        if not self.is_distributed():
            return ok
        try:
            t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            return bool(t.item())
        except Exception as e:
            if tag:
                hone.logger.error(f"[sync:{tag}] all_reduce failed: {e}")
            return False

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        if self.is_distributed():
            dist.broadcast(tensor, src=src)

    def gather_object(
        self, obj: Any, object_list: list | None = None, dst: int = 0
    ) -> list | None:
        if not self.is_distributed():
            return [obj] if self.rank == dst else None
        if self.rank == dst and object_list is None:
            object_list = [None] * self.world_size
        dist.gather_object(obj, object_list, dst=dst)
        return object_list if self.rank == dst else None

    def all_gather_object(self, obj: Any, object_list: list | None = None) -> list:
        if not self.is_distributed():
            return [obj]
        if object_list is None:
            object_list = [None] * self.world_size
        dist.all_gather_object(object_list, obj)
        return object_list

    def _force_reshard(self, model: torch.nn.Module) -> None:
        """Force every FSDP2 unit in ``model`` back to the sharded state.

        ``_pp_run_1f1b`` calls leaf submodules (``embed_tokens``,
        ``lm_head``, ``model.norm``, ``output_boundary.encoder``,
        ``output_boundary.decoder``, ...) directly, NOT through the
        root model's forward. FSDP2's post-backward reshard callback
        is hooked off the root forward; when that root forward never
        runs, some leaf units are left in the *unsharded* state where
        ``param.data`` is the full-shape plain Tensor instead of the
        local-shard DTensor. Iterating ``model.parameters()`` then
        observes a mix of sharded (DTensor, local shape (32, 2048))
        and unsharded (plain Tensor, full shape (64, 2048)) params,
        which breaks any code (offload/restore/checkpoint/etc.) that
        assumes a consistent state across all params.

        ``FSDPModule.reshard()`` (added by ``fully_shard``) puts the
        unit back to the sharded state synchronously. Walking the
        module tree and calling it on every FSDP unit is cheap and
        idempotent; for already-sharded units it's a no-op.
        """
        try:
            from torch.distributed.fsdp import FSDPModule
        except ImportError:
            try:
                from torch.distributed._composable.fsdp import FSDPModule
            except ImportError:
                return
        for sub in model.modules():
            if isinstance(sub, FSDPModule):
                try:
                    sub.reshard()
                except Exception:
                    # Best-effort: if reshard fails for any reason,
                    # the meta-driven branch in restore_offloaded_params
                    # below will still cope. Don't kill the run.
                    pass

    def get_offloaded_params(self, model: torch.nn.Module) -> tuple:
        # Make sure every FSDP unit is in its sharded (DTensor) state
        # before snapshotting; otherwise we'd save full-shape buffers
        # for some params and local-shape buffers for others, and the
        # corresponding restore would then mismatch sizes.
        self._force_reshard(model)
        params_offloaded = []
        param_info = []
        stream = self._get_offload_stream(model)
        t0 = time.time()
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode():
            ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with ctx:
                for p in model.parameters():
                    if isinstance(p, DT):
                        src_local = p.to_local()
                        cpu_buf = getattr(p, "_cpu_offload_buf", None)
                        if (
                            cpu_buf is None
                            or cpu_buf.shape != src_local.shape
                            or cpu_buf.dtype != src_local.dtype
                        ):
                            cpu_buf = torch.empty_like(
                                src_local, device="cpu", pin_memory=True
                            )
                            setattr(p, "_cpu_offload_buf", cpu_buf)
                        cpu_buf.copy_(src_local, non_blocking=True)
                        params_offloaded.append(cpu_buf)
                        param_info.append({
                            "is_dtensor": True,
                            "device_mesh": p.device_mesh,
                            "placements": p.placements,
                            "local_shape": src_local.shape,
                        })
                    else:
                        src = p.data
                        cpu_buf = getattr(p, "_cpu_offload_buf", None)
                        if (
                            cpu_buf is None
                            or cpu_buf.shape != src.shape
                            or cpu_buf.dtype != src.dtype
                        ):
                            cpu_buf = torch.empty_like(
                                src, device="cpu", pin_memory=True
                            )
                            setattr(p, "_cpu_offload_buf", cpu_buf)
                        cpu_buf.copy_(src, non_blocking=True)
                        params_offloaded.append(cpu_buf)
                        param_info.append({"is_dtensor": False})
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
            stream.synchronize()
        hone.logger.info(
            f"[ParamOffload] snap {len(params_offloaded)} params "
            f"in {time.time() - t0:.3f}s"
        )
        return params_offloaded, param_info

    def restore_offloaded_params(
        self,
        model: torch.nn.Module,
        params_offloaded: list,
        param_specs: list,
    ) -> None:
        # Same rationale as in ``get_offloaded_params``: FSDP2 may have
        # left some leaf units in the unsharded state after
        # ``_pp_run_1f1b``'s direct submodule calls. Reshard before
        # iterating so ``isinstance(p, DT)`` agrees with what we saved.
        self._force_reshard(model)
        stream = self._get_offload_stream(model)
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
        t0 = time.time()
        with torch.no_grad():
            ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with ctx:
                it = zip(zip(params_offloaded, param_specs), model.parameters())
                for (saved_cpu, meta), p in it:
                    # Drive the branch off the saved meta, NOT off
                    # ``isinstance(p, DT)``. The meta is the source of
                    # truth: it tells us whether ``saved_cpu`` is a
                    # DTensor's local shard (``is_dtensor=True``) or a
                    # plain-tensor full snapshot (``is_dtensor=False``).
                    # If FSDP unfortunately leaves ``p`` in the wrong
                    # state we still know how to interpret the buffer.
                    if meta.get("is_dtensor", False):
                        # After ``_force_reshard`` ``p`` MUST be a
                        # DTensor whose local shape matches what we
                        # saved. If not, fail loudly with the actual
                        # mismatch -- silently coercing would corrupt
                        # the outer-step delta math (``saved - p``)
                        # and poison every subsequent step.
                        if not isinstance(p, DT):
                            raise RuntimeError(
                                "[restore_offloaded_params] meta says "
                                "param was a DTensor at offload but at "
                                f"restore p is plain (shape={tuple(p.shape)}); "
                                "FSDP _force_reshard didn't put this unit "
                                "back to sharded state. Saved local_shape="
                                f"{meta.get('local_shape')}."
                            )
                        local = p.to_local()
                        if local.shape != saved_cpu.shape:
                            raise RuntimeError(
                                "[restore_offloaded_params] local shard "
                                f"shape {tuple(local.shape)} != saved CPU "
                                f"shape {tuple(saved_cpu.shape)}; saved meta "
                                f"local_shape={meta.get('local_shape')}"
                            )
                        g_loc = torch.empty_like(local, device=local.device)
                        g_loc.copy_(saved_cpu, non_blocking=True)
                        grad_dt = DT.from_local(
                            g_loc,
                            device_mesh=meta["device_mesh"],
                            placements=meta["placements"],
                            run_check=False,
                        )
                        p.grad = grad_dt
                        grad_dt.sub_(p)
                        p.data.add_(grad_dt)
                    else:
                        tensor = p.data
                        if (
                            p.grad is None
                            or p.grad.shape != tensor.shape
                            or p.grad.device != tensor.device
                        ):
                            p.grad = torch.empty_like(tensor)
                        p.grad.copy_(saved_cpu, non_blocking=True)
                        p.grad.sub_(tensor)
                        tensor.add_(p.grad)
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
            stream.synchronize()
        hone.logger.info(
            f"[ParamRestore] restored {len(params_offloaded)} params "
            f"in {time.time() - t0:.3f}s"
        )

    def _get_offload_stream(self, model: torch.nn.Module):
        if not torch.cuda.is_available():
            return None
        stream = getattr(model, "_offload_stream", None)
        if stream is None:
            device = next(model.parameters()).device
            stream = torch.cuda.Stream(device=device)
            setattr(model, "_offload_stream", stream)
        return stream


dist_helper = DistributedHelper()
