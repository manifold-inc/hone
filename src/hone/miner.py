"""Miner: inner SparseLoCo steps, compressed pseudo-gradients, outer aggregation."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import signal
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from hone import __version__
from hone.chain import ChainManager, get_highest_stake_uid
from hone.compress import aggregate as aggregate_compressed, compress
from hone.comms import GradientStore
from hone.config import HoneConfig
from hone.data import build_dataloader, download_shards_from_r2, load_shards
from hone.distributed import barrier, cleanup, gather_object, get_world_size, init_distributed, is_main_process
from hone.model import build_model
from hone.report import HoneReporter

logger = logging.getLogger(__name__)


class Miner:
    def __init__(self, config: HoneConfig) -> None:
        self.config = config
        self.chain = ChainManager(config)
        self.comms = GradientStore(config)
        self.reporter = HoneReporter(config)
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.error_bufs: dict[str, torch.Tensor] = {}
        self._loader: DataLoader | None = None
        self._dataloader_iter: object = None
        self._device = torch.device("cpu")
        self._uid = -1
        self._stop = asyncio.Event()
        self.current_window = 0
        self.outer_step = 0

    async def setup(self) -> None:
        init_distributed()
        await asyncio.to_thread(self.chain.connect)
        uid = await asyncio.to_thread(self.chain.get_uid)
        if uid is None:
            raise RuntimeError("Hotkey not registered on metagraph; cannot mine.")
        self._uid = int(uid)
        if self.config.is_rank_zero:
            b = {
                "account_id": self.config.r2_gradients_account_id,
                "bucket_name": self.config.r2_gradients_bucket_name,
                "access_key_id": self.config.r2_gradients_read_access_key_id,
                "secret_access_key": self.config.r2_gradients_read_secret_access_key,
            }
            await asyncio.to_thread(self.chain.commit_bucket, b)
        barrier()
        hp = await asyncio.to_thread(self.chain.get_hparams)
        if hp:
            flds = type(self.config).model_fields
            self.config = self.config.model_copy(update={k: v for k, v in hp.items() if k in flds})
        self.model = build_model(self.config)
        self._device = torch.device(
            f"cuda:{self.config.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self._device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.inner_lr,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )
        if self.config.is_rank_zero:
            self.error_bufs = {
                n: torch.zeros(p.shape, dtype=torch.float32, device="cpu")
                for n, p in self.model.named_parameters()
            }
        shards = await asyncio.to_thread(load_shards, self.config.dataset_bins_path)
        if not shards:
            logger.info("No local shards found; attempting R2 download...")
            shards = await asyncio.to_thread(download_shards_from_r2, self.config)
        if not shards:
            raise RuntimeError(
                f"No .bin data shards found in {self.config.dataset_bins_path!r} and R2 download failed. "
                "Set HONE_DATASET_BINS_PATH to a directory with pretokenized .bin files, "
                "or configure HONE_R2_DATASET_* env vars to download from R2."
            )
        self._loader = build_dataloader(self.config, shards, seed=self._uid)
        self._dataloader_iter = itertools.cycle(self._loader)
        if self.config.is_rank_zero and self.chain.wallet is not None:
            await self.reporter.report_run(
                hotkey=self.chain.wallet.hotkey.ss58_address,
                role="miner",
                netuid=self.config.netuid,
                uid=self._uid,
                version=__version__,
                config=self.config.model_dump(),
            )

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, self._stop.set)
        try:
            async for _block, window in self.chain.block_listener():
                if self._stop.is_set():
                    logger.info("Stop requested; exiting miner loop.")
                    break
                try:
                    await self.window_step(int(window))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("window_step failed for window %s", window)
        except asyncio.CancelledError:
            logger.info("Miner run cancelled.")
            raise

    async def window_step(self, window: int) -> None:
        self.current_window = window
        if window % 10 == 0:
            await asyncio.to_thread(self.chain.sync_metagraph)
        if self._loader is not None:
            s = getattr(self._loader, "sampler", None)
            if s is not None and hasattr(s, "set_epoch"):
                s.set_epoch(window)
        m, opt, it = self.model, self.optimizer, self._dataloader_iter
        assert m is not None and opt is not None and it is not None
        theta_old = {n: p.detach().clone() for n, p in m.named_parameters()}
        avg_loss, tps, total_tokens = self._inner_training()
        pseudo = {n: theta_old[n] - p.data for n, p in m.named_parameters() if n in theta_old}
        if self.config.fsdp_enabled and get_world_size() > 1:
            g = gather_object({k: v.detach().cpu() for k, v in pseudo.items()}, dst=0)
            pseudo_merged = (
                {
                    n: torch.stack([d[n].float() for d in g if isinstance(d, dict) and n in d], 0).mean(0)
                    for n in g[0]
                }
                if is_main_process() and g and isinstance(g[0], dict)
                else {}
            )
        else:
            pseudo_merged = {k: v.detach().cpu() for k, v in pseudo.items()} if is_main_process() else {}
        if is_main_process() and pseudo_merged:
            cfg = SimpleNamespace(**self.config.model_dump(), outer_step=self.outer_step)
            comp, self.error_bufs = compress(pseudo_merged, self.error_bufs, cfg)
            await self.comms.put(self._uid, window, comp)
        barrier()
        await self._apply_outer_step(window)
        self._broadcast_params_from_rank0()
        barrier()
        self.outer_step += 1
        if self.config.is_rank_zero:
            await self.reporter.report_miner(
                window=window,
                global_step=self.outer_step,
                loss=avg_loss,
                tokens_per_sec=tps,
                batch_tokens=total_tokens,
            )

    def _inner_training(self) -> tuple[float, float, int]:
        m, opt, it = self.model, self.optimizer, self._dataloader_iter
        assert m is not None and opt is not None and it is not None
        m.train()
        cuda = self._device.type == "cuda"
        dt = torch.float16 if cuda else torch.bfloat16
        losses: list[float] = []
        total_tokens, t0 = 0, time.perf_counter()
        for _ in range(self.config.inner_steps):
            batch = next(it)
            input_ids = batch["input_ids"].to(self._device, non_blocking=True)
            labels = batch["labels"].to(self._device, non_blocking=True)
            total_tokens += int(input_ids.numel())
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self._device.type, dtype=dt, enabled=cuda):
                loss = m(input_ids=input_ids, labels=labels).loss
            loss.backward()
            clip_grad_norm_(m.parameters(), self.config.grad_clip)
            opt.step()
            losses.append(float(loss.detach().item()))
        elapsed = max(time.perf_counter() - t0, 1e-6)
        return sum(losses) / max(len(losses), 1), float(total_tokens / elapsed), total_tokens

    async def _apply_outer_step(self, window: int) -> None:
        assert self.model is not None
        if is_main_process():
            buckets = self.chain.get_commitments()
            mg = self.chain.metagraph
            val_uid = get_highest_stake_uid(mg) if mg is not None else -1
            peers = sorted(u for u in buckets if u not in (self._uid, val_uid))[: self.config.gather_peer_count]
            if not peers:
                logger.warning("No peer buckets for outer step (window %s).", window)
            else:
                valid, _skipped = await self.comms.gather(peers, window, buckets)
                if not valid:
                    logger.warning("No valid peer gradients (window %s).", window)
                else:
                    agg = aggregate_compressed(list(valid.values()))
                    olr = float(self.config.outer_lr)
                    with torch.no_grad():
                        for n, p in self.model.named_parameters():
                            if n in agg:
                                p.data.sub_(olr * agg[n].to(device=p.device, dtype=p.dtype))
        barrier()

    def _broadcast_params_from_rank0(self) -> None:
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        assert self.model is not None
        for p in self.model.parameters():
            dist.broadcast(p.data, src=0)

def run_miner(config: HoneConfig) -> None:
    asyncio.run(_async_main(Miner(config)))

async def _async_main(miner: Miner) -> None:
    try:
        await miner.setup()
        await miner.run()
    finally:
        cleanup()
        await miner.reporter.close()
