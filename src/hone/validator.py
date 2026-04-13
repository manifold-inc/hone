"""Validator neuron: gather pseudo-gradients, verify, score, outer SGD, set weights."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

import hone
from hone.chain import ChainManager, get_highest_stake_uid
from hone.comms import GradientStore
from hone.compress import CompressedTensor, decompress
from hone.config import HoneConfig
from hone.data import (
    build_dataloader,
    deterministic_sample_indices,
    download_shards_from_r2,
    load_shards,
)
from hone.distributed import barrier, cleanup, init_distributed, is_main_process
from hone.model import build_model
from hone.report import HoneReporter
from hone.score import Scorer
from hone.verify import GradientVerifier, VerifyResult

logger = logging.getLogger(__name__)


def _metagraph_n(metagraph: Any) -> int:
    return int(np.asarray(metagraph.n).item())


def _collate(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples], dim=0),
        "labels": torch.stack([s["labels"] for s in samples], dim=0),
    }


class Validator:
    def __init__(self, config: HoneConfig) -> None:
        self.config = config
        self.chain = ChainManager(config)
        self.comms = GradientStore(config)
        self.reporter = HoneReporter(config)
        self.model: nn.Module | None = None
        self.verifier: GradientVerifier | None = None
        self.scorer: Scorer | None = None
        self._eval_loader: Any = None
        self._eval_iter: Any = None
        self._device = torch.device(
            f"cuda:{config.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self._own_uid: int | None = None
        self.current_window: int = 0
        self.outer_step: int = 0

    def _hparams_payload(self) -> dict[str, Any]:
        c = self.config
        return {
            "model_type": c.model_type,
            "model_name_or_path": c.model_name_or_path,
            "model_args": dict(c.model_args),
            "inner_steps": c.inner_steps,
            "inner_lr": c.inner_lr,
            "outer_lr": c.outer_lr,
            "weight_decay": c.weight_decay,
            "adam_beta1": c.adam_beta1,
            "adam_beta2": c.adam_beta2,
            "grad_clip": c.grad_clip,
            "batch_size": c.batch_size,
            "sequence_length": c.sequence_length,
            "topk_k": c.topk_k,
            "chunk_size": c.chunk_size,
            "quant_bits": c.quant_bits,
            "ef_momentum": c.ef_momentum,
            "ef_freeze_pct": c.ef_freeze_pct,
            "blocks_per_window": c.blocks_per_window,
        }

    def _next_eval_batch(self) -> dict[str, Tensor] | None:
        if self._eval_loader is None:
            return None
        try:
            batch = next(self._eval_iter)
        except (StopIteration, TypeError):
            self._eval_iter = iter(self._eval_loader)
            batch = next(self._eval_iter)
        return {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}

    def _provenance_batches(
        self, uid: int, window: int, ds_len: int,
    ) -> tuple[dict[str, Tensor] | None, dict[str, Tensor] | None]:
        if self._eval_loader is None:
            return None, None
        ds = self._eval_loader.dataset
        bs = int(self.config.batch_size)
        ia = deterministic_sample_indices(uid, window, ds_len, bs, seed=42)
        ir = deterministic_sample_indices(uid + 1, window + 7, ds_len, bs, seed=43)
        if len(ia) < bs or len(ir) < bs:
            return None, None
        try:
            a = _collate([ds[int(i)] for i in ia[:bs]])
            r = _collate([ds[int(i)] for i in ir[:bs]])
        except (IndexError, RuntimeError):
            return None, None
        return (
            {k: v.to(self._device, non_blocking=True) for k, v in a.items()},
            {k: v.to(self._device, non_blocking=True) for k, v in r.items()},
        )

    async def setup(self) -> None:
        init_distributed()
        self.chain.connect()
        self._own_uid = self.chain.get_uid()
        mg = self.chain.metagraph
        if mg is not None and self._own_uid is not None:
            if self._own_uid == get_highest_stake_uid(mg):
                self.chain.commit_hparams(self._hparams_payload())
        self.model = build_model(self.config)
        self.model.to(self._device)
        self.model.eval()
        shards = load_shards(self.config.dataset_bins_path)
        if not shards:
            logger.info("No local shards found; attempting R2 download...")
            shards = download_shards_from_r2(self.config)
        if shards:
            self._eval_loader = build_dataloader(self.config, shards, seed=0, shuffle=True)
            self._eval_iter = iter(self._eval_loader)
        else:
            logger.warning("No data shards available; verification and provenance checks will be skipped")
            self._eval_loader = None
            self._eval_iter = None
        if mg is None:
            raise RuntimeError("Metagraph not loaded after connect()")
        n_uids = _metagraph_n(mg)
        self.verifier = GradientVerifier(self.config)
        self.scorer = Scorer(self.config, max_uids=max(n_uids, 256), device=self._device)
        hk = self.chain.wallet.hotkey.ss58_address if self.chain.wallet else ""
        cfg_dump = self.config.model_dump() if hasattr(self.config, "model_dump") else {}
        await self.reporter.report_run(
            hotkey=hk, role="validator", netuid=self.config.netuid,
            uid=self._own_uid, version=hone.__version__, config=cfg_dump,
        )
        if is_main_process():
            c = self.config
            self.chain.commit_bucket(
                {
                    "account_id": c.r2_gradients_account_id,
                    "bucket_name": c.r2_gradients_bucket_name,
                    "access_key_id": c.r2_gradients_read_access_key_id,
                    "secret_access_key": c.r2_gradients_read_secret_access_key,
                }
            )

    async def run(self) -> None:
        assert self.chain.subtensor is not None
        async for _block, window in self.chain.block_listener():
            target = window - int(self.config.validator_offset)
            if target > self.current_window:
                await self.window_step(target)

    async def window_step(self, window: int) -> None:
        t0 = time.perf_counter()
        assert self.model is not None and self.verifier is not None and self.scorer is not None
        per_uid: dict[int, dict[str, Any]] = {}
        window_scores: dict[int, float] = {}
        evaluated_uids: list[int] = []
        valid_grads: dict[int, dict[str, CompressedTensor]] = {}
        skipped: list[int] = []
        t_peer = t_gather = t_eval = t_update = t0
        grad_stats: dict[str, Any] = {}
        try:
            self.chain.sync_metagraph()
            mg = self.chain.metagraph
            assert mg is not None
            buckets = self.chain.get_commitments()
            t_peer = time.perf_counter()
            n = _metagraph_n(mg)
            uids = [u for u in range(n) if u in buckets]
            valid_grads, skipped = await self.comms.gather(uids, window, buckets)
            t_gather = time.perf_counter()
            verify_passed: dict[int, dict[str, Tensor]] = {}
            with torch.no_grad():
                eval_batch = self._next_eval_batch()
                ds_len = len(self._eval_loader.dataset) if self._eval_loader else 0
                for uid, comp in valid_grads.items():
                    dense = decompress(comp)
                    evaluated_uids.append(uid)
                    if eval_batch is not None:
                        ab, rb = self._provenance_batches(uid, window, ds_len)
                        vr: VerifyResult = self.verifier.verify(
                            dense, self.model, eval_batch, ab, rb,
                        )
                        per_uid[uid] = {
                            "passed": vr.passed,
                            "loss_score": vr.loss_score,
                            "checks": [
                                {"name": c.name, "passed": c.passed, "score": c.score, "detail": c.detail}
                                for c in vr.checks
                            ],
                        }
                        window_scores[uid] = float(vr.loss_score) if vr.passed else 0.0
                        if vr.passed:
                            verify_passed[uid] = dense
                    else:
                        window_scores[uid] = 1.0
                        verify_passed[uid] = dense
                normalized = GradientVerifier.normalize_norms(
                    {u: verify_passed[u] for u in verify_passed},
                )
            t_eval = time.perf_counter()
            self.scorer.update_scores(window_scores, evaluated_uids)
            self.scorer.compute_weights()
            top_k = int(self.config.gather_peer_count)
            ranked = sorted(normalized.keys(), key=lambda u: window_scores.get(u, 0.0), reverse=True)
            selected = ranked[: max(1, min(top_k, len(ranked)))] if ranked else []
            agg: dict[str, Tensor] = {}
            if selected:
                u0 = selected[0]
                keys = normalized[u0].keys()
                with torch.no_grad():
                    for name in keys:
                        agg[name] = torch.stack([normalized[u][name] for u in selected], dim=0).mean(0)
                    lr = float(self.config.outer_lr)
                    for name, p in self.model.named_parameters():
                        if name in agg:
                            p.data.add_(agg[name], alpha=-lr)
            t_update = time.perf_counter()
            barrier()
            if is_main_process() and self.outer_step % int(self.config.windows_per_weights) == 0:
                w_uids, w_vals = self.scorer.get_weights()
                if w_uids:
                    self.chain.set_weights(w_uids, w_vals)
            grad_stats = self._compute_gradient_stats(normalized) if normalized else {}
        except Exception:
            logger.exception("window_step failed window=%s", window)
        finally:
            self.current_window = window
            dt = time.perf_counter() - t0
            if is_main_process():
                scores_list = [
                    {
                        "uid": uid,
                        "gradientScore": info.get("loss_score"),
                        "finalScore": float(self.scorer.final_scores[uid].item()) if self.scorer and uid < self.scorer.max_uids else None,
                        "weight": float(self.scorer.weights[uid].item()) if self.scorer and uid < self.scorer.max_uids else None,
                    }
                    for uid, info in per_uid.items()
                ] if per_uid else None
                await self.reporter.report_window(
                    window=window,
                    global_step=self.outer_step,
                    gather_peers=len(valid_grads),
                    gather_success_rate=len(valid_grads) / max(len(valid_grads) + len(skipped), 1),
                    timing_window_total=dt,
                    timing_gather=t_gather - t_peer,
                    timing_evaluation=t_eval - t_gather,
                    timing_model_update=t_update - t_eval,
                    uid_scores=scores_list,
                    gradient_stats=grad_stats,
                )
            self.outer_step += 1

    def _compute_gradient_stats(self, grad_dicts: dict[int, dict[str, Tensor]]) -> dict[str, float]:
        if not grad_dicts or self.model is None:
            return {}
        norms: list[Tensor] = []
        for g in grad_dicts.values():
            sq = torch.stack([t.detach().float().norm() ** 2 for t in g.values()]).sum()
            norms.append(torch.sqrt(sq))
        nstack = torch.stack(norms)
        keys = set(next(iter(grad_dicts.values())).keys())
        w_list = []
        for k in keys:
            try:
                w_list.append(self.model.get_parameter(k).detach().float().norm())
            except (AttributeError, ValueError):
                for n, p in self.model.named_parameters():
                    if n == k:
                        w_list.append(p.detach().float().norm())
                        break
        mean_w = torch.stack(w_list).mean() if w_list else torch.tensor(0.0)
        eps = 1e-8
        return {
            "grad_norm_mean": float(nstack.mean().item()),
            "grad_norm_max": float(nstack.max().item()),
            "grad_norm_min": float(nstack.min().item()),
            "grad_norm_median": float(torch.median(nstack).item()),
            "grad_norm_std": float(nstack.std(unbiased=False).item()),
            "weight_norm_mean": float(mean_w.item()),
            "grad_to_weight_ratio": float((nstack.mean() / (mean_w + eps)).item()),
        }


def run_validator(config: HoneConfig) -> None:
    validator = Validator(config)
    asyncio.run(_async_main(validator))


async def _async_main(validator: Validator) -> None:
    try:
        await validator.setup()
        await validator.run()
    finally:
        cleanup()
        await validator.reporter.close()
