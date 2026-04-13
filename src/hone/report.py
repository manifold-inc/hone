"""Async client for hone-api ingest telemetry."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import aiohttp

from .config import HoneConfig

logger = logging.getLogger(__name__)


class HoneReporter:
    def __init__(self, config: HoneConfig) -> None:
        self._url = config.hone_api_url.rstrip("/")
        self._key = config.hone_api_key
        self._session: aiohttp.ClientSession | None = None
        self.run_id = str(uuid.uuid4())

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def _headers(self) -> dict[str, str]:
        if not self._key:
            return {}
        return {"x-api-key": self._key}

    async def _post(self, path: str, body: dict[str, Any]) -> None:
        try:
            await self._ensure_session()
            assert self._session is not None
            async with self._session.post(
                f"{self._url}{path}", json=body, headers=self._headers(),
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.warning("Hone API %s failed: %s %s", path, resp.status, text[:500])
                    return
                logger.debug("Hone API %s ok (%s)", path, resp.status)
        except Exception as exc:
            logger.warning("Hone API %s error: %s", path, exc)

    async def report_run(
        self,
        hotkey: str,
        role: str,
        netuid: int,
        uid: int | None,
        version: str,
        config: dict | None = None,
    ) -> None:
        await self._post("/ingest/run", {
            "id": self.run_id,
            "hotkey": hotkey,
            "role": role,
            "netuid": netuid,
            "uid": uid,
            "version": version,
            "config": config if config is not None else {},
        })

    async def report_window(
        self,
        window: int,
        global_step: int,
        block: int | None = None,
        loss_own_before: float | None = None,
        loss_own_after: float | None = None,
        loss_random_before: float | None = None,
        loss_random_after: float | None = None,
        outer_lr: float | None = None,
        inner_lr: float | None = None,
        active_miners: int | None = None,
        gather_success_rate: float | None = None,
        gather_peers: int | None = None,
        timing_window_total: float | None = None,
        timing_gather: float | None = None,
        timing_evaluation: float | None = None,
        timing_model_update: float | None = None,
        uid_scores: list[dict[str, Any]] | None = None,
        gradient_stats: dict[str, float] | None = None,
        **extra: Any,
    ) -> None:
        wm: dict[str, Any] = {
            "runId": self.run_id,
            "window": window,
            "globalStep": global_step,
        }
        if block is not None:
            wm["block"] = block
        if loss_own_before is not None:
            wm["lossOwnBefore"] = loss_own_before
        if loss_own_after is not None:
            wm["lossOwnAfter"] = loss_own_after
        if loss_random_before is not None:
            wm["lossRandomBefore"] = loss_random_before
        if loss_random_after is not None:
            wm["lossRandomAfter"] = loss_random_after
        if outer_lr is not None:
            wm["outerLr"] = outer_lr
        if inner_lr is not None:
            wm["innerLr"] = inner_lr
        if active_miners is not None:
            wm["activeMiners"] = active_miners
        if gather_success_rate is not None:
            wm["gatherSuccessRate"] = gather_success_rate
        if gather_peers is not None:
            wm["gatherPeers"] = gather_peers
        if timing_window_total is not None:
            wm["timingWindowTotal"] = timing_window_total
        if timing_gather is not None:
            wm["timingGather"] = timing_gather
        if timing_evaluation is not None:
            wm["timingEvaluation"] = timing_evaluation
        if timing_model_update is not None:
            wm["timingModelUpdate"] = timing_model_update

        body: dict[str, Any] = {"windowMetrics": wm}
        if uid_scores:
            body["uidScores"] = uid_scores
        gs = gradient_stats or {}
        if gs:
            body["gradientStats"] = {
                "meanGradNorm": gs.get("grad_norm_mean"),
                "maxGradNorm": gs.get("grad_norm_max"),
                "minGradNorm": gs.get("grad_norm_min"),
                "medianGradNorm": gs.get("grad_norm_median"),
                "gradNormStd": gs.get("grad_norm_std"),
                "meanWeightNorm": gs.get("weight_norm_mean"),
                "gradToWeightRatio": gs.get("grad_to_weight_ratio"),
            }
        await self._post("/ingest/window", body)

    async def report_miner(
        self,
        window: int,
        global_step: int,
        loss: float | None = None,
        tokens_per_sec: float | None = None,
        batch_tokens: int | None = None,
        grad_norm: float | None = None,
        weight_norm: float | None = None,
        inner_lr: float | None = None,
        timing: dict | None = None,
        **extra: Any,
    ) -> None:
        body: dict[str, Any] = {
            "runId": self.run_id,
            "window": window,
            "globalStep": global_step,
        }
        if loss is not None:
            body["loss"] = loss
        if tokens_per_sec is not None:
            body["tokensPerSec"] = tokens_per_sec
        if batch_tokens is not None:
            body["batchTokens"] = batch_tokens
        if grad_norm is not None:
            body["gradNorm"] = grad_norm
        if weight_norm is not None:
            body["weightNorm"] = weight_norm
        if inner_lr is not None:
            body["innerLr"] = inner_lr
        if timing is not None:
            body["timing"] = timing
        await self._post("/ingest/miner", body)
