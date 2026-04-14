"""Dashboard metrics reporter -- async HTTP client that POSTs training data to hone-api."""

import asyncio
import hashlib
import json
import os
import time
import uuid
from typing import Any

import aiohttp

from .logging import logger


class DashboardReporter:
    """Fire-and-forget reporter that sends metrics to the hone-api ingest endpoints.

    Authentication uses Bittensor sr25519 hotkey signatures. Each request is signed
    with the miner/validator's hotkey so the API can verify identity without a shared
    secret.

    Never raises or blocks the training loop -- all errors are logged and swallowed.
    """

    def __init__(
        self,
        *,
        hotkey: str,
        role: str,
        netuid: int,
        uid: int | None = None,
        version: str | None = None,
        config: dict[str, Any] | None = None,
        api_url: str | None = None,
        wallet: Any | None = None,
    ):
        self.api_url = (api_url or os.environ.get("DASHBOARD_API_URL", "")).rstrip("/")
        self.wallet = wallet
        self.enabled = bool(self.api_url)

        self.run_id = str(uuid.uuid4())
        self.hotkey = hotkey
        self.role = role
        self.netuid = netuid
        self.uid = uid
        self.version = version
        self.config = config

        self._session: aiohttp.ClientSession | None = None

        if not self.enabled:
            logger.info("[DashboardReporter] disabled (no DASHBOARD_API_URL set)")
        elif not self.wallet:
            logger.warning(
                "[DashboardReporter] no wallet provided -- requests will be unsigned"
            )

    def _sign_payload(self, body_bytes: bytes) -> dict[str, str]:
        """Create auth headers by signing a nonce + body hash with the hotkey."""
        headers: dict[str, str] = {"Content-Type": "application/json"}

        if not self.wallet:
            return headers

        try:
            nonce = str(int(time.time()))
            body_hash = hashlib.sha256(body_bytes).hexdigest()
            message = f"{nonce}:{body_hash}"

            signature = self.wallet.hotkey.sign(message.encode("utf-8"))
            sig_hex = signature.hex() if isinstance(signature, bytes) else str(signature)

            headers["x-hotkey"] = self.hotkey
            headers["x-nonce"] = nonce
            headers["x-signature"] = sig_hex
        except Exception as e:
            logger.warning(f"[DashboardReporter] failed to sign request: {e}")

        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _post(self, path: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            session = await self._get_session()
            url = f"{self.api_url}{path}"
            body_bytes = json.dumps(payload).encode("utf-8")
            headers = self._sign_payload(body_bytes)
            async with session.post(url, data=body_bytes, headers=headers) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning(
                        f"[DashboardReporter] POST {path} returned {resp.status}: {body[:200]}"
                    )
        except Exception as e:
            logger.warning(f"[DashboardReporter] POST {path} failed: {e}")

    # ── Run registration ─────────────────────────────────────────────────

    async def register_run(self) -> None:
        await self._post(
            "/ingest/run",
            {
                "id": self.run_id,
                "hotkey": self.hotkey,
                "role": self.role,
                "netuid": self.netuid,
                "uid": self.uid,
                "version": self.version,
                "config": self.config,
            },
        )

    # ── Validator window metrics ──────────────────────────────────────────

    async def report_window(
        self,
        *,
        window: int,
        global_step: int,
        block: int | None = None,
        loss_own_before: float | None = None,
        loss_own_after: float | None = None,
        loss_random_before: float | None = None,
        loss_random_after: float | None = None,
        loss_own_improvement: float | None = None,
        loss_random_improvement: float | None = None,
        outer_lr: float | None = None,
        inner_lr: float | None = None,
        active_miners: int | None = None,
        gather_success_rate: float | None = None,
        gather_peers: int | None = None,
        positive_peers_ratio: float | None = None,
        reserve_used: int | None = None,
        overlap_mean: float | None = None,
        overlap_max: float | None = None,
        overlap_pairs_checked: int | None = None,
        overlap_pairs_over_threshold: int | None = None,
        overlap_ratio_over_threshold: float | None = None,
        compress_min_median_norm: float | None = None,
        compress_max_median_norm: float | None = None,
        gather_intended_mean_final: float | None = None,
        gather_actual_mean_final: float | None = None,
        timing_window_total: float | None = None,
        timing_peer_update: float | None = None,
        timing_gather: float | None = None,
        timing_evaluation: float | None = None,
        timing_model_update: float | None = None,
        evaluated_uids: int | None = None,
        total_negative_evals: int | None = None,
        total_excluded: int | None = None,
        uid_scores: list[dict[str, Any]] | None = None,
        gradient_stats: dict[str, Any] | None = None,
    ) -> None:
        wm: dict[str, Any] = {
            "runId": self.run_id,
            "window": window,
            "globalStep": global_step,
        }
        local_vars = locals()
        field_map = {
            "block": "block",
            "loss_own_before": "lossOwnBefore",
            "loss_own_after": "lossOwnAfter",
            "loss_random_before": "lossRandomBefore",
            "loss_random_after": "lossRandomAfter",
            "loss_own_improvement": "lossOwnImprovement",
            "loss_random_improvement": "lossRandomImprovement",
            "outer_lr": "outerLr",
            "inner_lr": "innerLr",
            "active_miners": "activeMiners",
            "gather_success_rate": "gatherSuccessRate",
            "gather_peers": "gatherPeers",
            "positive_peers_ratio": "positivePeersRatio",
            "reserve_used": "reserveUsed",
            "overlap_mean": "overlapMean",
            "overlap_max": "overlapMax",
            "overlap_pairs_checked": "overlapPairsChecked",
            "overlap_pairs_over_threshold": "overlapPairsOverThreshold",
            "overlap_ratio_over_threshold": "overlapRatioOverThreshold",
            "compress_min_median_norm": "compressMinMedianNorm",
            "compress_max_median_norm": "compressMaxMedianNorm",
            "gather_intended_mean_final": "gatherIntendedMeanFinal",
            "gather_actual_mean_final": "gatherActualMeanFinal",
            "timing_window_total": "timingWindowTotal",
            "timing_peer_update": "timingPeerUpdate",
            "timing_gather": "timingGather",
            "timing_evaluation": "timingEvaluation",
            "timing_model_update": "timingModelUpdate",
            "evaluated_uids": "evaluatedUids",
            "total_negative_evals": "totalNegativeEvals",
            "total_excluded": "totalExcluded",
        }
        for py_name, js_name in field_map.items():
            val = local_vars.get(py_name)
            if val is not None:
                wm[js_name] = val

        payload: dict[str, Any] = {"windowMetrics": wm}
        if uid_scores:
            payload["uidScores"] = uid_scores
        if gradient_stats:
            gs_map = {
                "mean_grad_norm": "meanGradNorm",
                "max_grad_norm": "maxGradNorm",
                "min_grad_norm": "minGradNorm",
                "median_grad_norm": "medianGradNorm",
                "grad_norm_std": "gradNormStd",
                "mean_weight_norm": "meanWeightNorm",
                "grad_to_weight_ratio": "gradToWeightRatio",
            }
            payload["gradientStats"] = {
                gs_map.get(k, k): v for k, v in gradient_stats.items() if v is not None
            }

        await self._post("/ingest/window", payload)

    # ── Miner metrics ─────────────────────────────────────────────────────

    async def report_miner(
        self,
        *,
        window: int,
        global_step: int,
        loss: float | None = None,
        window_entry_loss: float | None = None,
        tokens_per_sec: float | None = None,
        batch_tokens: int | None = None,
        grad_norm: float | None = None,
        weight_norm: float | None = None,
        momentum_norm: float | None = None,
        gather_success_rate: float | None = None,
        gather_peers: int | None = None,
        gpu_memory_allocated: float | None = None,
        gpu_memory_cached: float | None = None,
        inner_lr: float | None = None,
        timing: dict[str, float] | None = None,
        gradient_l2_norm: float | None = None,
        gradient_total_elements: int | None = None,
        cpu_usage: float | None = None,
        gpu_utilization: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "runId": self.run_id,
            "window": window,
            "globalStep": global_step,
        }
        field_map = {
            "loss": "loss",
            "window_entry_loss": "windowEntryLoss",
            "tokens_per_sec": "tokensPerSec",
            "batch_tokens": "batchTokens",
            "grad_norm": "gradNorm",
            "weight_norm": "weightNorm",
            "momentum_norm": "momentumNorm",
            "gather_success_rate": "gatherSuccessRate",
            "gather_peers": "gatherPeers",
            "gpu_memory_allocated": "gpuMemoryAllocated",
            "gpu_memory_cached": "gpuMemoryCached",
            "inner_lr": "innerLr",
            "timing": "timing",
            "gradient_l2_norm": "gradientL2Norm",
            "gradient_total_elements": "gradientTotalElements",
            "cpu_usage": "cpuUsage",
            "gpu_utilization": "gpuUtilization",
        }
        local_vars = locals()
        for py_name, js_name in field_map.items():
            val = local_vars.get(py_name)
            if val is not None:
                payload[js_name] = val

        await self._post("/ingest/miner", payload)

    # ── Sync scores (validator → per-UID sync data) ───────────────────────

    async def report_sync_scores(
        self,
        *,
        window: int,
        scores: list[dict[str, Any]],
    ) -> None:
        await self._post(
            "/ingest/sync-scores",
            {
                "runId": self.run_id,
                "window": window,
                "scores": scores,
            },
        )

    # ── Slash events ──────────────────────────────────────────────────────

    async def report_slash_event(
        self,
        *,
        window: int,
        uid: int,
        score_before: float,
        score_after: float,
        reason: str,
    ) -> None:
        await self._post(
            "/ingest/slash",
            {
                "runId": self.run_id,
                "window": window,
                "uid": uid,
                "scoreBefore": score_before,
                "scoreAfter": score_after,
                "reason": reason,
            },
        )

    # ── Inactivity events ─────────────────────────────────────────────────

    async def report_inactivity(
        self,
        *,
        window: int,
        uid: int,
        score_before: float,
        score_after: float,
    ) -> None:
        await self._post(
            "/ingest/inactivity",
            {
                "runId": self.run_id,
                "window": window,
                "uid": uid,
                "scoreBefore": score_before,
                "scoreAfter": score_after,
            },
        )

    # ── Cleanup ───────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
