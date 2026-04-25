"""Dashboard metrics reporter -- async HTTP + WebSocket + Redis client for hone-api."""

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

    Transport priority:
      1. Redis Streams (if REDIS_URL is set) -- simplest, most reliable
      2. WebSocket to hone-api /ws/ingest -- low-latency, keeps liveness
      3. HTTP POST to hone-api /ingest/* -- universal fallback

    Each Redis message carries an SR25519 signature so hone-api verifies
    authenticity before persisting. Even if Redis is compromised, forged
    messages are rejected without a valid Bittensor hotkey.

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
        project: str | None = None,
        model_size: str | None = None,
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
        self.project = project
        self.model_size = model_size
        self.config = config

        self._session: aiohttp.ClientSession | None = None

        # Redis state
        self._redis_url = os.environ.get("REDIS_URL", "")
        self._redis: Any | None = None
        self._redis_available = bool(self._redis_url)
        self._redis_stream_key = f"metrics:{netuid}:{hotkey}"

        # WebSocket state
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_authenticated = False
        self._ws_connect_failures = 0
        # Backoff window after a burst of failed reconnects. The miner
        # asyncio loop shares time with CUDA-bound training steps; a
        # 12s inner step can starve the heartbeat task long enough for
        # the server to disconnect us. Instead of permanently disabling
        # the WS after N failures (the old behaviour, which silently
        # locked the reporter into HTTP-only fallback for the rest of
        # the run), back off for ``_ws_backoff_until`` and try again.
        self._ws_backoff_threshold = 5
        self._ws_backoff_seconds = 30.0
        self._ws_backoff_until: float = 0.0
        self._ws_heartbeat_task: asyncio.Task | None = None
        self._ws_connecting = False

        if not self.enabled and not self._redis_available:
            logger.info("[DashboardReporter] disabled (no DASHBOARD_API_URL or REDIS_URL set)")
        elif not self.wallet:
            logger.warning(
                "[DashboardReporter] no wallet provided -- requests will be unsigned"
            )
        if self._redis_available:
            logger.info(f"[DashboardReporter] Redis transport enabled → {self._redis_stream_key}")

    # ── Redis Streams transport ──────────────────────────────────────────

    async def _get_redis(self) -> Any:
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            await self._redis.ping()
            logger.info("[DashboardReporter] Redis connection established")
            return self._redis
        except Exception as e:
            logger.warning(f"[DashboardReporter] Redis connect failed: {e}")
            self._redis = None
            self._redis_available = False
            return None

    def _sign_message(self, data_json: str) -> tuple[str, str, str]:
        """Sign data_json and return (nonce, body_hash, signature_hex)."""
        nonce = str(int(time.time()))
        body_hash = hashlib.sha256(data_json.encode("utf-8")).hexdigest()
        message = f"{nonce}:{body_hash}"
        sig = self.wallet.hotkey.sign(message.encode("utf-8"))
        sig_hex = sig.hex() if isinstance(sig, bytes) else str(sig)
        return nonce, body_hash, sig_hex

    async def _send_redis(self, msg_type: str, data: dict[str, Any]) -> bool:
        if not self._redis_available or not self.wallet:
            return False
        try:
            r = await self._get_redis()
            if r is None:
                return False

            data_json = json.dumps(data, default=str)
            nonce, _, sig_hex = self._sign_message(data_json)

            await r.xadd(
                self._redis_stream_key,
                {
                    "type": msg_type,
                    "hotkey": self.hotkey,
                    "runId": self.run_id,
                    "nonce": nonce,
                    "signature": sig_hex,
                    "data": data_json,
                },
                maxlen=10000,
            )
            return True
        except Exception as e:
            logger.warning(f"[DashboardReporter] Redis XADD failed: {e}")
            return False

    # ── WebSocket connection ──────────────────────────────────────────────

    def _ws_url(self) -> str:
        url = self.api_url.replace("https://", "wss://").replace("http://", "ws://")
        return f"{url}/ws/ingest"

    async def _connect_ws(self) -> bool:
        if not self.enabled or not self.wallet:
            return False
        if self._ws_connecting:
            return False
        # Soft backoff: after a burst of N failures we wait
        # ``_ws_backoff_seconds`` before retrying. Crucially this is a
        # *temporary* freeze, not a permanent one -- the WS will come
        # back automatically once the asyncio loop has time to breathe
        # again. The old code permanently disabled WS after 5 fails,
        # which silently locked the reporter into HTTP-only fallback
        # for the remainder of the training run.
        now = time.monotonic()
        if (
            self._ws_connect_failures >= self._ws_backoff_threshold
            and now < self._ws_backoff_until
        ):
            return False

        self._ws_connecting = True
        try:
            session = await self._get_session()
            ws_url = self._ws_url()
            self._ws = await session.ws_connect(ws_url, timeout=10)

            nonce = str(int(time.time()))
            message = f"{nonce}:ws-auth"
            signature = self.wallet.hotkey.sign(message.encode("utf-8"))
            sig_hex = signature.hex() if isinstance(signature, bytes) else str(signature)

            await self._ws.send_json({
                "type": "auth",
                "hotkey": self.hotkey,
                "nonce": nonce,
                "signature": sig_hex,
                "runId": self.run_id,
            })

            resp = await asyncio.wait_for(self._ws.receive_json(), timeout=5)
            if resp.get("type") == "auth-ok":
                self._ws_authenticated = True
                self._ws_connect_failures = 0
                self._ws_backoff_until = 0.0
                self._start_heartbeat()
                logger.info("[DashboardReporter] WebSocket connected and authenticated")
                return True
            else:
                logger.warning(f"[DashboardReporter] WS auth rejected: {resp}")
                await self._ws.close()
                self._ws = None
                self._record_ws_failure()
                return False

        except Exception as e:
            # Use repr() so e.g. ``asyncio.TimeoutError`` (whose str()
            # is empty) shows up as "TimeoutError()" in the log instead
            # of an unhelpful blank "WS connect failed:" line.
            logger.warning(f"[DashboardReporter] WS connect failed: {e!r}")
            self._ws = None
            self._ws_authenticated = False
            self._record_ws_failure()
            return False
        finally:
            self._ws_connecting = False

    def _record_ws_failure(self) -> None:
        """Bump the failure counter and arm the backoff window if we
        crossed the threshold. Resets on the next successful auth."""
        self._ws_connect_failures += 1
        if self._ws_connect_failures >= self._ws_backoff_threshold:
            self._ws_backoff_until = (
                time.monotonic() + self._ws_backoff_seconds
            )

    def _start_heartbeat(self):
        if self._ws_heartbeat_task and not self._ws_heartbeat_task.done():
            return

        async def heartbeat_loop():
            while self._ws and not self._ws.closed and self._ws_authenticated:
                try:
                    await self._ws.send_json({"type": "heartbeat"})
                except Exception:
                    break
                await asyncio.sleep(15)

        self._ws_heartbeat_task = asyncio.create_task(heartbeat_loop())

    async def _send_ws(self, msg_type: str, data: dict[str, Any]) -> bool:
        if not self._ws or self._ws.closed or not self._ws_authenticated:
            if not await self._connect_ws():
                return False

        try:
            await self._ws.send_json({"type": msg_type, "data": data})  # type: ignore
            return True
        except Exception as e:
            logger.warning(f"[DashboardReporter] WS send failed: {e}")
            self._ws_authenticated = False
            self._ws = None
            return False

    # ── HTTP fallback ─────────────────────────────────────────────────────

    def _sign_payload(self, body_bytes: bytes) -> dict[str, str]:
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
            # ``repr`` so empty-string exceptions like
            # ``asyncio.TimeoutError`` are still identifiable in logs.
            logger.warning(f"[DashboardReporter] POST {path} failed: {e!r}")

    # ── Unified send: try Redis → WS → HTTP ────────────────────────────

    async def _send(self, msg_type: str, http_path: str, payload: dict[str, Any]) -> None:
        if not self.enabled and not self._redis_available:
            return
        if await self._send_redis(msg_type, payload):
            return
        if await self._send_ws(msg_type, payload):
            return
        await self._post(http_path, payload)

    # ── Run registration (always HTTP -- must exist before WS auth) ───────

    async def register_run(self) -> None:
        payload = {
            "id": self.run_id,
            "hotkey": self.hotkey,
            "role": self.role,
            "netuid": self.netuid,
        }
        if self.uid is not None:
            payload["uid"] = self.uid
        if self.version is not None:
            payload["version"] = self.version
        if self.project is not None:
            payload["project"] = self.project
        if self.model_size is not None:
            payload["modelSize"] = self.model_size
        if self.config is not None:
            payload["config"] = self.config

        for attempt in range(3):
            try:
                if not self.enabled:
                    logger.warning("[DashboardReporter] Cannot register run: no DASHBOARD_API_URL")
                    return
                session = await self._get_session()
                url = f"{self.api_url}/ingest/run"
                body_bytes = json.dumps(payload).encode("utf-8")
                headers = self._sign_payload(body_bytes)
                async with session.post(url, data=body_bytes, headers=headers) as resp:
                    body = await resp.text()
                    if resp.status < 400:
                        logger.info(f"[DashboardReporter] Run registered: {self.run_id}")
                        break
                    logger.error(
                        f"[DashboardReporter] register_run attempt {attempt + 1}/3 "
                        f"returned {resp.status}: {body[:300]}"
                    )
            except Exception as e:
                logger.error(
                    f"[DashboardReporter] register_run attempt {attempt + 1}/3 failed: {e}"
                )
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

        await self._connect_ws()

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

        await self._send("window", "/ingest/window", payload)

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
        outer_step_applied: bool | None = None,
        compressed_size_mb: float | None = None,
        upload_size_mb: float | None = None,
        offload_time: float | None = None,
        restore_time: float | None = None,
        skipped_peers: int | None = None,
        gather_peer_list: list[int] | None = None,
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
            "outer_step_applied": "outerStepApplied",
            "compressed_size_mb": "compressedSizeMb",
            "upload_size_mb": "uploadSizeMb",
            "offload_time": "offloadTime",
            "restore_time": "restoreTime",
            "skipped_peers": "skippedPeers",
            "gather_peer_list": "gatherPeerList",
        }
        local_vars = locals()
        for py_name, js_name in field_map.items():
            val = local_vars.get(py_name)
            if val is not None:
                payload[js_name] = val

        await self._send("miner", "/ingest/miner", payload)

    # ── Sync scores ───────────────────────────────────────────────────────

    async def report_sync_scores(
        self,
        *,
        window: int,
        scores: list[dict[str, Any]],
    ) -> None:
        payload = {
            "runId": self.run_id,
            "window": window,
            "scores": scores,
        }
        await self._send("sync-scores", "/ingest/sync-scores", payload)

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
        payload = {
            "runId": self.run_id,
            "window": window,
            "uid": uid,
            "scoreBefore": score_before,
            "scoreAfter": score_after,
            "reason": reason,
        }
        await self._send("slash", "/ingest/slash", payload)

    # ── Inactivity events ─────────────────────────────────────────────────

    async def report_inactivity(
        self,
        *,
        window: int,
        uid: int,
        score_before: float,
        score_after: float,
    ) -> None:
        payload = {
            "runId": self.run_id,
            "window": window,
            "uid": uid,
            "scoreBefore": score_before,
            "scoreAfter": score_after,
        }
        await self._send("inactivity", "/ingest/inactivity", payload)

    # ── Inner step metrics (high frequency, per optimizer step) ─────────

    async def report_inner_step(
        self,
        *,
        window: int,
        inner_step: int,
        global_step: int,
        loss: float | None = None,
        batch_size: int | None = None,
        batch_tokens: int | None = None,
        inner_lr: float | None = None,
        grad_norm: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "runId": self.run_id,
            "window": window,
            "innerStep": inner_step,
            "globalStep": global_step,
        }
        field_map = {
            "loss": "loss",
            "batch_size": "batchSize",
            "batch_tokens": "batchTokens",
            "inner_lr": "innerLr",
            "grad_norm": "gradNorm",
        }
        local_vars = locals()
        for py_name, js_name in field_map.items():
            val = local_vars.get(py_name)
            if val is not None:
                payload[js_name] = val

        await self._send("inner-step", "/ingest/inner-step", payload)

    # ── Gather status (per-UID gather outcome) ─────────────────────────

    async def report_gather_status(
        self,
        *,
        window: int,
        results: list[dict[str, Any]],
    ) -> None:
        payload = {
            "runId": self.run_id,
            "window": window,
            "results": results,
        }
        await self._send("gather-status", "/ingest/gather-status", payload)

    # ── Cleanup ───────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._ws_heartbeat_task and not self._ws_heartbeat_task.done():
            self._ws_heartbeat_task.cancel()
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None
