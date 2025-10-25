from __future__ import annotations
import asyncio
import time
from typing import Optional, Tuple
import httpx
from loguru import logger


class TelemetryClient:
    def __init__(
        self,
        endpoint_base_url: Optional[str],
        max_queue_size: int = 1000,
        flush_interval_s: float = 1.0,
        request_timeout_s: float = 3.0,
        max_retries: int = 3,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
    
        self.endpoint_base_url = (endpoint_base_url or "").rstrip("/")
        self.enabled = bool(self.endpoint_base_url)

        self.queue: asyncio.Queue[
            Tuple[str, dict, float]
        ] = asyncio.Queue(maxsize=max_queue_size)

        self.flush_interval_s = flush_interval_s
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries

        self._stopping = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None
        self._loop = loop or asyncio.get_event_loop()

        # httpx async client, reused for connection pooling
        self._client = httpx.AsyncClient(timeout=self.request_timeout_s)

        # spawn the worker immediately
        self._worker_task = self._loop.create_task(self._worker_loop(), name="telemetry-worker")

        logger.info("TelemetryClient initialized (enabled=%s)", self.enabled)

    def publish(self, route: str, payload: dict) -> None:

        if not self.enabled:
            return

        try:
            if asyncio.get_event_loop() is self._loop:
                try:
                    self.queue.put_nowait((route, payload, time.time()))
                except asyncio.QueueFull:
                    logger.warning("Telemetry queue full, dropping metric")
            else:
                # called from another thread (or different loop)
                def _enqueue():
                    try:
                        self.queue.put_nowait((route, payload, time.time()))
                    except asyncio.QueueFull:
                        logger.warning("Telemetry queue full, dropping metric")

                self._loop.call_soon_threadsafe(_enqueue)

        except Exception as e:
            logger.warning(f"Telemetry publish() swallowed exception: {e}")

    async def _worker_loop(self) -> None:

        if not self.enabled:
            logger.info("Telemetry disabled, worker going into idle mode.")
            await self._stopping.wait()
            return

        logger.info("Telemetry worker started")

        while not self._stopping.is_set():
            try:
                try:
                    route, payload, ts = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.flush_interval_s,
                    )
                except asyncio.TimeoutError:
                    continue  # just loop back and check stopping flag

                url = f"{self.endpoint_base_url}/{route.lstrip('/')}"
                sent = False

                for attempt in range(1, self.max_retries + 1):
                    if self._stopping.is_set():
                        break

                    try:
                        r = await self._client.post(url, json=payload)
                        status_ok = 200 <= r.status_code < 300
                        if status_ok:
                            sent = True
                            break
                        else:
                            logger.warning(
                                f"Telemetry send failed {r.status_code} {r.text} "
                                f"(attempt {attempt}/{self.max_retries})"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Telemetry exception: {e} "
                            f"(attempt {attempt}/{self.max_retries})"
                        )
                        logger.exception(e)
                        await asyncio.sleep(0.5)

                if not sent:
                    logger.error("Dropping telemetry after max retries")

                self.queue.task_done()

            except Exception as e:
                logger.error(f"Unhandled telemetry worker error: {e}")
                await asyncio.sleep(0.5)

        logger.info("Telemetry worker exiting")

    async def shutdown(self, drain: bool = False) -> None:
        """
        - if drain=True we try to flush what's left in the queue before closing.
        - we always close the httpx client.
        """

        if self._stopping.is_set():
            return  # already shutting down

        self._stopping.set()

        if drain and self.enabled:
            logger.info("Telemetry drain requested, flushing remaining queue...")
            drain_deadline = time.time() + 2.0

            while not self.queue.empty() and time.time() < drain_deadline:
                await asyncio.sleep(0.05)

        if self._worker_task:
            await asyncio.sleep(0)
            if not self._worker_task.done():
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass

        try:
            await self._client.aclose()
        except Exception as e:
            logger.warning(f"Error closing telemetry http client: {e}")

        logger.info("Telemetry shutdown complete")
