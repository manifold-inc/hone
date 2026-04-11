"""Base node for Hone miners and validators."""

import abc
import asyncio
import functools
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import bittensor as bt
import websockets.exceptions
from bittensor.core.subtensor import ScaleObj

import hone
from hone.distributed import dist_helper

CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))


class BaseNode(abc.ABC):
    executor: ThreadPoolExecutor
    world_size: int = 1
    comms: hone.Comms
    config: Any = None
    hparams: SimpleNamespace
    subtensor: bt.Subtensor
    ckpt: hone.DCPCheckpointer

    stop_event: asyncio.Event
    window_changed: asyncio.Event | None
    _notify_loop: asyncio.AbstractEventLoop | None

    def __init__(self):
        self.stop_event = asyncio.Event()
        self._bg_tasks: set[asyncio.Task] = set()
        self._threads: list[threading.Thread] = []

        self.window_changed = None
        self._notify_loop = None

        self.current_block = 0
        self.current_window = 0
        self.subtensor_rpc: bt.Subtensor | None = None
        self.subtensor_client: bt.Subtensor | None = None

    async def main(self):
        loop = asyncio.get_running_loop()
        self._setup_signal_handlers(loop)

        self.window_changed = asyncio.Event()
        self._notify_loop = loop

        t = threading.Thread(target=self.block_listener, name="blocks", daemon=True)
        t.start()
        self._threads.append(t)

        try:
            await self.run()
        except Exception:
            hone.logger.error("Unhandled exception in run()", exc_info=True)
        finally:
            if not self.stop_event.is_set():
                await self._graceful_shutdown(signal.SIGTERM)

    @abc.abstractmethod
    async def run(self):
        raise NotImplementedError

    async def wait_until_window(self, target_window: int) -> None:
        evt = self.window_changed

        if evt is not None and evt.is_set():
            evt.clear()

        while not self.stop_event.is_set():
            if self.current_window >= target_window:
                return

            remaining_windows = target_window - self.current_window
            blocks_into_window = self.current_block % self.hparams.blocks_per_window
            remaining_blocks = (
                remaining_windows * self.hparams.blocks_per_window - blocks_into_window
            )
            eta_seconds = max(0, remaining_blocks * 12)
            mins, secs = divmod(int(eta_seconds), 60)

            hone.logger.info(
                f"Waiting for window {target_window} "
                f"(~{mins}m {secs:02d}s, {remaining_blocks} blocks)"
            )

            if evt is None:
                await asyncio.sleep(0.5)
            else:
                await evt.wait()
                evt.clear()

    def _setup_signal_handlers(self, loop: asyncio.AbstractEventLoop):
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                functools.partial(asyncio.create_task, self._graceful_shutdown(sig)),
            )

    async def _graceful_shutdown(self, sig):
        hone.logger.warning(f"Received {sig.name} - shutting down")
        self.stop_event.set()
        await self.ckpt.flush_background_uploads()

        for task in list(self._bg_tasks):
            task.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)

        for th in self._threads:
            th.join(timeout=2)

        if self.subtensor_rpc is not None:
            try:
                self.subtensor_rpc.close()
            except Exception:
                pass

        if self.subtensor_client is not None:
            try:
                self.subtensor_client.close()
            except Exception:
                pass

        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, "world_size"):
            dist_helper.destroy_process_group()

        await asyncio.sleep(0.2)
        hone.logger.info("Shutdown complete")

    def query_block_timestamp(
        self,
        block: int,
        *,
        retries: int = 5,
        init_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> float | None:
        if self.subtensor_client is None:
            self.subtensor_client = bt.subtensor(config=self.config)

        delay = init_delay
        for attempt in range(1, retries + 1):
            try:
                resp = self.subtensor_client.query_module(
                    "Timestamp", "Now", block=block
                )
                if resp is None or not isinstance(resp, ScaleObj):
                    raise ValueError(f"Could not query timestamp for {block}")
                return cast(int, resp.value) / 1000
            except Exception as e:
                hone.logger.warning(
                    f"[timestamp] block {block} attempt {attempt}/{retries}: {e}"
                )
                if attempt == retries:
                    return None
                try:
                    self.subtensor_client.substrate.close()
                except Exception:
                    pass
                self.subtensor_client.substrate.initialize()
                time.sleep(delay)
                delay = min(delay * 2, max_delay)

    def block_listener(self):
        backoff, max_backoff = 1, 60

        if self.subtensor_rpc is None:
            self.subtensor_rpc = bt.subtensor(config=self.config)

        def handler(event):
            try:
                self.current_block = int(event["header"]["number"])
                new_window = self.current_block // self.hparams.blocks_per_window
                if new_window != self.current_window:
                    self.current_window = new_window
                    if hasattr(self, "comms"):
                        self.comms.current_window = self.current_window
                    hone.logger.info(f"Window -> {self.current_window}")

                    if self.window_changed and self._notify_loop:
                        self._notify_loop.call_soon_threadsafe(self.window_changed.set)
            except Exception as e:
                hone.logger.error(f"block-handler err: {e}")

        while not self.stop_event.is_set():
            try:
                self.subtensor_rpc.substrate.initialize()
                self.subtensor_rpc.substrate.subscribe_block_headers(handler)
                backoff = 1
            except websockets.exceptions.ConnectionClosedError as e:
                if self.stop_event.is_set():
                    break
                hone.logger.warning(f"ws closed: {e} - retrying in {backoff}s")
            except Exception as e:
                if self.stop_event.is_set():
                    break
                hone.logger.error(f"block subscription err: {e} - retrying in {backoff}s")
            finally:
                try:
                    self.subtensor_rpc.substrate.close()
                except Exception:
                    pass

            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
