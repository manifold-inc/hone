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
        self.external_id = str(uuid.uuid4())

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

    async def _post_json(self, path: str, body: dict[str, Any]) -> None:
        try:
            await self._ensure_session()
            assert self._session is not None
            async with self._session.post(
                f"{self._url}{path}",
                json=body,
                headers=self._headers(),
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.warning(
                        "Hone API %s failed: %s %s", path, resp.status, text[:500]
                    )
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
        await self._post_json(
            "/ingest/run",
            {
                "external_id": self.external_id,
                "hotkey": hotkey,
                "role": role,
                "netuid": netuid,
                "uid": uid,
                "version": version,
                "config": config if config is not None else {},
            },
        )

    async def report_window(self, **metrics: Any) -> None:
        await self._post_json("/ingest/window", {**metrics, "external_id": self.external_id})

    async def report_miner(self, **metrics: Any) -> None:
        await self._post_json("/ingest/miner", {**metrics, "external_id": self.external_id})
