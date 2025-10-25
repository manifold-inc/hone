import os
import asyncpg
from loguru import logger

_pool = None


async def init_pool():
    global _pool
    if _pool is not None:
        return _pool

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")

    _pool = await asyncpg.create_pool(
        db_url,
        min_size=1,
        max_size=10,
        timeout=3.0,
    )
    logger.info("Postgres pool initialized")
    return _pool


async def close_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        logger.info("Postgres pool closed")
        _pool = None


async def execute(query: str, *args):
    if _pool is None:
        raise RuntimeError("DB pool not initialized")

    async with _pool.acquire() as conn:
        await conn.execute(query, *args)
