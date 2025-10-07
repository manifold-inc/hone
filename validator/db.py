import os
import asyncio
import asyncpg
from typing import Dict, List, Optional
from datetime import datetime, timezone
from loguru import logger
import json


class Database:
    def __init__(self, dsn: Optional[str] = None, schema: str = "hone"):
        self.dsn = dsn or os.getenv("DB_URL") or os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@localhost:5432/hone"
        self.schema = schema
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        if self.pool:
            return
        attempts = 0
        delay = 0.5
        last_exc = None
        while attempts < 10:
            try:
                self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
                logger.info("Connected to Postgres.")
                return
            except Exception as e:
                last_exc = e
                attempts += 1
                logger.warning(f"DB connect attempt {attempts}/10 failed ({e}); retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)

    async def close(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def upsert_miner(self, uid: int, hotkey: str, ip: Optional[str], port: Optional[int], stake: Optional[float], last_update_block: Optional[int]):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO miners (uid, hotkey, ip, port, stake, last_update_block, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                ON CONFLICT (uid) DO UPDATE SET
                    hotkey = EXCLUDED.hotkey,
                    ip = EXCLUDED.ip,
                    port = EXCLUDED.port,
                    stake = EXCLUDED.stake,
                    last_update_block = EXCLUDED.last_update_block,
                    updated_at = NOW()
                """,
                uid, hotkey, ip, port, stake, last_update_block
            )

    async def get_miners(self) -> List[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            return await conn.fetch("SELECT * FROM miners ORDER BY uid ASC")

    async def record_query_result(
        self, 
        block: int, 
        uid: int, 
        success: bool, 
        response: Optional[dict], 
        error: Optional[str], 
        response_time: Optional[float], 
        ts: datetime,
        exact_match: bool = False,
        partial_correctness: float = 0.0,
        grid_similarity: float = 0.0,
        efficiency_score: float = 0.0,
        problem_id: Optional[str] = None
    ):
        async with self.pool.acquire() as conn:
            response_json = json.dumps(response) if response else None
            
            await conn.execute(
                """
                INSERT INTO query_results (
                    block, uid, success, response, error, response_time, timestamp,
                    exact_match, partial_correctness, grid_similarity, efficiency_score,
                    problem_id
                )
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                block, uid, success, response_json, error, response_time, ts,
                exact_match, partial_correctness, grid_similarity, efficiency_score,
                problem_id
            )

    async def get_recent_results(self, window_blocks: int, current_block: int) -> List[asyncpg.Record]:
        min_block = max(0, current_block - window_blocks)
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT * FROM query_results
                WHERE block >= $1
                ORDER BY timestamp DESC
                """,
                min_block
            )

    async def save_scores(self, scores: Dict[int, Dict[str, float]]):
        """
        Save scores with detailed metrics.
        scores format: {uid: {"score": float, "exact_match_rate": float, ...}}
        """
        ts = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.pool.acquire() as conn:
            for uid, metrics in scores.items():
                await conn.execute(
                    """
                    INSERT INTO scores (
                        uid, score, exact_match_rate, partial_correctness_avg, 
                        efficiency_avg, timestamp
                    ) 
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    uid, 
                    float(metrics.get("score", 0.0)),
                    float(metrics.get("exact_match_rate", 0.0)),
                    float(metrics.get("partial_correctness_avg", 0.0)),
                    float(metrics.get("efficiency_avg", 0.0)),
                    ts
                )

    async def get_scores_last_hours(self, hours: int = 24) -> List[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM scores WHERE timestamp > NOW() - ($1 || ' hours')::interval",
                str(hours)
            )
            return rows

    async def get_miner_performance_stats(self, uid: int, window_blocks: int, current_block: int) -> Dict:
        """Get detailed performance statistics for a specific miner"""
        min_block = max(0, current_block - window_blocks)
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(CASE WHEN exact_match THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as exact_match_rate,
                    AVG(partial_correctness) as avg_partial_correctness,
                    AVG(grid_similarity) as avg_grid_similarity,
                    AVG(efficiency_score) as avg_efficiency,
                    AVG(response_time) as avg_response_time
                FROM query_results
                WHERE uid = $1 AND block >= $2
                """,
                uid, min_block
            )
            return dict(stats) if stats else {}