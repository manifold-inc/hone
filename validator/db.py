import os
import asyncio
import asyncpg
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, date
from loguru import logger
import json
import hashlib


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
            # remove any stale entry where a different uid has this hotkey
            await conn.execute(
                "DELETE FROM miners WHERE hotkey = $1 AND uid != $2",
                hotkey, uid
            )
            
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

    async def get_miner_by_hotkey(self, hotkey: str) -> Optional[asyncpg.Record]:
        """Get miner by hotkey"""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow("SELECT * FROM miners WHERE hotkey = $1", hotkey)

    # ===== SUBMISSION HISTORY METHODS =====
    
    async def check_daily_submission_limit(self, hotkey: str, max_per_day: int) -> Tuple[bool, int]:
        """
        Check if miner has reached daily submission limit
        Returns: (can_submit, current_count)
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT submission_count 
                FROM daily_submissions 
                WHERE hotkey = $1 AND submission_date = CURRENT_DATE
                """,
                hotkey
            )
            
            current_count = row['submission_count'] if row else 0
            can_submit = current_count < max_per_day
            
            return can_submit, current_count

    async def increment_daily_submissions(self, hotkey: str):
        """Increment daily submission count for a miner"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO daily_submissions (hotkey, submission_date, submission_count, last_submission_time)
                VALUES ($1, CURRENT_DATE, 1, NOW())
                ON CONFLICT (hotkey, submission_date) DO UPDATE SET
                    submission_count = daily_submissions.submission_count + 1,
                    last_submission_time = NOW()
                """,
                hotkey
            )

    async def get_submission_history(
        self, 
        hotkey: str,
        repo_url: str,
        repo_branch: str,
        repo_commit: Optional[str],
        repo_path: str,
        weight_class: str
    ) -> Optional[asyncpg.Record]:
        """
        Check if identical solution has been evaluated before
        Returns cached metrics if found
        """
        repo_commit = repo_commit or ''
        repo_path = repo_path or ''
        
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                """
                SELECT * FROM submission_history
                WHERE hotkey = $1 
                AND repo_url = $2 
                AND repo_branch = $3 
                AND repo_commit = $4
                AND repo_path = $5
                AND weight_class = $6
                """,
                hotkey, repo_url, repo_branch, repo_commit, repo_path, weight_class
            )

    async def save_submission_history(
        self,
        hotkey: str,
        repo_url: str,
        repo_branch: str,
        repo_commit: Optional[str],
        repo_path: str,
        weight_class: str,
        use_vllm: bool,
        vllm_config: Optional[Dict],
        exact_match_rate: float
    ):
        """Save or update submission history with exact_match_rate"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO submission_history (
                    hotkey, repo_url, repo_branch, repo_commit, repo_path, weight_class,
                    use_vllm, vllm_config,
                    exact_match_rate,
                    first_submitted_at, last_evaluated_at, evaluation_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, NOW(), NOW(), 1)
                ON CONFLICT (hotkey, repo_url, repo_branch, COALESCE(repo_commit, ''), repo_path, weight_class)
                DO UPDATE SET
                    exact_match_rate = EXCLUDED.exact_match_rate,
                    last_evaluated_at = NOW(),
                    evaluation_count = submission_history.evaluation_count + 1
                """,
                hotkey, repo_url, repo_branch, repo_commit or '', repo_path, weight_class,
                use_vllm, json.dumps(vllm_config) if vllm_config else None,
                exact_match_rate
            )

    async def record_query_result(
        self, 
        block: int, 
        uid: int,
        hotkey: str,
        success: bool, 
        response: Optional[dict], 
        error: Optional[str], 
        response_time: Optional[float], 
        ts: datetime,
        exact_match_rate: float = 0.0,
        repo_url: Optional[str] = None,
        repo_branch: Optional[str] = None,
        repo_commit: Optional[str] = None,
        repo_path: Optional[str] = None,
        weight_class: Optional[str] = None,
        from_cache: bool = False
    ):
        async with self.pool.acquire() as conn:
            response_json = json.dumps(response) if response else None
            
            await conn.execute(
                """
                INSERT INTO query_results (
                    block, uid, hotkey, success, response, error, response_time, timestamp,
                    exact_match_rate,
                    repo_url, repo_branch, repo_commit, repo_path, weight_class, from_cache
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                block, uid, hotkey, success, response_json, error, response_time, ts,
                exact_match_rate,
                repo_url, repo_branch, repo_commit, repo_path, weight_class, from_cache
            )

    # ===== LEADERBOARD METHODS =====
    
    async def get_leaderboard(self, limit: int = 5) -> List[asyncpg.Record]:
        """Get current top miners"""
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT * FROM leaderboard
                ORDER BY exact_match_rate DESC
                LIMIT $1
                """,
                limit
            )

    async def update_leaderboard(
        self,
        hotkey: str,
        uid: int,
        exact_match_rate: float,
        repo_url: str,
        repo_branch: str,
        repo_commit: Optional[str],
        repo_path: str
    ):
        """Update or insert miner into leaderboard"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO leaderboard (
                    hotkey, uid, exact_match_rate,
                    repo_url, repo_branch, repo_commit, repo_path,
                    first_achieved_at, last_updated_at, evaluation_count
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW(), 1)
                ON CONFLICT (hotkey) DO UPDATE SET
                    uid = EXCLUDED.uid,
                    exact_match_rate = EXCLUDED.exact_match_rate,
                    repo_url = EXCLUDED.repo_url,
                    repo_branch = EXCLUDED.repo_branch,
                    repo_commit = EXCLUDED.repo_commit,
                    repo_path = EXCLUDED.repo_path,
                    last_updated_at = NOW(),
                    evaluation_count = leaderboard.evaluation_count + 1
                """,
                hotkey, uid, exact_match_rate, repo_url, repo_branch, repo_commit, repo_path
            )

    async def remove_from_leaderboard(self, hotkey: str):
        """Remove miner from leaderboard"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM leaderboard WHERE hotkey = $1",
                hotkey
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

    async def save_scores(self, scores: Dict[int, float], hotkey_map: Dict[int, str]):
        """
        Save scores (exact_match_rate based)
        scores format: {uid: exact_match_rate}
        hotkey_map: {uid: hotkey}
        """
        ts = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.pool.acquire() as conn:
            for uid, exact_match_rate in scores.items():
                hotkey = hotkey_map.get(uid)
                if not hotkey:
                    continue
                    
                await conn.execute(
                    """
                    INSERT INTO scores (uid, hotkey, exact_match_rate, timestamp) 
                    VALUES ($1, $2, $3, $4)
                    """,
                    uid, hotkey, float(exact_match_rate), ts
                )

    async def get_scores_last_hours(self, hours: int = 24) -> List[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM scores WHERE timestamp > NOW() - ($1 || ' hours')::interval",
                str(hours)
            )
            return rows

    async def get_miner_performance_stats(self, uid: int, window_blocks: int, current_block: int) -> Dict:
        """Get performance statistics for a specific miner"""
        min_block = max(0, current_block - window_blocks)
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(exact_match_rate) as avg_exact_match_rate,
                    MAX(exact_match_rate) as max_exact_match_rate,
                    AVG(response_time) as avg_response_time
                FROM query_results
                WHERE uid = $1 AND block >= $2 AND success = true
                """,
                uid, min_block
            )
            return dict(stats) if stats else {}
    
    async def cleanup_old_data(self, retention_days: int):
        """Delete data older than retention_days to prevent disk overflow"""
        async with self.pool.acquire() as conn:
            deleted_queries = await conn.execute(
                """
                DELETE FROM query_results
                WHERE timestamp < NOW() - ($1 || ' days')::interval
                """,
                str(retention_days)
            )
            
            deleted_scores = await conn.execute(
                """
                DELETE FROM scores
                WHERE timestamp < NOW() - ($1 || ' days')::interval
                """,
                str(retention_days)
            )
            
            logger.info(f"Cleaned up data older than {retention_days} days")
            return deleted_queries, deleted_scores