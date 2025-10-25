from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, status
from pydantic import BaseModel
from loguru import logger

import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_pool()
    logger.info("API ready.")
    try:
        yield
    finally:
        await db.close_pool()
        logger.info("API shutdown complete.")


app = FastAPI(
    title="Validator Telemetry API",
    version="1.0.0",
    lifespan=lifespan,
)


class HeartbeatIn(BaseModel):
    ts: datetime
    version: str
    cycle_count: Optional[int] = None
    netuid: Optional[int] = None
    wallet_hotkey: Optional[str] = None


class MinerMetricsDetails(BaseModel):
    exact_match: bool
    partial_correctness: float
    grid_similarity: float
    efficiency_score: float


class MinerMetricsIn(BaseModel):
    ts: datetime
    block: int
    uid: int
    problem_id: str
    success: bool
    response_time: float  # seconds
    metrics: MinerMetricsDetails


class BatchSummaryIn(BaseModel):
    ts: datetime
    block: int
    num_miners: int
    num_problems: int
    total_queries: int
    successful: int
    exact_matches: int


@app.post(
    "/validator/heartbeat",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def ingest_heartbeat(body: HeartbeatIn):
    """
    Validator heartbeat: version, cycle, etc.
    """
    query = """
        INSERT INTO validator_heartbeat
            (ts, version, cycle_count, netuid, wallet_hotkey)
        VALUES ($1, $2, $3, $4, $5)
    """

    await db.execute(
        query,
        body.ts,
        body.version,
        body.cycle_count,
        body.netuid,
        body.wallet_hotkey,
    )

    return


@app.post(
    "/validator/ingest_miner_metrics",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def ingest_miner_metrics(body: MinerMetricsIn):
    """
    Per miner/problem result.
    """
    query = """
        INSERT INTO miner_metrics
            (ts, block, uid, problem_id, success,
             response_time,
             exact_match,
             partial_correctness,
             grid_similarity,
             efficiency_score)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
    """

    await db.execute(
        query,
        body.ts,
        body.block,
        body.uid,
        body.problem_id,
        body.success,
        body.response_time,
        body.metrics.exact_match,
        body.metrics.partial_correctness,
        body.metrics.grid_similarity,
        body.metrics.efficiency_score,
    )

    return


@app.post(
    "/validator/batch_summary",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def ingest_batch_summary(body: BatchSummaryIn):
    """
    Aggregated stats per cycle / batch.
    """
    query = """
        INSERT INTO batch_summary
            (ts, block,
             num_miners,
             num_problems,
             total_queries,
             successful,
             exact_matches)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
    """

    await db.execute(
        query,
        body.ts,
        body.block,
        body.num_miners,
        body.num_problems,
        body.total_queries,
        body.successful,
        body.exact_matches,
    )

    return
