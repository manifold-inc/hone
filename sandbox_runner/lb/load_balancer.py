"""
Sandbox Runner Load Balancer

Routes validator requests to multiple sandbox runners with:
- Solution caching (repo + branch + commit)
- Sticky routing (job_id -> runner)
- Health checks and failover
- Persistent cache (memory + SQLite)

Usage:
    pm2 start load_balancer.py --interpreter python3 --name sandbox-lb -- \
        --runners https://runner1.example.com https://runner2.example.com \
        --port 8080
"""

import argparse
import asyncio
import hashlib
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import subprocess
import re

import aiohttp
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger


class Config:
    def __init__(
        self,
        runner_urls: List[str],
        port: int = 8080,
        cache_dir: str = "./cache",
        cache_ttl_days: int = 7,
        health_check_interval: int = 30,
        github_timeout: int = 10,
    ):
        self.runner_urls = [url.rstrip('/') for url in runner_urls]
        self.port = port
        self.cache_dir = Path(cache_dir)
        self.cache_ttl_days = cache_ttl_days
        self.cache_ttl_seconds = cache_ttl_days * 24 * 3600
        self.health_check_interval = health_check_interval
        self.github_timeout = github_timeout
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class VLLMConfig(BaseModel):
    model: Optional[str] = None
    dtype: Optional[str] = None
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    extra_args: Optional[Dict[str, Any]] = None


class JobSubmitRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub/GitLab repository URL")
    repo_branch: str = Field(default="main", description="Git branch")
    repo_commit: Optional[str] = Field(None, description="Specific commit hash")
    repo_path: str = Field(default="", description="Subdirectory path within repo")
    weight_class: str = Field(..., description="GPU weight class (1xH200, 2xH200, etc)")
    priority: int = Field(default=5, ge=0, le=10, description="Job priority")
    validator_hotkey: Optional[str] = Field(None, description="Validator's hotkey")
    miner_hotkey: str = Field(..., description="Miner's hotkey")
    custom_env_vars: Optional[Dict[str, str]] = Field(default_factory=dict)
    use_vllm: bool = Field(default=False)
    vllm_config: Optional[VLLMConfig] = None


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    estimated_start_time: Optional[str] = None
    queue_position: int
    from_cache: bool = False
    cache_hit: bool = False


class CachedMetrics(BaseModel):
    repo_url: str
    repo_branch: str
    repo_commit: str
    repo_path: str
    weight_class: str
    miner_hotkey: str
    metrics: Dict[str, Any]
    cached_at: str
    expires_at: str


class CacheManager:
    """Dual-layer cache: in-memory for speed, SQLite for persistence"""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_path = config.cache_dir / "cache.db"
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self._init_db()
        self._load_cache_to_memory()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solution_cache (
                cache_key TEXT PRIMARY KEY,
                repo_url TEXT NOT NULL,
                repo_branch TEXT NOT NULL,
                repo_commit TEXT NOT NULL,
                repo_path TEXT DEFAULT '',
                weight_class TEXT NOT NULL,
                miner_hotkey TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                cached_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_expires 
            ON solution_cache(expires_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_miner 
            ON solution_cache(miner_hotkey)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Cache database initialized at {self.db_path}")
    
    def _load_cache_to_memory(self):
        """Load non-expired cache entries to memory on startup"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = time.time()
        cursor.execute(
            "SELECT cache_key, repo_url, repo_branch, repo_commit, repo_path, "
            "weight_class, miner_hotkey, metrics_json, cached_at, expires_at "
            "FROM solution_cache WHERE expires_at > ?",
            (now,)
        )
        
        loaded = 0
        for row in cursor.fetchall():
            cache_key = row[0]
            self.memory_cache[cache_key] = {
                "repo_url": row[1],
                "repo_branch": row[2],
                "repo_commit": row[3],
                "repo_path": row[4],
                "weight_class": row[5],
                "miner_hotkey": row[6],
                "metrics": json.loads(row[7]),
                "cached_at": row[8],
                "expires_at": row[9]
            }
            loaded += 1
        
        conn.close()
        logger.info(f"Loaded {loaded} cached entries to memory")
    
    def _make_cache_key(
        self, 
        repo_url: str, 
        repo_branch: str, 
        repo_commit: str, 
        repo_path: str,
        weight_class: str
    ) -> str:
        """Generate deterministic cache key"""
        normalized_url = repo_url.lower().rstrip('/').rstrip('.git')
        key_data = f"{normalized_url}|{repo_branch}|{repo_commit}|{repo_path}|{weight_class}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(
        self,
        repo_url: str,
        repo_branch: str,
        repo_commit: str,
        repo_path: str,
        weight_class: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached metrics if available and not expired"""
        cache_key = self._make_cache_key(repo_url, repo_branch, repo_commit, repo_path, weight_class)
        
        # check memory first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if entry["expires_at"] > time.time():
                logger.debug(f"Cache HIT (memory): {cache_key[:16]}...")
                return entry
            else:
                # expired, remove from memory
                del self.memory_cache[cache_key]
        
        # check disk
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT repo_url, repo_branch, repo_commit, repo_path, weight_class, "
            "miner_hotkey, metrics_json, cached_at, expires_at "
            "FROM solution_cache WHERE cache_key = ? AND expires_at > ?",
            (cache_key, time.time())
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            entry = {
                "repo_url": row[0],
                "repo_branch": row[1],
                "repo_commit": row[2],
                "repo_path": row[3],
                "weight_class": row[4],
                "miner_hotkey": row[5],
                "metrics": json.loads(row[6]),
                "cached_at": row[7],
                "expires_at": row[8]
            }
            # reload to memory
            self.memory_cache[cache_key] = entry
            logger.debug(f"Cache HIT (disk): {cache_key[:16]}...")
            return entry
        
        logger.debug(f"Cache MISS: {cache_key[:16]}...")
        return None
    
    def set(
        self,
        repo_url: str,
        repo_branch: str,
        repo_commit: str,
        repo_path: str,
        weight_class: str,
        miner_hotkey: str,
        metrics: Dict[str, Any]
    ):
        """Store metrics in cache"""
        cache_key = self._make_cache_key(repo_url, repo_branch, repo_commit, repo_path, weight_class)
        now = time.time()
        expires_at = now + self.config.cache_ttl_seconds
        
        entry = {
            "repo_url": repo_url,
            "repo_branch": repo_branch,
            "repo_commit": repo_commit,
            "repo_path": repo_path,
            "weight_class": weight_class,
            "miner_hotkey": miner_hotkey,
            "metrics": metrics,
            "cached_at": now,
            "expires_at": expires_at
        }
        
        # save to memory
        self.memory_cache[cache_key] = entry
        
        # save to disk
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO solution_cache 
            (cache_key, repo_url, repo_branch, repo_commit, repo_path, weight_class,
             miner_hotkey, metrics_json, cached_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (cache_key, repo_url, repo_branch, repo_commit, repo_path, weight_class,
             miner_hotkey, json.dumps(metrics), now, expires_at)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cached metrics for {repo_url} @ {repo_commit[:8]}... (expires in {self.config.cache_ttl_days} days)")
    
    def cleanup_expired(self):
        """Remove expired entries from both memory and disk"""
        now = time.time()
        
        # cleanup memory
        expired_keys = [k for k, v in self.memory_cache.items() if v["expires_at"] <= now]
        for key in expired_keys:
            del self.memory_cache[key]
        
        # cleanup disk
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM solution_cache WHERE expires_at <= ?", (now,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted > 0 or expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} memory + {deleted} disk expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM solution_cache WHERE expires_at > ?", (time.time(),))
        disk_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM solution_cache")
        total_disk = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "memory_entries": len(self.memory_cache),
            "disk_entries_valid": disk_count,
            "disk_entries_total": total_disk,
            "cache_ttl_days": self.config.cache_ttl_days
        }


class GitHubResolver:
    """Resolve latest commit hash from GitHub without authentication"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self._commit_cache: Dict[str, tuple] = {}  # (commit, timestamp)
        self._cache_ttl = 60  # cache commit lookups for 60 seconds
    
    async def get_latest_commit(self, repo_url: str, branch: str) -> Optional[str]:
        """
        Get latest commit hash for a branch using git ls-remote
        No authentication needed for public repos
        """
        cache_key = f"{repo_url}|{branch}"
        
        # check local cache (avoid hammering git)
        if cache_key in self._commit_cache:
            commit, ts = self._commit_cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return commit
        
        # normalize URL for git
        git_url = repo_url
        if not git_url.endswith('.git'):
            git_url = git_url + '.git'
        
        try:
            proc = await asyncio.create_subprocess_exec(
                'git', 'ls-remote', git_url, f'refs/heads/{branch}',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout
            )
            
            if proc.returncode != 0:
                logger.warning(f"git ls-remote failed for {repo_url}: {stderr.decode()}")
                return None
            
            output = stdout.decode().strip()
            if not output:
                logger.warning(f"No commit found for {repo_url} branch {branch}")
                return None
            
            # output format: "commit_hash\trefs/heads/branch"
            commit = output.split()[0]
            
            # cache it
            self._commit_cache[cache_key] = (commit, time.time())
            
            logger.debug(f"Resolved {repo_url}@{branch} -> {commit[:8]}...")
            return commit
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout resolving commit for {repo_url}")
            return None
        except Exception as e:
            logger.error(f"Error resolving commit for {repo_url}: {e}")
            return None


class RunnerHealthManager:
    """Track health status of sandbox runners"""
    
    def __init__(self, runner_urls: List[str], check_interval: int = 30):
        self.runner_urls = runner_urls
        self.check_interval = check_interval
        self.health_status: Dict[str, Dict[str, Any]] = {
            url: {"healthy": True, "last_check": 0, "error": None, "queue_depth": 0}
            for url in runner_urls
        }
        self._check_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start background health check task"""
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health check task started")
    
    async def stop(self):
        """Stop health check task"""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self):
        """Periodically check runner health"""
        while True:
            try:
                await self._check_all_runners()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
    
    async def _check_all_runners(self):
        """Check health of all runners"""
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._check_runner(session, url) for url in self.runner_urls]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_runner(self, session: aiohttp.ClientSession, runner_url: str):
        """Check single runner health"""
        try:
            async with session.get(
                f"{runner_url}/v1/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    self.health_status[runner_url] = {
                        "healthy": True,
                        "last_check": time.time(),
                        "error": None,
                        "queue_depth": 0
                    }
                    
                    # try to get queue depth from status endpoint
                    try:
                        async with session.get(
                            f"{runner_url}/v1/status",
                            headers={"X-API-Key": "dev-key-12345"},  # TODO: change 
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as status_resp:
                            if status_resp.status == 200:
                                data = await status_resp.json()
                                self.health_status[runner_url]["queue_depth"] = data.get("queue_depth", 0)
                                self.health_status[runner_url]["active_jobs"] = data.get("active_jobs", 0)
                    except:
                        pass
                else:
                    self.health_status[runner_url] = {
                        "healthy": False,
                        "last_check": time.time(),
                        "error": f"HTTP {resp.status}",
                        "queue_depth": 0
                    }
        except Exception as e:
            self.health_status[runner_url] = {
                "healthy": False,
                "last_check": time.time(),
                "error": str(e),
                "queue_depth": 0
            }
    
    def get_healthy_runners(self) -> List[str]:
        """Get list of healthy runner URLs"""
        return [url for url, status in self.health_status.items() if status["healthy"]]
    
    def get_best_runner(self) -> Optional[str]:
        """Get healthiest runner with lowest queue depth"""
        healthy = [(url, status) for url, status in self.health_status.items() if status["healthy"]]
        
        if not healthy:
            return None
        
        # sort by queue_depth + active_jobs
        healthy.sort(key=lambda x: x[1].get("queue_depth", 0) + x[1].get("active_jobs", 0))
        return healthy[0][0]
    
    def is_any_healthy(self) -> bool:
        """Check if at least one runner is healthy"""
        return any(status["healthy"] for status in self.health_status.values())


class JobRouter:
    """Routes jobs to runners with sticky job_id -> runner mapping"""
    
    def __init__(self, health_manager: RunnerHealthManager):
        self.health_manager = health_manager
        self.job_to_runner: Dict[str, str] = {}  # job_id -> runner_url
        self.round_robin_index = 0
        self._lock = asyncio.Lock()
    
    async def assign_runner(self, job_id: str) -> Optional[str]:
        """Assign a runner to a new job"""
        async with self._lock:
            runner = self.health_manager.get_best_runner()
            
            if not runner:
                return None
            
            self.job_to_runner[job_id] = runner
            logger.debug(f"Assigned job {job_id} to runner {runner}")
            return runner
    
    def get_runner(self, job_id: str) -> Optional[str]:
        """Get the assigned runner for a job"""
        return self.job_to_runner.get(job_id)
    
    def release_job(self, job_id: str):
        """Release job_id -> runner mapping after job completes"""
        if job_id in self.job_to_runner:
            del self.job_to_runner[job_id]
            logger.debug(f"Released job {job_id} from routing table")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        runner_job_counts = {}
        for runner in self.health_manager.runner_urls:
            runner_job_counts[runner] = sum(
                1 for r in self.job_to_runner.values() if r == runner
            )
        
        return {
            "active_job_mappings": len(self.job_to_runner),
            "jobs_per_runner": runner_job_counts
        }


class APIKeyManager:
    """Simple API key validation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.keys_file = config.cache_dir / "api_keys.json"
        self.valid_keys: Set[str] = set()
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from file"""
        if self.keys_file.exists():
            with open(self.keys_file, 'r') as f:
                data = json.load(f)
                self.valid_keys = set(data.get("keys", []))
        else:
            # TODO : change
            default_key = "lb-dev-key-12345"
            self.valid_keys = {default_key}
            self._save_keys()
    
    def _save_keys(self):
        """Save API keys to file"""
        with open(self.keys_file, 'w') as f:
            json.dump({"keys": list(self.valid_keys)}, f, indent=2)
    
    def validate(self, api_key: str) -> bool:
        """Validate an API key"""
        return api_key in self.valid_keys
    
    def add_key(self, api_key: str):
        """Add a new API key"""
        self.valid_keys.add(api_key)
        self._save_keys()
    
    def remove_key(self, api_key: str):
        """Remove an API key"""
        self.valid_keys.discard(api_key)
        self._save_keys()


class StatsTracker:
    """Track load balancer statistics"""
    
    def __init__(self):
        self.started_at = time.time()
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.jobs_submitted = 0
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.errors = 0
    
    def record_cache_hit(self):
        self.cache_hits += 1
        self.total_requests += 1
    
    def record_cache_miss(self):
        self.cache_misses += 1
        self.total_requests += 1
    
    def record_job_submitted(self):
        self.jobs_submitted += 1
    
    def record_job_completed(self):
        self.jobs_completed += 1
    
    def record_job_failed(self):
        self.jobs_failed += 1
    
    def record_error(self):
        self.errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.started_at
        cache_total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0
        
        return {
            "uptime_seconds": uptime,
            "uptime_human": f"{uptime / 3600:.1f} hours",
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": round(hit_rate, 2),
            "jobs_submitted": self.jobs_submitted,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "errors": self.errors
        }


class LoadBalancer:
    """Main load balancer orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache = CacheManager(config)
        self.github = GitHubResolver(timeout=config.github_timeout)
        self.health_manager = RunnerHealthManager(config.runner_urls, config.health_check_interval)
        self.router = JobRouter(self.health_manager)
        self.api_keys = APIKeyManager(config)
        self.stats = StatsTracker()
        
        # job_id -> {runner_url, metrics cached after completion}
        self.pending_jobs: Dict[str, Dict[str, Any]] = {}
    
    async def start(self):
        """Start background tasks"""
        await self.health_manager.start()
        
        # start cache cleanup task
        asyncio.create_task(self._cache_cleanup_loop())
        
        logger.info(f"Load balancer started with {len(self.config.runner_urls)} runners")
    
    async def stop(self):
        """Stop background tasks"""
        await self.health_manager.stop()
    
    async def _cache_cleanup_loop(self):
        """Periodically cleanup expired cache entries"""
        while True:
            try:
                await asyncio.sleep(3600)  # every hour
                self.cache.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def submit_job(self, request: JobSubmitRequest, api_key: str) -> Dict[str, Any]:
        """Submit a job - check cache first, then route to runner"""
        
        # resolve commit if not provided
        repo_commit = request.repo_commit
        if not repo_commit:
            repo_commit = await self.github.get_latest_commit(request.repo_url, request.repo_branch)
            if not repo_commit:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not resolve commit for {request.repo_url}@{request.repo_branch}"
                )
        
        # check cache
        cached = self.cache.get(
            repo_url=request.repo_url,
            repo_branch=request.repo_branch,
            repo_commit=repo_commit,
            repo_path=request.repo_path,
            weight_class=request.weight_class
        )
        
        if cached:
            self.stats.record_cache_hit()
            
            # generate a fake job_id for cache hits
            cache_job_id = f"cache_{hashlib.md5(f'{repo_commit}{time.time()}'.encode()).hexdigest()[:12]}"
            
            # store in pending_jobs so metrics endpoint works
            self.pending_jobs[cache_job_id] = {
                "from_cache": True,
                "metrics": cached["metrics"],
                "repo_commit": repo_commit,
                "status": "completed"
            }
            
            logger.info(f"Cache HIT for {request.repo_url}@{repo_commit[:8]}... -> {cache_job_id}")
            
            return {
                "job_id": cache_job_id,
                "status": "completed",
                "queue_position": 0,
                "from_cache": True,
                "cache_hit": True,
                "resolved_commit": repo_commit
            }
        
        self.stats.record_cache_miss()
        
        # check runner health
        if not self.health_manager.is_any_healthy():
            self.stats.record_error()
            raise HTTPException(
                status_code=503,
                detail="All sandbox runners are unavailable"
            )
        
        # assign runner
        runner_url = await self.router.assign_runner(f"pending_{time.time()}")
        if not runner_url:
            self.stats.record_error()
            raise HTTPException(
                status_code=503,
                detail="No healthy sandbox runners available"
            )
        
        # forward request to runner
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                payload = {
                    "repo_url": request.repo_url,
                    "repo_branch": request.repo_branch,
                    "repo_commit": repo_commit,
                    "repo_path": request.repo_path,
                    "weight_class": request.weight_class,
                    "priority": request.priority,
                    "validator_hotkey": request.validator_hotkey,
                    "miner_hotkey": request.miner_hotkey,
                    "custom_env_vars": request.custom_env_vars or {},
                    "use_vllm": request.use_vllm,
                    "vllm_config": request.vllm_config.dict() if request.vllm_config else None
                }
                
                # remove None values
                payload = {k: v for k, v in payload.items() if v is not None}
                
                async with session.post(
                    f"{runner_url}/v1/jobs/submit",
                    json=payload,
                    headers={"X-API-Key": "dev-key-12345"},  # TODO: change
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 201:
                        text = await resp.text()
                        self.stats.record_error()
                        raise HTTPException(
                            status_code=resp.status,
                            detail=f"Runner error: {text}"
                        )
                    
                    result = await resp.json()
        
        except aiohttp.ClientError as e:
            self.stats.record_error()
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to sandbox runner: {e}"
            )
        
        job_id = result.get("job_id")
        
        # update routing with real job_id
        self.router.release_job(f"pending_{time.time()}")
        self.router.job_to_runner[job_id] = runner_url
        
        # track pending job for caching later
        self.pending_jobs[job_id] = {
            "runner_url": runner_url,
            "from_cache": False,
            "repo_url": request.repo_url,
            "repo_branch": request.repo_branch,
            "repo_commit": repo_commit,
            "repo_path": request.repo_path,
            "weight_class": request.weight_class,
            "miner_hotkey": request.miner_hotkey,
            "status": "pending"
        }
        
        self.stats.record_job_submitted()
        
        logger.info(f"Submitted job {job_id} to {runner_url} for {request.repo_url}@{repo_commit[:8]}...")
        
        return {
            "job_id": job_id,
            "status": result.get("status", "pending"),
            "queue_position": result.get("queue_position", 0),
            "estimated_start_time": result.get("estimated_start_time"),
            "from_cache": False,
            "cache_hit": False,
            "resolved_commit": repo_commit,
            "assigned_runner": runner_url
        }
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status - from cache or forward to runner"""
        
        # check if it's a cache hit job
        if job_id in self.pending_jobs and self.pending_jobs[job_id].get("from_cache"):
            return {
                "job_id": job_id,
                "status": "completed",
                "from_cache": True,
                "progress_percentage": 100.0,
                "current_phase": "completed"
            }
        
        # find the runner for this job
        runner_url = self.router.get_runner(job_id)
        
        if not runner_url:
            # check pending_jobs as fallback
            if job_id in self.pending_jobs:
                runner_url = self.pending_jobs[job_id].get("runner_url")
        
        if not runner_url:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
        
        # forward to runner
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    f"{runner_url}/v1/jobs/{job_id}",
                    headers={"X-API-Key": "dev-key-12345"}, # TODO : change
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 404:
                        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
                    if resp.status != 200:
                        text = await resp.text()
                        raise HTTPException(status_code=resp.status, detail=text)
                    
                    result = await resp.json()
                    
                    # update status in pending_jobs
                    if job_id in self.pending_jobs:
                        self.pending_jobs[job_id]["status"] = result.get("status")
                    
                    return result
        
        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to sandbox runner: {e}"
            )
    
    async def get_job_metrics(self, job_id: str) -> Dict[str, Any]:
        """Get job metrics - from cache or forward to runner, then cache result"""
        
        # check if it's a cache hit job
        if job_id in self.pending_jobs and self.pending_jobs[job_id].get("from_cache"):
            return {
                "job_id": job_id,
                "status": "completed",
                "metrics": self.pending_jobs[job_id]["metrics"],
                "from_cache": True
            }
        
        # find the runner for this job
        runner_url = self.router.get_runner(job_id)
        
        if not runner_url and job_id in self.pending_jobs:
            runner_url = self.pending_jobs[job_id].get("runner_url")
        
        if not runner_url:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
        
        # forward to runner
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    f"{runner_url}/v1/jobs/{job_id}/metrics",
                    headers={"X-API-Key": "dev-key-12345"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 404:
                        raise HTTPException(status_code=404, detail=f"Metrics not found for job: {job_id}")
                    if resp.status == 400:
                        text = await resp.text()
                        raise HTTPException(status_code=400, detail=text)
                    if resp.status != 200:
                        text = await resp.text()
                        raise HTTPException(status_code=resp.status, detail=text)
                    
                    result = await resp.json()
        
        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to sandbox runner: {e}"
            )
        
        # cache the result if job completed successfully
        if result.get("status") == "completed" and result.get("metrics"):
            job_info = self.pending_jobs.get(job_id, {})
            
            if job_info and not job_info.get("from_cache"):
                self.cache.set(
                    repo_url=job_info["repo_url"],
                    repo_branch=job_info["repo_branch"],
                    repo_commit=job_info["repo_commit"],
                    repo_path=job_info["repo_path"],
                    weight_class=job_info["weight_class"],
                    miner_hotkey=job_info["miner_hotkey"],
                    metrics=result["metrics"]
                )
                
                self.stats.record_job_completed()
                
                # cleanup
                self.router.release_job(job_id)
                if job_id in self.pending_jobs:
                    del self.pending_jobs[job_id]
                
                logger.info(f"Cached metrics for completed job {job_id}")
        
        return result
    
    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a job"""
        
        # can't cancel cache hits
        if job_id in self.pending_jobs and self.pending_jobs[job_id].get("from_cache"):
            return {"job_id": job_id, "status": "cancelled"}
        
        runner_url = self.router.get_runner(job_id)
        
        if not runner_url and job_id in self.pending_jobs:
            runner_url = self.pending_jobs[job_id].get("runner_url")
        
        if not runner_url:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        try:
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.delete(
                    f"{runner_url}/v1/jobs/{job_id}",
                    headers={"X-API-Key": "dev-key-12345"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 404:
                        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
                    
                    result = await resp.json()
        
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=503, detail=f"Failed to connect to sandbox runner: {e}")
        
        # cleanup
        self.router.release_job(job_id)
        if job_id in self.pending_jobs:
            del self.pending_jobs[job_id]
        
        self.stats.record_job_failed()
        
        return result


# global instance
lb: Optional[LoadBalancer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global lb
    
    logger.info("Starting Sandbox Load Balancer...")
    await lb.start()
    
    yield
    
    logger.info("Shutting down Sandbox Load Balancer...")
    await lb.stop()


def create_app(config: Config) -> FastAPI:
    """Create FastAPI application"""
    global lb
    
    lb = LoadBalancer(config)
    
    app = FastAPI(
        title="Sandbox Runner Load Balancer",
        description="Routes validator requests to multiple sandbox runners with caching",
        version="1.0.0",
        lifespan=lifespan
    )
    
    async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
        if not lb.api_keys.validate(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return x_api_key
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint (no auth required)"""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runners": {
                url: status["healthy"]
                for url, status in lb.health_manager.health_status.items()
            }
        }
    
    @app.get("/v1/health")
    async def health_check_v1():
        """Health check endpoint v1 (no auth required)"""
        return await health_check()
    
    @app.post("/v1/jobs/submit", dependencies=[Depends(verify_api_key)])
    async def submit_job(request: JobSubmitRequest, api_key: str = Depends(verify_api_key)):
        """Submit a job for execution"""
        return await lb.submit_job(request, api_key)
    
    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
    async def get_job_status(job_id: str):
        """Get job status"""
        return await lb.get_job_status(job_id)
    
    @app.get("/v1/jobs/{job_id}/metrics", dependencies=[Depends(verify_api_key)])
    async def get_job_metrics(job_id: str):
        """Get job metrics"""
        return await lb.get_job_metrics(job_id)
    
    @app.delete("/v1/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
    async def cancel_job(job_id: str):
        """Cancel a job"""
        return await lb.cancel_job(job_id)
    
    @app.get("/v1/status", dependencies=[Depends(verify_api_key)])
    async def get_status():
        """Get load balancer status"""
        return {
            "status": "operational",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runners": lb.health_manager.health_status,
            "routing": lb.router.get_stats(),
            "cache": lb.cache.get_stats(),
            "stats": lb.stats.get_stats()
        }
    
    @app.get("/v1/stats", dependencies=[Depends(verify_api_key)])
    async def get_stats():
        """Get detailed statistics"""
        return {
            "load_balancer": lb.stats.get_stats(),
            "cache": lb.cache.get_stats(),
            "routing": lb.router.get_stats(),
            "runners": {
                url: {
                    **status,
                    "last_check_ago": f"{time.time() - status['last_check']:.1f}s" if status['last_check'] else "never"
                }
                for url, status in lb.health_manager.health_status.items()
            }
        }
    
    @app.get("/v1/cache/entries", dependencies=[Depends(verify_api_key)])
    async def list_cache_entries(limit: int = 100):
        """List cached entries"""
        entries = []
        for key, entry in list(lb.cache.memory_cache.items())[:limit]:
            entries.append({
                "cache_key": key[:16] + "...",
                "repo_url": entry["repo_url"],
                "repo_branch": entry["repo_branch"],
                "repo_commit": entry["repo_commit"][:8] + "...",
                "weight_class": entry["weight_class"],
                "miner_hotkey": entry["miner_hotkey"][:12] + "...",
                "cached_at": datetime.fromtimestamp(entry["cached_at"]).isoformat(),
                "expires_at": datetime.fromtimestamp(entry["expires_at"]).isoformat()
            })
        
        return {
            "count": len(entries),
            "total": len(lb.cache.memory_cache),
            "entries": entries
        }
    
    return app


def main():
    parser = argparse.ArgumentParser(
        description="Sandbox Runner Load Balancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python load_balancer.py --runners https://runner1.com https://runner2.com --port 8080
    
    pm2 start load_balancer.py --interpreter python3 --name sandbox-lb -- \\
        --runners https://runner1.com https://runner2.com --port 8080
        """
    )
    
    parser.add_argument(
        "--runners",
        nargs="+",
        required=True,
        help="Sandbox runner URLs (space-separated)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./lb_cache",
        help="Directory for cache storage (default: ./lb_cache)"
    )
    
    parser.add_argument(
        "--cache-ttl-days",
        type=int,
        default=7,
        help="Cache TTL in days (default: 7)"
    )
    
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=30,
        help="Health check interval in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=args.log_level
    )
    
    logger.info(f"Starting Sandbox Load Balancer")
    logger.info(f"  Runners: {args.runners}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Cache dir: {args.cache_dir}")
    logger.info(f"  Cache TTL: {args.cache_ttl_days} days")
    
    config = Config(
        runner_urls=args.runners,
        port=args.port,
        cache_dir=args.cache_dir,
        cache_ttl_days=args.cache_ttl_days,
        health_check_interval=args.health_check_interval
    )
    
    app = create_app(config)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level=args.log_level.lower()
    )


if __name__ == "__main__":
    main()