"""
API Routes Module
"""

from typing import Optional, List
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field, HttpUrl, validator
import logging

from config import Config
from api.auth import AuthenticationManager, RateLimiter

logger = logging.getLogger("api.routes")


# ============================================================================
# Request/Response Models
# ============================================================================

class WeightClass(str, Enum):
    """GPU weight class for job execution."""
    ONE_GPU = "1xH200"
    TWO_GPU = "2xH200"
    FOUR_GPU = "4xH200"
    EIGHT_GPU = "8xH200"


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    CLONING = "cloning"
    BUILDING = "building"
    PREP = "prep"
    INFERENCE = "inference"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class JobSubmitRequest(BaseModel):
    """Request body for job submission."""
    
    repo_url: HttpUrl = Field(
        ...,
        description="GitHub/GitLab repository URL",
        example="https://github.com/user/miner-repo"
    )
    
    repo_branch: str = Field(
        default="main",
        description="Git branch",
        example="main"
    )
    
    repo_commit: Optional[str] = Field(
        None,
        description="Specific commit hash (optional)",
        example="abc123"
    )

    repo_path: str = Field(
        default="",
        description="Subdirectory path within repo (e.g., 'miner', 'src/solver')",
        example="miner"
    )
    
    weight_class: WeightClass = Field(
        ...,
        description="GPU weight class",
        example="1xH200"
    )
    
    input_data_s3_path: str = Field(
        ...,
        description="S3 path to input data",
        example="s3://bucket/inputs/dataset.json"
    )
    
    output_data_s3_path: str = Field(
        ...,
        description="S3 path for results",
        example="s3://bucket/outputs/results.json"
    )
    
    priority: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Job priority (0-10)",
        example=5
    )
    
    validator_hotkey: Optional[str] = Field(
        None,
        description="Validator's hotkey",
        example="5Abc...xyz"
    )
    
    miner_hotkey: str = Field(
        ...,
        description="Miner's hotkey",
        example="5Def...uvw"
    )
    
    custom_env_vars: Optional[dict] = Field(
        default_factory=dict,
        description="Custom environment variables"
    )
    
    @validator('repo_url')
    def validate_repo_url(cls, v):
        """Validate repository URL."""
        allowed_hosts = ['github.com', 'gitlab.com']
        if not any(host in str(v) for host in allowed_hosts):
            raise ValueError(f"Repository must be from: {', '.join(allowed_hosts)}")
        return v


class JobSubmitResponse(BaseModel):
    """Response after job submission."""
    job_id: str
    status: JobStatus
    estimated_start_time: Optional[datetime] = None
    queue_position: int


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: JobStatus
    weight_class: WeightClass
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_phase: Optional[str] = None
    progress_percentage: Optional[float] = Field(None, ge=0, le=100)
    assigned_gpus: Optional[List[int]] = None
    error_message: Optional[str] = None
    input_data_s3_path: str
    output_data_s3_path: str
    validator_hotkey: Optional[str] = None
    miner_hotkey: str
    priority: int


class RunnerStatus(BaseModel):
    """Overall runner status."""
    runner_id: str
    status: str
    total_gpus: int
    available_gpus: int
    allocated_gpus: int
    queue_depth: int
    active_jobs: int
    total_submitted: int
    total_completed: int
    total_failed: int
    execution_mode: str


async def get_auth_manager(request: Request) -> AuthenticationManager:
    """Get authentication manager from app state."""
    return request.app.state.auth_manager


async def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter from app state."""
    return request.app.state.rate_limiter


async def get_config(request: Request) -> Config:
    """Get configuration from app state."""
    return request.app.state.config


async def get_meta_manager(request: Request):
    """Get meta-manager from app state."""
    return request.app.state.meta_manager


async def authenticate_request(
    request: Request,
    auth_manager: AuthenticationManager = Depends(get_auth_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
) -> tuple[str, str]:
    """Authenticate request and check rate limits."""
    validator_id = None
    auth_method = "none"
    
    if auth_manager.config.require_epistula:
        validator_id = await auth_manager.verify_epistula_signature(request)
        auth_method = "epistula"
    
    if auth_manager.config.require_api_key:
        api_key = await auth_manager.verify_api_key()
        if auth_method == "epistula":
            auth_method = "dual"
        else:
            validator_id = api_key
            auth_method = "api_key"
    
    if validator_id:
        await rate_limiter.check_rate_limit(validator_id)
    
    return validator_id, auth_method


def create_router(config: Config) -> APIRouter:
    """Create and configure the API router."""
    router = APIRouter()
    
    @router.post(
        "/jobs/submit",
        response_model=JobSubmitResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Submit a new job"
    )
    async def submit_job(
        request: JobSubmitRequest,
        auth: tuple = Depends(authenticate_request),
        meta_manager = Depends(get_meta_manager)
    ):
        """Submit a new job for execution."""
        validator_id, auth_method = auth
        
        logger.info(
            f"Job submission from {validator_id}",
            extra={
                "validator_id": validator_id,
                "miner_hotkey": request.miner_hotkey,
                "weight_class": request.weight_class
            }
        )
        
        job_request = {
            "repo_url": str(request.repo_url),
            "repo_branch": request.repo_branch,
            "repo_commit": request.repo_commit,
            "repo_path": request.repo_path,
            "weight_class": request.weight_class.value,
            "input_data_s3_path": request.input_data_s3_path,
            "output_data_s3_path": request.output_data_s3_path,
            "priority": request.priority,
            "validator_hotkey": request.validator_hotkey or validator_id,
            "miner_hotkey": request.miner_hotkey,
            "custom_env_vars": request.custom_env_vars
        }
        
        response = await meta_manager.submit_job(job_request)
        
        return JobSubmitResponse(
            job_id=response["job_id"],
            status=JobStatus(response["status"]),
            estimated_start_time=datetime.fromisoformat(response["estimated_start_time"]) if response.get("estimated_start_time") else None,
            queue_position=response["queue_position"]
        )
    
    @router.get(
        "/jobs/{job_id}",
        response_model=JobStatusResponse,
        summary="Get job status"
    )
    async def get_job_status(
        job_id: str,
        auth: tuple = Depends(authenticate_request),
        meta_manager = Depends(get_meta_manager)
    ):
        """Get detailed status of a job."""
        validator_id, _ = auth
        
        logger.info(f"Job status query: {job_id} from {validator_id}")
        
        job_data = await meta_manager.get_job_status(job_id)
        
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}"
            )
        
        return JobStatusResponse(
            job_id=job_data["job_id"],
            status=JobStatus(job_data["status"]),
            weight_class=WeightClass(job_data["weight_class"]),
            submitted_at=datetime.fromisoformat(job_data["submitted_at"]),
            started_at=datetime.fromisoformat(job_data["started_at"]) if job_data.get("started_at") else None,
            completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
            current_phase=job_data.get("current_phase"),
            progress_percentage=job_data.get("progress_percentage"),
            assigned_gpus=job_data.get("assigned_gpus"),
            error_message=job_data.get("error_message"),
            input_data_s3_path=job_data["input_s3_path"],
            output_data_s3_path=job_data["output_s3_path"],
            validator_hotkey=job_data.get("validator_hotkey"),
            miner_hotkey=job_data["miner_hotkey"],
            priority=job_data["priority"]
        )
    
    @router.delete(
        "/jobs/{job_id}",
        summary="Cancel a job"
    )
    async def cancel_job(
        job_id: str,
        auth: tuple = Depends(authenticate_request),
        meta_manager = Depends(get_meta_manager)
    ):
        """Cancel a pending or running job."""
        validator_id, _ = auth
        
        logger.info(f"Job cancellation: {job_id} from {validator_id}")
        
        cancelled = await meta_manager.cancel_job(job_id)
        
        if not cancelled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}"
            )
        
        return {"job_id": job_id, "status": "cancelled"}
    
    @router.get(
        "/status",
        response_model=RunnerStatus,
        summary="Get runner status"
    )
    async def get_runner_status(
        auth: tuple = Depends(authenticate_request),
        meta_manager = Depends(get_meta_manager)
    ):
        """Get overall runner status."""
        status_data = await meta_manager.get_runner_status()
        
        gpu_stats = status_data["gpu_stats"]
        queue_stats = status_data["queue_stats"]
        
        return RunnerStatus(
            runner_id=status_data["runner_id"],
            status=status_data["status"],
            total_gpus=gpu_stats["total_gpus"],
            available_gpus=gpu_stats["free_gpus"],
            allocated_gpus=gpu_stats["allocated_gpus"],
            queue_depth=queue_stats["total_jobs"],
            active_jobs=status_data["active_jobs"],
            total_submitted=status_data["total_submitted"],
            total_completed=status_data["total_completed"],
            total_failed=status_data["total_failed"],
            execution_mode=status_data["execution_mode"]
        )
    
    @router.get(
        "/health",
        summary="Health check"
    )
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return router