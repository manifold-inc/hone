from __future__ import annotations
from fastapi import APIRouter, Request
from typing import Dict, Any, Optional
import os
from loguru import logger

router = APIRouter()

@router.get("/info")
async def get_miner_info(request: Request) -> Dict[str, Any]:
    """
    Provide miner configuration information to validators
    
    This endpoint allows validators to discover:
    - Miner's repository URL for sandbox execution
    - Required GPU resources (weight class)
    - Optional configurations (vLLM, custom env vars)
    
    Returns:
        {
            "repo_url": "https://github.com/user/miner-repo",
            "repo_branch": "main",
            "repo_commit": "abc123" (optional),
            "repo_path": "miner" (optional subdirectory),
            "weight_class": "1xH200" | "2xH200" | "4xH200" | "8xH200",
            "use_vllm": false,
            "vllm_config": {...} (optional),
            "custom_env_vars": {...} (optional),
            "version": "1.0.0",
            "hotkey": "5Abc...xyz"
        }
    """
    
    # read from environment variables or config

    use_vllm = os.getenv("MINER_USE_VLLM", "true").lower() == "true"
    
    miner_info = {
        "repo_url": os.getenv("MINER_REPO_URL", "https://github.com/manifold-inc/hone"),
        "repo_branch": os.getenv("MINER_REPO_BRANCH", "main"),
        "repo_commit": os.getenv("MINER_REPO_COMMIT"),  # optional - None means use latest
        "repo_path": os.getenv("MINER_REPO_PATH", "miner-solution-example"),  # optional subdirectory within repo
        "weight_class": os.getenv("MINER_WEIGHT_CLASS", "1xH200"),  # how many GPUs needed, use 1xH200 up to 8xH200
        "use_vllm": use_vllm,
        "vllm_config": _get_vllm_config() if use_vllm else None,
        "custom_env_vars": _get_custom_env_vars(),
        "version": os.getenv("MINER_VERSION", "1.0.0"),
        "hotkey": request.app.state.cfg.hotkey if hasattr(request.app.state, 'cfg') else None
    }
    
    # remove None values for cleaner response
    miner_info = {k: v for k, v in miner_info.items() if v is not None}
    
    logger.debug(f"Info endpoint called, returning: {miner_info}")
    
    return miner_info


def _get_vllm_config() -> Optional[Dict[str, Any]]:
    """Parse vLLM configuration from environment"""
    model = os.getenv("VLLM_MODEL")
    if not model:
        return None
    
    return {
        "model": model,
        "dtype": os.getenv("VLLM_DTYPE", "half"),
        "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTIL", "0.8")),
        "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "12000"))
    }


def _get_custom_env_vars() -> Dict[str, str]:
    """Parse custom environment variables for sandbox execution"""
    # miners can specify additional env vars needed for their solution
    # format: MINER_ENV_VAR1=value1,MINER_ENV_VAR2=value2
    env_vars = {}
    
    custom_env_str = os.getenv("MINER_CUSTOM_ENV_VARS", "")
    if custom_env_str:
        for pair in custom_env_str.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                env_vars[key.strip()] = value.strip()
    
    return env_vars