"""
GPU Pool Manager Module

Manages the allocation and tracking of GPU resources across multiple jobs
Handles:
- Tracking which GPUs are free/allocated
- Weight class allocation (1x, 2x, 4x, 8x)
- Contiguous GPU block allocation
- Real-time GPU utilization monitoring
- Thread-safe async operations
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class WeightClass(str, Enum):
    """GPU weight class for job execution"""
    ONE_GPU = "1xH200"
    TWO_GPU = "2xH200"
    FOUR_GPU = "4xH200"
    EIGHT_GPU = "8xH200"
    
    def gpu_count(self) -> int:
        """Return number of GPUs required for this weight class"""
        return {
            "1xH200": 1,
            "2xH200": 2,
            "4xH200": 4,
            "8xH200": 8,
        }[self.value]


class GPUStatus(str, Enum):
    """GPU allocation status"""
    FREE = "free"
    ALLOCATED = "allocated"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class GPUInfo:
    """Information about a single GPU"""
    gpu_id: int
    status: GPUStatus
    allocated_to_job: Optional[str] = None
    utilization_percent: float = 0.0
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    temperature_celsius: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class GPUPoolManager:
    """
    Manages GPU resource allocation across jobs
    
    Key features:
    - Tracks 8 H200 GPUs (configurable)
    - Supports weight classes (1x, 2x, 4x, 8x)
    - Prefers contiguous GPU allocation for multi-GPU jobs
    - Thread-safe with async locks
    - Real-time utilization monitoring
    """
    
    def __init__(self, gpu_count: int = 8):
        """
        Initialize GPU pool manager
        
        Args:
            gpu_count: Total number of GPUs to manage (default: 8)
        """
        self.gpu_count = gpu_count
        self._lock = asyncio.Lock()
        
        self._gpus: Dict[int, GPUInfo] = {}
        for i in range(gpu_count):
            self._gpus[i] = GPUInfo(
                gpu_id=i,
                status=GPUStatus.FREE,
                memory_total_mb=81920  # H200 = 80GB = 81920MB
            )
        
        self._job_allocations: Dict[str, List[int]] = {}        
        self._allocation_history: List[Dict] = []
        
        logger.info(f"GPU Pool Manager initialized with {gpu_count} GPUs")
    
    async def allocate_gpus(
        self,
        weight_class: WeightClass,
        job_id: str
    ) -> Optional[List[int]]:
        """
        Allocate GPUs for a job based on weight class
        
        Allocation strategy:
        1. Determine number of GPUs needed from weight class
        2. Try to find contiguous block of free GPUs (optimal for performance)
        3. If contiguous not possible, allocate any available GPUs
        4. Mark GPUs as allocated and track job mapping
        
        Args:
            weight_class: GPU weight class (1x, 2x, 4x, 8x)
            job_id: Unique job identifier
            
        Returns:
            List of allocated GPU IDs, or None if allocation failed
        """
        async with self._lock:
            required_gpus = weight_class.gpu_count()
            
            available_count = sum(
                1 for gpu in self._gpus.values()
                if gpu.status == GPUStatus.FREE
            )
            
            if available_count < required_gpus:
                logger.warning(
                    f"Insufficient GPUs for job {job_id}: "
                    f"need {required_gpus}, have {available_count}"
                )
                return None
            
            allocated_gpus = self._find_contiguous_gpus(required_gpus)
            
            if not allocated_gpus:
                allocated_gpus = self._find_any_free_gpus(required_gpus)
            
            if not allocated_gpus:
                logger.error(
                    f"Failed to allocate {required_gpus} GPUs for job {job_id}"
                )
                return None
            
            for gpu_id in allocated_gpus:
                self._gpus[gpu_id].status = GPUStatus.ALLOCATED
                self._gpus[gpu_id].allocated_to_job = job_id
                self._gpus[gpu_id].last_updated = datetime.utcnow()
            
            self._job_allocations[job_id] = allocated_gpus
            
            self._allocation_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "job_id": job_id,
                "weight_class": weight_class.value,
                "gpus": allocated_gpus,
                "action": "allocate"
            })
            
            logger.info(
                f"Allocated GPUs {allocated_gpus} to job {job_id} "
                f"({weight_class.value})"
            )
            
            return allocated_gpus
    
    async def release_gpus(self, job_id: str) -> None:
        """
        Release GPUs allocated to a job
        
        Args:
            job_id: Job identifier whose GPUs should be released
        """
        async with self._lock:
            if job_id not in self._job_allocations:
                logger.warning(f"No GPU allocation found for job {job_id}")
                return
            
            allocated_gpus = self._job_allocations[job_id]
            
            for gpu_id in allocated_gpus:
                if gpu_id in self._gpus:
                    self._gpus[gpu_id].status = GPUStatus.FREE
                    self._gpus[gpu_id].allocated_to_job = None
                    self._gpus[gpu_id].last_updated = datetime.utcnow()
            
            del self._job_allocations[job_id]
            
            self._allocation_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "job_id": job_id,
                "gpus": allocated_gpus,
                "action": "release"
            })
            
            logger.info(f"Released GPUs {allocated_gpus} from job {job_id}")
    
    async def get_available_gpu_count(self) -> int:
        """
        Get count of currently available (free) GPUs
        
        Returns:
            Number of free GPUs
        """
        async with self._lock:
            return sum(
                1 for gpu in self._gpus.values()
                if gpu.status == GPUStatus.FREE
            )
    
    async def get_gpu_status(self) -> Dict[int, GPUInfo]:
        """
        Get status of all GPUs
        
        Returns:
            Dictionary mapping GPU ID to GPUInfo
        """
        async with self._lock:
            return {
                gpu_id: GPUInfo(
                    gpu_id=info.gpu_id,
                    status=info.status,
                    allocated_to_job=info.allocated_to_job,
                    utilization_percent=info.utilization_percent,
                    memory_used_mb=info.memory_used_mb,
                    memory_total_mb=info.memory_total_mb,
                    temperature_celsius=info.temperature_celsius,
                    last_updated=info.last_updated
                )
                for gpu_id, info in self._gpus.items()
            }
    
    async def can_allocate(self, weight_class: WeightClass) -> bool:
        """
        Check if GPUs can be allocated for a weight class
        
        Args:
            weight_class: GPU weight class to check
            
        Returns:
            True if allocation is possible, False otherwise
        """
        required_gpus = weight_class.gpu_count()
        available = await self.get_available_gpu_count()
        return available >= required_gpus
    
    async def get_job_gpus(self, job_id: str) -> Optional[List[int]]:
        """
        Get GPU IDs allocated to a specific job
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of GPU IDs, or None if job not found
        """
        async with self._lock:
            return self._job_allocations.get(job_id)
    
    async def update_gpu_utilization(
        self,
        gpu_id: int,
        utilization_percent: float,
        memory_used_mb: int,
        temperature_celsius: float = 0.0
    ):
        """
        Update real-time GPU utilization metrics
        
        This should be called periodically by a monitoring task
        to keep GPU stats up to date.
        
        Args:
            gpu_id: GPU identifier
            utilization_percent: GPU utilization (0-100)
            memory_used_mb: GPU memory used in MB
            temperature_celsius: GPU temperature in Celsius
        """
        async with self._lock:
            if gpu_id in self._gpus:
                self._gpus[gpu_id].utilization_percent = utilization_percent
                self._gpus[gpu_id].memory_used_mb = memory_used_mb
                self._gpus[gpu_id].temperature_celsius = temperature_celsius
                self._gpus[gpu_id].last_updated = datetime.utcnow()
    
    async def get_allocation_stats(self) -> Dict:
        """
        Get allocation statistics for monitoring
        
        Returns:
            Dictionary with allocation statistics
        """
        async with self._lock:
            free_count = sum(
                1 for gpu in self._gpus.values()
                if gpu.status == GPUStatus.FREE
            )
            allocated_count = sum(
                1 for gpu in self._gpus.values()
                if gpu.status == GPUStatus.ALLOCATED
            )
            
            return {
                "total_gpus": self.gpu_count,
                "free_gpus": free_count,
                "allocated_gpus": allocated_count,
                "utilization_percent": (allocated_count / self.gpu_count * 100),
                "active_jobs": len(self._job_allocations),
                "job_allocations": dict(self._job_allocations)
            }
    
    def _find_contiguous_gpus(self, count: int) -> Optional[List[int]]:
        """
        Find a contiguous block of free GPUs
        
        Contiguous allocation is preferred for multi-GPU jobs because:
        1. Better NVLink connectivity between adjacent GPUs
        2. Simplified GPU topology
        3. Better performance for distributed training/inference
        
        Args:
            count: Number of contiguous GPUs needed
            
        Returns:
            List of contiguous GPU IDs, or None if not available
        """
        for start_id in range(self.gpu_count - count + 1):
            gpus = list(range(start_id, start_id + count))
            if all(
                self._gpus[gpu_id].status == GPUStatus.FREE
                for gpu_id in gpus
            ):
                return gpus
        
        return None
    
    def _find_any_free_gpus(self, count: int) -> Optional[List[int]]:
        """
        Find any available free GPUs (not necessarily contiguous)
        
        Args:
            count: Number of GPUs needed
            
        Returns:
            List of GPU IDs, or None if not enough available
        """
        free_gpus = [
            gpu_id for gpu_id, gpu in self._gpus.items()
            if gpu.status == GPUStatus.FREE
        ]
        
        if len(free_gpus) >= count:
            return free_gpus[:count]
        
        return None
    
    async def get_cuda_visible_devices(self, job_id: str) -> Optional[str]:
        """
        Get CUDA_VISIBLE_DEVICES environment variable value for a job
        
        This is used to restrict a job to only see its allocated GPUs
        
        Args:
            job_id: Job identifier
            
        Returns:
            Comma-separated GPU IDs (e.g., "0,1,2,3"), or None if not found
        """
        gpus = await self.get_job_gpus(job_id)
        if gpus:
            return ",".join(str(gpu_id) for gpu_id in gpus)
        return None
    
    async def mark_gpu_error(self, gpu_id: int, error_message: str):
        """
        Mark a GPU as in error state
        
        Args:
            gpu_id: GPU identifier
            error_message: Error description
        """
        async with self._lock:
            if gpu_id in self._gpus:
                self._gpus[gpu_id].status = GPUStatus.ERROR
                self._gpus[gpu_id].last_updated = datetime.utcnow()
                logger.error(f"GPU {gpu_id} marked as error: {error_message}")
    
    async def reset_gpu(self, gpu_id: int):
        """
        Reset a GPU to free state (for error recovery)
        
        Args:
            gpu_id: GPU identifier
        """
        async with self._lock:
            if gpu_id in self._gpus:
                self._gpus[gpu_id].status = GPUStatus.FREE
                self._gpus[gpu_id].allocated_to_job = None
                self._gpus[gpu_id].last_updated = datetime.utcnow()
                logger.info(f"GPU {gpu_id} reset to free state")