from __future__ import annotations
import asyncio
import os
import shutil
import uuid
import logging
from typing import Dict, Any, Optional


from .config import settings
from .models import TaskState
from .docker_runner import drive_miner_lifecycle
from .evaluator import evaluate
from .s3_utils import download_to_dir, upload_dir_to_s3


log = logging.getLogger(__name__)


class TaskRecord:
    def __init__(self, task_id: str, work_dir: str) -> None:
        self.task_id = task_id
        self.work_dir = work_dir
        self.state: TaskState = TaskState.QUEUED
        self.message: str = "queued"
        self.metrics: Optional[Dict[str, Any]] = None
        self.outputs_s3_uri: Optional[str] = None
        self.logs_tail: Optional[str] = None


class TaskManager:
    def __init__(self) -> None:
        self._tasks: Dict[str, TaskRecord] = {}
        self._sem = asyncio.Semaphore(settings.max_concurrency)
        os.makedirs(settings.data_root, exist_ok=True)
        os.makedirs(settings.out_root, exist_ok=True)
        os.makedirs(settings.tmp_root, exist_ok=True)


    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)
    
    async def submit(self, *, miner_uid: str, s3_dataset_url: str, docker_image: str) -> TaskRecord:
        task_id = uuid.uuid4().hex
        work_root = os.path.join(settings.tmp_root, task_id)
        data_dir = os.path.join(work_root, "data")
        out_dir = os.path.join(work_root, "out")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)


        rec = TaskRecord(task_id, work_root)
        self._tasks[task_id] = rec

        async def _run():
            async with self._sem:
                try:
                    rec.state = TaskState.RUNNING
                    rec.message = "downloading dataset"
                    await asyncio.to_thread(download_to_dir, s3_dataset_url, data_dir, settings.aws_region)


                    rec.message = "executing miner"
                    final_state, err = await asyncio.to_thread(
                    drive_miner_lifecycle,
                    docker_image,
                    data_dir,
                    out_dir,
                    os.path.join(os.path.dirname(__file__), "seccomp-restrictive.json") if settings.secure_mode else None,
                    settings.child_timeout_min,
                    )
                    if final_state == "DONE":
                        rec.message = "evaluating"
                        metrics = await asyncio.to_thread(evaluate, data_dir, out_dir)
                        rec.metrics = metrics
                        rec.state = TaskState.DONE
                        rec.message = "uploading results"
                        if settings.s3_metrics_bucket:
                            prefix = f"s3://{settings.s3_metrics_bucket}/arcagi2/results/{task_id}"
                            rec.outputs_s3_uri = await asyncio.to_thread(upload_dir_to_s3, out_dir, prefix, settings.aws_region)
                        rec.message = "complete"
                    elif final_state == "TIMEOUT":
                        rec.state = TaskState.TIMEOUT
                        rec.message = f"timeout after {settings.child_timeout_min} min"
                    else:
                        rec.state = TaskState.ERROR
                        rec.message = f"miner error: {err}"
                except Exception as e:
                    rec.state = TaskState.ERROR
                    rec.message = f"orchestrator error: {e}"
                finally:
                    try:
                        shutil.rmtree(work_root, ignore_errors=True)
                    except Exception:
                        pass

        asyncio.create_task(_run())
        return rec