from __future__ import annotations
import os
import time
import uuid
import logging
from typing import Optional, Tuple
import docker
from docker.models.containers import Container
from .config import settings


log = logging.getLogger(__name__)

def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def start_miner_container(
    image: str,
    dataset_dir: str,
    out_dir: str,
    seccomp_path: Optional[str],
    ) -> Container:
        
        client = docker.from_env()
        if settings.pull_always:
            log.info({"event": "docker_pull", "image": image})
            client.images.pull(image)
        
        # security & resource limits
        mem_limit = settings.child_memory
        nano_cpus = int(settings.child_cpus * 1e9)
        pids_limit = settings.child_pids
        
        # read-only root FS and tmpfs for /tmp
        host_config = client.api.create_host_config(
            network_mode="none", # absolutely no internet/egress
            binds={
            os.path.abspath(dataset_dir): {"bind": "/data", "mode": "ro"},
            os.path.abspath(out_dir): {"bind": "/out", "mode": "rw"},
            },
            read_only=True,
            tmpfs={"/tmp": "rw,noexec,nosuid,nodev,size=64m"},
            mem_limit=mem_limit,
            nano_cpus=nano_cpus,
            pids_limit=pids_limit,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            seccomp_profile=seccomp_path if settings.secure_mode and seccomp_path else None,
            auto_remove=False,
            )
        container = client.api.create_container(
            image=image,
            name=f"miner-{uuid.uuid4().hex[:12]}",
            host_config=host_config,
            environment={
            "ARC_DATA_DIR": "/data",
            "ARC_OUT_DIR": "/out",
            },
            tty=False,
            stdin_open=False,
            )

        cid = container.get("Id")
        client.api.start(cid)
        return client.containers.get(cid)

def _curl_in_namespace(miner: Container, method: str, path: str, timeout: int = 10, payload: Optional[str] = None) -> Tuple[int, str]:
    """
    Launch a short‑lived curl helper container **sharing the miner's network namespace**
    to call 127.0.0.1:8085/<path>. This preserves the miner's "network none" isolation
    while letting us hit its localhost HTTP API
    """
    client = docker.from_env()
    args = [
        "--silent",
        "--show-error",
        "--max-time", str(timeout),
        "--retry", "0",
        "--fail-early",
        "--write-out", "\n%{http_code}",
        ]
    url = f"http://127.0.0.1:8085/{path.lstrip('/')}"
    if method.upper() == "POST":
        args.extend(["-X", "POST", "-H", "Content-Type: application/json", "--data", payload or "{}"])
    else:
        args.extend(["-X", "GET"])
    
    args.append(url)
    image = "curlimages/curl:8.10.1"
    
    try:
        client.images.pull(image)
    except Exception:
        pass

    res = client.api.create_container(
        image=image,
        name=f"curl-{uuid.uuid4().hex[:10]}",
        host_config=client.api.create_host_config(network_mode=f"container:{miner.id}"),
        command=args,
        )
    cid = res.get("Id")
    client.api.start(cid)
    exit_status = client.api.wait(cid)
    logs = client.api.logs(cid).decode("utf-8", errors="ignore")

    try:
        client.api.remove_container(cid, force=True)
    except Exception:
        pass

    if "\n" in logs:
        body, status_line = logs.rsplit("\n", 1)
        try:
            code = int(status_line.strip())
        except ValueError:
            code = int(exit_status.get("StatusCode", 1))
            body = logs
    else:
        code = int(exit_status.get("StatusCode", 1))
        body = logs

    return code, body


def drive_miner_lifecycle(
    image: str,
    dataset_dir: str,
    out_dir: str,
    seccomp_path: Optional[str],
    timeout_min: int,
    ) -> Tuple[str, Optional[dict]]:
    """
    Start container, POST /start-prediction, poll GET /status until DONE/ERROR/timeout.
    Returns (final_state, optional_error_info)
    """
    miner = start_miner_container(image, dataset_dir, out_dir, seccomp_path)
    try:
        code, body = _curl_in_namespace(miner, "POST", "/start-prediction")
        if code >= 400:
            return "ERROR", {"stage": "start", "status": code, "body": body}

        deadline = time.time() + timeout_min * 60
        last = None
        while time.time() < deadline:
            code, body = _curl_in_namespace(miner, "GET", "/status")
            last = (code, body)
            if code == 200 and "DONE" in body:
                return "DONE", None
            if code >= 400 or "ERROR" in body:
                return "ERROR", {"stage": "status", "status": code, "body": body}
            time.sleep(3)
        return "TIMEOUT", {"stage": "poll", "last": last}
    finally:
        try:
            miner.reload()
            logs = miner.logs(tail=200).decode("utf-8", errors="ignore")
        except Exception:
            logs = ""
        try:
            miner.remove(force=True)
        except Exception:
            pass
        log.info({"event": "miner_finished", "container": miner.id, "logs_tail": logs[-1000:]})