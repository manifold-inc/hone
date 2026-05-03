# The MIT License (MIT)
# © 2025 hone.training

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# type: ignore
import asyncio
import concurrent.futures
import json
import math
import os
import random
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from functools import partial
from types import SimpleNamespace
from typing import Any, Literal, cast

import aiofiles
import bittensor as bt
import boto3
import botocore
import torch
from aiobotocore.client import AioBaseClient
from aiobotocore.session import get_session
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, ConnectionClosedError
from tqdm import tqdm as std_tqdm

import hone
from hone.chain import ChainManager
from hone.compress import TopKCompressor, unpack_12bit_indices
from hone.config import BUCKET_SECRETS, client_config
from hone.schemas import Bucket, CommsGetResult

# Constants
CF_REGION_NAME: str = "enam"
LOCAL_TMP_DIR = "/tmp/local_store"
PEERS_FILE_PREFIX = "peers_"
CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))

SAFE_BUCKET_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
# Allow path-like keys. Tighten if you need to.
SAFE_KEY_RE = re.compile(r"^[a-zA-Z0-9._/-]+$")


class Comms(ChainManager):
    """
    The Comms class handles all communication and data transfer operations for the templar network.
    It is responsible for managing S3 client connections, handling file uploads and downloads,
    and managing data for gradients, checkpoints, and other operational data.

    This class builds upon the ChainManager to interact with the blockchain while providing
    specialized methods for efficient data handling in a distributed environment.
    """

    def __init__(
        self,
        wallet: bt.Wallet | None,
        save_location: str = "/tmp",
        key_prefix: str = "model",
        config=None,
        hparams=None,
        uid: int | None = None,
        **kwargs,
    ):
        """
        Initializes the Comms object, setting up necessary configurations, directories,
        and background tasks for communication.

        Args:
            wallet (bt.Wallet | None): The bittensor wallet instance.
            save_location (str, optional): The base directory for saving local files. Defaults to "/tmp".
            key_prefix (str, optional): A prefix for keys used in storage. Defaults to "model".
            config (object, optional): Configuration object. Defaults to None.
            hparams (object, optional): Hyperparameters object. Defaults to None.
            uid (int | None, optional): The UID of the neuron. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.uid = uid
        self.wallet = wallet

        # Create temp directory for this instance
        self.temp_dir = os.path.join("/tmp", f"templar_{self.uid}")
        os.makedirs(self.temp_dir, exist_ok=True)
        # Get the bucket directly
        self.bucket = self.get_own_bucket("gradients", "write")
        # Now initialize ChainManager with the bucket
        super().__init__(
            config=config,
            hparams=hparams,
            wallet=self.wallet,
            bucket=self.bucket,
        )

        # Use the hotkey directly in the save_location
        if self.wallet is not None:
            hotkey = self.wallet.hotkey.ss58_address
            self.save_location = os.path.join("/tmp", f"hotkey_{hotkey}")
            os.makedirs(self.save_location, exist_ok=True)
        self.key_prefix = key_prefix

        ## a single aiobotocore session and a dictionary of clients
        self.session = get_session()
        self._s3_clients: dict[
            tuple[str, str, str], AioBaseClient
        ] = {}  # (acc_key, sec_key, account_id) -> s3_client

        self.lock = asyncio.Lock()
        self.active_peers = set()  # Set to store active peers
        self.active_check_interval = (
            self.hparams.active_check_interval
        )  # Interval in seconds
        self.recent_windows = (
            self.hparams.recent_windows
        )  # Number of recent windows to check
        self.peers: list[int] = []
        self.reserve_peers: list[int] = []

        self.client_semaphore = asyncio.Semaphore(CPU_MAX_CONNECTIONS // 2)
        self.gather_semaphore = asyncio.Semaphore(CPU_MAX_CONNECTIONS // 2)
        # Limit how many TransferManagers run concurrently (protects threads/conn pool)
        self.upload_sem = asyncio.Semaphore(4)

        # P3: K_quorum + adaptive grace window state. ``_ema_step_seconds``
        # tracks the full outer-step wall-clock (validator feeds back via
        # ``record_step_time``); ``_ema_quorum_seconds`` is measured in the
        # gather loops below; ``_ema_sync_seconds`` tracks post-gather
        # outer_step apply+merge (validator feeds back via
        # ``record_sync_time``). Bootstrap: ``_ema_step_seconds`` seeds to
        # ``blocks_per_window * 12s`` until the first real sample lands so
        # ``_compute_grace_window_seconds`` returns a conservative
        # ~4.8 * blocks_per_window bound on the very first gather (≈ 144s
        # at the default ``blocks_per_window=30``, well inside the 600s
        # hard timeout).
        _bpw = int(getattr(self.hparams, "blocks_per_window", 30)) if self.hparams else 30
        self._ema_step_seconds: float = float(_bpw) * 12.0
        self._ema_quorum_seconds: float = 0.0
        self._ema_sync_seconds: float = 0.0
        self._ema_step_samples: int = 0
        self._ema_sync_samples: int = 0
        self._ema_quorum_samples: int = 0
        # Hysteresis signal: set to True whenever the previous window hit
        # ``K_quorum`` inside its overall timeout. Exposed for dashboards
        # + future policy (don't drop the K_quorum floor unless the prior
        # window was grace-window-late by >= ξ_step per plan line 137).
        # The 600s overall timeout always wins; this flag is observational.
        self._last_window_quorum_hit: bool = False
        # Telemetry feed-forward: last gather's measured quorum_hit time
        # (seconds from gather start) and the adaptive grace deadline that
        # was armed. ``None`` when K_quorum=0 or K_quorum was not reached.
        # Validator reads these after gather_with_reserve to populate the
        # new ``timing_gather_quorum_seconds`` / ``timing_gather_grace_seconds``
        # fields on ``DashboardReporter.report_window``.
        self._last_gather_quorum_seconds: float | None = None
        self._last_gather_grace_seconds: float | None = None

    def _compute_grace_window_seconds(self) -> float:
        """Decoupled DiLoCo eq. 3 adaptive grace window.

        ``ξ_grace = γ · (τ · ξ_step − ξ_quorum − ξ_sync)`` with γ from
        ``gather_grace_window_factor`` and τ from ``gather_grace_tau``.
        Bootstrap note: ``_ema_step_seconds`` seeds to
        ``blocks_per_window * 12s`` so the first gather gets a conservative
        bound (no live step sample yet).

        Clamped to ``[0.0, overall_timeout)`` so we can never set a grace
        deadline past the 600s hard ceiling.
        """
        gamma = float(getattr(self.hparams, "gather_grace_window_factor", 0.2)) if self.hparams else 0.2
        tau = float(getattr(self.hparams, "gather_grace_tau", 2.0)) if self.hparams else 2.0
        grace = gamma * (
            tau * self._ema_step_seconds
            - self._ema_quorum_seconds
            - self._ema_sync_seconds
        )
        if grace < 0.0:
            grace = 0.0
        # Pin below the hard 600s overall ceiling even if ema_step is huge.
        if grace > 600.0:
            grace = 600.0
        return grace

    def record_step_time(self, seconds: float) -> None:
        """Feed back the full-window wall-clock (``timing_window_total``).

        Alpha = ``gather_grace_ema_alpha``. Validator calls this once per
        window near ``report_window`` so the next gather's
        ``_compute_grace_window_seconds`` reflects recent step durations.
        """
        if not (seconds > 0.0 and math.isfinite(seconds)):
            return
        alpha = float(getattr(self.hparams, "gather_grace_ema_alpha", 0.2)) if self.hparams else 0.2
        if self._ema_step_samples == 0:
            self._ema_step_seconds = float(seconds)
        else:
            self._ema_step_seconds = (
                alpha * float(seconds) + (1.0 - alpha) * self._ema_step_seconds
            )
        self._ema_step_samples += 1

    def record_sync_time(self, seconds: float) -> None:
        """Feed back post-gather sync+apply wall-clock.

        Per plan: ``outer_step_timings["apply_seconds"] + outer_step_timings["merge_seconds"]``.
        """
        if not (seconds >= 0.0 and math.isfinite(seconds)):
            return
        alpha = float(getattr(self.hparams, "gather_grace_ema_alpha", 0.2)) if self.hparams else 0.2
        if self._ema_sync_samples == 0:
            self._ema_sync_seconds = float(seconds)
        else:
            self._ema_sync_seconds = (
                alpha * float(seconds) + (1.0 - alpha) * self._ema_sync_seconds
            )
        self._ema_sync_samples += 1

    def _record_quorum_time(self, seconds: float) -> None:
        """Internal: update ``_ema_quorum_seconds`` from inside the gather loop."""
        if not (seconds >= 0.0 and math.isfinite(seconds)):
            return
        alpha = float(getattr(self.hparams, "gather_grace_ema_alpha", 0.2)) if self.hparams else 0.2
        if self._ema_quorum_samples == 0:
            self._ema_quorum_seconds = float(seconds)
        else:
            self._ema_quorum_seconds = (
                alpha * float(seconds) + (1.0 - alpha) * self._ema_quorum_seconds
            )
        self._ema_quorum_samples += 1

    async def _get_s3_client(self, bucket: Bucket) -> AioBaseClient:
        """
        Retrieves or creates a persistent S3 client for the given bucket.

        This method ensures that a single client is reused for each set of bucket
        credentials to optimize resource usage. If a client for the specified
        bucket does not exist, it is created and stored for future use.

        Args:
            bucket (Bucket): The bucket for which to get the S3 client.

        Returns:
            AioBaseClient: The asynchronous S3 client for the specified bucket.
        """
        key = (bucket.access_key_id, bucket.secret_access_key, bucket.account_id)
        if key in self._s3_clients:
            return self._s3_clients[key]

        # Create the base URL
        endpoint_url = self.get_base_url(bucket.account_id)

        # Create the client (equivalent to `async with`, but we store the client persistently)
        new_client = self.session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=CF_REGION_NAME,
            config=client_config,
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
        )
        # We must manually do an async enter
        new_client = await new_client.__aenter__()

        self._s3_clients[key] = new_client
        return new_client

    async def close_all_s3_clients(self):
        """
        Closes all active S3 clients.

        This method iterates through all created S3 clients and closes them
        gracefully. It's intended to be called during shutdown to release
        all network resources.
        """
        for key, client in list(self._s3_clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                hone.logger.warning(f"Error closing s3_client {key}: {e}")
        self._s3_clients.clear()

    async def _purge_s3_client(self, bucket: Bucket) -> None:
        """
        Removes a specific S3 client from the managed pool.

        This is typically used when a client encounters a persistent error
        (e.g., ConnectionClosedError) and needs to be recreated on the next request.

        Args:
            bucket (Bucket): The bucket whose client needs to be purged.
        """
        key = (bucket.access_key_id, bucket.secret_access_key, bucket.account_id)
        if key in self._s3_clients:
            del self._s3_clients[key]

    def start_background_tasks(self):
        """
        Initializes and starts background tasks for the Comms instance.

        This method sets up a thread pool executor and starts the `track_active_peers`
        task to run in the background, continuously monitoring peer activity.
        """
        self.loop = asyncio.get_running_loop()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        # Start background tasks
        self.loop.create_task(self.track_active_peers())

    def get_own_bucket(
        self,
        bucket_type: Literal["gradients", "dataset", "aggregator"],
        access_type: Literal["read", "write"] | None = None,
    ) -> Bucket:
        """
        Retrieves bucket configuration from environment variables.

        This method constructs a `Bucket` object based on the specified type and
        access level, using credentials stored in `BUCKET_SECRETS`.

        Args:
            bucket_type (Literal["gradients", "dataset", "aggregator"]): The type of bucket to retrieve.
            access_type (Literal["read", "write"] | None, optional): The access level required.
                Required for 'gradients' and 'aggregator' buckets. Defaults to None.

        Returns:
            Bucket: The configured bucket object.

        Raises:
            ValueError: If `bucket_type` or `access_type` is invalid.
            KeyError: If required R2 configuration is missing.
        """
        try:
            if bucket_type not in ["gradients", "dataset", "aggregator"]:
                raise ValueError("bucket_type must be either 'gradients' or 'dataset'")

            if bucket_type in ["gradients", "aggregator"]:
                if access_type not in ["read", "write"]:
                    raise ValueError(
                        f"For {bucket_type} bucket, access_type must be either 'read' or 'write'"
                    )

                bucket_config = BUCKET_SECRETS[bucket_type]
                credentials = bucket_config["credentials"][access_type]  # type: ignore
            else:  # dataset bucket
                bucket_config = BUCKET_SECRETS["dataset"]
                # For dataset, we'll use read credentials by default
                credentials = bucket_config["credentials"]["read"]  # type: ignore

            # Create a Bucket object using specified credentials
            bucket = Bucket(
                name=bucket_config["name"],
                account_id=bucket_config["account_id"],
                access_key_id=credentials["access_key_id"],
                secret_access_key=credentials["secret_access_key"],
            )

            hone.logger.debug(
                f"Created {bucket_type} bucket with {'read/write' if bucket_type == 'dataset' else access_type} access: {bucket}"
            )
            return bucket

        except KeyError as e:
            hone.logger.error(f"Missing required R2 configuration: {e}")
            raise
        except Exception as e:
            hone.logger.error(f"Error creating bucket: {e}")
            raise

    def get_base_url(self, account_id: str) -> str:
        """
        Constructs the base URL for the R2 storage endpoint.

        Args:
            account_id (str): The R2 account ID.

        Returns:
            str: The full endpoint URL for the R2 storage.
        """
        return f"https://{account_id}.r2.cloudflarestorage.com"

    def delete_local_directory(self, path: str):
        """
        Safely removes a local directory and all its contents.

        Args:
            path (str): The path to the directory to be deleted.
        """
        if not os.path.exists(path):
            return
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(path)

    # Convert all the existing functions to methods
    async def cleanup_local_data(
        self, uid: str, current_window: int, stale_retention: int
    ):
        """
        Cleans up stale local data for a given UID.

        Removes directories and files that are older than the specified retention window
        to manage local disk space.

        Args:
            uid (str): The UID whose local data needs to be cleaned.
            current_window (int): The current training window, used as a reference point.
            stale_retention (int): The number of windows to keep before data is considered stale.
        """
        user_dir = os.path.join(LOCAL_TMP_DIR, str(uid))
        if not os.path.exists(user_dir):
            return

        min_allowed_window = current_window - stale_retention
        for wdir in os.listdir(user_dir):
            if wdir.isdigit():
                w = int(wdir)
                if w < min_allowed_window:
                    old_path = os.path.join(user_dir, wdir)
                    hone.logger.debug(f"Removing stale local directory: {old_path}")
                    try:
                        self.delete_local_directory(old_path)
                    except Exception as e:
                        hone.logger.debug(
                            f"Error removing stale directory {old_path}: {e}"
                        )

    async def cleanup_s3_data(
        self, uid: str, current_window: int, stale_retention: int
    ):
        """
        Cleans up stale S3 data for a given UID.

        Deletes objects from the S3 bucket that are older than the specified retention
        window to manage storage costs and keep the bucket clean.

        Args:
            uid (str): The UID whose S3 data needs to be cleaned.
            current_window (int): The current training window.
            stale_retention (int): The number of windows to keep.
        """
        try:
            min_allowed_window = current_window - stale_retention

            # Regex pattern to match filenames of the form:
            #   gradient-<window>-<uid>-v<version>.pt          (legacy / non-PP)
            #   gradient-<window>-<uid>-stage<N>-v<version>.pt (per-stage upload)
            pattern = re.compile(
                rf"^gradient-(\d+)-{uid}(?:-stage\d+)?-v"
                rf"{re.escape(hone.__version__)}\.pt$"
            )

            prefix = "gradient"

            # so we get the same credentials as `self.bucket`
            s3_client = await self._get_s3_client(self.bucket)

            continuation_token = None
            while True:
                list_args = {
                    "Bucket": self.bucket.name,
                    "Prefix": prefix,
                    "MaxKeys": 1000,
                }
                if continuation_token:
                    list_args["ContinuationToken"] = continuation_token

                response = await s3_client.list_objects_v2(**list_args)
                contents = response.get("Contents", [])

                # Identify stale objects to delete
                stale_objects = []
                for obj in contents:
                    key = obj["Key"]

                    # Attempt to match our known filename pattern
                    match = pattern.match(key)
                    if match is None:
                        continue

                    try:
                        file_window = int(match.group(1))
                    except ValueError:
                        continue

                    if file_window < min_allowed_window:
                        stale_objects.append({"Key": key})

                # Batch delete stale objects
                if len(stale_objects) > 0:
                    hone.logger.debug(
                        f"Removing stale S3 objects for {uid}: {stale_objects}"
                    )
                    await s3_client.delete_objects(
                        Bucket=self.bucket.name,
                        Delete={"Objects": stale_objects},
                    )

                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break
        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(self.bucket)

    async def s3_put_object(
        self,
        key: str,
        file_path: str | None = None,
        bucket: Bucket | None = None,
    ) -> None:
        """
        Uploads an object to S3, handling different file types and sizes.

        This method supports both regular and multipart uploads for large files.
        It also handles JSON and PyTorch tensor files appropriately.

        Args:
            key (str): The S3 object key.
            file_path (str | None, optional): The local path to the file to upload.
                Required for non-JSON files. Defaults to None.
            bucket (Bucket | None, optional): The target bucket. If None, uses the
                default instance bucket. Defaults to None.

        Raises:
            ValueError: If `file_path` is not provided for a non-JSON file.
            Exception: Propagates exceptions from the S3 client.
        """
        try:
            if bucket is None:
                bucket = self.bucket

            s3_client = await self._get_s3_client(bucket)

            # Handle JSON files
            if key.endswith(".json") or "start_window" in key:
                if file_path:
                    async with aiofiles.open(file_path, "r") as f:
                        data = await f.read()
                        data_bytes = data.encode("utf-8")
                else:
                    raise ValueError(f"file_path required for JSON file: {key}")

                await s3_client.put_object(Bucket=bucket.name, Key=key, Body=data_bytes)
                return

            # Otherwise, likely PyTorch files
            file_size = os.path.getsize(file_path)
            multipart_threshold = 100 * 1024 * 1024  # 100MB

            if file_size <= multipart_threshold:
                # Simple upload for small files
                async with aiofiles.open(file_path, "rb") as f:
                    data = await f.read()
                    await s3_client.put_object(Bucket=bucket.name, Key=key, Body=data)
            else:
                # Multipart upload for large files -> boto3 TransferManager in a thread
                await self._upload_large_file_via_boto3(file_path, key, bucket)

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
        except Exception as e:
            hone.logger.error(f"Error uploading {key} to S3: {e}")
            raise

    async def s3_get_object(
        self,
        key: str,
        bucket: Bucket | None = None,
        timeout: int = 30,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        load_data: bool = True,
        show_progress: bool = True,
        map_location: str | None = None,
    ):
        """Download object from S3 using asynchronous streaming.

        Args:
            key: The S3 object key to download
            bucket: The bucket configuration (defaults to self.bucket)
                        timeout (int, optional): Download timeout in seconds. Defaults to 30.
            time_min (datetime | None, optional): The minimum modification time for the object.
                If the object is older, it's skipped. Defaults to None.
            time_max (datetime | None, optional): The maximum modification time for the object.
                If the object is newer, it's skipped. Defaults to None.
            load_data (bool, optional): If True, loads the object into memory.
                If False, moves it to a local path. Defaults to True.
            map_location (str | None, optional): Device to map tensors to when loading PyTorch files.
                If None, defaults to self.config.device. Use "cpu" to avoid GPU OOM. Defaults to None.

        Returns:
            Any | None: The loaded data (e.g., dict from JSON, tensor from .pt),
                a status dictionary if skipped, or None if not found or on error.
        """

        # Replace forward slashes in key to avoid creating subdirectories
        safe_key = key.replace("/", "_")
        temp_file_path = os.path.join(
            self.temp_dir, f"temp_{safe_key}_{uuid.uuid4().hex}.pt"
        )
        s3_client = await self._get_s3_client(bucket)
        try:
            # Normalize timezone info
            if time_min is not None and not time_min.tzinfo:
                time_min = time_min.replace(tzinfo=timezone.utc)
            if time_max is not None and not time_max.tzinfo:
                time_max = time_max.replace(tzinfo=timezone.utc)

            # HEAD the object
            try:
                response = await asyncio.wait_for(
                    s3_client.head_object(Bucket=bucket.name, Key=key),
                    timeout=timeout,
                )
                last_modified = response.get("LastModified")
                if last_modified is None:
                    hone.logger.info(f"Object does not exist: {key}")
                    return None

                if time_min is not None and last_modified < time_min:
                    time_diff = (time_min - last_modified).total_seconds()
                    hone.logger.info(
                        f"Object {key} was uploaded {time_diff:.2f}s before time_min."
                    )
                    return {"__status": "TOO_EARLY"}

                if time_max is not None and last_modified > time_max:
                    time_diff = (last_modified - time_max).total_seconds()
                    hone.logger.info(
                        f"Object {key} was uploaded {time_diff:.2f}s after time_max."
                    )
                    return {"__status": "TOO_LATE"}

            except asyncio.TimeoutError:
                hone.logger.debug(f"Timeout checking for {key}")
                return None
            except (ConnectionClosedError, ClientError) as e:
                await self._purge_s3_client(bucket)
                if "404" in str(e):
                    hone.logger.debug(f"Object {key} not found in bucket {bucket.name}")
                    return None
            except Exception as e:
                hone.logger.debug(f"Some other exception occurred: {e=}")
                raise e

            file_size = response["ContentLength"]  # type: ignore

            # Download the object - choose method based on file size
            if file_size <= 1 * 1024 * 1024 * 1024:  # 1GB - use simple download
                response = await asyncio.wait_for(
                    s3_client.get_object(Bucket=bucket.name, Key=key),
                    timeout=timeout,
                )
                async with aiofiles.open(temp_file_path, "wb") as f:
                    async with response["Body"] as stream:
                        data = await asyncio.wait_for(stream.read(), timeout=timeout)
                        await f.write(data)
                success = True
            else:
                # Use our optimized Python implementation for all large files
                # Scale timeout with file size: minimum 300s, or 1 second per 25MB
                download_timeout = max(300, file_size / (25 * 1024 * 1024))
                success = await asyncio.wait_for(
                    self.download_large_file(
                        s3_client=s3_client,
                        bucket=bucket,
                        key=key,
                        file_size=file_size,
                        temp_file_path=temp_file_path,
                        show_progress=show_progress,
                    ),
                    timeout=download_timeout,
                )

                if not success:
                    return None

            loaded_data = None
            if load_data:
                # Now load the data
                if key.endswith(".json") or "start_window" in key:
                    async with aiofiles.open(temp_file_path, "r") as f:
                        data = await f.read()
                        loaded_data = json.loads(data)
                else:
                    # Use provided map_location or default to self.config.device
                    device_location = (
                        map_location if map_location is not None else self.config.device
                    )
                    loaded_data = torch.load(
                        temp_file_path,
                        map_location=device_location,
                        weights_only=True,
                    )
            else:
                if success:
                    target_directory = os.path.dirname(key)
                    if target_directory:
                        os.makedirs(target_directory, exist_ok=True)

                    # with a cross-device-safe move:
                    loaded_data = shutil.move(temp_file_path, key)
                else:
                    raise FileNotFoundError(
                        f"Download not successful for file at: {temp_file_path}"
                    )

            return loaded_data

        except asyncio.TimeoutError:
            hone.logger.debug(f"Timeout downloading {key}")
            return None
        except Exception as e:
            hone.logger.error(f"Error in s3_get_object for {key}: {e}")
            return None
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    #  Large File Operations

    async def upload_large_file(
        self,
        file_path: str,
        key: str,
        s3_client: Any,
        bucket: Bucket | None = None,
    ) -> None:
        """
        Uploads a large file to S3 using a robust, asynchronous multipart strategy.

        This method handles large file uploads by splitting them into chunks and
        uploading them in parallel. It includes retry logic for transient network
        errors and dynamically adjusts part sizes based on the file's total size
        to optimize performance.

        Args:
            file_path (str): The local path to the file to be uploaded.
            key (str): The destination S3 object key.
            s3_client: The `AioBaseClient` instance to use for the upload.
            bucket (Bucket | None, optional): The target S3 bucket. If None, the
                instance's default bucket is used. Defaults to None.

        Raises:
            Exception: Propagates exceptions from the S3 client if the upload
                fails after multiple retries.
        """
        # ---- parameters & helpers -------------------------------------------
        bucket = bucket or self.bucket
        file_size = os.path.getsize(file_path)

        # Keep parts under ~9k to stay safely below the 10k S3 limit
        MIN_PART = 64 * 1024 * 1024  # 64 MiB (S3 min is 5 MiB; use 64 MiB alignment)
        MAX_PARTS = 9000
        part_size = max(MIN_PART, math.ceil(file_size / MAX_PARTS))
        # Round up to 8 MiB alignment for nicer server-side buffering
        part_size = ((part_size + MIN_PART - 1) // MIN_PART) * MIN_PART

        # Optional overrides via env for tuning
        part_size = int(os.getenv("TPLR_S3_PART_SIZE", part_size))
        # Concurrency local to this MPU; also gated by self.client_semaphore
        concurrency = int(os.getenv("TPLR_S3_CONCURRENCY", "32"))
        concurrency = max(2, min(concurrency, 32))

        MAX_RETRIES = 5
        JITTER = 0.1

        def _b64_md5(data: bytes) -> str:
            return base64.b64encode(__import__("hashlib").md5(data).digest()).decode()

        total_parts = max(1, math.ceil(file_size / part_size))
        hone.logger.info(
            f"[S3][MPU] start key={key} size={file_size}B "
            f"parts={total_parts} part_size={part_size}B concurrency={concurrency}"
        )

        # ---- helpers ---------------------------------------------------------
        async def _create_mpu() -> str:
            for attempt in range(MAX_RETRIES):
                try:
                    async with self.client_semaphore:
                        resp = await s3_client.create_multipart_upload(
                            Bucket=bucket.name, Key=key
                        )
                    return resp["UploadId"]
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    backoff = (2**attempt) + random.random() * JITTER
                    hone.logger.warning(
                        f"[S3][MPU] create failed (attempt {attempt + 1}) – retrying in {backoff:.2f}s: {e}"
                    )
                    await asyncio.sleep(backoff)
            raise RuntimeError("unreachable")

        async def _complete_mpu(upload_id: str, parts: list[dict[str, object]]) -> None:
            for attempt in range(MAX_RETRIES):
                try:
                    async with self.client_semaphore:
                        await s3_client.complete_multipart_upload(
                            Bucket=bucket.name,
                            Key=key,
                            UploadId=upload_id,
                            MultipartUpload={"Parts": parts},
                        )
                    hone.logger.info(f"[S3][MPU] complete ok key={key}")
                    return
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    backoff = (2**attempt) + random.random() * JITTER
                    hone.logger.warning(
                        f"[S3][MPU] complete failed (attempt {attempt + 1}) – retrying in {backoff:.2f}s: {e}"
                    )
                    await asyncio.sleep(backoff)

        async def _abort_mpu(upload_id: str) -> None:
            try:
                async with self.client_semaphore:
                    await s3_client.abort_multipart_upload(
                        Bucket=bucket.name, Key=key, UploadId=upload_id
                    )
                hone.logger.info(f"[S3][MPU] aborted key={key} upload_id={upload_id}")
            except Exception as e:
                hone.logger.warning(f"[S3][MPU] abort failed: {e}")

        # ---- outer restart loop (for NoSuchUpload) ---------------------------
        upload_attempt = 0
        while True:
            upload_attempt += 1
            start_all = time.perf_counter()
            uploaded_bytes = 0
            uploaded_lock = asyncio.Lock()

            upload_id = await _create_mpu()
            hone.logger.info(
                f"[S3][MPU] created key={key} upload_id={upload_id} (attempt {upload_attempt})"
            )

            # Work queue of part numbers
            q: asyncio.Queue[int] = asyncio.Queue()
            for pn in range(1, total_parts + 1):
                q.put_nowait(pn)

            cancel_event = asyncio.Event()
            parts_done: dict[int, str] = {}
            errors: list[BaseException] = []

            async def _worker(wid: int) -> None:
                nonlocal uploaded_bytes
                while not cancel_event.is_set():
                    try:
                        pn = await asyncio.wait_for(q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if q.empty():
                            return
                        continue

                    byte_start = (pn - 1) * part_size
                    byte_end = min(byte_start + part_size, file_size)
                    size = byte_end - byte_start

                    for attempt in range(MAX_RETRIES):
                        try:
                            async with aiofiles.open(file_path, "rb") as f:
                                await f.seek(byte_start)
                                data = await f.read(size)

                            headers = {"ContentMD5": _b64_md5(data)}
                            t0 = time.perf_counter()
                            async with self.client_semaphore:
                                resp = await s3_client.upload_part(
                                    Bucket=bucket.name,
                                    Key=key,
                                    PartNumber=pn,
                                    UploadId=upload_id,
                                    Body=data,
                                    **headers,
                                )
                            dt = time.perf_counter() - t0
                            async with uploaded_lock:
                                uploaded_bytes += size
                            parts_done[pn] = resp["ETag"]
                            speed = (size / (1024 * 1024)) / max(dt, 1e-6)
                            hone.logger.debug(
                                f"[S3][MPU] ↑ part {pn}/{total_parts} "
                                f"{size / 1024 / 1024:.2f} MiB in {dt:.2f}s ({speed:.2f} MiB/s) "
                                f"[worker {wid}]"
                            )
                            break
                        except ClientError as e:
                            code = e.response.get("Error", {}).get("Code")
                            if code == "NoSuchUpload":
                                # The MPU has been invalidated; request a restart.
                                cancel_event.set()
                                errors.append(e)
                                hone.logger.warning(
                                    f"[S3][MPU] part {pn} got NoSuchUpload – restarting MPU"
                                )
                                break
                            if attempt == MAX_RETRIES - 1:
                                cancel_event.set()
                                errors.append(e)
                                hone.logger.error(
                                    f"[S3][MPU] part {pn} failed after {MAX_RETRIES} attempts: {e}"
                                )
                                break
                            backoff = (2**attempt) + random.random() * JITTER
                            await asyncio.sleep(backoff)
                        except (
                            EndpointConnectionError,
                            ClientOSError,
                            ServerDisconnectedError,
                            asyncio.TimeoutError,
                        ) as e:
                            if attempt == MAX_RETRIES - 1:
                                cancel_event.set()
                                errors.append(e)
                                hone.logger.error(
                                    f"[S3][MPU] part {pn} network failure after {MAX_RETRIES} attempts: {e}"
                                )
                                break
                            backoff = (2**attempt) + random.random() * JITTER
                            await asyncio.sleep(backoff)
                        except Exception as e:
                            cancel_event.set()
                            errors.append(e)
                            hone.logger.error(
                                f"[S3][MPU] part {pn} unexpected error: {e}"
                            )
                            break
                    q.task_done()

            workers = [asyncio.create_task(_worker(i)) for i in range(concurrency)]
            await q.join()
            cancel_event.set()
            await asyncio.gather(*workers, return_exceptions=True)

            dt_all = time.perf_counter() - start_all
            mb = uploaded_bytes / (1024 * 1024)
            hone.logger.info(
                f"[S3][MPU] uploaded ~{mb:.2f} MiB in {dt_all:.2f}s "
                f"(~{(mb / dt_all) if dt_all > 0 else 0:.2f} MiB/s) key={key}"
            )

            # If we saw a fatal error (e.g., NoSuchUpload), abort and restart.
            if errors:
                await _abort_mpu(upload_id)
                if any(
                    isinstance(e, ClientError)
                    and e.response.get("Error", {}).get("Code") == "NoSuchUpload"
                    for e in errors
                ):
                    if upload_attempt < 3:
                        hone.logger.warning(
                            f"[S3][MPU] restarting MPU for key={key} (attempt {upload_attempt + 1})"
                        )
                        continue
                # Otherwise propagate the first error.
                raise errors[0]

            # Complete MPU
            if len(parts_done) != total_parts:
                await _abort_mpu(upload_id)
                raise RuntimeError(
                    f"[S3][MPU] missing parts: have {len(parts_done)} expected {total_parts}"
                )

            parts_for_complete = [
                {"PartNumber": pn, "ETag": parts_done[pn]} for pn in sorted(parts_done)
            ]
            try:
                await _complete_mpu(upload_id, parts_for_complete)
                return
            except Exception as e:
                # best effort cleanup, then rethrow
                await _abort_mpu(upload_id)
                raise
        # end while

    async def _upload_large_file_via_boto3(
        self, file_path: str, key: str, bucket: Bucket
    ) -> None:
        """
        Upload large files using boto3's TransferManager inside a worker thread.
        Bounded by self.upload_sem to avoid spinning up many managers at once.
        """
        MB = 1024 * 1024
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        if file_size_gb > 10:
            part_size = 128 * MB
            max_conc = 16
        elif file_size_gb > 1:
            part_size = 64 * MB
            max_conc = 12
        else:
            part_size = 32 * MB
            max_conc = 8

        tconf = TransferConfig(
            multipart_threshold=part_size,
            multipart_chunksize=part_size,
            max_concurrency=max_conc,
            use_threads=True,
        )

        endpoint = self.get_base_url(bucket.account_id)

        async with self.upload_sem:

            def _do():
                c = boto3.client(
                    "s3",
                    endpoint_url=endpoint,
                    region_name=CF_REGION_NAME,
                    config=client_config,  # reuses your existing botocore.Config
                    aws_access_key_id=bucket.access_key_id,
                    aws_secret_access_key=bucket.secret_access_key,
                )
                c.upload_file(file_path, bucket.name, key, Config=tconf)

            await asyncio.to_thread(_do)

    async def download_large_file(
        self,
        s3_client,
        bucket: Bucket,
        key: str,
        file_size: int,
        temp_file_path: str,
        show_progress: bool = True,
    ) -> bool:
        """
        Downloads a large file from S3 using a parallel, multipart strategy.

        This method optimizes large file downloads by fetching multiple chunks
        concurrently. It dynamically adjusts the chunk size and the number of
        parallel workers based on available system resources (GPU or CPU) to
        maximize download speed. It also includes a progress bar and robust
        error handling with retries.

        Args:
            s3_client: The `AioBaseClient` instance for the download.
            bucket (Bucket): The S3 bucket to download from.
            key (str): The S3 object key.
            file_size (int): The total size of the file in bytes.
            temp_file_path (str): The local path to save the downloaded file.

        Returns:
            bool: True if the download was successful, False otherwise.

        """
        try:
            # Centralised parameter selection (no logic duplication)
            params = self._calc_download_params(
                file_size, os.getenv("DOWNLOAD_MAX_WORKERS")
            )
            chunk_size = params["chunk_size"]
            max_workers = params["max_workers"]
            total_chunks = params["total_chunks"]
            file_size_gb = params["file_size_gb"]

            hone.logger.info(
                f"Downloading {file_size_gb:.1f}GB file with {chunk_size // (1024 * 1024)}MB chunks, {max_workers} workers"
            )

            # Resume capability - check for existing partial file
            resume_info = await self._get_download_resume_info(
                temp_file_path, total_chunks, chunk_size, file_size
            )
            downloaded_chunks = resume_info["completed_chunks"]
            remaining_chunks = resume_info["remaining_chunks"]

            if downloaded_chunks:
                hone.logger.info(
                    f"Resuming download: {len(downloaded_chunks)}/{total_chunks} chunks already completed"
                )

            # Create directory if needed
            target_directory = os.path.dirname(temp_file_path)
            if target_directory:
                os.makedirs(target_directory, exist_ok=True)

            # Use a memory-efficient approach - don't pre-allocate the entire file
            # Instead, create sparse file or extend as needed
            if not os.path.exists(temp_file_path):
                # Create file
                async with aiofiles.open(temp_file_path, "wb") as f:
                    pass  # Just create the file

            # Create progress bar if requested
            already_downloaded = sum(
                chunk["size"] for chunk in downloaded_chunks.values()
            )
            if show_progress:
                pbar = std_tqdm(
                    total=file_size,
                    initial=already_downloaded,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {key} ({max_workers} workers)",
                )
            else:
                pbar = None

            # Memory-controlled semaphore to prevent excessive memory usage
            download_semaphore = asyncio.Semaphore(max_workers)

            async def download_chunk_streaming(chunk_number: int, max_retries: int = 3):
                """Download chunk directly to file using streaming (no data loss, no over-read)."""
                if chunk_number in downloaded_chunks:
                    return chunk_number

                for attempt in range(max_retries):
                    async with download_semaphore:
                        start = chunk_number * chunk_size
                        end = min(start + chunk_size, file_size) - 1
                        expected_len = end - start + 1

                        try:
                            # Per-chunk timeout (60 seconds for S3 request)
                            response = await asyncio.wait_for(
                                s3_client.get_object(
                                    Bucket=bucket.name,
                                    Key=key,
                                    Range=f"bytes={start}-{end}",
                                ),
                                timeout=60,
                            )

                            bytes_written = 0
                            buffer_size = min(8 * 1024 * 1024, expected_len)  # ≤ 8MB

                            async with aiofiles.open(temp_file_path, "rb+") as f:
                                await f.seek(start)
                                stream = response[
                                    "Body"
                                ]  # aiobotocore AIOStreamingBody
                                try:
                                    while bytes_written < expected_len:
                                        to_read = min(
                                            buffer_size, expected_len - bytes_written
                                        )
                                        # Timeout for each stream read (60 seconds)
                                        chunk = await asyncio.wait_for(
                                            stream.read(to_read), timeout=60
                                        )
                                        if not chunk:
                                            raise Exception(
                                                f"Unexpected EOF in range {start}-{end}; "
                                                f"wrote {bytes_written} of {expected_len}"
                                            )
                                        await f.write(chunk)
                                        bytes_written += len(chunk)
                                finally:
                                    # Return connection to pool properly
                                    if hasattr(stream, "release_conn"):
                                        try:
                                            await stream.release_conn()
                                        except TypeError:
                                            # some versions define it sync
                                            stream.release_conn()
                                    else:
                                        # fallback: sync close
                                        try:
                                            stream.close()
                                        except Exception:
                                            pass

                            if bytes_written != expected_len:
                                raise Exception(
                                    f"Chunk write mismatch: wrote {bytes_written}, expected {expected_len}"
                                )
                            if pbar:
                                pbar.update(expected_len)

                            downloaded_chunks[chunk_number] = {
                                "start": start,
                                "end": end + 1,
                                "size": bytes_written,
                            }
                            return chunk_number

                        except asyncio.CancelledError:
                            raise
                        except asyncio.TimeoutError:
                            hone.logger.warning(
                                f"Timeout downloading chunk {chunk_number} (attempt {attempt + 1}/{max_retries}) - "
                                f"possible rate limiting or slow network"
                            )
                            if attempt == max_retries - 1:
                                raise
                            await asyncio.sleep((2**attempt) + random.uniform(0, 1))
                        except Exception as e:
                            # Check for rate limiting indicators
                            error_str = str(e).lower()
                            if any(
                                indicator in error_str
                                for indicator in [
                                    "429",
                                    "throttl",
                                    "slow down",
                                    "rate limit",
                                    "503",
                                    "service unavailable",
                                    "too many",
                                ]
                            ):
                                hone.logger.error(
                                    f"RATE LIMITING DETECTED downloading chunk {chunk_number}: {e}"
                                )
                            else:
                                hone.logger.error(
                                    f"Error downloading chunk {chunk_number} (attempt {attempt + 1}/{max_retries}): {e}"
                                )
                            if attempt == max_retries - 1:
                                raise
                            await asyncio.sleep((2**attempt) + random.uniform(0, 1))

            try:
                # Download remaining chunks
                if remaining_chunks:
                    tasks = [
                        asyncio.create_task(download_chunk_streaming(c))
                        for c in remaining_chunks
                    ]
                    try:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        # propagate first real error (skip CancelledError/None)
                        for r in results:
                            if isinstance(r, Exception) and not isinstance(
                                r, asyncio.CancelledError
                            ):
                                raise r
                    finally:
                        # ➌ make sure **all** tasks are finished before we return
                        for t in tasks:
                            t.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)

                # Final verification
                if len(downloaded_chunks) != total_chunks:
                    missing_chunks = set(range(total_chunks)) - set(
                        downloaded_chunks.keys()
                    )
                    raise Exception(f"Missing chunks: {missing_chunks}")

                downloaded_size = sum(
                    chunk["size"] for chunk in downloaded_chunks.values()
                )
                if downloaded_size != file_size:
                    raise Exception(
                        f"Downloaded size ({downloaded_size}) != expected size ({file_size})"
                    )

                # Verify file integrity
                actual_file_size = os.path.getsize(temp_file_path)
                if actual_file_size != file_size:
                    raise Exception(
                        f"File size mismatch: {actual_file_size} != {file_size}"
                    )

                hone.logger.info(f"Successfully downloaded {file_size_gb:.1f}GB file")
                return True

            finally:
                if pbar:
                    pbar.close()

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(bucket)
            return False
        except Exception as e:
            hone.logger.error(f"Error in download_large_file for {key}: {e}")
            return False

    async def _get_download_resume_info(
        self, temp_file_path: str, total_chunks: int, chunk_size: int, file_size: int
    ):
        """Check existing partial file and determine which chunks need to be downloaded."""
        completed_chunks = {}
        remaining_chunks = list(range(total_chunks))

        if not os.path.exists(temp_file_path):
            return {
                "completed_chunks": completed_chunks,
                "remaining_chunks": remaining_chunks,
            }

        try:
            current_file_size = os.path.getsize(temp_file_path)
            if current_file_size == 0:
                return {
                    "completed_chunks": completed_chunks,
                    "remaining_chunks": remaining_chunks,
                }

            # For resume capability, we need to verify which chunks are complete
            # We'll do a basic verification by checking file size and assuming
            # sequential chunks are complete up to the current size
            completed_bytes = min(current_file_size, file_size)
            completed_full_chunks = completed_bytes // chunk_size

            # Mark completed chunks
            for chunk_num in range(min(completed_full_chunks, total_chunks)):
                start = chunk_num * chunk_size
                end = min(start + chunk_size, file_size)
                completed_chunks[chunk_num] = {
                    "start": start,
                    "end": end,
                    "size": end - start,
                }
                if chunk_num in remaining_chunks:
                    remaining_chunks.remove(chunk_num)

            # If there's a partial chunk at the end, we'll re-download it
            # This is safer than trying to resume from the middle of a chunk

            hone.logger.debug(
                f"Resume analysis: {len(completed_chunks)}/{total_chunks} chunks verified complete"
            )

        except Exception as e:
            hone.logger.warning(
                f"Error analyzing partial file for resume: {e}. Starting fresh download."
            )
            # If we can't analyze the partial file, start fresh
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            completed_chunks = {}
            remaining_chunks = list(range(total_chunks))

        return {
            "completed_chunks": completed_chunks,
            "remaining_chunks": remaining_chunks,
        }

    @staticmethod
    def _calc_download_params(
        file_size: int, custom_workers: str | None = None
    ) -> dict[str, float | int]:
        """
        Decide chunk size & worker count according to file size, with an
        optional explicit override (DOWNLOAD_MAX_WORKERS).
        """
        file_size_gb = file_size / (1024 * 1024 * 1024)

        if file_size_gb > 100:  # ≥100 GB
            chunk_size = 512 * 1024 * 1024  # 512 MB
            max_workers = min(32, max(8, CPU_COUNT))
        elif file_size_gb > 10:  # 10–100 GB
            chunk_size = 256 * 1024 * 1024  # 256 MB
            max_workers = min(16, max(8, CPU_COUNT))
        else:  # <10 GB
            chunk_size = 64 * 1024 * 1024  # 64 MB
            max_workers = min(8, max(6, CPU_COUNT * 2))

        # Optional override
        if custom_workers:
            try:
                max_workers = int(custom_workers)
            except ValueError:
                pass

        total_chunks = math.ceil(file_size / chunk_size)
        max_workers = min(max_workers, total_chunks)

        return {
            "chunk_size": chunk_size,
            "max_workers": max_workers,
            "total_chunks": total_chunks,
            "file_size_gb": file_size_gb,
        }

    async def put(
        self,
        state_dict: dict[str, Any],
        window: int,
        # P1: widened from a strict Literal to ``str`` so the fragmented
        # upload path can pass ``"gradient-frag{i:02d}"`` without dropping
        # the type-system contract for the legacy keys (filename derivation
        # below treats anything not equal to "aggregator" as a per-uid blob,
        # which is correct for both legacy "gradient"/"debug" and new
        # "gradient-frag*" keys).
        key: str,
        uid: str | None = None,
        global_step: int = 0,
        local: bool = True,
        stale_retention: int = 10,
    ) -> float:
        """
        Saves a state dictionary to either local storage or a remote S3 bucket.

        This method acts as a high-level interface for persisting data like model
        checkpoints, gradients, or other artifacts. It handles file serialization,
        temporary file management, and routing to the correct storage backend (local
        or S3). It also triggers cleanup routines to remove stale data.

        Args:
            state_dict (dict[str, torch.Tensor]): The state dictionary to save.
            window (int): The current training window, used for organizing files.
            key (Literal["checkpoint", "debug", "gradient", "aggregator"]): The type of data being saved.
            uid (str | None, optional): The UID associated with the data, used for creating
                unique file paths. Defaults to None.
            global_step (int, optional): The global training step. Defaults to 0.
            local (bool, optional): If True, saves the file to the local filesystem.
                If False, uploads to the remote S3 bucket. Defaults to True.
            stale_retention (int, optional): The number of windows to retain before
                data is considered stale and eligible for cleanup. Defaults to 10.

        Returns:
            float: The time taken for the save operation in seconds.
        """
        if key == "aggregator":
            filename = f"{key}-{window}-v{hone.__version__}.pt"
            bucket_config = BUCKET_SECRETS["aggregator"]
            credentials = bucket_config["credentials"]["write"]

            # Create a Bucket object using specified credentials
            bucket = Bucket(
                name=bucket_config["name"],
                account_id=bucket_config["account_id"],
                access_key_id=credentials["access_key_id"],
                secret_access_key=credentials["secret_access_key"],
            )
        else:
            filename = f"{key}-{window}-{uid}-v{hone.__version__}.pt"
            bucket = None
        hone.logger.debug(f"PUT {filename} -->")

        put_start = hone.T()

        # Create per-uid temp directory
        temp_dir = os.path.join("/tmp", str(self.uid))
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_{filename}")

        try:
            # Prepare the data to be saved
            save_data = {
                "state_dict": state_dict,
                "global_step": global_step,
            }

            # Save to temp file
            await asyncio.to_thread(torch.save, save_data, temp_file_path)

            if local:
                # Local storage with per-uid directories
                await self.cleanup_local_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                local_dir = os.path.join(LOCAL_TMP_DIR, str(uid), str(window))
                os.makedirs(local_dir, exist_ok=True)
                final_path = os.path.join(local_dir, filename)
                os.replace(temp_file_path, final_path)
            else:
                await self.s3_put_object(filename, temp_file_path, bucket)
                # Remote storage with automatic handling of large files
                asyncio.create_task(
                    self.cleanup_s3_data(
                        uid=uid, current_window=window, stale_retention=stale_retention
                    )
                )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        put_end = hone.T()
        hone.logger.info(f"{hone.P(window, put_end - put_start)} PUT {filename} <--")
        return put_end - put_start

    async def gradient_timestamp(
        self,
        uid: int,
        window: int,
        version: str = hone.__version__,
    ) -> float:
        """
        Retrieves the last-modified timestamp of a gradient file from S3.

        This method performs a HEAD request to get the metadata of an S3 object
        without downloading its content, providing an efficient way to check for
        the existence and modification time of a gradient file.

        Args:
            uid (int): The UID of the miner who owns the gradient.
            window (int): The window number for the gradient.
            version (str, optional): The templar version string. Defaults to `hone.__version__`.

        Returns:
            float: The POSIX timestamp (seconds since epoch) of the file's
                last modification, or 0.0 if the file is not found or an
                error occurs.
        """
        bucket = self.commitments.get(int(uid))
        if not bucket:
            return 0.0
        try:
            s3 = await self._get_s3_client(bucket)
            key = f"gradient-{window}-{uid}-v{version}.pt"
            hdr = await s3.head_object(Bucket=bucket.name, Key=key)
            return hdr["LastModified"].timestamp()
        except Exception:
            await self._purge_s3_client(bucket)
            return 0.0

    async def get(
        self,
        uid: str,
        window: int,
        # P1: widened from Literal to str to accept per-fragment subkeys
        # (``"gradient-frag{i:02d}"``). Aggregator/debug/gradient continue
        # to dispatch through the same filename derivation below.
        key: str,
        local: bool = True,
        stale_retention: int = 10,
        timeout: int = 30,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        show_progress: bool = True,
        map_location: str | None = None,
    ) -> CommsGetResult:
        """
        Retrieves an object from storage, either locally or from a remote S3 bucket.

        This method serves as a high-level interface for fetching data like checkpoints
        or gradients. It supports time-based validation to skip files that are too
        old or too new, and it can handle both local file paths and remote S3 keys.

        Args:
            uid (str): The UID associated with the data.
            window (int): The training window of the data.
            key (Literal["checkpoint", "debug", "gradient", "aggregator"]): The type of data.
            local (bool, optional): If True, retrieves from local storage. If False,
                fetches from the remote S3 bucket. Defaults to True.
            stale_retention (int, optional): The number of windows to keep before
                considering local data stale. Defaults to 10.
            timeout (int, optional): Timeout in seconds for S3 requests. Defaults to 30.
            time_min (datetime | None, optional): The minimum modification time for the
                object. If the object is older, it's skipped. Defaults to None.
            time_max (datetime | None, optional): The maximum modification time for the
                object. If the object is newer, it's skipped. Defaults to None.
            map_location (str | None, optional): Device to map tensors to when loading PyTorch files.
                If None, defaults to self.config.device. Use "cpu" to avoid GPU OOM. Defaults to None.

        Returns:
            CommsGetResult: An object containing the status of the operation and the
                retrieved data if successful.
        """
        if key == "aggregator":
            filename = f"{key}-{window}-v{hone.__version__}.pt"
        else:
            filename = f"{key}-{window}-{uid}-v{hone.__version__}.pt"
        hone.logger.debug(f"GET {filename} -->")

        try:
            if local:
                # Local storage logic
                await self.cleanup_local_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                local_path = os.path.join(
                    LOCAL_TMP_DIR, str(uid), str(window), filename
                )
                if not os.path.exists(local_path):
                    hone.logger.debug(f"Local file not found: {local_path}")
                    return CommsGetResult(status="NOT_FOUND")
                loaded_data = torch.load(local_path, weights_only=True)
                state_dict = loaded_data.get("state_dict")
                global_step = loaded_data.get("global_step", 0)
                return CommsGetResult(data=state_dict, global_step=global_step)

            # Remote storage logic
            if key == "aggregator":
                bucket_config = BUCKET_SECRETS["aggregator"]
                credentials = bucket_config["credentials"]["read"]

                # Create a Bucket object using specified credentials
                bucket = Bucket(
                    name=bucket_config["name"],
                    account_id=bucket_config["account_id"],
                    access_key_id=credentials["access_key_id"],
                    secret_access_key=credentials["secret_access_key"],
                )
            else:
                bucket = self.commitments.get(int(uid))
            hone.logger.debug(f"Peer bucket : {bucket}")
            if not bucket:
                return CommsGetResult(status="NOT_FOUND")

            loaded_data = await self.s3_get_object(
                key=filename,
                bucket=bucket,
                timeout=timeout,
                time_min=time_min,
                time_max=time_max,
                show_progress=show_progress,
                map_location=map_location,
            )

            if loaded_data is None:
                return CommsGetResult(status="NOT_FOUND")

            # Check for TOO_LATE/TOO_EARLY marker
            if isinstance(loaded_data, dict):
                if loaded_data.get("__status") == "TOO_LATE":
                    hone.logger.info(
                        f"Object for UID {uid}, window {window}, key {key} was uploaded too late. Skipping."
                    )
                    return CommsGetResult(status="TOO_LATE")
                elif loaded_data.get("__status") == "TOO_EARLY":
                    hone.logger.info(
                        f"Object for UID {uid}, window {window}, key {key} was uploaded too early. Skipping."
                    )
                    return CommsGetResult(status="TOO_EARLY")

            state_dict = loaded_data.get("state_dict")
            global_step = loaded_data.get("global_step", 0)
            return CommsGetResult(data=state_dict, global_step=global_step)

        except Exception as e:
            hone.logger.debug(f"GET error {filename}: {e}")
            return CommsGetResult(status="ERROR")
        finally:
            hone.logger.debug(f"GET {filename} <--")

    async def get_with_retry(
        self,
        uid: str,
        window: int,
        key: str,
        timeout: int,
        local: bool = True,
        stale_retention: int = 10,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        show_progress: bool = False,
        map_location: str | None = None,
    ) -> CommsGetResult | None:
        """
        Attempts to retrieve an object from storage with a retry mechanism.

        This wrapper around the `get` method provides resilience against transient
        issues by retrying the fetch operation until a timeout is reached. It is
        particularly useful for fetching gradients that may not be immediately
        available. It also includes a grace period for time-based validation to
        accommodate minor clock discrepancies.

        Args:
            uid (str): The UID associated with the data.
            window (int): The training window of the data.
            key (str): The type of data to retrieve.
            timeout (int): The total time in seconds to keep retrying.
            local (bool, optional): Whether to fetch from local or remote storage.
                Defaults to True.
            stale_retention (int, optional): The retention period for local data.
                Defaults to 10.
            time_min (datetime | None, optional): The minimum modification time for the
                object. Defaults to None.
            time_max (datetime | None, optional): The maximum modification time for the
                object. Defaults to None.

        Returns:
            CommsGetResult | None: A result object if successful, or None if the
                operation times out or is permanently skipped.
        """
        start_time = time.time()
        end_time = start_time + timeout
        tried_after_time_max = False
        time_max_grace_period = 3.0

        while True:
            # Check if we've timed out
            if time.time() >= end_time:
                hone.logger.debug(f"GET {uid}/{window}/{key} timed out.")
                return None

            # Check if we're past time_max with grace period
            now = datetime.now(timezone.utc)

            # Only consider it "past time_max" if we're 3 seconds beyond time_max
            past_time_max = False
            if time_max is not None and now > time_max:
                seconds_past_time_max = (now - time_max).total_seconds()
                past_time_max = seconds_past_time_max > time_max_grace_period

            # If we're past time_max (with grace period) and already tried once, don't retry again
            if past_time_max and tried_after_time_max:
                hone.logger.debug(
                    f"Already tried once after time_max + {time_max_grace_period}s for UID {uid}, window {window}. Stopping retries."
                )
                return None

            # If we're past time_max (with grace period), mark that we've tried once
            if past_time_max:
                tried_after_time_max = True
                hone.logger.debug(
                    f"Past time_max + {time_max_grace_period}s for UID {uid}, window {window}. This is the final retry."
                )

            # Make the request
            result = await self.get(
                uid=uid,
                window=window,
                key=key,
                local=local,
                stale_retention=stale_retention,
                time_min=time_min,
                time_max=time_max,
                show_progress=show_progress,
                map_location=map_location,
            )

            if result.success:
                return result

            if result.status in ["TOO_LATE", "TOO_EARLY"]:
                formatted_status = result.status.lower().split("_")
                formatted_status = " ".join(formatted_status)
                hone.logger.info(
                    f"Gradient for UID {uid}, window {window} exists but was uploaded {formatted_status}. Skipping."
                )
                return None

            # For NOT_FOUND or ERROR, we retry.
            # Short delay before retrying
            await asyncio.sleep(0.5)

    async def gather(
        self,
        my_uid: int | None,
        uids: list[int],
        window: int,
        key: str,
        timeout: int,
        device: str,
        totalks: dict[str, torch.Tensor],
        compressor: TopKCompressor,
        expected_compressed_params: set[str] | None = None,
        local: bool = True,
        stale_retention: int = 10,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        xshapes: dict[str, tuple] | None = None,
        num_fragments: int | None = None,
    ) -> SimpleNamespace | None:
        """
        Gathers and processes gradients from a list of peer UIDs.

        This is a core method for distributed training, responsible for fetching
        compressed gradients from multiple peers, validating them, and aggregating
        them into a single structure. It handles decompression, de-quantization,
        and robust error checking to ensure data integrity.

        Args:
            my_uid (int | None): The UID of the neuron performing the gather.
            uids (list[int]): A list of peer UIDs to gather gradients from.
            window (int): The current training window.
            key (str): The key identifying the data to gather (e.g., "gradient").
            timeout (int): The timeout for fetching data from each peer.
            device (str): The device to move tensors to (e.g., "cuda" or "cpu").
            totalks (dict[str, torch.Tensor]): A dictionary mapping parameter names
                to their total number of elements, used for validation.
            compressor (TopKCompressor): The compressor instance for de-quantization.
            expected_compressed_params (set[str] | None, optional): A set of parameter
                names that are expected to be in the compressed state dict. Defaults to None.
            local (bool, optional): Whether to fetch from local or remote storage.
                Defaults to True.
            stale_retention (int, optional): The retention period for local data.
                Defaults to 10.
            time_min (datetime | None, optional): The minimum modification time for
                gradients. Defaults to None.
            time_max (datetime | None, optional): The maximum modification time for
                gradients. Defaults to None.
            xshapes (dict[str, tuple] | None, optional): Expected shapes for gradient
                tensors, used to validate that received gradients have the correct
                dimensions. Defaults to None.
            num_fragments (int | None, optional): When ``None`` or ``1``, fetches a
                single blob per UID (legacy path; key is ``key``). When > 1, fans
                out to ``num_fragments × len(uids)`` GETs against per-fragment keys
                ``f"{key}-frag{i:02d}"``. Out-of-order arrival is fine; the result
                preserves the legacy ``state_dict`` list-per-key shape (each
                fragment contributes disjoint tensor keys, so per-key list length
                equals the number of UIDs that uploaded a fragment containing that
                key). The result also exposes ``fragments_seen_per_uid`` and
                ``uids_per_param`` so downstream consumers (notably
                ``check_uid_index_overlap``) can disambiguate which UID contributed
                each tensor instead of assuming positional alignment with ``uids``.

        Returns:
            SimpleNamespace | None: A namespace containing the aggregated state dict,
                a list of UIDs from which gradients were successfully gathered, and
                performance metrics. Returns None if no valid gradients are received.
        """
        if not expected_compressed_params:
            expected_compressed_params = set()
        if num_fragments is not None and num_fragments > 1:
            return await self._gather_fragmented(
                my_uid=my_uid,
                uids=uids,
                window=window,
                key=key,
                timeout=timeout,
                device=device,
                totalks=totalks,
                compressor=compressor,
                num_fragments=num_fragments,
                expected_compressed_params=expected_compressed_params,
                local=local,
                stale_retention=stale_retention,
                time_min=time_min,
                time_max=time_max,
                xshapes=xshapes,
            )

        start_time = time.time()
        metrics = {"upload_bytes": 0, "download_bytes": 0, "successes": []}

        hone.logger.debug(
            f"Starting gather for window {window} with time window: {time_min} to {time_max}"
        )
        hone.logger.debug(
            f"Gather operation - my_uid: {my_uid}, window: {window}, key: {key}, timeout: {timeout}"
        )
        hone.log_with_context(
            level="debug",
            message=f"Target UIDs for gathering: {uids}",
            current_window=window,
        )

        aggregated_state_dict = {}
        valid_uids = []
        skipped_uids = []
        skip_reasons: dict[int, str] = {}
        global_steps = []
        # FU4: per-UID ``UploadMetadata.to_wire()`` dict captured at gather
        # time so the validator can populate ``_peer_c_tokens`` /
        # ``_peer_c_steps`` BEFORE the parallel outer_step dispatch.
        # Without this, ``token_weighted_aggregation`` was incompatible
        # with ``parallel_outer_step`` because the eval-loop metadata
        # read raced with the outer_step kernels. Peers without a
        # ``metadata`` field (legacy / malformed) are silently skipped
        # to keep parity with the pre-FU4 path.
        metadata_per_uid: dict[int, dict] = {}

        # Ensure deterministic order across processes/ranks
        uids = sorted(uids)

        async def _fetch_uid(uid: int) -> CommsGetResult | None:
            return await self.get_with_retry(
                uid=str(uid),
                window=window,
                key=key,
                timeout=timeout,
                local=local,
                stale_retention=stale_retention,
                time_min=time_min,
                time_max=time_max,
                map_location=device,
            )

        async with self.gather_semaphore:
            # P3: per-UID task map so we can resolve a future back to its UID
            # when using ``asyncio.wait(FIRST_COMPLETED)`` (as_completed loses
            # the per-task identity we need). All tasks start concurrently;
            # the early-fire loop below cancels pending tasks once the grace
            # deadline expires.
            tasks: dict[asyncio.Task, int] = {
                asyncio.create_task(_fetch_uid(uid)): uid for uid in uids
            }
            pending: set[asyncio.Task] = set(tasks.keys())
            responses_by_uid: dict[int, Any] = {}

            try:
                download_start = hone.T()
                # Overall timeout for all downloads: 10 minutes max
                # This prevents indefinite hangs from rate limiting or network issues
                gather_timeout = 600  # 10 minutes
                # P3: K_quorum throughput trigger. ``gather_min_quorum=0``
                # (default in hparams.json) disables the early-fire loop and
                # falls back to "wait for all tasks up to 600s", byte-for-byte
                # the pre-P3 behaviour. Operator flips to 4 after fleet soak
                # per plan line 139. The overall 600s gather_timeout is the
                # hard ceiling; K_quorum + grace just triggers EARLIER success.
                k_quorum = int(
                    getattr(self.hparams, "gather_min_quorum", 0)
                    if self.hparams is not None
                    else 0
                )
                start_t_monotonic = time.monotonic()
                successful = 0
                quorum_hit_time: float | None = None
                grace_seconds: float = 0.0
                deadline_by_quorum_grace: float | None = None

                while pending:
                    now = time.monotonic()
                    elapsed = now - start_t_monotonic
                    overall_remaining = float(gather_timeout) - elapsed
                    if overall_remaining <= 0.0:
                        break
                    if deadline_by_quorum_grace is not None:
                        grace_remaining = deadline_by_quorum_grace - now
                        if grace_remaining <= 0.0:
                            break
                        wait_timeout = min(grace_remaining, overall_remaining)
                    else:
                        wait_timeout = overall_remaining

                    done, pending = await asyncio.wait(
                        pending,
                        timeout=wait_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        # Outer timeout fired before any completion.
                        break

                    for task in done:
                        task_uid = tasks[task]
                        try:
                            resp = task.result()
                        except asyncio.CancelledError:
                            responses_by_uid[task_uid] = None
                            continue
                        except Exception as exc:
                            responses_by_uid[task_uid] = exc
                            continue
                        responses_by_uid[task_uid] = resp
                        # Count a response as "successful" for the quorum
                        # count iff the peer returned a non-None payload.
                        # Validation (quant params, shapes, etc.) happens
                        # below — if validation fails post-hoc the UID just
                        # gets added to skipped_uids, identical to the
                        # pre-P3 contract.
                        if resp is not None:
                            successful += 1
                            if (
                                k_quorum > 0
                                and successful >= k_quorum
                                and quorum_hit_time is None
                            ):
                                quorum_hit_time = time.monotonic()
                                grace_seconds = self._compute_grace_window_seconds()
                                deadline_by_quorum_grace = (
                                    quorum_hit_time + grace_seconds
                                )
                                hone.logger.info(
                                    f"[gather] K_quorum={k_quorum} hit at "
                                    f"{quorum_hit_time - start_t_monotonic:.1f}s; "
                                    f"grace deadline in {grace_seconds:.1f}s"
                                )

                # Cancel stragglers (grace-window-cut peers). Their slots
                # are filled either by whatever already landed or, below in
                # ``gather_with_reserve``, by the reserve tier.
                for task in pending:
                    if not task.done():
                        task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                for task in pending:
                    task_uid = tasks[task]
                    if task_uid not in responses_by_uid:
                        responses_by_uid[task_uid] = None

                # Record EMAs + surface per-gather telemetry on the Comms
                # instance for validator's report_window feed-forward.
                if k_quorum > 0 and quorum_hit_time is not None:
                    self._record_quorum_time(
                        quorum_hit_time - start_t_monotonic
                    )
                    self._last_gather_quorum_seconds = float(
                        quorum_hit_time - start_t_monotonic
                    )
                    self._last_gather_grace_seconds = float(grace_seconds)
                    self._last_window_quorum_hit = True
                else:
                    self._last_gather_quorum_seconds = None
                    self._last_gather_grace_seconds = None
                    self._last_window_quorum_hit = k_quorum == 0 and successful > 0

                total_elapsed = time.monotonic() - start_t_monotonic
                _quorum_log = (
                    f"{(quorum_hit_time - start_t_monotonic):.1f}s"
                    if quorum_hit_time is not None
                    else "-"
                )
                hone.logger.info(
                    f"[gather] uids={len(uids)} successful={successful} "
                    f"elapsed={total_elapsed:.1f}s "
                    f"quorum_hit={_quorum_log} grace={grace_seconds:.1f}s "
                    f"ema_step={self._ema_step_seconds:.1f}s "
                    f"ema_quorum={self._ema_quorum_seconds:.1f}s "
                    f"fragmented=False"
                )

                # Hysteresis: if K_quorum is armed and we didn't land enough
                # successful fetches inside the overall 600s budget, abort
                # the window (mirrors the pre-P3 "return None on timeout"
                # semantics). When K_quorum=0 this is a no-op; we fall
                # through to the "if not valid_uids: return None" check.
                if k_quorum > 0 and successful < k_quorum:
                    hone.logger.error(
                        f"Gather K_quorum={k_quorum} not hit within "
                        f"{gather_timeout}s (successful={successful}). "
                        f"Aborting gather."
                    )
                    return None

                hone.logger.info(
                    f"{hone.P(window, hone.T() - download_start)} Downloaded peer gradients <--"
                )
                process_start = hone.T()
                # Preserve the pre-P3 iteration order (sorted UIDs) so rank
                # positions in ``aggregated_state_dict[k]`` lists stay
                # deterministic across processes.
                for uid in uids:
                    response = responses_by_uid.get(uid)
                    received_compressed_params = set()

                    if isinstance(response, Exception):
                        hone.log_with_context(
                            level="debug",
                            message=f"Error from UID {uid}: {str(response)}",
                            current_window=window,
                        )
                        skipped_uids.append(uid)
                        skip_reasons[uid] = f"error response: {str(response)[:200]}"
                        continue
                    if response is None:
                        hone.logger.info(f"Skipped UID {uid} - gradient not found.")
                        skipped_uids.append(uid)
                        skip_reasons[uid] = "gradient not found"
                        continue

                    try:
                        # This is where get response uses the step
                        response = cast(CommsGetResult, response)
                        state_dict_resp, global_step_resp = (
                            response.data,
                            response.global_step,
                        )
                        hone.logger.debug(
                            f"Received state dict and global step {global_step_resp} from UID {uid}"
                        )
                    except (TypeError, ValueError) as e:
                        hone.log_with_context(
                            level="debug",
                            message=f"Invalid response from UID {uid}: {e}",
                            current_window=window,
                        )
                        skipped_uids.append(uid)
                        skip_reasons[uid] = f"invalid response: {str(e)[:200]}"
                        continue

                    if state_dict_resp is None:
                        hone.logger.debug(f"Empty state dict from UID {uid}")
                        skipped_uids.append(uid)
                        skip_reasons[uid] = "empty state dict"
                        continue

                    # ---------- Begin Compressed Indices and Values Check ----------
                    valid_response = True
                    validation_failure = ""
                    for param_name, tensor in state_dict_resp.items():
                        received_compressed_params.add(param_name)

                        # FU1 (P6b) TurboQuant payload. The legacy validation
                        # below (12-bit index unpack, ``totalks`` lookup, qparams
                        # tuple shape) doesn't apply to the ``tq_idxs`` /
                        # ``tq_codes`` / ``tq_meta`` triple: tq_idxs is a raw
                        # int64 top-K index tensor (NOT a 12-bit packed
                        # uint8), tq_codes is a raw uint8 codebook tensor
                        # (no qparams companion), and tq_meta is a Python
                        # dict (not a 5/7-tuple). Skip the legacy suffix-
                        # based checks for these keys; the downstream
                        # ``batch_decompress_turboquant`` in ``outer_step``
                        # is what actually consumes them and will reject
                        # malformed payloads loudly. Matches how the legacy
                        # ``maybe_dequantize_values`` path tolerates a
                        # missing / non-tuple qparams on vals-only uploads.
                        if (
                            param_name.endswith("tq_idxs")
                            or param_name.endswith("tq_codes")
                            or param_name.endswith("tq_meta")
                        ):
                            continue

                        # ----------------------------------------------------------
                        # (1)  Validate quantisation parameters themselves
                        # ----------------------------------------------------------
                        if param_name.endswith("quant_params"):
                            # P0b: qparams may be a 5-tuple (legacy raw uint8
                            # values) or a 7-tuple when the sender opted into
                            # 2-bit value packing. The tail (pack_version,
                            # original_last_dim) is consumed by ``_dequantize_values``;
                            # the validator only needs shift/scale/lookup
                            # finiteness checks here.
                            shift, scale, offset, lookup, dtype, *_extras = tensor
                            if (
                                (not torch.isfinite(shift))
                                or isinstance(scale, float)
                                and (
                                    not math.isfinite(scale)
                                    or abs(scale) < 1e-12
                                    or abs(scale) > 1e4
                                )
                            ):
                                hone.logger.warning(
                                    f"Bad quant‑params in {param_name} from UID {uid}; "
                                    f"shift={shift}, scale={scale}"
                                )
                                validation_failure = f"bad quant params in {param_name}"
                                valid_response = False
                                break
                            if torch.is_tensor(lookup) and (
                                not torch.isfinite(lookup).all()
                            ):
                                hone.logger.warning(
                                    f"Lookup table contains non‑finite values in {param_name} "
                                    f"from UID {uid}"
                                )
                                validation_failure = f"non-finite lookup table in {param_name}"
                                valid_response = False
                                break

                        if param_name.endswith("idxs"):
                            base_name = param_name[:-4]
                            totalk_value = totalks.get(base_name)
                            if totalk_value is None:
                                hone.logger.warning(
                                    f"Missing totalk for parameter {base_name} from UID {uid}, skipping UID."
                                )
                                validation_failure = f"missing totalk for {base_name}"
                                valid_response = False
                                break
                            # totalks stores integers, not tensors
                            totalk = (
                                totalk_value
                                if isinstance(totalk_value, int)
                                else totalk_value.numel()
                            )
                            # Get corresponding vals tensor for 12-bit unpacking.
                            # qparams may be 5-tuple (legacy raw uint8 vals) or
                            # 7-tuple (PACK_VERSION_2BIT). The 2-bit-packed
                            # path needs the original_last_dim from qparams to
                            # know how many indices the row encodes.
                            vals_tensor = state_dict_resp.get(base_name + "vals", None)
                            qparams_tensor = state_dict_resp.get(
                                base_name + "quant_params", None
                            )
                            try:
                                self.check_compressed_indices(
                                    param_name,
                                    tensor,
                                    totalk,
                                    allowed_topk=self.hparams.topk_compression,
                                    vals=vals_tensor,
                                    qparams=qparams_tensor,
                                )
                            except Exception as e:
                                hone.logger.warning(
                                    f"Compressed indices check failed for parameter {param_name} from UID {uid}: {e}"
                                )
                                validation_failure = f"compressed indices check failed for {param_name}: {str(e)[:150]}"
                                valid_response = False
                                break
                        # Check if values are valid (not NaN, not Inf) - validate without dequantizing
                        elif param_name.endswith("vals"):
                            # Only move to device for validation if needed
                            if tensor.dtype == torch.uint8:
                                if tensor.nelement() == 0:
                                    hone.logger.warning(
                                        f"Empty tensor in {param_name} from UID {uid}, skipping"
                                    )
                                    validation_failure = f"empty tensor in {param_name}"
                                    valid_response = False
                                    break
                            else:
                                tensor_to_check = tensor.to(device)
                                if (
                                    torch.isnan(tensor_to_check).any()
                                    or torch.isinf(tensor_to_check).any()
                                ):
                                    hone.logger.warning(
                                        f"NaN/Inf in {param_name} from UID {uid}, skipping"
                                    )
                                    validation_failure = f"NaN/Inf in {param_name}"
                                    valid_response = False
                                    break
                                # Clean up temporary tensor
                                del tensor_to_check

                            # ------------------------------------------------------
                            # (2)  Only validate quantization params exist, don't dequantize
                            # ------------------------------------------------------
                            qparams = state_dict_resp.get(
                                param_name[:-4] + "quant_params", None
                            )
                            if qparams is None and tensor.dtype == torch.uint8:
                                hone.logger.warning(
                                    f"Missing quant_params for quantized {param_name} from UID {uid}"
                                )
                                validation_failure = f"missing quant_params for {param_name}"
                                valid_response = False
                                break

                            # ------------------------------------------------------
                            # (3)  Validate vals shape matches expected model shape
                            # ------------------------------------------------------
                            if xshapes is not None:
                                base_name = param_name[:-4]  # Remove "vals" suffix
                                expected_shape = xshapes.get(base_name)
                                if expected_shape is not None:
                                    # After compression:
                                    # - 4D xshape [y, x, h, w] -> rearrange to [y, x, h*w] -> vals [y, x, topk]
                                    # - 2D xshape [x, w] -> no rearrange -> vals [x, topk]
                                    # So vals.shape[:-1] should match xshape[:2] for 4D, xshape[:1] for 2D
                                    if len(expected_shape) == 4:
                                        expected_vals_prefix = expected_shape[:2]
                                    else:
                                        expected_vals_prefix = expected_shape[:1]
                                    actual_vals_prefix = tensor.shape[:-1]
                                    if tuple(actual_vals_prefix) != tuple(
                                        expected_vals_prefix
                                    ):
                                        hone.logger.warning(
                                            f"Shape mismatch for {param_name} from UID {uid}: "
                                            f"expected shape prefix {expected_vals_prefix}, "
                                            f"got {actual_vals_prefix} (likely sharded gradient)"
                                        )
                                        validation_failure = (
                                            f"shape mismatch for {param_name}: "
                                            f"expected {expected_vals_prefix}, got {actual_vals_prefix}"
                                        )
                                        valid_response = False
                                        break

                    # FU1: a TurboQuant-encoded param ships ``cname +
                    # {'tq_idxs','tq_codes','tq_meta'}`` INSTEAD of
                    # ``cname + {'idxs','vals','quant_params'}``. The
                    # ``expected_compressed_params`` set (built by
                    # ``Trainer.get_expected_params`` from the legacy
                    # suffix triple) would therefore flag every
                    # TurboQuant-encoded param as missing. Expand
                    # received_compressed_params so each tq_ triple
                    # "satisfies" the legacy expectation for the same
                    # canonical name. This keeps a TurboQuant-only peer
                    # valid without changing the expected set (and a
                    # mixed peer stays valid on both codecs for free).
                    virtual_received = set(received_compressed_params)
                    for rk in received_compressed_params:
                        for tq_suffix, legacy_suffix in (
                            ("tq_idxs", "idxs"),
                            ("tq_codes", "vals"),
                            ("tq_meta", "quant_params"),
                        ):
                            if rk.endswith(tq_suffix):
                                base = rk[: -len(tq_suffix)]
                                virtual_received.add(base + legacy_suffix)
                                break

                    missing_params = (
                        expected_compressed_params - virtual_received
                    )
                    if missing_params:
                        hone.logger.warning(
                            f"UID {uid} missing compressed parameters: {missing_params}, skipping UID."
                        )
                        validation_failure = f"missing compressed parameters: {sorted(missing_params)}"
                        valid_response = False

                    if not valid_response:
                        hone.logger.info(
                            f"Skipping UID {uid} due to validation failures"
                        )
                        skipped_uids.append(uid)
                        skip_reasons[uid] = validation_failure or "validation failure"
                        continue
                    # ---------- End Compressed Indices and Values Check ----------

                    # Process tensors - keep everything quantized to save memory
                    # FU1: a key like ``cname + "tq_idxs"`` also ends with
                    # ``"idxs"``, so the legacy ``endswith("idxs")`` arm
                    # would catch it — BUT its payload bytes still need
                    # accounting and we want the comms contract to be
                    # explicit about which codec we're routing. The
                    # dedicated ``tq_*`` arms below handle the three
                    # TurboQuant keys before the legacy arms see them;
                    # ``tq_codes`` / ``tq_meta`` don't match any legacy
                    # suffix so they'd be silently dropped otherwise.
                    for param_name, tensor in state_dict_resp.items():
                        # FU1 TurboQuant payload ----------------------------------------------
                        if param_name.endswith("tq_idxs"):
                            aggregated_state_dict.setdefault(param_name, []).append(
                                tensor
                            )
                            if isinstance(tensor, torch.Tensor):
                                metrics["download_bytes"] += (
                                    tensor.element_size() * tensor.nelement()
                                )
                            continue
                        if param_name.endswith("tq_codes"):
                            aggregated_state_dict.setdefault(param_name, []).append(
                                tensor
                            )
                            if isinstance(tensor, torch.Tensor):
                                metrics["download_bytes"] += (
                                    tensor.element_size() * tensor.nelement()
                                )
                            continue
                        if param_name.endswith("tq_meta"):
                            # tq_meta is a Python dict (not a tensor).
                            # torch.save pickles it; we pass the object
                            # through unchanged so ``outer_step``'s
                            # decode branch can feed it to
                            # ``dequantize_turboquant`` as-is.
                            aggregated_state_dict.setdefault(param_name, []).append(
                                tensor
                            )
                            continue

                        # 1️⃣  Indices are kept as‑is -----------------------------------------
                        if param_name.endswith("idxs"):
                            aggregated_state_dict.setdefault(param_name, []).append(
                                tensor
                            )
                            # Handle 12-bit packed format (uint8 tensor)
                            metrics["download_bytes"] += (
                                tensor.element_size() * tensor.nelement()
                            )

                        # 2️⃣  Values → keep quantized, store with quant_params ---------------
                        elif param_name.endswith("vals"):
                            # Keep values quantized - just store the raw tensor
                            aggregated_state_dict.setdefault(param_name, []).append(
                                tensor  # Keep original dtype (uint8 if quantized)
                            )
                            metrics["download_bytes"] += (
                                tensor.element_size() * tensor.nelement()
                            )

                        # 3️⃣  Store quantization parameters for later use --------------------
                        elif param_name.endswith("quant_params"):
                            aggregated_state_dict.setdefault(param_name, []).append(
                                tensor
                            )

                    # FU4: capture per-peer ``UploadMetadata.to_wire()`` dict
                    # before it gets dropped on the floor below. Legacy peers
                    # (pre-P2 miners) don't ship this field; ``meta`` will be
                    # ``None`` for them and the UID is simply absent from
                    # ``metadata_per_uid``. Validator-side
                    # ``token_weighted_aggregation`` falls back to a uniform
                    # weight for any UID missing here, identical to the
                    # pre-FU4 contract.
                    meta = state_dict_resp.get("metadata")
                    if isinstance(meta, dict):
                        metadata_per_uid[uid] = meta
                    else:
                        hone.logger.info(
                            f"[gather] uid={uid} no metadata field in payload "
                            f"(legacy or malformed); skipping metadata_per_uid"
                        )

                    valid_uids.append(uid)
                    global_steps.append(global_step_resp)

                hone.logger.info(
                    f"{hone.P(window, hone.T() - process_start)} Processed peer gradients <--"
                )

            except Exception as e:
                hone.logger.error(f"Error processing uid batch: {str(e)}")

        if not valid_uids:
            hone.logger.info("No valid gradients received from any UID")
            return None

        total_time = time.time() - start_time
        # P1 R2 amplification telemetry (legacy single-blob branch). Op-side
        # comparison against the fragmented branch's same line — risk #6 in
        # the plan's register tracks R2 P95 latency once 24× fan-out lands.
        hone.logger.info(
            f"[gather] r2_fetches total={len(uids)} successful={len(valid_uids)} "
            f"wall_clock_seconds={total_time:.2f} fragmented=False"
        )
        hone.logger.info(
            f"Gather done in {total_time:.2f}s. Success rate: {len(valid_uids)}/{len(uids)}, "
            f"Upload: {metrics['upload_bytes']} bytes, Download: {metrics['download_bytes']} bytes"
        )

        # ``fragments_seen_per_uid`` and ``uids_per_param`` are populated only
        # by the fragmented branch (see ``_gather_fragmented``). Here we set
        # the canonical "every UID has 1 fragment, every param sees all valid
        # UIDs in order" so callers like ``check_uid_index_overlap`` can use
        # the same lookup pattern in both modes without branching.
        uids_per_param: dict[str, list[int]] = {
            k: list(valid_uids) for k in aggregated_state_dict.keys()
        }
        fragments_seen_per_uid: dict[int, set[int]] = {
            uid: {0} for uid in valid_uids
        }

        result = SimpleNamespace(
            time=total_time,
            upload_bytes=metrics["upload_bytes"],
            download_bytes=metrics["download_bytes"],
            success_rate=len(valid_uids) / len(uids),
            state_dict=SimpleNamespace(**aggregated_state_dict),
            uids=valid_uids,
            global_steps=global_steps,
            skipped_uids=skipped_uids,
            skip_reasons=skip_reasons,
            fragments_seen_per_uid=fragments_seen_per_uid,
            uids_per_param=uids_per_param,
            fragment_byte_counts=[],  # populated only in fragmented mode
            # FU4: per-UID ``UploadMetadata.to_wire()`` dict. Used by the
            # validator to populate ``_peer_c_tokens`` / ``_peer_c_steps``
            # immediately after gather, BEFORE outer_step dispatch — so
            # ``token_weighted_aggregation`` and ``parallel_outer_step``
            # can finally coexist.
            metadata_per_uid=metadata_per_uid,
        )
        return result

    # ------------------------------------------------------------------
    # _gather_fragmented –– per-(uid, fragment) fan-out for streaming uploads
    # ------------------------------------------------------------------
    async def _gather_fragmented(
        self,
        *,
        my_uid: int | None,
        uids: list[int],
        window: int,
        key: str,
        timeout: int,
        device: str,
        totalks: dict[str, torch.Tensor],
        compressor: TopKCompressor,
        num_fragments: int,
        expected_compressed_params: set[str],
        local: bool,
        stale_retention: int,
        time_min: datetime | None,
        time_max: datetime | None,
        xshapes: dict[str, tuple] | None,
    ) -> SimpleNamespace | None:
        """Fragmented sibling of ``gather``.

        Fans out ``num_fragments × len(uids)`` GETs against per-fragment keys
        ``f"{key}-frag{i:02d}"``. Each fragment file contributes a *disjoint*
        subset of (idxs, vals, quant_params) triplets per the upstream
        ``prepare_gradient_buckets`` partition. Per-fragment validation is
        identical to the legacy per-blob check; an invalid fragment drops
        only that fragment, leaving the rest of the UID's contributions intact.

        Returns the same ``SimpleNamespace`` shape as ``gather`` plus:
          * ``fragments_seen_per_uid: dict[int, set[int]]`` — the fragment
            indexes each UID successfully uploaded.
          * ``uids_per_param: dict[str, list[int]]`` — for each tensor key in
            ``state_dict``, the UIDs that contributed (in the same order as
            the per-key tensor list). Required by ``check_uid_index_overlap``
            to drop the broken positional ``idxs_all[i]==uids[i]`` assumption.
        """
        start_time = time.time()
        metrics = {"upload_bytes": 0, "download_bytes": 0, "successes": []}

        hone.logger.info(
            f"[gather] fragmented start window={window} key={key} "
            f"num_fragments={num_fragments} peers={len(uids)} "
            f"total_fetches={len(uids) * num_fragments}"
        )

        uids = sorted(uids)
        # Flat fan-out: every (uid, frag_idx) gets its own task. Out-of-order
        # arrival is fine — we group results by uid below before validating.
        fragment_keys = [f"{key}-frag{i:02d}" for i in range(num_fragments)]

        async def _fetch(uid: int, frag_idx: int) -> CommsGetResult | None:
            return await self.get_with_retry(
                uid=str(uid),
                window=window,
                key=fragment_keys[frag_idx],
                timeout=timeout,
                local=local,
                stale_retention=stale_retention,
                time_min=time_min,
                time_max=time_max,
                map_location=device,
            )

        fetch_specs: list[tuple[int, int]] = [
            (uid, frag_idx) for uid in uids for frag_idx in range(num_fragments)
        ]

        async with self.gather_semaphore:
            try:
                download_start = hone.T()
                gather_timeout = 600  # 10 minutes — same budget as legacy
                # P3: K_quorum throughput trigger — default-off
                # (``gather_min_quorum=0`` in hparams.json reproduces today's
                # "wait for all tasks up to 600s" semantics). In fragmented
                # mode, "successful" means "the UID returned a non-None
                # response for at least one fragment"; per-fragment
                # validation happens unchanged below so partially-uploaded
                # peers still count toward quorum and their validated
                # fragments land in the aggregated state_dict.
                k_quorum = int(
                    getattr(self.hparams, "gather_min_quorum", 0)
                    if self.hparams is not None
                    else 0
                )
                fetch_tasks: dict[asyncio.Task, tuple[int, int]] = {
                    asyncio.create_task(_fetch(uid, frag_idx)): (uid, frag_idx)
                    for uid, frag_idx in fetch_specs
                }
                pending: set[asyncio.Task] = set(fetch_tasks.keys())
                responses_by_uid: dict[int, list[tuple[int, Any]]] = {}
                peers_with_hit: set[int] = set()
                start_t_monotonic = time.monotonic()
                quorum_hit_time: float | None = None
                grace_seconds: float = 0.0
                deadline_by_quorum_grace: float | None = None

                while pending:
                    now = time.monotonic()
                    elapsed = now - start_t_monotonic
                    overall_remaining = float(gather_timeout) - elapsed
                    if overall_remaining <= 0.0:
                        break
                    if deadline_by_quorum_grace is not None:
                        grace_remaining = deadline_by_quorum_grace - now
                        if grace_remaining <= 0.0:
                            break
                        wait_timeout = min(grace_remaining, overall_remaining)
                    else:
                        wait_timeout = overall_remaining

                    done, pending = await asyncio.wait(
                        pending,
                        timeout=wait_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        break

                    for task in done:
                        task_uid, task_frag = fetch_tasks[task]
                        try:
                            resp = task.result()
                        except asyncio.CancelledError:
                            responses_by_uid.setdefault(task_uid, []).append(
                                (task_frag, None)
                            )
                            continue
                        except Exception as exc:
                            responses_by_uid.setdefault(task_uid, []).append(
                                (task_frag, exc)
                            )
                            continue
                        responses_by_uid.setdefault(task_uid, []).append(
                            (task_frag, resp)
                        )
                        # Count this peer as contributing to quorum on the
                        # first non-None fragment response. Post-hoc per-
                        # fragment validation may still drop this fragment
                        # (and even all of the peer's fragments) — the
                        # peer then falls into ``skipped_uids``, same
                        # semantics as the legacy path.
                        if (
                            resp is not None
                            and task_uid not in peers_with_hit
                        ):
                            peers_with_hit.add(task_uid)
                            if (
                                k_quorum > 0
                                and len(peers_with_hit) >= k_quorum
                                and quorum_hit_time is None
                            ):
                                quorum_hit_time = time.monotonic()
                                grace_seconds = self._compute_grace_window_seconds()
                                deadline_by_quorum_grace = (
                                    quorum_hit_time + grace_seconds
                                )
                                hone.logger.info(
                                    f"[gather] K_quorum={k_quorum} hit at "
                                    f"{quorum_hit_time - start_t_monotonic:.1f}s; "
                                    f"grace deadline in {grace_seconds:.1f}s "
                                    f"(fragmented)"
                                )

                # Cancel stragglers — peers who haven't finished uploading
                # fragments by the grace deadline just contribute whatever
                # landed in time. ``gather_with_reserve`` may promote
                # reserve peers if fewer than ``K_safety`` UIDs made the
                # cutoff.
                for task in pending:
                    if not task.done():
                        task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                for task in pending:
                    task_uid, task_frag = fetch_tasks[task]
                    existing = responses_by_uid.setdefault(task_uid, [])
                    if not any(f == task_frag for f, _ in existing):
                        existing.append((task_frag, None))

                # Record EMAs + surface per-gather telemetry on the Comms
                # instance for validator's report_window feed-forward.
                successful_peers = len(peers_with_hit)
                if k_quorum > 0 and quorum_hit_time is not None:
                    self._record_quorum_time(
                        quorum_hit_time - start_t_monotonic
                    )
                    self._last_gather_quorum_seconds = float(
                        quorum_hit_time - start_t_monotonic
                    )
                    self._last_gather_grace_seconds = float(grace_seconds)
                    self._last_window_quorum_hit = True
                else:
                    self._last_gather_quorum_seconds = None
                    self._last_gather_grace_seconds = None
                    self._last_window_quorum_hit = (
                        k_quorum == 0 and successful_peers > 0
                    )

                total_elapsed = time.monotonic() - start_t_monotonic
                _quorum_log = (
                    f"{(quorum_hit_time - start_t_monotonic):.1f}s"
                    if quorum_hit_time is not None
                    else "-"
                )
                hone.logger.info(
                    f"[gather] uids={len(uids)} successful={successful_peers} "
                    f"elapsed={total_elapsed:.1f}s "
                    f"quorum_hit={_quorum_log} grace={grace_seconds:.1f}s "
                    f"ema_step={self._ema_step_seconds:.1f}s "
                    f"ema_quorum={self._ema_quorum_seconds:.1f}s "
                    f"fragmented=True num_fragments={num_fragments}"
                )

                if k_quorum > 0 and successful_peers < k_quorum:
                    hone.logger.error(
                        f"Fragmented gather K_quorum={k_quorum} not hit "
                        f"within {gather_timeout}s "
                        f"(successful_peers={successful_peers}). "
                        f"Aborting gather."
                    )
                    return None

                _total_responses = sum(
                    len(v) for v in responses_by_uid.values()
                )
                hone.logger.info(
                    f"{hone.P(window, hone.T() - download_start)} "
                    f"Downloaded {_total_responses} peer fragments <--"
                )
            except Exception as e:
                hone.logger.error(
                    f"Error during fragmented fetch dispatch: {e}", exc_info=True
                )
                return None

        process_start = hone.T()
        aggregated_state_dict: dict[str, list] = {}
        uids_per_param: dict[str, list[int]] = {}
        valid_uids: list[int] = []
        skipped_uids: list[int] = []
        skip_reasons: dict[int, str] = {}
        global_steps: list[int] = []
        fragments_seen_per_uid: dict[int, set[int]] = {}
        fragment_byte_counts: list[int] = []
        successful_fetches = 0
        # FU4: per-UID canonical ``UploadMetadata.to_wire()`` dict captured
        # at gather time. In fragmented mode the miner generates the
        # metadata once per window before ``prepare_gradient_buckets``
        # (see ``hone.neurons.miner`` ~L862-882), so every fragment of
        # a given UID carries the SAME dict. We pick the most-common
        # entry across that UID's valid fragments and WARN if any
        # disagree — disagreement means a buggy / malicious miner spliced
        # different metadata into different fragments and we want
        # operators to see it.
        metadata_per_uid: dict[int, dict] = {}

        # Iterate UIDs in sorted order so per-key tensor lists end up in a
        # deterministic order across ranks (the existing outer_step merge
        # iterates them positionally).
        for uid in uids:
            uid_responses = responses_by_uid.get(uid, [])
            # Each entry: (frag_idx, valid_state_dict_resp, global_step_resp,
            # received_param_keys, fragment_bytes)
            valid_fragments: list[tuple[int, dict, int, set[str], int]] = []

            for frag_idx, response in sorted(uid_responses, key=lambda t: t[0]):
                if isinstance(response, Exception):
                    hone.logger.debug(
                        f"[gather-frag] uid={uid} frag={frag_idx} error: {response}"
                    )
                    continue
                if response is None:
                    continue
                response = cast(CommsGetResult, response)
                state_dict_resp = response.data
                global_step_resp = response.global_step
                if state_dict_resp is None:
                    continue

                # Run the same per-tensor validation legacy gather does, but
                # scoped to this fragment only. A failed check drops the
                # fragment, not the whole UID — partial uploads are by design
                # in the streaming path.
                fragment_valid = True
                fragment_failure = ""
                received: set[str] = set()
                fragment_bytes = 0
                for param_name, tensor in state_dict_resp.items():
                    received.add(param_name)
                    # FU1 (P6b) TurboQuant payload — skip the legacy
                    # suffix-based validation; see the same comment in
                    # the legacy gather path above for rationale.
                    # ``tq_idxs`` matches ``endswith("idxs")`` in the
                    # legacy arm but is a raw int64 top-K index tensor,
                    # not a 12-bit-packed uint8 tensor, so the legacy
                    # unpack would fail — the dedicated skip keeps the
                    # fragment valid end-to-end.
                    if (
                        param_name.endswith("tq_idxs")
                        or param_name.endswith("tq_codes")
                        or param_name.endswith("tq_meta")
                    ):
                        if isinstance(tensor, torch.Tensor):
                            fragment_bytes += (
                                tensor.element_size() * tensor.nelement()
                            )
                        continue
                    if param_name.endswith("quant_params"):
                        # Mirror legacy 5-tuple/7-tuple unpack from comms.gather
                        # (P0b: pack_version + original_last_dim are the trailing
                        # two when 2-bit pack is on; we only verify shift/scale/
                        # lookup finiteness here, same as the legacy path).
                        shift, scale, _offset, lookup, _dtype, *_extras = tensor
                        if (
                            (not torch.isfinite(shift))
                            or (
                                isinstance(scale, float)
                                and (
                                    not math.isfinite(scale)
                                    or abs(scale) < 1e-12
                                    or abs(scale) > 1e4
                                )
                            )
                        ):
                            fragment_valid = False
                            fragment_failure = (
                                f"bad quant params in {param_name}"
                            )
                            break
                        if torch.is_tensor(lookup) and (
                            not torch.isfinite(lookup).all()
                        ):
                            fragment_valid = False
                            fragment_failure = (
                                f"non-finite lookup table in {param_name}"
                            )
                            break
                        continue

                    if param_name.endswith("idxs"):
                        base_name = param_name[:-4]
                        totalk_value = totalks.get(base_name)
                        if totalk_value is None:
                            fragment_valid = False
                            fragment_failure = f"missing totalk for {base_name}"
                            break
                        totalk = (
                            totalk_value
                            if isinstance(totalk_value, int)
                            else totalk_value.numel()
                        )
                        vals_tensor = state_dict_resp.get(
                            base_name + "vals", None
                        )
                        qparams_tensor = state_dict_resp.get(
                            base_name + "quant_params", None
                        )
                        try:
                            self.check_compressed_indices(
                                param_name,
                                tensor,
                                totalk,
                                allowed_topk=self.hparams.topk_compression,
                                vals=vals_tensor,
                                qparams=qparams_tensor,
                            )
                        except Exception as e:
                            fragment_valid = False
                            fragment_failure = (
                                f"compressed indices check failed for "
                                f"{param_name}: {str(e)[:120]}"
                            )
                            break
                        if isinstance(tensor, torch.Tensor):
                            fragment_bytes += (
                                tensor.element_size() * tensor.nelement()
                            )
                        continue

                    if param_name.endswith("vals"):
                        if isinstance(tensor, torch.Tensor):
                            if tensor.dtype == torch.uint8:
                                if tensor.nelement() == 0:
                                    fragment_valid = False
                                    fragment_failure = (
                                        f"empty tensor in {param_name}"
                                    )
                                    break
                            else:
                                tensor_to_check = tensor.to(device)
                                if (
                                    torch.isnan(tensor_to_check).any()
                                    or torch.isinf(tensor_to_check).any()
                                ):
                                    fragment_valid = False
                                    fragment_failure = f"NaN/Inf in {param_name}"
                                    del tensor_to_check
                                    break
                                del tensor_to_check
                            fragment_bytes += (
                                tensor.element_size() * tensor.nelement()
                            )
                        qparams = state_dict_resp.get(
                            param_name[:-4] + "quant_params", None
                        )
                        if (
                            qparams is None
                            and isinstance(tensor, torch.Tensor)
                            and tensor.dtype == torch.uint8
                        ):
                            fragment_valid = False
                            fragment_failure = (
                                f"missing quant_params for {param_name}"
                            )
                            break
                        if xshapes is not None:
                            base_name = param_name[:-4]
                            expected_shape = xshapes.get(base_name)
                            if expected_shape is not None and isinstance(
                                tensor, torch.Tensor
                            ):
                                if len(expected_shape) == 4:
                                    expected_vals_prefix = expected_shape[:2]
                                else:
                                    expected_vals_prefix = expected_shape[:1]
                                actual_vals_prefix = tensor.shape[:-1]
                                if tuple(actual_vals_prefix) != tuple(
                                    expected_vals_prefix
                                ):
                                    fragment_valid = False
                                    fragment_failure = (
                                        f"shape mismatch for {param_name}: "
                                        f"expected {expected_vals_prefix}, "
                                        f"got {actual_vals_prefix}"
                                    )
                                    break

                if not fragment_valid:
                    hone.logger.warning(
                        f"[gather-frag] uid={uid} frag={frag_idx} dropped: "
                        f"{fragment_failure}"
                    )
                    continue

                valid_fragments.append(
                    (
                        frag_idx,
                        state_dict_resp,
                        global_step_resp,
                        received,
                        fragment_bytes,
                    )
                )

            if not valid_fragments:
                # Either no fragment arrived or every fragment failed validation.
                # Treat this UID as fully skipped, mirroring the legacy contract.
                if uid in responses_by_uid and any(
                    r is None or isinstance(r, Exception)
                    for _, r in responses_by_uid[uid]
                ):
                    skip_reasons[uid] = "all fragments failed/missing"
                else:
                    skip_reasons[uid] = "no valid fragments"
                skipped_uids.append(uid)
                continue

            # Append this UID's contributions to the per-key tensor lists in
            # frag_idx order so the layout is deterministic across ranks.
            # Match the legacy gather contract: only compressed-tensor keys
            # (idxs/vals/quant_params, or the FU1 TurboQuant
            # tq_idxs/tq_codes/tq_meta triple) end up in
            # ``aggregated_state_dict`` — ``metadata`` rides along inside
            # each fragment file and is surfaced separately via
            # ``metadata_per_uid`` (FU4) for validator-side
            # ``token_weighted_aggregation`` consumption.
            for (
                frag_idx,
                state_dict_resp,
                global_step_resp,
                _received,
                fragment_bytes,
            ) in valid_fragments:
                fragment_byte_counts.append(fragment_bytes)
                metrics["download_bytes"] += fragment_bytes
                successful_fetches += 1
                fragments_seen_per_uid.setdefault(uid, set()).add(frag_idx)
                for param_name, tensor in state_dict_resp.items():
                    # FU1 TurboQuant keys: checked BEFORE the legacy
                    # ``endswith("idxs")`` arm because ``tq_idxs`` also
                    # matches ``idxs``. Without the explicit check the
                    # legacy arm would accept the tq_idxs into the
                    # aggregated dict but ``tq_codes`` / ``tq_meta``
                    # would still be dropped (neither ends with a legacy
                    # suffix) and the validator-side decode would fail
                    # with a KeyError. An explicit allowlist keeps the
                    # three-key triple coherent in the aggregated dict.
                    if (
                        param_name.endswith("tq_idxs")
                        or param_name.endswith("tq_codes")
                        or param_name.endswith("tq_meta")
                    ):
                        aggregated_state_dict.setdefault(param_name, []).append(
                            tensor
                        )
                        uids_per_param.setdefault(param_name, []).append(uid)
                        continue
                    if not (
                        param_name.endswith("idxs")
                        or param_name.endswith("vals")
                        or param_name.endswith("quant_params")
                    ):
                        continue
                    aggregated_state_dict.setdefault(param_name, []).append(
                        tensor
                    )
                    uids_per_param.setdefault(param_name, []).append(uid)

            # FU4: extract the canonical ``UploadMetadata.to_wire()`` for
            # this UID. Miners generate the dict once per window before
            # ``prepare_gradient_buckets`` so every fragment carries an
            # identical copy. We tally frozen-dict keys across the UID's
            # valid fragments and pick the most common entry; any
            # disagreement is logged at WARN. If a UID's fragments all
            # lack a metadata field (legacy / malformed), the UID is
            # absent from ``metadata_per_uid`` and the validator falls
            # back to a uniform weight for that peer.
            frag_metas: list[dict] = []
            for _frag_idx, frag_state, _gstep, _recv, _bytes in valid_fragments:
                m = frag_state.get("metadata")
                if isinstance(m, dict):
                    frag_metas.append(m)
            if frag_metas:
                # Use a hashable canonical form so we can tally identical
                # dicts. ``json.dumps(sort_keys=True)`` survives the
                # ``UploadMetadata.to_wire()`` payload (all values are
                # ints / strs / a stable str digest).
                tally: dict[str, tuple[int, dict]] = {}
                for m in frag_metas:
                    try:
                        canon = json.dumps(m, sort_keys=True, default=str)
                    except (TypeError, ValueError):
                        # A peer who shipped a non-JSON metadata payload
                        # is malformed; skip without crashing the gather.
                        continue
                    if canon not in tally:
                        tally[canon] = (1, m)
                    else:
                        cnt, ref = tally[canon]
                        tally[canon] = (cnt + 1, ref)
                if tally:
                    if len(tally) > 1:
                        hone.logger.warning(
                            f"[gather-frag] uid={uid} fragments disagree on "
                            f"metadata: {len(tally)} distinct dicts across "
                            f"{len(frag_metas)} fragment(s); using majority "
                            f"value (FU4)."
                        )
                    # Majority value (ties broken by first-seen order via
                    # dict insertion order, which is fragment-idx-sorted).
                    _best = max(tally.values(), key=lambda t: t[0])
                    metadata_per_uid[uid] = _best[1]
            else:
                hone.logger.info(
                    f"[gather-frag] uid={uid} no metadata in any fragment "
                    f"(legacy or malformed); skipping metadata_per_uid"
                )

            # Use the global_step of the *first* (lowest-frag-idx) successful
            # fragment for this UID. They should all be identical (one upload
            # round per window) but defending against malformed peers.
            valid_uids.append(uid)
            global_steps.append(valid_fragments[0][2])

        hone.logger.info(
            f"{hone.P(window, hone.T() - process_start)} "
            f"Processed peer fragments <--"
        )

        if not valid_uids:
            hone.logger.info("No valid fragmented gradients received from any UID")
            return None

        total_time = time.time() - start_time
        # P1 R2 amplification telemetry. The 24× fan-out is the headline risk
        # called out in the plan's risk register (#6); operators tail this
        # line to monitor R2 P95 latency before flipping ``fragmented_uploads``
        # default-on across the fleet.
        total_fetches = len(uids) * num_fragments
        hone.logger.info(
            f"[gather] r2_fetches total={total_fetches} "
            f"successful={successful_fetches} "
            f"wall_clock_seconds={total_time:.2f} fragmented=True "
            f"num_fragments={num_fragments}"
        )
        hone.logger.info(
            f"Fragmented gather done in {total_time:.2f}s. "
            f"Success rate: {len(valid_uids)}/{len(uids)} peers, "
            f"{successful_fetches}/{total_fetches} fragments. "
            f"Download: {metrics['download_bytes']} bytes"
        )

        result = SimpleNamespace(
            time=total_time,
            upload_bytes=metrics["upload_bytes"],
            download_bytes=metrics["download_bytes"],
            success_rate=len(valid_uids) / len(uids) if uids else 0.0,
            state_dict=SimpleNamespace(**aggregated_state_dict),
            uids=valid_uids,
            global_steps=global_steps,
            skipped_uids=skipped_uids,
            skip_reasons=skip_reasons,
            fragments_seen_per_uid=fragments_seen_per_uid,
            uids_per_param=uids_per_param,
            fragment_byte_counts=fragment_byte_counts,
            # FU4: per-UID ``UploadMetadata.to_wire()`` dict — same shape
            # as the legacy gather branch so callers can read it
            # uniformly across fragmented and non-fragmented modes.
            metadata_per_uid=metadata_per_uid,
        )
        return result

    # ------------------------------------------------------------------
    # gather_with_reserve –– retry skipped gather peers with reserve tier
    # ------------------------------------------------------------------
    async def gather_with_reserve(
        self,
        *,
        my_uid: int | None,
        gather_uids: list[int],
        reserve_uids: list[int],
        expected_compressed_params: set[str] | None = None,
        **kwargs,
    ) -> SimpleNamespace | None:
        """
        Gathers gradients with a fallback mechanism using a reserve set of UIDs.

        This method first attempts to gather gradients from a primary list of UIDs.
        If any of these fail, it replaces the failed UIDs with UIDs from a reserve
        list and retries the gather operation. The results from both attempts are
        merged to maximize the number of successfully collected gradients.

        Args:
            my_uid (int | None): The UID of the neuron performing the gather.
            gather_uids (list[int]): The primary list of UIDs to gather from.
            reserve_uids (list[int]): A list of fallback UIDs to use if the
                primary gather fails for some peers.
            expected_compressed_params (set[str] | None, optional): A set of
                parameter names expected in the compressed state dict. Defaults to None.
            **kwargs: Additional arguments to be passed to the `gather` method.

        Returns:
            SimpleNamespace | None: A merged namespace containing the aggregated
                state dict and metrics from both primary and reserve gathers, or
                None if no gradients could be collected.
        """
        if len(gather_uids + reserve_uids) == 0:
            return None

        if not expected_compressed_params:
            expected_compressed_params = set()

        window = kwargs.get("window", None)  # for contextual logs
        context_log = partial(hone.log_with_context, level="info", window=window)

        context_log(
            message=f"[gather_with_reserve] ⏩ start | "
            f"gather={gather_uids} reserve={reserve_uids}"
        )

        primary = await self.gather(
            my_uid=my_uid,
            uids=gather_uids,
            expected_compressed_params=expected_compressed_params,
            **kwargs,
        )

        if primary is None:
            primary = SimpleNamespace(
                time=0.0,
                upload_bytes=0,
                download_bytes=0,
                success_rate=0.0,
                state_dict=SimpleNamespace(),
                uids=[],
                global_steps=[],
                skipped_uids=gather_uids.copy(),
                skip_reasons={uid: "no valid gradients received" for uid in gather_uids},
                fragments_seen_per_uid={},
                uids_per_param={},
                fragment_byte_counts=[],
                # FU4: empty metadata dict so downstream callers can read
                # ``primary.metadata_per_uid`` unconditionally.
                metadata_per_uid={},
            )

        context_log(
            message=f"[gather_with_reserve] ✅ primary gather "
            f"{len(primary.uids)}/{len(gather_uids)} succeeded | "
            f"skipped={primary.skipped_uids}"
        )

        # ── 2. Retry the misses with reserve peers ─────────────────────
        # P3 rebalance: a UID is "missing" iff it contributed zero fragments.
        # With K_quorum cutting the primary gather short at the grace
        # deadline, ``primary.uids`` can be well below ``len(gather_uids)``
        # even on a healthy fleet. The reserve's purpose is to backfill to
        # at least ``K_safety`` peers so the per-peer 1/K_safety cap in
        # ``_apply_k_safety_cap`` has enough peers to redistribute mass
        # to. Above that floor the reserve adds no further safety value
        # (the cap already constrains the dominant peer), so we skip the
        # reserve call entirely and avoid the second R2 round-trip.
        #
        # Below ``K_quorum`` (i.e. ``primary is None``), gather already
        # aborted up in ``self.gather`` and returned the empty primary
        # above. Reserve cannot recover from a missed quorum — that's a
        # hard window skip. So the reserve window of relevance is
        # ``K_quorum <= successful < K_safety``.
        k_safety = int(
            getattr(self.hparams, "gather_safety_cap", 0)
            if self.hparams is not None
            else 0
        )
        missing = set(gather_uids) - set(primary.uids)
        if (
            missing
            and reserve_uids
            and (k_safety <= 0 or len(primary.uids) < k_safety)
        ):
            # P3: only promote enough reserve peers to reach K_safety (no
            # point pulling the whole reserve tier when the cap already
            # guarantees no single peer exceeds 1/K_safety). When
            # ``k_safety <= 0`` the cap is disabled and we fall back to
            # "fill every missing slot" for byte-for-byte parity with
            # the pre-P3 path.
            if k_safety > 0:
                shortfall = max(k_safety - len(primary.uids), 0)
                reserve_budget = min(len(missing), shortfall)
            else:
                reserve_budget = len(missing)
            replacements = [uid for uid in reserve_uids if uid not in primary.uids][
                :reserve_budget
            ]

            if replacements:
                context_log(
                    message=f"[gather_with_reserve] 🔄 retrying with reserve "
                    f"uids={replacements}"
                )
                fallback = await self.gather(
                    my_uid=my_uid,
                    uids=replacements,
                    **kwargs,
                )
                if fallback:
                    # merge tensor‑lists inside the nested state_dict
                    for k, v in vars(fallback.state_dict).items():
                        merged = getattr(primary.state_dict, k, []) + v
                        setattr(primary.state_dict, k, merged)

                    primary.uids.extend(fallback.uids)
                    primary.global_steps.extend(fallback.global_steps)
                    primary.skipped_uids.extend(fallback.skipped_uids)
                    primary.skip_reasons.update(fallback.skip_reasons)
                    primary.upload_bytes += fallback.upload_bytes
                    primary.download_bytes += fallback.download_bytes

                    # P1: keep the per-UID and per-param dicts consistent
                    # across primary + reserve so ``check_uid_index_overlap``
                    # sees the merged contributor map. Order-merge into the
                    # per-key list so ``uids_per_param[k][i]`` still aligns
                    # with the i-th tensor in ``state_dict.<k>``.
                    primary_fspu = getattr(primary, "fragments_seen_per_uid", {})
                    fallback_fspu = getattr(fallback, "fragments_seen_per_uid", {})
                    if primary_fspu or fallback_fspu:
                        merged_fspu = dict(primary_fspu)
                        for uid, frags in fallback_fspu.items():
                            merged_fspu.setdefault(uid, set()).update(frags)
                        primary.fragments_seen_per_uid = merged_fspu

                    primary_upp = getattr(primary, "uids_per_param", {})
                    fallback_upp = getattr(fallback, "uids_per_param", {})
                    if primary_upp or fallback_upp:
                        merged_upp = {k: list(v) for k, v in primary_upp.items()}
                        for k, v in fallback_upp.items():
                            merged_upp.setdefault(k, []).extend(v)
                        primary.uids_per_param = merged_upp

                    primary_fbc = getattr(primary, "fragment_byte_counts", [])
                    fallback_fbc = getattr(fallback, "fragment_byte_counts", [])
                    primary.fragment_byte_counts = list(primary_fbc) + list(
                        fallback_fbc
                    )

                    # FU4: merge per-UID metadata dicts so the validator
                    # gets ``c_tokens`` / ``c_steps`` for both primary and
                    # reserve peers in one place. Reserve UIDs are
                    # disjoint from primary by construction
                    # (``replacements = [uid for uid in reserve_uids if
                    # uid not in primary.uids]`` above), so a plain
                    # ``update`` is collision-free.
                    primary_mpu = getattr(primary, "metadata_per_uid", {})
                    fallback_mpu = getattr(fallback, "metadata_per_uid", {})
                    if primary_mpu or fallback_mpu:
                        merged_mpu = dict(primary_mpu)
                        merged_mpu.update(fallback_mpu)
                        primary.metadata_per_uid = merged_mpu

                    context_log(
                        message=f"[gather_with_reserve] ✅ reserve gather "
                        f"{len(fallback.uids)}/{len(replacements)} "
                        f"succeeded | skipped={fallback.skipped_uids}"
                    )

        # recompute success‑rate with respect to the *original* gather tier
        target = len(gather_uids)
        primary.success_rate = len(primary.uids) / target if target else 0.0

        context_log(
            message=f"[gather_with_reserve] 🏁 done | "
            f"final_success={len(primary.uids)}/{target} "
            f"({primary.success_rate:.1%}) | total_skipped={primary.skipped_uids}"
        )

        # Return None if no gradients were successfully gathered
        if len(primary.uids) == 0:
            return None

        return primary

    ## Peer Management

    async def is_miner_active(self, uid: int, recent_windows: int = 3) -> bool:
        """
        Checks if a miner is active by verifying recent gradient uploads.

        This method determines a miner's activity by checking for the existence of
        gradient files in their S3 bucket within a specified number of recent
        windows. This is used to filter out inactive or unresponsive peers.

        Args:
            uid (int): The UID of the miner to check.
            recent_windows (int, optional): The number of recent windows to check for
                gradient uploads. Defaults to 3.

        Returns:
            bool: True if the miner has uploaded a gradient in the specified
                recent windows, False otherwise.
        """
        hone.logger.debug(f"Checking if UID {uid} is active")
        current_window = self.current_window

        peer_bucket = self.commitments.get(uid)
        if not peer_bucket:
            hone.log_with_context(
                level="debug",
                message=f"No bucket committed for UID {uid}",
                current_window=self.current_window,
            )
            return False

        try:
            s3_client = await self._get_s3_client(peer_bucket)

            if not hasattr(self, "current_window") or self.current_window is None:
                hone.logger.error(
                    "current_window is not set in comms. Please set comms.current_window."
                )
                return False

            current_window = self.current_window
            if current_window is None:
                return False
            # ListObjectsV2 with the per-(window, uid) prefix matches both
            # legacy ``gradient-{w}-{uid}-vX.pt`` and per-stage
            # ``gradient-{w}-{uid}-stage{N}-vX.pt`` uploads in one call,
            # so PP miners (which never write the legacy filename) are
            # correctly counted as active.
            #
            # Rollout 3 hotfix (2026-05-03): under ``fragmented_uploads:
            # true`` miners write ``gradient-frag{NN}-{w}-{uid}-vX.pt``
            # instead of ``gradient-{w}-{uid}-vX.pt``. The legacy prefix
            # above cannot match fragmented keys — once the legacy
            # pre-flip blobs age out of the ``recent_windows`` lookback
            # the activity check returns False for every peer on the new
            # code, ``active_peers`` empties, ``eval_peers`` empties,
            # and outer-steps freeze silently. Probing BOTH prefixes per
            # window makes the check tolerant of mixed fleets: legacy
            # miners match ``gradient-{w}-{uid}-``; fragmented miners
            # match ``gradient-frag00-{w}-{uid}-`` (fragment 00 is
            # always written when ``num_fragments >= 1``). Any match in
            # any window returns True, so a single successful upload in
            # the lookback keeps the peer counted active.
            for window in range(current_window - recent_windows, current_window + 1):
                for prefix in (
                    f"gradient-{window}-{uid}-",
                    f"gradient-frag00-{window}-{uid}-",
                ):
                    hone.logger.debug(
                        f"Listing {prefix}* in {peer_bucket.name}"
                    )
                    try:
                        resp = await s3_client.list_objects_v2(
                            Bucket=peer_bucket.name, Prefix=prefix, MaxKeys=1
                        )
                        if resp.get("KeyCount", 0) > 0:
                            hone.logger.debug(
                                f"Found {resp['Contents'][0]['Key']} for UID {uid}"
                            )
                            return True
                        hone.logger.debug(f"No {prefix}* in {peer_bucket.name}")
                    except botocore.exceptions.ClientError as e:
                        if e.response["Error"]["Code"] not in ["404", "403", "401"]:
                            hone.logger.error(f"Error checking activity for {uid}: {e}")
                            return False
                        hone.logger.debug(f"{prefix}* list error for UID {uid}")

        except (ConnectionClosedError, ClientError):
            await self._purge_s3_client(peer_bucket)
            return False
        except Exception as e:
            hone.logger.error(f"Error accessing bucket for UID {uid}: {e}")
            return False

        return False

    async def track_active_peers(self) -> None:
        """
        Periodically checks for and updates the set of active peers.

        This background task runs continuously, iterating through all known peers
        (from commitments) and using `is_miner_active` to determine if they are
        still responsive. The set of active peers is updated and used for
        subsequent operations like gradient gathering.
        """
        while True:
            active_peers = set()
            max_concurrent = min(30, len(self.commitments) if self.commitments else 10)
            semaphore = asyncio.Semaphore(max_concurrent)

            hone.logger.debug(f"Commitments: {self.commitments}")

            async def check_peer(uid):
                async with semaphore:
                    is_active = await self.is_miner_active(
                        uid, recent_windows=self.recent_windows
                    )
                    if is_active:
                        active_peers.add(uid)

            # Buffer the processing of commitments to avoid blocking the event loop (over resourcing)
            batch_size = 50
            commitment_uids = list(self.commitments.keys())

            for i in range(0, len(commitment_uids), batch_size):
                batch_uids = commitment_uids[i : i + batch_size]
                batch_tasks = [check_peer(uid) for uid in batch_uids]
                await asyncio.gather(*batch_tasks)

            self.active_peers = active_peers

            hone.logger.info(
                f"Updated active peers: {[int(uid) for uid in self.active_peers]}"
            )

            await asyncio.sleep(self.active_check_interval)

    async def _get_highest_stake_validator_bucket(self):
        """
        Retrieves the bucket information for the validator with the highest stake.

        This method identifies the validator with the most stake in the network,
        fetches their committed bucket details, and returns the bucket object
        along with the validator's UID.

        Returns:
            tuple[Bucket | None, int | None]: A tuple containing the `Bucket` object
            and the UID of the highest-staked validator, or (None, None) if not found.
        """
        # Get validator with highest stake
        if self.metagraph is None:
            return None, None
        validator_uid = self.metagraph.S.argmax().item()
        hone.logger.info(f"Found validator with highest stake: {validator_uid}")

        if validator_uid is None:
            hone.logger.info("No active validators found")
            return None, None

        validator_bucket = self.commitments.get(int(validator_uid))
        if not validator_bucket:
            return None, None

        hone.logger.info(f"Validator Bucket: {validator_bucket}")
        return validator_bucket, validator_uid

    async def post_peer_list(
        self,
        *,
        peers: list[int],
        reserve_peers: list[int] | None = None,
        first_effective_window: int,
        sync_window: int,
        initial_selection: bool,
    ) -> None:
        """
        Uploads the selected peer list to the node's S3 bucket.

        This method is used by validators to publish the list of miners that should
        participate in the upcoming training windows. The list is stored as a JSON
        object in a location that is accessible to all miners.

        The following debugging fields are included in the JSON:
        - sync_window: when the peer list was updated in "validator time"
        - weights: weights for all UIDs, which were used to update the peer
          list (except for during the initial peer selection)
        - initial selection: whether this peer list is the first one in the
          current run

        Args:
            peers (list[int]): The list of primary peer UIDs.
            reserve_peers (list[int] | None, optional): A list of reserve peer UIDs.
                Defaults to None.
            first_effective_window (int): The window from which this peer list becomes active.
            sync_window (int): The window at which the peer list was generated.
            initial_selection (bool): A flag indicating if this is the first peer
                selection of the run.
        """

        key = f"{PEERS_FILE_PREFIX}{first_effective_window}_v{hone.__version__}.json"
        peers_and_weights = {
            "peers": peers,
            "reserve_peers": reserve_peers or [],
            "initial_selection": initial_selection,
            "sync_window": sync_window,
            "first_effective_window": first_effective_window,
        }

        # Create temporary JSON file
        temp_file = os.path.join(self.temp_dir, f"temp_{key}")
        try:
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(peers_and_weights))

            await self.s3_put_object(key=key, file_path=temp_file)
            hone.logger.info(f"PUT {key} <--")
        except Exception as e:
            hone.logger.info(f"Failed to upload peer list: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Start Window Operations
    async def post_start_window(self, start_window: int) -> None:
        """
        Uploads the starting window number to the node's S3 bucket.

        This method allows a validator to broadcast the official starting window for
        a training run. This is essential for synchronizing all participating
        neurons and ensuring they start from the same point.

        Args:
            start_window (int): The window number to be set as the starting point.
        """
        key = f"start_window_v{hone.__version__}.json"
        start_window_data = {"start_window": start_window}

        # Create temporary JSON file
        temp_file = os.path.join(self.temp_dir, f"temp_{key}")
        try:
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(start_window_data))

            await self.s3_put_object(key=key, file_path=temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    async def get_peer_list(
        self, fetch_previous: bool = False
    ) -> tuple[list[int], list[int], int] | None:
        """
        Retrieves the peer list from the highest-staked validator's bucket.

        Miners use this method to discover which peers they should collaborate with
        for gradient aggregation. It fetches the JSON file posted by the validator
        and parses it to return the primary and reserve peer lists.

        Args:
            fetch_previous (bool, optional): If True, attempts to fetch the second
                most recent peer list instead of the latest one. Defaults to False.

        Returns:
            tuple[list[int], list[int], int] | None: A tuple containing the list of
            primary peers, reserve peers, and the window from which the list is
            effective. Returns None if no list is found.
        """
        hone.logger.info(
            f"Looking for a {'previous' if fetch_previous else 'current'} peer list on a validator bucket"
        )
        while True:
            validator_bucket = None
            try:
                (
                    validator_bucket,
                    validator_uid,
                ) = await self._get_highest_stake_validator_bucket()

                if validator_bucket is None:
                    hone.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds."
                    )
                    await asyncio.sleep(10)
                    continue

                hone.logger.info(
                    f"Attempting to fetch peer list from UID {validator_uid} bucket {validator_bucket.name}"
                )

                s3_client = await self._get_s3_client(validator_bucket)
                pattern = rf"^{PEERS_FILE_PREFIX}(?P<window>\d+)_v{re.escape(hone.__version__)}\.json$"
                keys = []
                continuation_token = None

                while True:
                    list_args = {
                        "Bucket": validator_bucket.name,
                        "Prefix": PEERS_FILE_PREFIX,
                    }
                    if continuation_token:
                        list_args["ContinuationToken"] = continuation_token

                    response = await s3_client.list_objects_v2(**list_args)

                    for obj in response.get("Contents", []):
                        if re.match(pattern, obj["Key"]):
                            keys.append(obj["Key"])

                    if response.get("IsTruncated"):
                        continuation_token = response.get("NextContinuationToken")
                    else:
                        break

                if len(keys) == 0:
                    hone.logger.info("No peer list files found")
                    return None

                # Parse windows from all keys
                window_to_key = {}
                for key in keys:
                    match = re.match(pattern, key)
                    if match:
                        window = int(match.group("window"))
                        window_to_key[window] = key

                if not window_to_key:
                    hone.logger.error(
                        f"Failed to parse windows from peer list files. First "
                        f"{len(keys[:5])} peer list files are {keys[:5]}"
                    )
                    return None

                # Sort windows to find the most recent or the previous one
                window_to_keys = window_to_key.keys()

                if len(window_to_keys) == 0:
                    return None

                # If fetching previous, get the second most recent (if available)
                selected_window = None
                if fetch_previous and len(window_to_keys) > 1:
                    sorted_windows = sorted(window_to_keys, reverse=True)
                    selected_window = sorted_windows[1]  # Second most recent
                    hone.logger.info(f"Selected previous window {selected_window}")
                elif fetch_previous and len(window_to_keys) <= 1:
                    hone.logger.info(f"Found no previous window {selected_window}")
                    return None
                else:
                    selected_window = max(window_to_keys)  # Most recent
                    hone.logger.info(f"Selected most recent window {selected_window}")

                selected_key = window_to_key[selected_window]

                peers_data = await self.s3_get_object(
                    key=selected_key, bucket=validator_bucket
                )

                if peers_data is None:
                    return None
                if isinstance(peers_data, dict):
                    peers_dict = peers_data
                else:
                    peers_dict = json.loads(peers_data.decode("utf-8"))

                reserves = peers_dict.get("reserve_peers", [])
                return (
                    peers_dict["peers"],
                    reserves,
                    peers_dict["first_effective_window"],
                )

            except (ConnectionClosedError, ClientError):
                if validator_bucket:
                    await self._purge_s3_client(validator_bucket)
                await asyncio.sleep(10)
            except Exception as e:
                hone.logger.error(f"Error fetching peer list: {e}")
                await asyncio.sleep(10)

    async def get_start_window(
        self, retries: int = -1, version: str | None = None
    ) -> int | None:
        """
        Retrieves the official start window from the highest-staked validator.

        This method repeatedly attempts to fetch the `start_window.json` file from
        the lead validator's bucket. This is a crucial step for a neuron that is
        just joining the network, as it needs to know the globally agreed-upon
        starting point for training.

        Args:
            retries (int, optional): The number of times to retry fetching the start
                window. A value of -1 means infinite retries. Defaults to -1.
            version (str, optional): Specific version to fetch start_window for.
                Defaults to current version.

        Returns:
            int | None: The start window number if successfully fetched, otherwise None.
        """
        attempt = 0
        while retries == -1 or attempt < retries:
            try:
                (
                    validator_bucket,
                    validator_uid,
                ) = await self._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    hone.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds"
                    )
                    attempt += 1
                    await asyncio.sleep(10)
                    continue

                hone.logger.info(
                    f"Attempting to fetch start_window from UID {validator_uid} bucket {validator_bucket.name}"
                )

                target_version = version if version else hone.__version__
                start_window_data = await self.s3_get_object(
                    key=f"start_window_v{target_version}.json",
                    bucket=validator_bucket,
                )

                if start_window_data is not None:
                    if isinstance(start_window_data, dict):
                        start_window_json = start_window_data
                    else:
                        start_window_json = json.loads(
                            start_window_data.decode("utf-8")
                        )

                    start_window = start_window_json["start_window"]
                    hone.logger.info(f"Fetched start_window: {start_window}")
                    return start_window

                hone.logger.warning(
                    "start_window.json not found or empty. Retrying in 10 seconds"
                )
                attempt += 1
                await asyncio.sleep(10)

            except Exception as e:
                hone.logger.error(f"Error fetching start_window: {e}")
                attempt += 1
                await asyncio.sleep(10)

        hone.logger.warning("Max retries exceeded while trying to fetch start_window")
        return None

    def check_compressed_indices(
        self,
        param_name: str,
        idxs: torch.Tensor,
        totalk: int,
        allowed_topk: int | None = None,
        vals: torch.Tensor | None = None,
        qparams: tuple | None = None,
    ) -> None:
        """
        Validates the integrity and format of compressed gradient indices.

        This is a crucial security and stability check to ensure that gradients
        received from peers are well-formed. It verifies that indices are within
        the expected bounds and that the compression format (e.g., 12-bit packing)
        is correctly applied.

        Args:
            param_name (str): The name of the parameter being checked.
            idxs (torch.Tensor): The tensor of indices.
            totalk (int): The total number of elements in the original uncompressed tensor.
            allowed_topk (int | None, optional): The expected number of top-k values.
                Defaults to the hparams configuration.
            vals (torch.Tensor | None, optional): The corresponding values tensor,
                required for validating 12-bit packed indices. Defaults to None.
            qparams (tuple | None, optional): The quantisation params tuple
                that travels alongside ``vals``. When the sender opted into
                PACK_VERSION_2BIT (P0b), qparams is a 7-tuple whose 7th
                element carries the original (pre-2-bit-pack) last-dim size,
                which is needed to reconstruct the indices shape. Legacy
                5-tuple qparams (raw uint8 values) are fully supported with
                the wire ``vals.shape`` interpreted directly. ``None`` falls
                back to legacy behaviour.

        Raises:
            ValueError: If any validation check fails, such as out-of-bounds
                indices, incorrect data types, or malformed packed data.
        """
        allowed_topk = (
            min(self.hparams.topk_compression, totalk)
            if allowed_topk is None
            else min(allowed_topk, totalk)
        )

        def _bounds_check(t: torch.Tensor):
            """fast min/max bounds check"""
            if t.numel() == 0:
                raise ValueError(f"[{param_name}] empty index list")
            if t.min().item() < 0 or t.max().item() >= totalk:
                bad = t[(t < 0) | (t >= totalk)][0].item()
                raise ValueError(
                    f"[{param_name}] Index {bad} out of bounds (totalk = {totalk})"
                )

        # Handle 12-bit packed index format only
        if isinstance(idxs, torch.Tensor):
            if idxs.dtype != torch.uint8:
                raise ValueError(
                    f"[{param_name}] Expected uint8 for 12-bit packed indices, got {idxs.dtype}"
                )
            # 12-bit packed format is the only supported format
            if vals is None:
                raise ValueError(
                    f"[{param_name}] Values tensor required to validate 12-bit packed indices"
                )
            if idxs.numel() == 0:
                raise ValueError(f"[{param_name}] Empty packed indices tensor")

            # Reconstruct the per-row index count. For PACK_VERSION_2BIT,
            # ``vals.shape[-1]`` is ``ceil(topk/4)`` which is wrong for the
            # 12-bit unpacker; the original ``topk`` lives in qparams[6].
            # ``vals.shape[:-1]`` is preserved across both wire formats so
            # we always trust the leading dims.
            indices_last_dim = vals.shape[-1]
            if (
                isinstance(qparams, tuple)
                and len(qparams) >= 7
                and int(qparams[5]) == 1  # PACK_VERSION_2BIT
            ):
                indices_last_dim = int(qparams[6])
            indices_shape = (*vals.shape[:-1], indices_last_dim)

            # Unpack using the reconstructed indices shape
            try:
                unpacked = unpack_12bit_indices(idxs, indices_shape)
                # Validate that the last dimension matches allowed_topk
                if unpacked.shape[-1] != allowed_topk:
                    raise ValueError(
                        f"[{param_name}] Invalid topk dimension: "
                        f"shape[-1]={unpacked.shape[-1]} but expected {allowed_topk}"
                    )
                _bounds_check(unpacked)
            except Exception as e:
                raise ValueError(f"[{param_name}] Failed to unpack 12-bit indices: {e}")
        else:
            raise ValueError(f"[{param_name}] Expected tensor but got {type(idxs)}")

    async def s3_get_object_size(self, bucket: Bucket, key: str) -> int | None:
        """
        Retrieves the size of an S3 object without downloading its content.

        This method uses an S3 HEAD request to efficiently fetch the metadata of an
        object, including its size in bytes. This is useful for pre-allocation or
        for making decisions about how to download a file.

        Args:
            bucket (Bucket): The S3 bucket containing the object.
            key (str): The key of the object.

        Returns:
            int | None: The size of the object in bytes, or None if the object
                is not found or an error occurs.
        """
        try:
            s3_client = await self._get_s3_client(bucket)

            response = await s3_client.head_object(Bucket=bucket.name, Key=key)
            file_size = response["ContentLength"]

            hone.logger.debug(f"Object {key} size: {file_size} bytes")
            return file_size

        except (ConnectionClosedError, ClientError) as e:
            await self._purge_s3_client(bucket)
            if "404" in str(e):
                hone.logger.debug(f"Object {key} not found in bucket {bucket.name}")
                return None
            hone.logger.error(f"Error getting object size for {key}: {e}")
            return None
        except Exception as e:
            hone.logger.error(f"Error getting object size for {key}: {e}")
            return None

    async def s3_get_object_range(
        self, bucket: Bucket, key: str, start: int, end: int, timeout: int = 30
    ) -> None | bytes:
        """
        Downloads a specific byte range from an S3 object.

        This is a low-level utility for partial file downloads. It's a key component
        of the parallel, multipart download strategy used for large files, allowing
        different parts of a file to be fetched concurrently.

        Args:
            bucket (Bucket): The S3 bucket containing the object.
            key (str): The key of the object.
            start (int): The starting byte position.
            end (int): The ending byte position.
            timeout (int, optional): The timeout for the request in seconds. Defaults to 30.

        Returns:
            None | bytes: The downloaded chunk of data as bytes, or None if an
                error occurs.
        """
        try:
            s3_client = await self._get_s3_client(bucket)

            response = await asyncio.wait_for(
                s3_client.get_object(
                    Bucket=bucket.name, Key=key, Range=f"bytes={start}-{end}"
                ),
                timeout=timeout,
            )

            # Read the chunk data
            async with response["Body"] as stream:
                chunk_data = await asyncio.wait_for(stream.read(), timeout=timeout)

            # Verify chunk size
            expected_size = end - start + 1
            if len(chunk_data) != expected_size:
                raise Exception(
                    f"Chunk size mismatch: got {len(chunk_data)}, expected {expected_size}"
                )

            return chunk_data

        except asyncio.TimeoutError:
            hone.logger.error(f"Timeout downloading range {start}-{end} for {key}")
            return None
        except (ConnectionClosedError, ClientError) as e:
            await self._purge_s3_client(bucket)
            hone.logger.error(
                f"Client error downloading range {start}-{end} for {key}: {e}"
            )
            return None
        except Exception as e:
            hone.logger.error(f"Error downloading range {start}-{end} for {key}: {e}")
            return None

    async def get_debug_dict(self, window: int) -> dict[str, Any] | None:
        """
        Retrieves a debug dictionary from the lead validator's bucket for a specific window.

        This method allows for the inspection of validator state at a particular point
        in time, which is invaluable for debugging and performance analysis. The debug
        dictionary can contain various metrics and metadata.

        Args:
            window (int): The specific window for which to retrieve the debug data.

        Returns:
            dict | None: The debug dictionary if found, otherwise None.
        """
        try:
            (
                validator_bucket,
                validator_uid,
            ) = await self._get_highest_stake_validator_bucket()
            if not validator_bucket or validator_uid is None:
                hone.logger.warning(
                    "No validator bucket - cannot proceed with debug fetch"
                )
                return

            key = f"debug-{window}-{validator_uid}-v{hone.__version__}.pt"
            hone.logger.info(
                f"Attempting to retrieve debug dictionary for window {window} from validator {validator_uid}"
            )

            result = await self.s3_get_object(
                key=key,
                bucket=validator_bucket,
                timeout=20,
            )

            if result is None:
                hone.logger.warning(f"No debug dictionary found for window {window}")
                return None

            hone.logger.info(
                f"Successfully retrieved debug dictionary for window {window}"
            )
            return result

        except Exception as e:
            hone.logger.error(
                f"Error getting debug dictionary for window {window}: {e}"
            )
            import traceback

            hone.logger.error(traceback.format_exc())
            return None

    def weighted_random_sample_no_replacement(
        self, candidates: list[int], weights: list[int], k: int
    ) -> list[int]:
        """
        Performs a weighted random sample without replacement.

        This utility function is used to select a subset of candidates (e.g., miners)
        based on their associated weights (e.g., stake or performance scores). It
        ensures that the selection is both random and biased towards higher-weighted
        candidates, and that no candidate is selected more than once.

        Args:
            candidates (list[int]): A list of items to sample from (e.g., UIDs).
            weights (list[int]): A list of corresponding weights for each candidate.
            k (int): The number of items to sample.

        Returns:
            list[int]: A list of the selected items.
        """
        hone.logger.debug("Starting weighted random sampling")
        hone.logger.debug(f"Candidates: {candidates}")
        hone.logger.debug(f"Weights: {weights}")
        hone.logger.debug(f"Sample size (k): {k}")

        # Safety checks
        if not candidates or not weights or k <= 0:
            hone.logger.warning("Invalid input detected. Returning empty list.")
            return []

        # Pair up each candidate with its weight
        pool = list(zip(candidates, weights))
        total_w = float(sum(weights))
        selected = []

        # If total weight is 0, return empty
        if total_w <= 0:
            hone.logger.warning("Total weight is zero. Returning empty list")
            return []

        hone.logger.debug(f"Initial total weight: {total_w}")

        for _ in range(min(k, len(candidates))):
            if total_w <= 0 or len(pool) == 0:
                hone.logger.info("No more items to sample. Stopping early.")
                break

            # Draw a uniform sample in [0, total_w]
            r = random.uniform(0.0, total_w)
            hone.logger.debug(f"Random threshold: {r}")
            cumulative = 0.0
            for idx, (uid, w) in enumerate(pool):
                cumulative += w
                if cumulative >= r:
                    # Found our pick
                    selected.append(uid)
                    hone.logger.info(f"Selected item: {uid} with weight {w}")
                    # Remove from pool and subtract from total_w
                    total_w -= w
                    pool.pop(idx)
                    hone.logger.debug(f"Updated total weight: {total_w}")
                    break

        hone.logger.debug(f"Final selected items: {selected}")
        return selected
