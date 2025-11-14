"""
S3 Manager Module - Updated with Presigned URL Support

Handles S3 data transfer for job input/output:
- Download input data before job execution (S3 URIs or presigned URLs)
- Upload output data after job completion
- Support for s3:// URLs and HTTPS presigned URLs
- Error handling and retries
"""

import asyncio
import logging
from pathlib import Path
import aiohttp

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config

from config import StorageConfig

logger = logging.getLogger(__name__)


class S3TransferError(Exception):
    """Raised when S3 transfer fails"""
    pass


class S3Manager:
    """
    Manages S3 data transfers for job execution
    
    Features:
    - Asynchronous file transfers
    - Support for presigned URLs (GET/PUT)
    - Support for s3:// URIs
    - Automatic retry with exponential backoff
    - Progress tracking
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize S3 manager"""
        self.config = config
        
        boto_config = Config(
            region_name=config.s3_region,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            }
        )
        
        if config.s3_access_key and config.s3_secret_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config.s3_access_key,
                aws_secret_access_key=config.s3_secret_key,
                endpoint_url=config.s3_endpoint,
                config=boto_config
            )
        else:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=config.s3_endpoint,
                config=boto_config
            )
        
        logger.info(f"S3Manager initialized (region={config.s3_region})")
    
    def _is_presigned_url(self, path: str) -> bool:
        """Check if path is a presigned URL."""
        return path.startswith('http://') or path.startswith('https://')
    
    async def download_input_data(
        self,
        s3_path: str,
        local_path: Path,
        max_retries: int = 3
    ) -> bool:
        """
        Download input data from S3 to local filesystem
        
        Supports both:
        - s3://bucket/key URIs (uses boto3)
        - Presigned URLs (uses direct HTTP GET)
        
        Args:
            s3_path: S3 URI or presigned URL
            local_path: Local destination path
            max_retries: Maximum retry attempts
            
        Returns:
            True if download successful
        """
        if self._is_presigned_url(s3_path):
            return await self._download_from_presigned_url(
                s3_path, local_path, max_retries
            )
        else:
            return await self._download_from_s3_uri(
                s3_path, local_path, max_retries
            )
    
    async def _download_from_presigned_url(
        self,
        presigned_url: str,
        local_path: Path,
        max_retries: int = 3
    ) -> bool:
        """Download from presigned URL using HTTP GET"""
        logger.info(f"Downloading from presigned URL to {local_path}")
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(presigned_url) as response:
                        if response.status == 200:
                            with open(local_path, 'wb') as f:
                                while True:
                                    chunk = await response.content.read(8192)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                            
                            if not local_path.exists():
                                raise S3TransferError(f"File not found after download")
                            
                            file_size = local_path.stat().st_size
                            logger.info(f"Downloaded {file_size} bytes from presigned URL")
                            return True
                        
                        else:
                            raise S3TransferError(
                                f"HTTP {response.status}: {await response.text()}"
                            )
                
            except Exception as e:
                logger.warning(
                    f"Presigned URL download attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                
                if attempt == max_retries - 1:
                    raise S3TransferError(
                        f"Failed to download from presigned URL after {max_retries} attempts: {e}"
                    )
                
                await asyncio.sleep(2 ** attempt)
        
        return False
    
    async def _download_from_s3_uri(
        self,
        s3_path: str,
        local_path: Path,
        max_retries: int = 3
    ) -> bool:
        """Download from S3 URI using boto3"""
        bucket, key = self._parse_s3_path(s3_path)
        
        logger.info(f"Downloading from s3://{bucket}/{key} to {local_path}")
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._download_file,
                    bucket,
                    key,
                    local_path
                )
                
                if not local_path.exists():
                    raise S3TransferError(f"File not found after download")
                
                file_size = local_path.stat().st_size
                logger.info(f"Downloaded {file_size} bytes from s3://{bucket}/{key}")
                
                return True
                
            except (ClientError, BotoCoreError) as e:
                logger.warning(
                    f"S3 download attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                
                if attempt == max_retries - 1:
                    raise S3TransferError(
                        f"Failed to download after {max_retries} attempts: {e}"
                    )
                
                await asyncio.sleep(2 ** attempt)
            
            except Exception as e:
                logger.exception(f"Unexpected error during S3 download: {e}")
                raise S3TransferError(f"Download failed: {e}")
        
        return False
    
    async def upload_output_data(
        self,
        local_path: Path,
        s3_path: str,
        max_retries: int = 3
    ) -> bool:
        """
        Upload output data from local filesystem to S3
        
        Supports both:
        - s3://bucket/key URIs (uses boto3)
        - Presigned URLs (uses direct HTTP PUT)
        
        Args:
            local_path: Local file path
            s3_path: S3 URI or presigned URL
            max_retries: Maximum retry attempts
            
        Returns:
            True if upload successful
        """
        if not local_path.exists():
            raise S3TransferError(f"Local file not found: {local_path}")
        
        if self._is_presigned_url(s3_path):
            return await self._upload_to_presigned_url(
                local_path, s3_path, max_retries
            )
        else:
            return await self._upload_to_s3_uri(
                local_path, s3_path, max_retries
            )
    
    async def _upload_to_presigned_url(
        self,
        local_path: Path,
        presigned_url: str,
        max_retries: int = 3
    ) -> bool:
        """Upload to presigned URL using HTTP PUT"""
        file_size = local_path.stat().st_size
        logger.info(f"Uploading {file_size} bytes to presigned URL")
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    with open(local_path, 'rb') as f:
                        async with session.put(presigned_url, data=f) as response:
                            if response.status in [200, 204]:
                                logger.info(f"Uploaded {file_size} bytes to presigned URL")
                                return True
                            else:
                                raise S3TransferError(
                                    f"HTTP {response.status}: {await response.text()}"
                                )
                
            except Exception as e:
                logger.warning(
                    f"Presigned URL upload attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                
                if attempt == max_retries - 1:
                    raise S3TransferError(
                        f"Failed to upload to presigned URL after {max_retries} attempts: {e}"
                    )
                
                await asyncio.sleep(2 ** attempt)
        
        return False
    
    async def _upload_to_s3_uri(
        self,
        local_path: Path,
        s3_path: str,
        max_retries: int = 3
    ) -> bool:
        """Upload to S3 URI using boto3"""
        bucket, key = self._parse_s3_path(s3_path)
        
        file_size = local_path.stat().st_size
        logger.info(f"Uploading {file_size} bytes to s3://{bucket}/{key}")
        
        for attempt in range(max_retries):
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._upload_file,
                    local_path,
                    bucket,
                    key
                )
                
                logger.info(f"Uploaded {file_size} bytes to s3://{bucket}/{key}")
                return True
                
            except (ClientError, BotoCoreError) as e:
                logger.warning(
                    f"S3 upload attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                
                if attempt == max_retries - 1:
                    raise S3TransferError(
                        f"Failed to upload after {max_retries} attempts: {e}"
                    )
                
                await asyncio.sleep(2 ** attempt)
        
        return False
    
    async def download_directory(
        self,
        s3_prefix: str,
        local_dir: Path,
        max_retries: int = 3
    ) -> bool:
        """Download directory - only works with S3 URIs, not presigned URLs"""
        if self._is_presigned_url(s3_prefix):
            filename = local_dir.name if local_dir.suffix else "input.json"
            return await self._download_from_presigned_url(
                s3_prefix,
                local_dir / filename,
                max_retries
            )
        
        bucket, prefix = self._parse_s3_path(s3_prefix)
        logger.info(f"Downloading directory s3://{bucket}/{prefix} to {local_dir}")
        
        try:
            objects = await asyncio.get_event_loop().run_in_executor(
                None,
                self._list_objects,
                bucket,
                prefix
            )
            
            if not objects:
                logger.warning(f"No objects found with prefix s3://{bucket}/{prefix}")
                return True
            
            download_tasks = []
            for obj in objects:
                key = obj['Key']
                rel_path = key[len(prefix):].lstrip('/')
                local_path = local_dir / rel_path
                
                task = self._download_from_s3_uri(
                    f"s3://{bucket}/{key}",
                    local_path,
                    max_retries
                )
                download_tasks.append(task)
            
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
            failures = [r for r in results if isinstance(r, Exception)]
            
            if failures:
                logger.error(f"Failed to download {len(failures)} files")
                return False
            
            logger.info(f"Downloaded {len(objects)} files")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to download directory: {e}")
            raise S3TransferError(f"Directory download failed: {e}")
    
    async def upload_directory(
        self,
        local_dir: Path,
        s3_prefix: str,
        max_retries: int = 3
    ) -> bool:
        """Upload directory - only works with S3 URIs"""
        if not local_dir.exists():
            raise S3TransferError(f"Local directory not found: {local_dir}")
        
        if self._is_presigned_url(s3_prefix):
            raise S3TransferError(
                "Cannot upload directory to presigned URL. Use S3 URI instead."
            )
        
        bucket, prefix = self._parse_s3_path(s3_prefix)
        logger.info(f"Uploading directory {local_dir} to s3://{bucket}/{prefix}")
        
        files = [f for f in local_dir.rglob('*') if f.is_file()]
        
        if not files:
            logger.warning(f"No files found in {local_dir}")
            return True
        
        upload_tasks = []
        for local_path in files:
            rel_path = local_path.relative_to(local_dir)
            key = f"{prefix.rstrip('/')}/{rel_path}".lstrip('/')
            
            task = self._upload_to_s3_uri(
                local_path,
                f"s3://{bucket}/{key}",
                max_retries
            )
            upload_tasks.append(task)
        
        results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        failures = [r for r in results if isinstance(r, Exception)]
        
        if failures:
            logger.error(f"Failed to upload {len(failures)} files")
            return False
        
        logger.info(f"Uploaded {len(files)} files")
        return True
    
    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and key"""
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
        
        parts = s3_path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path format: {s3_path}")
        
        return parts[0], parts[1]
    
    def _download_file(self, bucket: str, key: str, local_path: Path):
        """Blocking S3 download"""
        self.s3_client.download_file(bucket, key, str(local_path))
    
    def _upload_file(self, local_path: Path, bucket: str, key: str):
        """Blocking S3 upload"""
        self.s3_client.upload_file(str(local_path), bucket, key)
    
    def _list_objects(self, bucket: str, prefix: str) -> list:
        """Blocking S3 list objects"""
        objects = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                objects.extend(page['Contents'])
        
        return objects