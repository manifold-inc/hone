"""
S3 Manager Module

Handles S3 data transfer for job input/output:
- Download input data before job execution
- Upload output data after job completion
- Support for s3:// URLs
- Error handling and retries
- Progress tracking for large files
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config

from config import StorageConfig

logger = logging.getLogger(__name__)


class S3TransferError(Exception):
    """Raised when S3 transfer fails."""
    pass


class S3Manager:
    """
    Manages S3 data transfers for job execution.
    
    Features:
    - Asynchronous file transfers
    - Automatic retry with exponential backoff
    - Progress tracking
    - Support for s3:// URLs
    - IAM role or credential-based authentication
    """
    
    def __init__(self, config: StorageConfig):
        """
        Initialize S3 manager.
        
        Args:
            config: Storage configuration with S3 settings
        """
        self.config = config
        
        # Configure boto3 with retries
        boto_config = Config(
            region_name=config.s3_region,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            }
        )
        
        # Initialize S3 client
        if config.s3_access_key and config.s3_secret_key:
            # Use explicit credentials
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config.s3_access_key,
                aws_secret_access_key=config.s3_secret_key,
                endpoint_url=config.s3_endpoint,
                config=boto_config
            )
        else:
            # Use IAM role or environment credentials
            self.s3_client = boto3.client(
                's3',
                endpoint_url=config.s3_endpoint,
                config=boto_config
            )
        
        logger.info(f"S3Manager initialized (region={config.s3_region})")
    
    async def download_input_data(
        self,
        s3_path: str,
        local_path: Path,
        max_retries: int = 3
    ) -> bool:
        """
        Download input data from S3 to local filesystem.
        
        Args:
            s3_path: S3 path (s3://bucket/key or full URL)
            local_path: Local destination path
            max_retries: Maximum retry attempts
            
        Returns:
            True if download successful, False otherwise
            
        Raises:
            S3TransferError: If download fails after retries
        """
        bucket, key = self._parse_s3_path(s3_path)
        
        logger.info(f"Downloading from s3://{bucket}/{key} to {local_path}")
        
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                # Use asyncio to run blocking S3 operation
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._download_file,
                    bucket,
                    key,
                    local_path
                )
                
                # Verify file was downloaded
                if not local_path.exists():
                    raise S3TransferError(f"File not found after download: {local_path}")
                
                file_size = local_path.stat().st_size
                logger.info(
                    f"Downloaded {file_size} bytes from s3://{bucket}/{key}",
                    extra={"bucket": bucket, "key": key, "size": file_size}
                )
                
                return True
                
            except (ClientError, BotoCoreError) as e:
                error_msg = str(e)
                logger.warning(
                    f"S3 download attempt {attempt + 1}/{max_retries} failed: {error_msg}"
                )
                
                if attempt == max_retries - 1:
                    raise S3TransferError(
                        f"Failed to download from s3://{bucket}/{key} after "
                        f"{max_retries} attempts: {error_msg}"
                    )
                
                # Exponential backoff
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
        Upload output data from local filesystem to S3.
        
        Args:
            local_path: Local file path
            s3_path: S3 destination path (s3://bucket/key or full URL)
            max_retries: Maximum retry attempts
            
        Returns:
            True if upload successful, False otherwise
            
        Raises:
            S3TransferError: If upload fails after retries
        """
        if not local_path.exists():
            raise S3TransferError(f"Local file not found: {local_path}")
        
        bucket, key = self._parse_s3_path(s3_path)
        
        file_size = local_path.stat().st_size
        logger.info(
            f"Uploading {file_size} bytes from {local_path} to s3://{bucket}/{key}"
        )
        
        for attempt in range(max_retries):
            try:
                # Use asyncio to run blocking S3 operation
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._upload_file,
                    local_path,
                    bucket,
                    key
                )
                
                # Verify upload
                if not await self.verify_object_exists(bucket, key):
                    raise S3TransferError(f"Object not found after upload: s3://{bucket}/{key}")
                
                logger.info(
                    f"Uploaded {file_size} bytes to s3://{bucket}/{key}",
                    extra={"bucket": bucket, "key": key, "size": file_size}
                )
                
                return True
                
            except (ClientError, BotoCoreError) as e:
                error_msg = str(e)
                logger.warning(
                    f"S3 upload attempt {attempt + 1}/{max_retries} failed: {error_msg}"
                )
                
                if attempt == max_retries - 1:
                    raise S3TransferError(
                        f"Failed to upload to s3://{bucket}/{key} after "
                        f"{max_retries} attempts: {error_msg}"
                    )
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
            
            except Exception as e:
                logger.exception(f"Unexpected error during S3 upload: {e}")
                raise S3TransferError(f"Upload failed: {e}")
        
        return False
    
    async def verify_object_exists(self, bucket: str, key: str) -> bool:
        """
        Verify that an S3 object exists.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            True if object exists, False otherwise
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.s3_client.head_object,
                bucket,
                key
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    async def download_directory(
        self,
        s3_prefix: str,
        local_dir: Path,
        max_retries: int = 3
    ) -> bool:
        """
        Download all objects with a given S3 prefix to a local directory.
        
        Args:
            s3_prefix: S3 prefix (s3://bucket/prefix/)
            local_dir: Local destination directory
            max_retries: Maximum retry attempts per file
            
        Returns:
            True if all files downloaded successfully
        """
        bucket, prefix = self._parse_s3_path(s3_prefix)
        
        logger.info(f"Downloading directory s3://{bucket}/{prefix} to {local_dir}")
        
        # List objects with prefix
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
            
            # Download each object
            download_tasks = []
            for obj in objects:
                key = obj['Key']
                # Calculate relative path
                rel_path = key[len(prefix):].lstrip('/')
                local_path = local_dir / rel_path
                
                # Create download task
                task = self.download_input_data(
                    f"s3://{bucket}/{key}",
                    local_path,
                    max_retries
                )
                download_tasks.append(task)
            
            # Download all files concurrently
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
            
            # Check for errors
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                logger.error(f"Failed to download {len(failures)} files")
                return False
            
            logger.info(f"Downloaded {len(objects)} files from s3://{bucket}/{prefix}")
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
        """
        Upload all files from a local directory to S3 with a given prefix.
        
        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix (s3://bucket/prefix/)
            max_retries: Maximum retry attempts per file
            
        Returns:
            True if all files uploaded successfully
        """
        if not local_dir.exists():
            raise S3TransferError(f"Local directory not found: {local_dir}")
        
        bucket, prefix = self._parse_s3_path(s3_prefix)
        
        logger.info(f"Uploading directory {local_dir} to s3://{bucket}/{prefix}")
        
        # Find all files in directory
        files = list(local_dir.rglob('*'))
        files = [f for f in files if f.is_file()]
        
        if not files:
            logger.warning(f"No files found in {local_dir}")
            return True
        
        # Upload each file
        upload_tasks = []
        for local_path in files:
            # Calculate relative path and S3 key
            rel_path = local_path.relative_to(local_dir)
            key = f"{prefix.rstrip('/')}/{rel_path}".lstrip('/')
            
            # Create upload task
            task = self.upload_output_data(
                local_path,
                f"s3://{bucket}/{key}",
                max_retries
            )
            upload_tasks.append(task)
        
        # Upload all files concurrently
        results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        
        # Check for errors
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            logger.error(f"Failed to upload {len(failures)} files")
            return False
        
        logger.info(f"Uploaded {len(files)} files to s3://{bucket}/{prefix}")
        return True
    
    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """
        Parse S3 path into bucket and key.
        
        Supports formats:
        - s3://bucket/key
        - https://bucket.s3.region.amazonaws.com/key
        - bucket/key
        
        Args:
            s3_path: S3 path string
            
        Returns:
            Tuple of (bucket, key)
            
        Raises:
            ValueError: If path format is invalid
        """
        # Remove s3:// prefix if present
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
        
        # Parse as URL if it's an HTTP(S) URL
        if s3_path.startswith('http://') or s3_path.startswith('https://'):
            parsed = urlparse(s3_path)
            # Extract bucket from hostname
            bucket = parsed.hostname.split('.')[0]
            # Key is the path without leading slash
            key = parsed.path.lstrip('/')
            return bucket, key
        
        # Parse as bucket/key format
        parts = s3_path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid S3 path format: {s3_path}. "
                "Expected: s3://bucket/key or bucket/key"
            )
        
        bucket, key = parts
        return bucket, key
    
    def _download_file(self, bucket: str, key: str, local_path: Path):
        """
        Blocking S3 download operation.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            local_path: Local destination path
        """
        self.s3_client.download_file(bucket, key, str(local_path))
    
    def _upload_file(self, local_path: Path, bucket: str, key: str):
        """
        Blocking S3 upload operation.
        
        Args:
            local_path: Local file path
            bucket: S3 bucket name
            key: S3 object key
        """
        self.s3_client.upload_file(str(local_path), bucket, key)
    
    def _list_objects(self, bucket: str, prefix: str) -> list:
        """
        Blocking S3 list objects operation.
        
        Args:
            bucket: S3 bucket name
            prefix: Object key prefix
            
        Returns:
            List of object metadata dictionaries
        """
        objects = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                objects.extend(page['Contents'])
        
        return objects