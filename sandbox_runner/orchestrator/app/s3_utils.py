from __future__ import annotations
import os
import tempfile
import urllib.parse
from typing import Tuple
import boto3


def parse_s3_url(url: str) -> Tuple[str, str]:
    if url.startswith("s3://"):
        _, _, rest = url.partition("s3://")
        bucket, _, key = rest.partition("/")
        return bucket, key
    parsed = urllib.parse.urlparse(url)
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError("Unsupported S3 URL format")

def download_to_dir(s3_url: str, dest_dir: str, region: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    bucket, key_prefix = parse_s3_url(s3_url)
    s3 = boto3.resource("s3", region_name=region)
    b = s3.Bucket(bucket)
    for obj in b.objects.filter(Prefix=key_prefix):
        rel = obj.key[len(key_prefix):].lstrip("/")
        target = os.path.join(dest_dir, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        b.download_file(obj.key, target)


def upload_dir_to_s3(src_dir: str, s3_prefix: str, region: str) -> str:
    bucket, key_prefix = parse_s3_url(s3_prefix)
    s3 = boto3.client("s3", region_name=region)
    for root, _, files in os.walk(src_dir):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, src_dir)
            s3_key = f"{key_prefix.rstrip('/')}/{rel}"
            s3.upload_file(full, bucket, s3_key)
    return f"s3://{bucket}/{key_prefix.rstrip('/')}"