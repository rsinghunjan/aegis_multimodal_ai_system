  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
#!/usr/bin/env python3
"""
Storage client factory.

Creates a StorageClient implementation based on configuration (cfg.OBJECT_STORE_TYPE).
Supported types: "minio", "s3", "gcs", "azure".
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path

from aegis_multimodal_ai_system import config

from .abstract import StorageClient

def create_storage_client(bucket: Optional[str] = None) -> StorageClient:
    typ = (config.cfg.OBJECT_STORE_TYPE or "s3").lower()
    bucket = bucket or config.cfg.OBJECT_STORE_BUCKET
    if typ == "minio":
        from .minio_adapter import MinIOStorageAdapter
        endpoint = config.cfg.OBJECT_STORE_ENDPOINT or "http://localhost:9000"
        access = config.cfg.OBJECT_STORE_ACCESS_KEY or "minioadmin"
        secret = config.cfg.OBJECT_STORE_SECRET_KEY or "minioadmin"
        return MinIOStorageAdapter(bucket=bucket, endpoint_url=endpoint, access_key=access, secret_key=secret, region=config.cfg.OBJECT_STORE_REGION)
    if typ == "s3":
        from .s3_adapter import S3StorageAdapter
        return S3StorageAdapter(bucket=bucket, region=config.cfg.OBJECT_STORE_REGION, endpoint_url=config.cfg.OBJECT_STORE_ENDPOINT, access_key=config.cfg.OBJECT_STORE_ACCESS_KEY, secret_key=config.cfg.OBJECT_STORE_SECRET_KEY)
    if typ == "gcs":
        from .gcs_adapter import GCSStorageAdapter
        # GCS adapter may rely on GOOGLE_APPLICATION_CREDENTIALS env var
        return GCSStorageAdapter(bucket=bucket)
    if typ in ("azure", "azureblob", "az"):
        from .azure_adapter import AzureBlobStorageAdapter
        return AzureBlobStorageAdapter(container=bucket)
    raise RuntimeError(f"Unsupported OBJECT_STORE_TYPE: {typ}")
aegis_multimodal_ai_system/storage/factory.py
