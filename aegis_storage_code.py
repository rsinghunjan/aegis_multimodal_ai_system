Python 3.13.7 (v3.13.7:bcee1c32211, Aug 14 2025, 19:10:51) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
import os
from abc import ABC, abstractmethod
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs_storage
import logging

logger = logging.getLogger(__name__)

# Define the interface all storage clients must implement
class StorageClient(ABC):
    @abstractmethod
    def download_file(self, bucket_name: str, source_blob_name: str, destination_file_name: str):
        pass

    @abstractmethod
    def upload_file(self, bucket_name: str, source_file_name: str, destination_blob_name: str):
        pass

# Concrete implementation for AWS S3 and S3-compatible APIs (like MinIO for on-prem)
class S3StorageClient(StorageClient):
    def __init__(self):
        # Credentials are loaded from environment variables via Boto3
        # (e.g., AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) or IAM role if on AWS.
        self.client = boto3.client('s3')

    def download_file(self, bucket_name: str, source_blob_name: str, destination_file_name: str):
        self.client.download_file(bucket_name, source_blob_name, destination_file_name)
        logger.info(f"Downloaded {source_blob_name} from S3 bucket {bucket_name}.")

    def upload_file(self, bucket_name: str, source_file_name: str, destination_blob_name: str):
        self.client.upload_file(source_file_name, bucket_name, destination_blob_name)
        logger.info(f"Uploaded {source_file_name} to S3 bucket {bucket_name} as {destination_blob_name}.")

# Concrete implementation for Azure Blob Storage
class AzureStorageClient(StorageClient):
    def __init__(self):
...         # Connection string is picked up from environment variable
...         connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
...         self.client = BlobServiceClient.from_connection_string(connect_str)
... 
...     def download_file(self, container_name: str, source_blob_name: str, destination_file_name: str):
...         blob_client = self.client.get_blob_client(container=container_name, blob=source_blob_name)
...         with open(destination_file_name, "wb") as download_file:
...             download_file.write(blob_client.download_blob().readall())
...         logger.info(f"Downloaded {source_blob_name} from Azure container {container_name}.")
... 
...     def upload_file(self, container_name: str, source_file_name: str, destination_blob_name: str):
...         blob_client = self.client.get_blob_client(container=container_name, blob=destination_blob_name)
...         with open(source_file_name, "rb") as data:
...             blob_client.upload_blob(data, overwrite=True)
...         logger.info(f"Uploaded {source_file_name} to Azure container {container_name} as {destination_blob_name}.")
... 
... # Factory function to create the appropriate client
... def get_storage_client() -> StorageClient:
...     provider = os.getenv('STORAGE_PROVIDER', 's3').lower() # Default to 's3'
... 
...     if provider == 's3':
...         return S3StorageClient()
...     elif provider == 'azure':
...         return AzureStorageClient()
...     elif provider == 'gcs':
...         # Implementation for Google Cloud Storage would go here
...         return GCSStorageClient()
...     else:
...         raise ValueError(f"Unsupported storage provider: {provider}")
... 
... # Example usage in another file (e.g., app.py):
... # storage_client = get_storage_client()
