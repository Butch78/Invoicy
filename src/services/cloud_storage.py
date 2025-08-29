"""
Multi-cloud storage services for Invoicy.
Demonstrates efficient data transfer between AWS and GCP environments using MinIO.
This addresses the tech stack requirements mentioned in the brief.
"""

import os
import asyncio
from typing import List, Tuple, Optional
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from minio import Minio
from minio.error import S3Error
import logging

from src.models import CloudProvider, FileType

logger = logging.getLogger(__name__)


class CloudStorageConfig:
    """Configuration for multi-cloud storage environments"""

    # AWS MinIO configuration (simulating AWS S3)
    AWS_ENDPOINT = os.getenv("MINIO_AWS_ENDPOINT", "localhost:9001")
    AWS_ACCESS_KEY = os.getenv("MINIO_AWS_ACCESS_KEY", "awsadmin")
    AWS_SECRET_KEY = os.getenv("MINIO_AWS_SECRET_KEY", "awspassword")
    AWS_BUCKET = "aws-invoices"

    # GCP MinIO configuration (simulating GCP Cloud Storage)
    GCP_ENDPOINT = os.getenv("MINIO_GCP_ENDPOINT", "localhost:9002")
    GCP_ACCESS_KEY = os.getenv("MINIO_GCP_ACCESS_KEY", "gcpadmin")
    GCP_SECRET_KEY = os.getenv("MINIO_GCP_SECRET_KEY", "gcppassword")
    GCP_BUCKET = "gcp-invoices"


class MultiCloudStorageService:
    """
    Service for managing files across multiple cloud providers.
    Demonstrates the multi-cloud architecture from the brief.
    """

    def __init__(self):
        self.config = CloudStorageConfig()
        self._aws_client = None
        self._gcp_client = None
        self._initialized = False

    def _setup_clients(self):
        """Initialize MinIO clients for both cloud providers"""
        if self._initialized:
            return

        try:
            # AWS MinIO client
            self._aws_client = Minio(
                self.config.AWS_ENDPOINT,
                access_key=self.config.AWS_ACCESS_KEY,
                secret_key=self.config.AWS_SECRET_KEY,
                secure=False  # HTTP for local development
            )

            # GCP MinIO client
            self._gcp_client = Minio(
                self.config.GCP_ENDPOINT,
                access_key=self.config.GCP_ACCESS_KEY,
                secret_key=self.config.GCP_SECRET_KEY,
                secure=False  # HTTP for local development
            )

            logger.info("Multi-cloud storage clients initialized successfully")
            self._ensure_buckets_exist()
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize storage clients: {e}")
            # Don't raise exception during initialization - allow graceful degradation

    @property
    def aws_client(self):
        """Lazy loading AWS client"""
        if not self._initialized:
            self._setup_clients()
        return self._aws_client

    @property
    def gcp_client(self):
        """Lazy loading GCP client"""
        if not self._initialized:
            self._setup_clients()
        return self._gcp_client

    def _ensure_buckets_exist(self):
        """Create buckets if they don't exist"""
        try:
            # Create AWS bucket - use _aws_client directly to avoid recursion
            if not self._aws_client.bucket_exists(self.config.AWS_BUCKET):
                self._aws_client.make_bucket(self.config.AWS_BUCKET)
                logger.info(f"Created AWS bucket: {self.config.AWS_BUCKET}")

            # Create GCP bucket - use _gcp_client directly to avoid recursion
            if not self._gcp_client.bucket_exists(self.config.GCP_BUCKET):
                self._gcp_client.make_bucket(self.config.GCP_BUCKET)
                logger.info(f"Created GCP bucket: {self.config.GCP_BUCKET}")

        except S3Error as e:
            logger.error(f"Failed to create buckets: {e}")
            raise

    def get_client(self, cloud_provider: CloudProvider) -> Minio:
        """Get the appropriate client for the cloud provider"""
        if cloud_provider == CloudProvider.AWS:
            return self.aws_client
        elif cloud_provider == CloudProvider.GCP:
            return self.gcp_client
        else:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider}")

    def get_bucket_name(self, cloud_provider: CloudProvider) -> str:
        """Get the bucket name for the cloud provider"""
        if cloud_provider == CloudProvider.AWS:
            return self.config.AWS_BUCKET
        elif cloud_provider == CloudProvider.GCP:
            return self.config.GCP_BUCKET
        else:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider}")

    async def upload_file(
        self,
        file_content: bytes,
        file_name: str,
        cloud_provider: CloudProvider,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload a file to the specified cloud provider.
        Returns the file URL.
        """
        try:
            client = self.get_client(cloud_provider)
            bucket_name = self.get_bucket_name(cloud_provider)

            # Generate unique file path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"invoices/{timestamp}_{file_name}"

            # Upload file
            from io import BytesIO
            client.put_object(
                bucket_name=bucket_name,
                object_name=file_path,
                data=BytesIO(file_content),
                length=len(file_content),
                content_type=content_type
            )

            # Generate file URL
            file_url = f"http://{client._base_url}/{bucket_name}/{file_path}"

            logger.info(
                f"Uploaded {file_name} to {cloud_provider.value}: {file_url}")
            return file_url

        except S3Error as e:
            logger.error(
                f"Failed to upload file to {cloud_provider.value}: {e}")
            raise

    async def cross_cloud_transfer(
        self,
        file_path: str,
        source_cloud: CloudProvider,
        target_cloud: CloudProvider
    ) -> str:
        """
        Transfer a file between cloud providers.
        This demonstrates efficient data transfer mentioned in the brief.
        """
        try:
            source_client = self.get_client(source_cloud)
            target_client = self.get_client(target_cloud)
            source_bucket = self.get_bucket_name(source_cloud)
            target_bucket = self.get_bucket_name(target_cloud)

            # Download from source
            logger.info(f"Downloading {file_path} from {source_cloud.value}")
            response = source_client.get_object(source_bucket, file_path)
            file_data = response.read()
            response.close()
            response.release_conn()

            # Upload to target
            logger.info(f"Uploading {file_path} to {target_cloud.value}")
            from io import BytesIO
            target_client.put_object(
                bucket_name=target_bucket,
                object_name=file_path,
                data=BytesIO(file_data),
                length=len(file_data)
            )

            # Generate target URL
            target_url = f"http://{target_client._base_url}/{target_bucket}/{file_path}"

            logger.info(
                f"Successfully transferred {file_path} from {source_cloud.value} to {target_cloud.value}")
            return target_url

        except S3Error as e:
            logger.error(f"Failed to transfer file between clouds: {e}")
            raise

    async def list_files(self, cloud_provider: CloudProvider, prefix: str = "invoices/") -> List[dict]:
        """List files in the specified cloud provider"""
        try:
            client = self.get_client(cloud_provider)
            bucket_name = self.get_bucket_name(cloud_provider)

            objects = client.list_objects(
                bucket_name, prefix=prefix, recursive=True)

            files = []
            for obj in objects:
                files.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag,
                    "url": f"http://{client._base_url}/{bucket_name}/{obj.object_name}"
                })

            return files

        except S3Error as e:
            logger.error(
                f"Failed to list files from {cloud_provider.value}: {e}")
            raise

    async def download_file(self, file_path: str, cloud_provider: CloudProvider) -> bytes:
        """Download a file from the specified cloud provider"""
        try:
            client = self.get_client(cloud_provider)
            bucket_name = self.get_bucket_name(cloud_provider)

            response = client.get_object(bucket_name, file_path)
            file_data = response.read()
            response.close()
            response.release_conn()

            return file_data

        except S3Error as e:
            logger.error(
                f"Failed to download file from {cloud_provider.value}: {e}")
            raise

    async def delete_file(self, file_path: str, cloud_provider: CloudProvider) -> bool:
        """Delete a file from the specified cloud provider"""
        try:
            client = self.get_client(cloud_provider)
            bucket_name = self.get_bucket_name(cloud_provider)

            client.remove_object(bucket_name, file_path)
            logger.info(f"Deleted {file_path} from {cloud_provider.value}")
            return True

        except S3Error as e:
            logger.error(
                f"Failed to delete file from {cloud_provider.value}: {e}")
            return False

    async def get_storage_metrics(self) -> dict:
        """Get storage usage metrics across cloud providers"""
        try:
            metrics = {}

            for cloud in [CloudProvider.AWS, CloudProvider.GCP]:
                client = self.get_client(cloud)
                bucket_name = self.get_bucket_name(cloud)

                objects = client.list_objects(bucket_name, recursive=True)
                total_size = sum(obj.size for obj in objects)
                file_count = len(
                    list(client.list_objects(bucket_name, recursive=True)))

                metrics[cloud.value] = {
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "file_count": file_count,
                    "bucket_name": bucket_name
                }

            return metrics

        except Exception as e:
            logger.error(f"Failed to get storage metrics: {e}")
            return {}

    async def health_check(self) -> dict:
        """Check health of all cloud storage connections"""
        health = {}

        for cloud in [CloudProvider.AWS, CloudProvider.GCP]:
            try:
                client = self.get_client(cloud)
                bucket_name = self.get_bucket_name(cloud)

                # Try to access the bucket
                client.bucket_exists(bucket_name)
                health[f"{cloud.value}_minio"] = True

            except Exception as e:
                logger.error(f"Health check failed for {cloud.value}: {e}")
                health[f"{cloud.value}_minio"] = False

        return health


# Global instance
storage_service = MultiCloudStorageService()
