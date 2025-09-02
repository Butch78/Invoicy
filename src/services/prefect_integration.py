"""
Prefect integration for pipeline orchestration.
Implements efficient data pipelines mentioned in the brief for smooth AI processing.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from prefect import flow, task, get_run_logger
from prefect.client.orchestration import PrefectClient
from prefect.deployments import run_deployment
from prefect.server.schemas.core import FlowRun
from prefect.server.schemas.states import StateType

from src.models import (
    CloudProvider, ProcessingStatus, InvoiceData,
    PipelineStatus, CrossCloudTransferRequest
)
from src.services.cloud_storage import storage_service
from src.services.vector_db import vector_db_service
from src.services.parquet_embeddings import (
    parquet_embedding_service,
    cross_cloud_parquet_service
)

logger = logging.getLogger(__name__)


class PrefectService:
    """
    Service for managing Prefect workflows and pipeline orchestration.
    Demonstrates efficient data pipelines for AI processing from the brief.
    """

    def __init__(self):
        self.prefect_api_url = os.getenv(
            "PREFECT_API_URL", "http://localhost:4200")
        self.client = None

    async def get_client(self) -> PrefectClient:
        """Get Prefect client"""
        if not self.client:
            self.client = PrefectClient(api=self.prefect_api_url)
        return self.client

    async def health_check(self) -> bool:
        """Check if Prefect server is healthy"""
        try:
            async with httpx.AsyncClient() as client:
                # Use admin/version endpoint which exists in Prefect 2.x
                response = await client.get(f"{self.prefect_api_url}/health", timeout=5.0)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Prefect health check failed: {e}")
            return False

    async def trigger_invoice_pipeline(
        self,
        file_id: str,
        filename: str,
        source_cloud: CloudProvider,
        target_cloud: Optional[CloudProvider] = None
    ) -> str:
        """
        Trigger the complete invoice processing pipeline.
        Returns the flow run ID.
        """
        try:
            # Create flow run
            client = await self.get_client()

            # Parameters for the flow
            parameters = {
                "file_id": file_id,
                "filename": filename,
                "source_cloud": source_cloud.value,
                "target_cloud": target_cloud.value if target_cloud else None
            }

            # Create deployment run
            flow_run = await client.create_flow_run_from_deployment(
                deployment_id="invoice-pipeline",  # This would be created during deployment
                parameters=parameters
            )

            logger.info(
                f"Triggered invoice pipeline for {filename}, flow run: {flow_run.id}")
            return str(flow_run.id)

        except Exception as e:
            logger.error(f"Failed to trigger invoice pipeline: {e}")
            raise

    async def get_pipeline_status(self, flow_run_id: str) -> Optional[PipelineStatus]:
        """Get the status of a pipeline run"""
        try:
            client = await self.get_client()
            flow_run = await client.read_flow_run(flow_run_id)

            if not flow_run:
                return None

            # Map Prefect state to our processing status
            status_mapping = {
                StateType.PENDING: ProcessingStatus.UPLOADED,
                StateType.RUNNING: ProcessingStatus.TRANSFERRED,
                StateType.COMPLETED: ProcessingStatus.COMPLETED,
                StateType.FAILED: ProcessingStatus.FAILED,
                StateType.CANCELLED: ProcessingStatus.FAILED,
                StateType.CRASHED: ProcessingStatus.FAILED
            }

            current_status = status_mapping.get(
                flow_run.state_type, ProcessingStatus.UPLOADED)

            return PipelineStatus(
                file_id=flow_run.parameters.get("file_id", "unknown"),
                filename=flow_run.parameters.get("filename", "unknown"),
                current_status=current_status,
                cloud_source=CloudProvider(
                    flow_run.parameters.get("source_cloud", "aws")),
                cloud_target=CloudProvider(flow_run.parameters.get(
                    "target_cloud")) if flow_run.parameters.get("target_cloud") else None,
                started_at=flow_run.created,
                completed_at=flow_run.state.timestamp if flow_run.state.is_final() else None,
                error_message=flow_run.state.message if flow_run.state_type == StateType.FAILED else None
            )

        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return None

    async def list_recent_runs(self, limit: int = 10) -> list:
        """List recent pipeline runs"""
        try:
            client = await self.get_client()
            flows = await client.read_flows(limit=limit)

            runs = []
            for flow in flows:
                flow_runs = await client.read_flow_runs(
                    flow_filter={"id": {"any_": [flow.id]}},
                    limit=5
                )

                for run in flow_runs:
                    runs.append({
                        "flow_run_id": str(run.id),
                        "flow_name": flow.name,
                        "state": run.state_type.value,
                        "created": run.created,
                        "parameters": run.parameters
                    })

            return sorted(runs, key=lambda x: x["created"], reverse=True)[:limit]

        except Exception as e:
            logger.error(f"Failed to list recent runs: {e}")
            return []


# Prefect Tasks and Flows
@task
async def upload_to_cloud_task(file_content: bytes, filename: str, cloud_provider: str):
    """Task to upload file to cloud storage"""
    logger = get_run_logger()

    try:
        cloud = CloudProvider(cloud_provider)
        url = await storage_service.upload_file(file_content, filename, cloud)
        logger.info(f"Uploaded {filename} to {cloud.value}: {url}")
        return url
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise


@task
async def cross_cloud_transfer_task(file_path: str, source_cloud: str, target_cloud: str):
    """Task to transfer file between cloud providers"""
    logger = get_run_logger()

    try:
        source = CloudProvider(source_cloud)
        target = CloudProvider(target_cloud)

        url = await storage_service.cross_cloud_transfer(file_path, source, target)
        logger.info(
            f"Transferred {file_path} from {source.value} to {target.value}")
        return url
    except Exception as e:
        logger.error(f"Failed to transfer file: {e}")
        raise


@task
async def extract_invoice_data_task(file_path: str, cloud_provider: str):
    """Task to extract structured data from invoice"""
    logger = get_run_logger()

    try:
        # This is a mock extraction - in real implementation, you'd use OCR/LLM
        # to extract actual invoice data from the file

        cloud = CloudProvider(cloud_provider)

        # Mock extracted data
        invoice_data = InvoiceData(
            file_id=f"inv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            vendor_name="Sample Vendor Corp",
            vendor_address="123 Business Street, City, State 12345",
            invoice_number=f"INV-{datetime.now().strftime('%Y%m%d')}",
            invoice_date=datetime.now(),
            total_amount=1250.50,
            currency="USD",
            line_items=[
                {"description": "Professional Services",
                    "quantity": 10, "price": 125.05}
            ]
        )

        logger.info(
            f"Extracted invoice data: {invoice_data.vendor_name}, ${invoice_data.total_amount}")
        return invoice_data.model_dump()

    except Exception as e:
        logger.error(f"Failed to extract invoice data: {e}")
        raise


@task
async def store_to_postgres_task(invoice_data: dict):
    """Task to store invoice data in PostgreSQL"""
    logger = get_run_logger()

    try:
        # Mock storage to PostgreSQL
        # In real implementation, you'd use SQLAlchemy to store to actual database
        logger.info(
            f"Stored invoice {invoice_data['invoice_number']} to PostgreSQL")
        return True
    except Exception as e:
        logger.error(f"Failed to store to PostgreSQL: {e}")
        raise


@task
async def create_parquet_embedding_dataset_task(invoices_data: list, dataset_name: str = None):
    """
    Task to create a Parquet dataset with invoices and their embeddings.
    Uses efficient columnar storage with automatic compression.
    """
    logger = get_run_logger()

    try:
        # Convert dict data to InvoiceData objects
        invoices = [InvoiceData(**data) for data in invoices_data]
        
        # Create Parquet dataset
        local_file_path = await parquet_embedding_service.create_invoice_embedding_dataset(
            invoices, dataset_name
        )
        
        # Get dataset info
        dataset_info = parquet_embedding_service.get_dataset_info(local_file_path)
        
        logger.info(
            f"Created Parquet dataset: {dataset_info['num_invoices']} invoices, "
            f"{dataset_info['file_size_mb']} MB, {dataset_info['embedding_dimensions']} dimensions"
        )
        
        return {
            "local_file_path": local_file_path,
            "dataset_info": dataset_info
        }

    except Exception as e:
        logger.error(f"Failed to create Parquet embedding dataset: {e}")
        raise


@task
async def transfer_parquet_dataset_task(
    source_cloud: str,
    target_cloud: str, 
    dataset_path: str,
    local_file_path: str = None
):
    """
    Task to transfer Parquet embedding dataset between clouds.
    Much more efficient than transferring individual embeddings.
    """
    logger = get_run_logger()

    try:
        source_provider = CloudProvider(source_cloud)
        target_provider = CloudProvider(target_cloud)
        
        # If we have a local file, upload it; otherwise download and transfer
        if local_file_path:
            # Upload to target cloud
            target_url = await parquet_embedding_service.upload_dataset_to_cloud(
                local_file_path, target_provider, dataset_path
            )
            
            # Get transfer stats
            file_info = parquet_embedding_service.get_dataset_info(local_file_path)
            
            result = {
                "source": "local",
                "target_cloud": target_cloud,
                "target_url": target_url,
                "file_size_mb": file_info["file_size_mb"],
                "num_invoices": file_info["num_invoices"]
            }
        else:
            # Cross-cloud transfer
            result = await cross_cloud_parquet_service.transfer_embedding_dataset(
                source_provider, target_provider, dataset_path
            )
        
        logger.info(
            f"Transferred Parquet dataset: {result.get('file_size_mb', 'unknown')} MB to {target_cloud}"
        )
        
        return result

    except Exception as e:
        logger.error(f"Failed to transfer Parquet dataset: {e}")
        raise


@task
async def query_parquet_embeddings_task(
    dataset_path: str, 
    query_text: str, 
    cloud_provider: str = None,
    filters: dict = None,
    k: int = 5
):
    """
    Task to query embeddings from a Parquet dataset.
    Demonstrates the power of combined storage and querying.
    """
    logger = get_run_logger()

    try:
        # Download dataset if needed
        if cloud_provider:
            cloud = CloudProvider(cloud_provider)
            local_file = await parquet_embedding_service.download_dataset_from_cloud(
                dataset_path, cloud
            )
        else:
            local_file = dataset_path
        
        # Load dataset
        df = parquet_embedding_service.load_embedding_dataset(local_file)
        
        # Find similar invoices
        similar_invoices = parquet_embedding_service.find_similar_invoices(
            df, query_text, k=k, filters=filters
        )
        
        # Convert results to dict for JSON serialization
        results = []
        for row in similar_invoices.iter_rows(named=True):
            # Remove embedding from result to save space
            result_row = {k: v for k, v in row.items() if k != 'embedding'}
            results.append(result_row)
        
        logger.info(f"Found {len(results)} similar invoices for query: '{query_text}'")
        return results

    except Exception as e:
        logger.error(f"Failed to query Parquet embeddings: {e}")
        raise


@task
async def embed_and_store_task(invoice_data: dict):
    """Task to create embeddings and store in vector database"""
    logger = get_run_logger()

    try:
        # Convert dict back to InvoiceData
        invoice = InvoiceData(**invoice_data)

        # Store in vector database (keeping original functionality)
        point_id = await vector_db_service.add_invoice(invoice)
        logger.info(f"Stored invoice embeddings in Qdrant: {point_id}")
        return point_id

    except Exception as e:
        logger.error(f"Failed to embed and store: {e}")
        raise


@flow(name="invoice-processing-pipeline")
async def invoice_pipeline_flow(
    file_id: str,
    filename: str,
    source_cloud: str,
    target_cloud: str = None,
    file_content: bytes = None
):
    """
    Complete invoice processing pipeline flow.
    Demonstrates the efficient data pipeline for AI processing from the brief.
    """
    logger = get_run_logger()
    logger.info(f"Starting invoice pipeline for {filename}")

    try:
        # Step 1: Upload to source cloud (if file_content provided)
        if file_content:
            await upload_to_cloud_task(file_content, filename, source_cloud)

        # Step 2: Cross-cloud transfer (if target_cloud specified)
        file_path = f"invoices/{filename}"
        if target_cloud and target_cloud != source_cloud:
            await cross_cloud_transfer_task(file_path, source_cloud, target_cloud)
            processing_cloud = target_cloud
        else:
            processing_cloud = source_cloud

        # Step 3: Extract invoice data
        invoice_data = await extract_invoice_data_task(file_path, processing_cloud)

        # Step 4: Store structured data in PostgreSQL
        await store_to_postgres_task(invoice_data)

        # Step 5: Create embeddings and store in vector database
        await embed_and_store_task(invoice_data)

        logger.info(f"Successfully completed pipeline for {filename}")
        return {
            "status": "completed",
            "file_id": file_id,
            "filename": filename,
            "invoice_data": invoice_data
        }

    except Exception as e:
        logger.error(f"Pipeline failed for {filename}: {e}")
        raise


# Global service instance
prefect_service = PrefectService()


@flow(name="parquet-embedding-pipeline")
async def parquet_embedding_pipeline_flow(
    invoices_data: list,  # List of invoice data dicts
    source_cloud: str,
    target_cloud: str = None,
    dataset_name: str = None
):
    """
    Simplified and efficient pipeline using Parquet for embedding storage.
    Much cleaner than complex compression schemes!
    """
    logger = get_run_logger()
    logger.info(f"Starting Parquet embedding pipeline for {len(invoices_data)} invoices")

    try:
        # Step 1: Create Parquet dataset with embeddings
        dataset_result = await create_parquet_embedding_dataset_task(
            invoices_data, dataset_name
        )
        
        local_file_path = dataset_result["local_file_path"]
        dataset_info = dataset_result["dataset_info"]
        
        # Step 2: Upload to source cloud
        source_dataset_path = f"embedding_datasets/{Path(local_file_path).name}"
        await transfer_parquet_dataset_task(
            source_cloud, source_cloud, source_dataset_path, local_file_path
        )
        
        # Step 3: Transfer to target cloud if different
        transfer_stats = None
        if target_cloud and target_cloud != source_cloud:
            transfer_result = await transfer_parquet_dataset_task(
                source_cloud, target_cloud, source_dataset_path
            )
            transfer_stats = transfer_result.get("transfer_stats", {})
        
        # Step 4: Store individual invoices in vector database for querying
        storage_results = []
        for invoice_data in invoices_data:
            point_id = await embed_and_store_task(invoice_data)
            storage_results.append(point_id)
        
        # Compile results
        results = {
            "status": "completed",
            "pipeline_type": "parquet_embedding",
            "processed_invoices": len(invoices_data),
            "dataset_info": dataset_info,
            "source_cloud": source_cloud,
            "target_cloud": target_cloud,
            "local_dataset_path": local_file_path,
            "cloud_dataset_path": source_dataset_path,
            "vector_storage_results": len(storage_results)
        }
        
        if transfer_stats:
            results["transfer_stats"] = transfer_stats
        
        logger.info(
            f"Parquet pipeline completed: {dataset_info['file_size_mb']} MB dataset, "
            f"{len(storage_results)} vectors stored"
        )
        
        return results

    except Exception as e:
        logger.error(f"Parquet pipeline failed: {e}")
        raise


# Global service instance
prefect_service = PrefectService()
