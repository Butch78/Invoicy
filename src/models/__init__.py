"""
Data models for the Invoicy multi-cloud invoice pipeline.
Demonstrates the tech stack mentioned in the brief: multi-cloud storage, data pipelines, and AI.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class CloudProvider(str, Enum):
    """Enum for cloud providers - simulating AWS and GCP environments"""
    AWS = "aws"
    GCP = "gcp"


class FileType(str, Enum):
    """Supported file types for invoice processing"""
    PDF = "pdf"
    CSV = "csv"
    PNG = "png"
    JPG = "jpg"


class ProcessingStatus(str, Enum):
    """Status of invoice processing pipeline"""
    UPLOADED = "uploaded"
    TRANSFERRED = "transferred"
    EXTRACTED = "extracted"
    STORED = "stored"
    EMBEDDED = "embedded"
    COMPLETED = "completed"
    FAILED = "failed"


# Request/Response Models for API
class FileUploadRequest(BaseModel):
    """Request model for file upload"""
    target_cloud: CloudProvider = Field(
        description="Target cloud provider (AWS or GCP)")
    process_immediately: bool = Field(
        default=True, description="Whether to trigger processing pipeline")


class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    file_id: str
    filename: str
    cloud_provider: CloudProvider
    bucket_url: str
    status: ProcessingStatus
    message: str


class InvoiceData(BaseModel):
    """Structured invoice data extracted from files"""
    file_id: str
    vendor_name: str
    vendor_address: Optional[str] = None
    invoice_number: str
    invoice_date: datetime
    due_date: Optional[datetime] = None
    total_amount: float
    currency: str = "USD"
    line_items: List[Dict[str, Any]] = []
    tax_amount: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "inv_123",
                "vendor_name": "Acme Corp",
                "vendor_address": "123 Business St, City, State",
                "invoice_number": "INV-2024-001",
                "invoice_date": "2024-08-15T00:00:00",
                "total_amount": 1500.00,
                "currency": "USD",
                "line_items": [
                    {"description": "Software License",
                        "quantity": 1, "price": 1500.00}
                ]
            }
        }


class InvoiceQueryRequest(BaseModel):
    """Request model for RAG-based invoice queries"""
    question: str = Field(
        description="Natural language question about invoices")
    limit: int = Field(default=5, ge=1, le=20,
                       description="Number of results to return")
    date_filter: Optional[str] = Field(
        default=None, description="Date filter (e.g., 'last month', '2024')")


class InvoiceQueryResponse(BaseModel):
    """Response model for invoice queries using RAG"""
    question: str
    answer: str
    relevant_invoices: List[InvoiceData]
    confidence_score: float
    processing_time_ms: int


class PipelineStatus(BaseModel):
    """Status of the data pipeline processing"""
    file_id: str
    filename: str
    current_status: ProcessingStatus
    cloud_source: CloudProvider
    cloud_target: Optional[CloudProvider] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    pipeline_steps: Dict[str, bool] = {
        "uploaded": False,
        "transferred": False,
        "extracted": False,
        "stored_postgres": False,
        "embedded_qdrant": False
    }


class CrossCloudTransferRequest(BaseModel):
    """Request to transfer data between cloud providers (AWS <-> GCP)"""
    file_id: str
    source_cloud: CloudProvider
    target_cloud: CloudProvider
    sync_mode: bool = Field(
        default=False, description="Whether to wait for completion")


class VectorSearchResult(BaseModel):
    """Result from vector database search"""
    invoice_id: str
    similarity_score: float
    invoice_data: InvoiceData
    metadata: Dict[str, Any] = {}


class HealthCheckResponse(BaseModel):
    """Health check response for monitoring"""
    status: str
    timestamp: datetime
    services: Dict[str, bool] = {
        "aws_minio": False,
        "gcp_minio": False,
        "postgres": False,
        "qdrant": False,
        "prefect": False
    }
    version: str = "1.0.0"


class DataPipelineMetrics(BaseModel):
    """Metrics for monitoring the data pipeline performance"""
    total_files_processed: int
    files_by_status: Dict[ProcessingStatus, int]
    avg_processing_time_seconds: float
    cross_cloud_transfers: int
    storage_usage_mb: Dict[CloudProvider, float]
    vector_db_size: int
    last_updated: datetime
