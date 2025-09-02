"""
Simplified embedding storage using Parquet files with Polars.
Based on Max Woolf's excellent approach for portable, efficient embedding storage.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import polars as pl
import numpy as np

from src.models import CloudProvider, InvoiceData
from src.services.cloud_storage import storage_service
from src.services.vector_db import EmbeddingService

logger = logging.getLogger(__name__)


class ParquetEmbeddingService:
    """
    Service for storing embeddings efficiently in Parquet format.
    Combines embeddings with metadata for portable, queryable storage.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.local_cache_dir = Path("./embedding_cache")
        self.local_cache_dir.mkdir(exist_ok=True)
    
    async def create_invoice_embedding_dataset(
        self, 
        invoices: List[InvoiceData],
        dataset_name: str = None
    ) -> str:
        """
        Create a Parquet dataset containing invoices and their embeddings.
        Returns the local file path.
        """
        try:
            if not dataset_name:
                dataset_name = f"invoices_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare data for each invoice
            rows = []
            embeddings_list = []
            
            logger.info(f"Processing {len(invoices)} invoices for embedding dataset")
            
            for invoice in invoices:
                # Prepare text for embedding
                invoice_text = self.embedding_service.prepare_invoice_text(invoice)
                
                # Generate embedding
                embedding = await self.embedding_service.get_embedding(invoice_text)
                embeddings_list.append(embedding)
                
                # Prepare row data
                row = {
                    "file_id": invoice.file_id,
                    "vendor_name": invoice.vendor_name,
                    "vendor_address": invoice.vendor_address,
                    "invoice_number": invoice.invoice_number,
                    "invoice_date": invoice.invoice_date,
                    "total_amount": invoice.total_amount,
                    "currency": invoice.currency,
                    "line_items": invoice.line_items,  # Will be stored as JSON
                    "text_content": invoice_text,
                    "created_at": datetime.now(),
                }
                rows.append(row)
            
            # Create DataFrame with embeddings
            df = pl.DataFrame(rows)
            
            # Add embeddings as a column (Polars handles this beautifully)
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            df = df.with_columns(
                embedding=pl.Series("embedding", embeddings_array.tolist())
            )
            
            # Save to Parquet
            file_path = self.local_cache_dir / f"{dataset_name}.parquet"
            df.write_parquet(file_path)
            
            # Log stats
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Created embedding dataset: {len(invoices)} invoices, "
                f"{file_size_mb:.2f} MB, saved to {file_path}"
            )
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to create embedding dataset: {e}")
            raise
    
    def load_embedding_dataset(self, file_path: str) -> pl.DataFrame:
        """Load an embedding dataset from Parquet file."""
        try:
            df = pl.read_parquet(file_path)
            logger.info(f"Loaded embedding dataset: {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load embedding dataset: {e}")
            raise
    
    def find_similar_invoices(
        self, 
        df: pl.DataFrame, 
        query_text: str, 
        k: int = 5,
        filters: Dict[str, Any] = None
    ) -> pl.DataFrame:
        """
        Find similar invoices using efficient dot product similarity.
        Supports filtering before similarity calculation.
        """
        try:
            # Generate query embedding
            query_embedding = asyncio.run(
                self.embedding_service.get_embedding(query_text)
            )
            query_array = np.array(query_embedding, dtype=np.float32)
            
            # Apply filters if provided
            filtered_df = df
            if filters:
                for column, value in filters.items():
                    if isinstance(value, str):
                        filtered_df = filtered_df.filter(
                            pl.col(column).str.contains(value)
                        )
                    elif isinstance(value, (int, float)):
                        filtered_df = filtered_df.filter(pl.col(column) == value)
                    elif isinstance(value, tuple) and len(value) == 2:
                        # Range filter: (min, max)
                        filtered_df = filtered_df.filter(
                            pl.col(column).is_between(value[0], value[1])
                        )
            
            if len(filtered_df) == 0:
                logger.warning("No invoices match the provided filters")
                return pl.DataFrame()
            
            # Extract embeddings matrix (zero-copy with Polars!)
            embeddings_matrix = filtered_df["embedding"].to_numpy(allow_copy=False)
            
            # Fast dot product similarity
            similarities = query_array @ embeddings_matrix.T
            
            # Get top-k indices
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
            
            # Return similar invoices with similarity scores
            similar_invoices = filtered_df[top_k_indices].with_columns(
                similarity_score=pl.Series("similarity_score", similarities[top_k_indices])
            )
            
            logger.info(f"Found {len(similar_invoices)} similar invoices")
            return similar_invoices
            
        except Exception as e:
            logger.error(f"Failed to find similar invoices: {e}")
            raise
    
    async def upload_dataset_to_cloud(
        self, 
        local_file_path: str, 
        cloud_provider: CloudProvider,
        remote_path: str = None
    ) -> str:
        """Upload Parquet dataset to cloud storage."""
        try:
            if not remote_path:
                filename = Path(local_file_path).name
                remote_path = f"embedding_datasets/{filename}"
            
            # Read file content
            with open(local_file_path, 'rb') as f:
                file_content = f.read()
            
            # Upload to cloud
            url = await storage_service.upload_file(
                file_content, 
                remote_path, 
                cloud_provider,
                content_type="application/octet-stream"
            )
            
            logger.info(f"Uploaded dataset to {cloud_provider.value}: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to upload dataset to cloud: {e}")
            raise
    
    async def download_dataset_from_cloud(
        self, 
        remote_path: str, 
        cloud_provider: CloudProvider,
        local_file_path: str = None
    ) -> str:
        """Download Parquet dataset from cloud storage."""
        try:
            if not local_file_path:
                filename = Path(remote_path).name
                local_file_path = str(self.local_cache_dir / filename)
            
            # Download from cloud
            file_content = await storage_service.download_file(remote_path, cloud_provider)
            
            # Save locally
            with open(local_file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Downloaded dataset from {cloud_provider.value} to {local_file_path}")
            return local_file_path
            
        except Exception as e:
            logger.error(f"Failed to download dataset from cloud: {e}")
            raise
    
    def merge_datasets(self, file_paths: List[str], output_path: str = None) -> str:
        """Merge multiple Parquet datasets into one."""
        try:
            if not output_path:
                output_path = str(self.local_cache_dir / f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
            
            # Load and concatenate all datasets
            dfs = [pl.read_parquet(path) for path in file_paths]
            merged_df = pl.concat(dfs)
            
            # Remove duplicates based on file_id
            merged_df = merged_df.unique(subset=["file_id"])
            
            # Save merged dataset
            merged_df.write_parquet(output_path)
            
            logger.info(f"Merged {len(file_paths)} datasets into {output_path} ({len(merged_df)} total invoices)")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to merge datasets: {e}")
            raise
    
    def get_dataset_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a Parquet dataset."""
        try:
            df = pl.read_parquet(file_path)
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            
            return {
                "file_path": file_path,
                "num_invoices": len(df),
                "file_size_mb": round(file_size_mb, 2),
                "columns": df.columns,
                "date_range": (
                    df["invoice_date"].min(),
                    df["invoice_date"].max()
                ) if "invoice_date" in df.columns else None,
                "vendors": df["vendor_name"].n_unique() if "vendor_name" in df.columns else None,
                "total_amount_sum": df["total_amount"].sum() if "total_amount" in df.columns else None,
                "embedding_dimensions": len(df["embedding"][0]) if len(df) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            return {}


class CrossCloudParquetService:
    """
    Service for efficiently transferring Parquet embedding datasets between clouds.
    Much simpler than the previous compression approach!
    """
    
    def __init__(self):
        self.parquet_service = ParquetEmbeddingService()
    
    async def transfer_embedding_dataset(
        self,
        source_cloud: CloudProvider,
        target_cloud: CloudProvider,
        remote_path: str,
        invoices: List[InvoiceData] = None
    ) -> Dict[str, Any]:
        """
        Transfer embedding dataset between clouds efficiently.
        If invoices provided, creates new dataset; otherwise downloads existing.
        """
        try:
            if invoices:
                # Create new dataset locally
                logger.info(f"Creating new embedding dataset with {len(invoices)} invoices")
                local_file = await self.parquet_service.create_invoice_embedding_dataset(invoices)
            else:
                # Download existing dataset
                logger.info(f"Downloading existing dataset from {source_cloud.value}")
                local_file = await self.parquet_service.download_dataset_from_cloud(
                    remote_path, source_cloud
                )
            
            # Get file info
            file_info = self.parquet_service.get_dataset_info(local_file)
            original_size_mb = file_info["file_size_mb"]
            
            # Upload to target cloud (Parquet compression is automatic)
            target_url = await self.parquet_service.upload_dataset_to_cloud(
                local_file, target_cloud, remote_path
            )
            
            # Calculate transfer efficiency
            # Parquet typically achieves 2-3x compression for mixed data
            estimated_uncompressed_mb = original_size_mb * 2.5
            compression_ratio = (1 - original_size_mb / estimated_uncompressed_mb) * 100
            
            # Estimate cost savings (AWS egress pricing)
            cost_per_gb = 0.09
            savings_usd = ((estimated_uncompressed_mb - original_size_mb) / 1024) * cost_per_gb
            
            result = {
                "source_cloud": source_cloud.value,
                "target_cloud": target_cloud.value,
                "target_url": target_url,
                "local_file": local_file,
                "file_info": file_info,
                "transfer_stats": {
                    "file_size_mb": original_size_mb,
                    "estimated_compression_ratio": round(compression_ratio, 1),
                    "estimated_cost_savings_usd": round(savings_usd, 4),
                    "transfer_method": "parquet_native_compression"
                }
            }
            
            logger.info(
                f"Successfully transferred dataset: {original_size_mb}MB, "
                f"~{compression_ratio:.1f}% compression, "
                f"~${savings_usd:.4f} savings"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to transfer embedding dataset: {e}")
            raise


# Global instances
parquet_embedding_service = ParquetEmbeddingService()
cross_cloud_parquet_service = CrossCloudParquetService()