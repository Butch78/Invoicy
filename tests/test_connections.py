#!/usr/bin/env python3
"""
Simple connection test script for Invoicy services
"""

import os
import sys
import asyncio
from qdrant_client import QdrantClient
from minio import Minio


async def test_connections():
    """Test connections to all external services"""
    print("🔍 Testing Invoicy Service Connections")
    print("=" * 40)
    
    # Test Qdrant connection
    print("🔗 Testing Qdrant connection...")
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        collections = qdrant_client.get_collections()
        print(f"✅ Qdrant connected at {qdrant_host}:{qdrant_port}")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
    
    # Test MinIO AWS connection
    print("\n🔗 Testing MinIO AWS connection...")
    try:
        aws_endpoint = os.getenv("MINIO_AWS_ENDPOINT", "localhost:9001")
        aws_client = Minio(
            aws_endpoint,
            access_key=os.getenv("MINIO_AWS_ACCESS_KEY", "awsadmin"),
            secret_key=os.getenv("MINIO_AWS_SECRET_KEY", "awspassword"),
            secure=False
        )
        # Try to list buckets
        buckets = list(aws_client.list_buckets())
        print(f"✅ MinIO AWS connected at {aws_endpoint}")
    except Exception as e:
        print(f"❌ MinIO AWS connection failed: {e}")
    
    # Test MinIO GCP connection
    print("\n🔗 Testing MinIO GCP connection...")
    try:
        gcp_endpoint = os.getenv("MINIO_GCP_ENDPOINT", "localhost:9002")
        gcp_client = Minio(
            gcp_endpoint,
            access_key=os.getenv("MINIO_GCP_ACCESS_KEY", "gcpadmin"),
            secret_key=os.getenv("MINIO_GCP_SECRET_KEY", "gcppassword"),
            secure=False
        )
        # Try to list buckets
        buckets = list(gcp_client.list_buckets())
        print(f"✅ MinIO GCP connected at {gcp_endpoint}")
    except Exception as e:
        print(f"❌ MinIO GCP connection failed: {e}")
    
    # Test OpenAI key
    print("\n🔗 Testing OpenAI configuration...")
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OpenAI API key configured (ends with: ...{openai_key[-4:]})")
    else:
        print("⚠️  OpenAI API key not set (will use mock embeddings)")
    
    print("\n🎯 Environment Variables:")
    relevant_vars = [
        "QDRANT_HOST", "QDRANT_PORT", 
        "MINIO_AWS_ENDPOINT", "MINIO_GCP_ENDPOINT",
        "OPENAI_API_KEY", "PREFECT_API_URL"
    ]
    for var in relevant_vars:
        value = os.getenv(var, "NOT SET")
        if "KEY" in var and value != "NOT SET":
            value = f"...{value[-4:]}" if len(value) > 4 else "***"
        print(f"  {var}: {value}")


if __name__ == "__main__":
    asyncio.run(test_connections())