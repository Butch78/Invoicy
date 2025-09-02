#!/usr/bin/env python3
"""
Deploy Prefect flows for the Invoicy project.
This script creates deployments that can be run by Prefect workers.
"""

import asyncio
from prefect import serve
from prefect.client.orchestration import get_client

# Import your flows
from prefect_flows.invoice_pipeline import invoice_pipeline
from prefect_flows.test_flow import test_pipeline
from src.services.prefect_integration import invoice_pipeline_flow


async def deploy_flows():
    """Deploy all flows to Prefect"""
    
    print("🚀 Deploying Prefect flows...")
    
    # Option 1: Simple serve() method for development
    # This creates temporary deployments that run as long as this script runs
    await serve(
        # Deploy the invoice pipeline flow
        invoice_pipeline_flow.to_deployment(
            name="invoice-processing-deployment",
            description="Complete invoice processing pipeline with multi-cloud support",
            tags=["invoice", "processing", "ai", "multi-cloud"],
            work_pool_name="default",
            parameters={
                "source_cloud": "aws",
                "target_cloud": None
            }
        ),
        
        # Deploy the simple test flow
        test_pipeline.to_deployment(
            name="test-pipeline-deployment", 
            description="Simple test pipeline for validation",
            tags=["test", "validation"],
            work_pool_name="default",
            parameters={
                "source": "test-source"
            }
        ),
        
        # Deploy the basic invoice pipeline
        invoice_pipeline.to_deployment(
            name="basic-invoice-pipeline",
            description="Basic invoice pipeline without multi-cloud features",
            tags=["invoice", "basic"],
            work_pool_name="default",
            parameters={
                "file_name": "sample.pdf"
            }
        ),
        
        # Keep the deployments running
        limit=10  # Max concurrent flow runs
    )


async def create_persistent_deployments():
    """Create persistent deployments that survive process restarts"""
    
    print("📋 Creating persistent deployments...")
    
    async with get_client() as client:
        
        # Create deployment for invoice processing pipeline
        deployment_1 = await invoice_pipeline_flow.to_deployment(
            name="invoice-processing-persistent",
            description="Persistent deployment for invoice processing",
            tags=["invoice", "processing", "persistent"],
            work_pool_name="default",
            parameters={
                "source_cloud": "aws",
                "target_cloud": None
            }
        ).apply()
        print(f"✅ Created deployment: {deployment_1.name} (ID: {deployment_1.id})")
        
        # Create deployment for test pipeline
        deployment_2 = await test_pipeline.to_deployment(
            name="test-pipeline-persistent",
            description="Persistent test pipeline deployment",
            tags=["test", "persistent"],
            work_pool_name="default"
        ).apply()
        print(f"✅ Created deployment: {deployment_2.name} (ID: {deployment_2.id})")
        
        # Create deployment for basic invoice pipeline  
        deployment_3 = await invoice_pipeline.to_deployment(
            name="basic-invoice-persistent",
            description="Persistent basic invoice pipeline",
            tags=["invoice", "basic", "persistent"], 
            work_pool_name="default"
        ).apply()
        print(f"✅ Created deployment: {deployment_3.name} (ID: {deployment_3.id})")
        
        print("\n🎉 All deployments created successfully!")
        print("\n📊 You can now:")
        print("   - View deployments in Prefect UI: http://localhost:4200")
        print("   - Run flows via the UI or API")
        print("   - Schedule flows for automated execution")


async def main():
    """Main deployment function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--persistent":
        await create_persistent_deployments()
    else:
        print("🔄 Starting serve mode (deployments active while this runs)")
        print("💡 Use --persistent flag to create permanent deployments")
        await deploy_flows()


if __name__ == "__main__":
    asyncio.run(main())