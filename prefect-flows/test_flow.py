"""
Simple test flow to verify Prefect orchestration is working
"""
from prefect import flow, task
import time
from datetime import datetime


@task
def fetch_data(source: str):
    """Simulate fetching data from a source"""
    print(f"📥 Fetching data from {source}...")
    time.sleep(1)  # Simulate some work
    return {"source": source, "timestamp": datetime.now().isoformat(), "data": "sample_data"}


@task  
def transform_data(data: dict):
    """Simulate data transformation"""
    print(f"🔄 Transforming data: {data['source']}")
    time.sleep(1)  # Simulate processing
    transformed = {
        **data,
        "processed": True,
        "processed_at": datetime.now().isoformat()
    }
    return transformed


@task
def load_data(data: dict):
    """Simulate loading data to destination"""
    print(f"💾 Loading processed data from {data['source']}")
    time.sleep(0.5)  # Simulate loading
    print(f"✅ Successfully processed data: {data}")
    return "SUCCESS"


@flow(name="test-pipeline")
def test_pipeline(source: str = "test-source"):
    """A simple ETL pipeline for testing Prefect orchestration"""
    print(f"🚀 Starting test pipeline for source: {source}")
    
    # ETL steps
    raw_data = fetch_data(source)
    transformed_data = transform_data(raw_data)
    result = load_data(transformed_data)
    
    print(f"🎉 Pipeline completed with result: {result}")
    return result


if __name__ == "__main__":
    # Run the flow
    test_pipeline("invoice-system")