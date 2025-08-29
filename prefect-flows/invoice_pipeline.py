from prefect import flow, task
import boto3


@task
def copy_from_gcp_to_aws(file_name: str):
    print(f"Copying {file_name} from GCP MinIO to AWS MinIO...")
    # Example: use boto3 to connect to MinIO endpoints
    # (both configured with S3-compatible API)


@task
def extract_invoice(file_name: str):
    print(f"Extracting structured data from {file_name}...")
    return {"vendor": "Acme Corp", "amount": 1500}


@task
def store_to_postgres(invoice_data: dict):
    print(f"Storing invoice in Postgres: {invoice_data}")


@task
def embed_and_store(invoice_data: dict):
    print(f"Creating embedding and storing in Qdrant...")


@flow
def invoice_pipeline(file_name: str):
    invoice = copy_from_gcp_to_aws(file_name)
    data = extract_invoice(invoice)
    store_to_postgres(data)
    embed_and_store(data)


if __name__ == "__main__":
    invoice_pipeline("invoice123.pdf")
