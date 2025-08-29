# Invoicy 🧾  
**Open-source invoice intelligence pipeline**  

Invoicy is a lightweight, open-source demo of an **AI-powered invoice pipeline**.  
It simulates a **multi-cloud environment** (AWS & GCP) using Docker and shows how invoices can be:  

- Ingested into cloud buckets  
- Transferred between cloud environments  
- Processed into structured data  
- Embedded into a **vector database** for semantic search  
- Queried via an **AI assistant (RAG)**  

This project is inspired by modern **data engineering practices** and the growing need for **intelligent invoice processing**.  

---

## 🚀 Features
- **Multi-cloud simulation** with two MinIO buckets (AWS + GCP)  
- **Prefect** (pipeline orchestration)
- **Invoice ingestion**: upload PDFs/CSVs  
- **Data extraction** with OCR/LLM (basic prototype step)  
- **Storage**: raw data in MinIO, structured data in Postgres  
- **Vector search**: embeddings stored in Qdrant DB  
- **AI Assistant (RAG)**: query invoices in natural language  
- **Streamlit UI**: simple frontend to upload invoices and chat  
- **pydantic-ai** (for structured RAG interfaces)
- **Qdrant** (vector database)

---

## 🏗️ Architecture
