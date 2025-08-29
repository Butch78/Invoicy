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

## 🎯 Perfect Example for the Brief

This project demonstrates exactly what was mentioned in the brief about the tech stack and requirements:

### ✅ **Multi-Cloud Data Storage**
> *"we'll host our models on GCP, whereas the base app is hosted on AWS so there will be data storing on both platforms"*

- **AWS simulation**: MinIO bucket on port 9001
- **GCP simulation**: MinIO bucket on port 9002  
- **Cross-cloud transfer**: Efficient data movement between providers

### ✅ **Efficient Data Pipelines**
> *"efficient data pipelines will be needed to ensure smooth processing of the AI module as well as semi-automated re-training"*

- **Prefect orchestration**: Complete pipeline automation
- **Background processing**: Async file processing with FastAPI
- **Pipeline monitoring**: Real-time status tracking and metrics

### ✅ **Vector Databases for RAG Applications**  
> *"building vector databases for RAG applications in the domain of AI assistant"*

- **Qdrant integration**: Vector storage for invoice embeddings
- **Semantic search**: Natural language invoice queries
- **AI assistant**: pydantic-ai powered RAG responses

---

## 🚀 Features
- **Multi-cloud simulation** with two MinIO buckets (AWS + GCP)  
- **Prefect** (pipeline orchestration)
- **Invoice ingestion**: upload PDFs/CSVs via FastAPI  
- **Data extraction** with OCR/LLM (prototype step)  
- **Storage**: raw data in MinIO, structured data in Postgres  
- **Vector search**: embeddings stored in Qdrant DB  
- **AI Assistant (RAG)**: query invoices in natural language  
- **FastAPI REST API**: comprehensive endpoints for all operations
- **pydantic-ai** (for structured RAG interfaces)
- **Docker Compose**: complete environment orchestration

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   AWS MinIO     │◄──►│   GCP MinIO     │
│  (Port 9001)    │    │  (Port 9002)    │
└─────────────────┘    └─────────────────┘
         ▲                       ▲
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│          FastAPI Application            │
│            (Port 8000)                  │
│  • File Upload/Download                 │
│  • Cross-Cloud Transfer                 │
│  • Pipeline Orchestration               │
│  • AI Query Interface                   │
└─────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│   Prefect    │ │ PostgreSQL  │ │   Qdrant     │
│ (Port 4200)  │ │ (Port 5432) │ │ (Port 6333)  │
│ Orchestrator │ │ Structured  │ │ Vector DB    │
│              │ │    Data     │ │  (RAG)       │
└──────────────┘ └─────────────┘ └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key (for AI features)

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/invoicy.git
cd invoicy

# Copy environment template
cp .env.template .env

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### 2. Start the Stack
```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8000/health
```

### 3. Access the Services
- **FastAPI Docs**: http://localhost:8000/docs
- **Prefect UI**: http://localhost:4200
- **MinIO AWS Console**: http://localhost:9011 (awsadmin/awspassword)
- **MinIO GCP Console**: http://localhost:9012 (gcpadmin/gcppassword)

---

## 📚 API Endpoints

### 🔄 **Multi-Cloud Operations**
```bash
# Upload invoice to AWS
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.pdf" \
  -F "target_cloud=aws"

# Transfer between clouds  
curl -X POST "http://localhost:8000/transfer" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "inv_abc123",
    "source_cloud": "aws", 
    "target_cloud": "gcp"
  }'

# List files in cloud
curl "http://localhost:8000/files/aws"
```

### 🤖 **AI Assistant (RAG)**
```bash
# Query invoices with natural language
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the total amount from Acme Corp last month?",
    "limit": 5
  }'
```

### 📊 **Pipeline Monitoring**
```bash
# Get pipeline metrics
curl "http://localhost:8000/metrics"

# List recent pipeline runs
curl "http://localhost:8000/pipeline/runs"

# Check pipeline status
curl "http://localhost:8000/pipeline/status/{flow_run_id}"
```

---

## 🎯 Key Demonstrations

### 1. **Multi-Cloud Data Transfer**
Shows efficient data movement between AWS and GCP environments, addressing the brief's requirement for data storage on both platforms.

### 2. **Pipeline Orchestration**
Prefect-based workflows demonstrate automated processing chains for AI model data preparation and semi-automated retraining scenarios.

### 3. **Vector Database RAG**
Qdrant integration shows how to build vector databases for RAG applications, exactly as mentioned in the brief for AI assistant domains.

### 4. **Scalable Architecture**
FastAPI async architecture demonstrates how to handle enterprise-scale data processing with proper monitoring and health checks.

---

## 🔧 Development

### Local Development
```bash
# Install dependencies
pip install -e .

# Run services (without FastAPI)
docker-compose up -d prefect minio-aws minio-gcp postgres qdrant

# Run FastAPI locally
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Adding New Features
- **Cloud Providers**: Extend `CloudProvider` enum and add new storage services
- **File Types**: Add support for new formats in upload validation
- **AI Models**: Integrate different LLM providers in the vector database service
- **Pipeline Steps**: Add new Prefect tasks for additional processing

---

## 🎯 Perfect for Data Engineering Roles

This project showcases exactly the skills mentioned in the brief:

- ✅ **Multi-cloud architecture** (AWS + GCP)
- ✅ **Efficient data transfer** between cloud providers  
- ✅ **Data pipeline orchestration** with monitoring
- ✅ **Vector databases** for AI applications
- ✅ **RAG implementation** for intelligent assistants
- ✅ **Modern Python stack** (FastAPI, Pydantic, async/await)
- ✅ **Production-ready patterns** (Docker, health checks, metrics)

---

## 📈 Next Steps

This foundation is ready for:
- **Production deployment** on actual AWS/GCP infrastructure
- **Team scaling** with proper CI/CD pipelines  
- **Advanced AI features** (custom models, retraining pipelines)
- **Enterprise integrations** (authentication, audit logs, compliance)

Perfect example of the tech stack and capabilities needed for the data engineering role mentioned in the brief! 🚀


# Development Commands 🛠️

## 🚀 **ONE COMMAND FOR EVERYTHING**

For the best development experience with **true auto-reload**, use this single command:

```bash
# Start everything with auto-reload (RECOMMENDED)
./run_local.sh
```

This command will:
- ✅ Install dependencies with `uv`
- ✅ Start all supporting services in Docker (Prefect, MinIO, PostgreSQL, Qdrant)
- ✅ Run FastAPI locally with **true auto-reload**
- ✅ Show you all the important URLs
- ✅ Auto-restart when you change any Python code

**Access your services:**
- 📡 **API**: http://localhost:8000
- 📖 **API Docs**: http://localhost:8000/docs  
- ❤️ **Health Check**: http://localhost:8000/health
- 🎛️ **Prefect UI**: http://localhost:4200
- ☁️ **AWS MinIO**: http://localhost:9011 (awsadmin/awspassword)
- ☁️ **GCP MinIO**: http://localhost:9012 (gcpadmin/gcppassword)

**To stop everything**: Press `Ctrl+C` (will cleanup Docker services automatically)

---

## 🔧 **Alternative Docker-Only Development** 

If you prefer Docker-only development (no auto-reload):

```bash
# Start all services in Docker
docker-compose up -d

# After code changes, restart FastAPI (fast ~2-3 seconds)
docker-compose restart invoicy-api

# View logs
docker-compose logs invoicy-api -f
```

## 🧹 **Cleanup Commands**
```bash
# Stop all Docker services
docker-compose down

# Clean up everything (free up space)
docker-compose down -v && docker system prune -f
``` 