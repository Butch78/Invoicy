#!/bin/bash

# 🧾 Invoicy Development Script - ONE COMMAND TO RULE THEM ALL! 🧾
echo "🧾 Starting Invoicy Development Environment"
echo "=========================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    docker-compose stop prefect minio-aws minio-gcp postgres qdrant 2>/dev/null
    exit 0
}

# Trap Ctrl+C to cleanup
trap cleanup SIGINT SIGTERM

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

echo "✅ uv version: $(uv --version)"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker."
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "💡 Please copy .env.template to .env and configure your settings."
    exit 1
fi

echo "✅ Loading environment variables from .env"

# Install dependencies
echo "📦 Installing dependencies with uv..."
uv sync

# Start supporting services in Docker (without FastAPI)
echo "🐳 Starting supporting services in Docker..."
docker-compose up -d prefect minio-aws minio-gcp postgres qdrant

# Wait a moment for services to start
echo "⏳ Waiting for services to start..."
sleep 3

# Start the FastAPI application locally with auto-reload
echo ""
echo "🚀 Starting FastAPI with AUTO-RELOAD enabled!"
echo "============================================"
echo "   📡 API: http://localhost:8000"
echo "   📖 Docs: http://localhost:8000/docs"
echo "   ❤️  Health: http://localhost:8000/health"
echo "   🎛️  Prefect: http://localhost:4200"
echo "   ☁️  AWS MinIO: http://localhost:9011"
echo "   ☁️  GCP MinIO: http://localhost:9012"
echo ""
echo "💡 Code changes will auto-reload! Press Ctrl+C to stop."
echo ""

# Run the application with auto-reload and environment variables
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload --env-file .env