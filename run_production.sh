#!/bin/bash

# 🧾 Invoicy Production Deployment Script 🧾
echo "🧾 Starting Invoicy Production Deployment"
echo "========================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Stopping production deployment..."
    docker-compose down
    exit 0
}

# Trap Ctrl+C to cleanup
trap cleanup SIGINT SIGTERM

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "💡 Please create .env file with your production settings."
    echo "   You can copy from .env.template and update the values."
    exit 1
fi

echo "✅ Found .env file"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker."
    exit 1
fi

# Check if OpenAI API key is set
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "⚠️  WARNING: OpenAI API key might not be properly configured in .env"
    echo "   Make sure OPENAI_API_KEY is set for AI features to work."
fi

# Build production images
echo "🏗️  Building production Docker images..."
docker-compose build --no-cache

# Start all services in production mode
echo "🚀 Starting all services in production mode..."
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
echo ""

# Check FastAPI health
if curl -f http://localhost:8000/health &>/dev/null; then
    echo "✅ FastAPI service is healthy"
else
    echo "❌ FastAPI service is not responding"
fi

# Check Prefect
if curl -f http://localhost:4200 &>/dev/null; then
    echo "✅ Prefect service is healthy"
else
    echo "❌ Prefect service is not responding"
fi

# Check MinIO services
if curl -f http://localhost:9011 &>/dev/null; then
    echo "✅ MinIO AWS service is healthy"
else
    echo "❌ MinIO AWS service is not responding"
fi

if curl -f http://localhost:9012 &>/dev/null; then
    echo "✅ MinIO GCP service is healthy"
else
    echo "❌ MinIO GCP service is not responding"
fi

echo ""
echo "🎉 Production deployment complete!"
echo "=================================="
echo "   📡 API: http://localhost:8000"
echo "   📖 API Docs: http://localhost:8000/docs"
echo "   ❤️  Health: http://localhost:8000/health"
echo "   🎛️  Prefect UI: http://localhost:4200"
echo "   ☁️  AWS MinIO: http://localhost:9011 (awsadmin/awspassword)"
echo "   ☁️  GCP MinIO: http://localhost:9012 (gcpadmin/gcppassword)"
echo ""
echo "📊 View logs: docker-compose logs -f"
echo "🛑 Stop services: docker-compose down"
echo "🔄 Restart API: docker-compose restart invoicy-api"
echo ""
echo "💡 Running in PRODUCTION mode (no auto-reload)"
echo "   For development with auto-reload, use: ./run_local.sh"
echo ""
echo "Press Ctrl+C to stop all services..."

# Keep the script running and show logs
docker-compose logs -f