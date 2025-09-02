"""
Vector database integration for RAG-based invoice querying.
Implements the AI assistant capabilities mentioned in the brief.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import ResponseHandlingException
import openai
from pydantic_ai import Agent, RunContext

from src.models import InvoiceData, VectorSearchResult, InvoiceQueryResponse

logger = logging.getLogger(__name__)


class VectorDatabaseConfig:
    """Configuration for Qdrant vector database"""
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    COLLECTION_NAME = "invoices"
    VECTOR_SIZE = 1536  # OpenAI embedding dimension
    DISTANCE_METRIC = qdrant_models.Distance.COSINE


class EmbeddingService:
    """Service for generating embeddings using OpenAI"""

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # For demo purposes, allow running without OpenAI key
            logger.warning(
                "OpenAI API key not provided - using mock embeddings")
            self.client = None
        self.model = "text-embedding-ada-002"

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text"""
        try:
            if not self.client:
                # Return dummy embedding for demo purposes when OpenAI is not available
                logger.info("Using mock embedding (OpenAI not configured)")
                return [0.1] * VectorDatabaseConfig.VECTOR_SIZE

            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return dummy embedding for demo purposes
            return [0.0] * VectorDatabaseConfig.VECTOR_SIZE

    def prepare_invoice_text(self, invoice: InvoiceData) -> str:
        """Prepare invoice data for embedding"""
        text_parts = [
            f"Vendor: {invoice.vendor_name}",
            f"Invoice Number: {invoice.invoice_number}",
            f"Date: {invoice.invoice_date.strftime('%Y-%m-%d')}",
            f"Amount: {invoice.total_amount} {invoice.currency}"
        ]

        if invoice.vendor_address:
            text_parts.append(f"Address: {invoice.vendor_address}")

        if invoice.line_items:
            items_text = ", ".join([
                f"{item.get('description', 'Item')}: {item.get('price', 0)}"
                for item in invoice.line_items
            ])
            text_parts.append(f"Items: {items_text}")

        return " | ".join(text_parts)


class VectorDatabaseService:
    """
    Service for managing invoice vectors in Qdrant.
    Implements vector search for RAG applications as mentioned in the brief.
    """

    def __init__(self):
        self.config = VectorDatabaseConfig()
        self.embedding_service = EmbeddingService()
        self._client = None
        self._initialized = False

    def _setup_client(self):
        """Initialize Qdrant client"""
        if self._initialized:
            return

        try:
            self._client = QdrantClient(
                host=self.config.QDRANT_HOST,
                port=self.config.QDRANT_PORT
            )
            logger.info("Connected to Qdrant vector database")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            # Don't raise exception during initialization

    @property
    def client(self):
        """Lazy loading Qdrant client"""
        if not self._initialized:
            self._setup_client()
        return self._client

    async def ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.config.COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=self.config.COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.config.VECTOR_SIZE,
                        distance=self.config.DISTANCE_METRIC
                    )
                )
                logger.info(
                    f"Created collection: {self.config.COLLECTION_NAME}")
            else:
                logger.info(
                    f"Collection {self.config.COLLECTION_NAME} already exists")

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    async def add_invoice(self, invoice: InvoiceData) -> str:
        """Add an invoice to the vector database"""
        try:
            await self.ensure_collection_exists()

            # Prepare text for embedding
            invoice_text = self.embedding_service.prepare_invoice_text(invoice)

            # Generate embedding
            embedding = await self.embedding_service.get_embedding(invoice_text)
            
            # Store the last generated embedding for caching purposes
            self._last_generated_embedding = embedding

            # Prepare payload
            payload = {
                "file_id": invoice.file_id,
                "vendor_name": invoice.vendor_name,
                "vendor_address": invoice.vendor_address,
                "invoice_number": invoice.invoice_number,
                "invoice_date": invoice.invoice_date.isoformat(),
                "total_amount": invoice.total_amount,
                "currency": invoice.currency,
                "line_items": invoice.line_items,
                "text_content": invoice_text,
                "indexed_at": datetime.now().isoformat()
            }

            # Generate point ID
            point_id = f"invoice_{invoice.file_id}_{invoice.invoice_number}"

            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )

            logger.info(
                f"Added invoice {invoice.invoice_number} to vector database")
            return point_id

        except Exception as e:
            logger.error(f"Failed to add invoice to vector database: {e}")
            raise

    async def add_invoice_with_embedding(self, invoice: InvoiceData, embedding: List[float]) -> str:
        """Add an invoice to the vector database using a pre-computed embedding"""
        try:
            await self.ensure_collection_exists()

            # Prepare text for reference
            invoice_text = self.embedding_service.prepare_invoice_text(invoice)

            # Prepare payload
            payload = {
                "file_id": invoice.file_id,
                "vendor_name": invoice.vendor_name,
                "vendor_address": invoice.vendor_address,
                "invoice_number": invoice.invoice_number,
                "invoice_date": invoice.invoice_date.isoformat(),
                "total_amount": invoice.total_amount,
                "currency": invoice.currency,
                "line_items": invoice.line_items,
                "text_content": invoice_text,
                "indexed_at": datetime.now().isoformat(),
                "cached_embedding": True  # Flag to indicate this used a cached embedding
            }

            # Generate point ID
            point_id = f"invoice_{invoice.file_id}_{invoice.invoice_number}"

            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )

            logger.info(
                f"Added invoice {invoice.invoice_number} to vector database using cached embedding")
            return point_id

        except Exception as e:
            logger.error(f"Failed to add invoice with embedding to vector database: {e}")
            raise

    async def search_invoices(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """Search for invoices using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.get_embedding(query)

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.config.COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )

            # Convert to VectorSearchResult objects
            results = []
            for hit in search_results:
                payload = hit.payload

                # Reconstruct InvoiceData
                invoice_data = InvoiceData(
                    file_id=payload["file_id"],
                    vendor_name=payload["vendor_name"],
                    vendor_address=payload.get("vendor_address"),
                    invoice_number=payload["invoice_number"],
                    invoice_date=datetime.fromisoformat(
                        payload["invoice_date"]),
                    total_amount=payload["total_amount"],
                    currency=payload["currency"],
                    line_items=payload.get("line_items", [])
                )

                results.append(VectorSearchResult(
                    invoice_id=hit.id,
                    similarity_score=hit.score,
                    invoice_data=invoice_data,
                    metadata={
                        "text_content": payload.get("text_content", ""),
                        "indexed_at": payload.get("indexed_at")
                    }
                ))

            logger.info(
                f"Found {len(results)} matching invoices for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Failed to search invoices: {e}")
            return []

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the invoice collection"""
        try:
            collection_info = self.client.get_collection(
                self.config.COLLECTION_NAME)
            return {
                "name": collection_info.config.params.vectors.size,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    async def delete_invoice(self, point_id: str) -> bool:
        """Delete an invoice from the vector database"""
        try:
            self.client.delete(
                collection_name=self.config.COLLECTION_NAME,
                points_selector=qdrant_models.PointIdsList(
                    points=[point_id]
                )
            )
            logger.info(f"Deleted invoice {point_id} from vector database")
            return True
        except Exception as e:
            logger.error(f"Failed to delete invoice: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False


class InvoiceRAGService:
    """
    RAG (Retrieval-Augmented Generation) service for invoice queries.
    Implements the AI assistant functionality mentioned in the brief.
    """

    def __init__(self):
        self.vector_service = VectorDatabaseService()
        self._agent = None
        self._agent_initialized = False

    def _setup_agent(self):
        """Setup the pydantic-ai agent for invoice queries"""
        if self._agent_initialized:
            return

        try:
            # Only initialize if OpenAI key is available
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning(
                    "OpenAI API key not available - RAG service will use mock responses")
                self._agent = None
                self._agent_initialized = True
                return

            self._agent = Agent(
                "openai:gpt-4o-mini",
                system_prompt="""
                You are an intelligent invoice assistant. Your job is to answer questions about invoices
                based on the retrieved invoice data provided to you.
                
                Guidelines:
                - Be precise and factual in your responses
                - Include specific invoice details (vendor, amount, date, invoice number) when relevant
                - If asked about totals or aggregations, calculate them accurately
                - If the retrieved data doesn't contain enough information, say so clearly
                - Format monetary amounts clearly with currency
                - Use dates in a readable format
                """,
            )
            self._agent_initialized = True
            logger.info("RAG agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG agent: {e}")
            self._agent = None
            self._agent_initialized = True

    @property
    def agent(self):
        """Lazy loading RAG agent"""
        if not self._agent_initialized:
            self._setup_agent()
        return self._agent

    async def query_invoices(self, question: str, limit: int = 5) -> InvoiceQueryResponse:
        """
        Process a natural language query about invoices using RAG.
        This demonstrates the AI assistant capabilities from the brief.
        """
        start_time = datetime.now()

        try:
            # Step 1: Retrieve relevant invoices using vector search
            search_results = await self.vector_service.search_invoices(
                query=question,
                limit=limit
            )

            if not search_results:
                return InvoiceQueryResponse(
                    question=question,
                    answer="I couldn't find any relevant invoices for your question.",
                    relevant_invoices=[],
                    confidence_score=0.0,
                    processing_time_ms=int(
                        (datetime.now() - start_time).total_seconds() * 1000)
                )

            # Step 2: Prepare context for the AI agent
            context_parts = []
            for result in search_results:
                invoice = result.invoice_data
                context_parts.append(
                    f"Invoice {invoice.invoice_number} from {invoice.vendor_name}: "
                    f"${invoice.total_amount} {invoice.currency} on {invoice.invoice_date.strftime('%Y-%m-%d')}"
                )

            context = "\n".join(context_parts)

            # Step 3: Generate response using AI agent
            if not self.agent:
                # Fallback response when OpenAI is not available
                answer = f"Found {len(search_results)} relevant invoice(s). " + \
                    "OpenAI integration not available for detailed analysis. " + \
                    "Here are the matching invoices: " + \
                    ", ".join([f"{r.invoice_data.vendor_name} (${r.invoice_data.total_amount})"
                               for r in search_results[:3]])
            else:
                prompt = f"""
                User question: {question}
                
                Relevant invoices found:
                {context}
                
                Please provide a clear, helpful answer based on this invoice data.
                """

                result = self.agent.run_sync(prompt)
                answer = str(result.data) if hasattr(
                    result, 'data') else str(result)

            # Calculate confidence score based on similarity scores
            avg_similarity = sum(
                r.similarity_score for r in search_results) / len(search_results)
            # Boost and cap at 1.0
            confidence_score = min(avg_similarity * 1.2, 1.0)

            processing_time = int(
                (datetime.now() - start_time).total_seconds() * 1000)

            return InvoiceQueryResponse(
                question=question,
                answer=answer,
                relevant_invoices=[r.invoice_data for r in search_results],
                confidence_score=confidence_score,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Failed to process invoice query: {e}")
            return InvoiceQueryResponse(
                question=question,
                answer=f"I encountered an error while processing your question: {str(e)}",
                relevant_invoices=[],
                confidence_score=0.0,
                processing_time_ms=int(
                    (datetime.now() - start_time).total_seconds() * 1000)
            )


# Global instances
vector_db_service = VectorDatabaseService()
rag_service = InvoiceRAGService()
