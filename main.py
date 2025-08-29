from typing import List
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from pydantic_ai import Agent, RunContext


def main():
    print("Hello from invoicy!")


if __name__ == "__main__":
    main()


# Connect to Qdrant
qdrant = QdrantClient(host="localhost", port=6333)

# Define model schema for response


class InvoiceAnswer(RunContext):
    vendor: str
    total_amount: float
    date: str


# Create an agent
agent = Agent(
    name="invoice-assistant",
    model="openai:gpt-4o-mini",   # or whichever model
    result_type=InvoiceAnswer,
)

# RAG function


def query_invoices(user_question: str) -> InvoiceAnswer:
    # Step 1: Retrieve embeddings / matches from Qdrant
    embedding = get_embedding(user_question)  # you define this
    search_results = qdrant.search(
        collection_name="invoices",
        query_vector=embedding,
        limit=3
    )

    context = [hit.payload for hit in search_results]

    # Step 2: Pass context into AI agent
    answer = agent.run_sync(
        f"User asked: {user_question}\n"
        f"Here are the top invoices: {context}\n"
        f"Answer clearly with vendor, amount, and date."
    )

    return answer


# Example
result = query_invoices("What was the total from Acme Corp last month?")
print(result)
