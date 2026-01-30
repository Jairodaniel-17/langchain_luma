# langchain-luma

[![PyPI version](https://badge.fury.io/py/langchain-luma.svg)](https://badge.fury.io/py/langchain-luma)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust Python client SDK for **Luma (RustKissVDB)**, designed for high-performance vector search and seamless integration with the LangChain ecosystem. 

Luma is not just a vector database; it's a modular system offering health monitoring, collection management, and efficient vector operations.

## Installation

Install the package from PyPI:

```bash
pip install langchain-luma

```

To include dependencies for LangChain integration:

```bash
pip install "langchain-luma[langchain]"

```

## Quick Start (Python SDK)

The core `LumaClient` provides direct access to the Luma API, organized into namespaces (e.g., `vectors`, `system`) for clarity.

```python
from langchain_luma import LumaClient
import time

# 1. Initialize the client
# Point this to your Luma instance (e.g., http://localhost:8000 or your production URL)
luma = LumaClient(url="http://localhost:1234", api_key="dev")

# 2. System Check
# Verify the database is reachable and running
health_status = luma.system.health()
print(f"Luma System Status: {health_status}")

# 3. Create a Collection
# Define the vector space (e.g., 384 dimensions for standard transformers)
try:
    luma.vectors.create_collection(name="knowledge_base", dim=384, metric="cosine")
    print("Collection 'knowledge_base' created.")
except Exception as e:
    print(f"Note: Collection might already exist. Details: {e}")

# 4. Upsert Data (Vectors + Metadata)
# Insert a dummy vector for demonstration
vector_data = [0.1] * 384
luma.vectors.upsert(
    collection="knowledge_base",
    id="doc_unique_id_001",
    vector=vector_data,
    meta={
        "source": "manual_entry",
        "category": "documentation",
        "timestamp": time.time()
    }
)
print("Vector inserted successfully.")

# 5. Semantic Search
results = luma.vectors.search(
    collection="knowledge_base",
    vector=vector_data,
    k=3  # Top 3 similar results
)

print("\n--- Search Results ---")
for hit in results:
    print(f"ID: {hit.id} | Score: {hit.score:.4f} | Meta: {hit.payload}")

```

## LangChain Integration

`langchain-luma` provides a first-class `VectorStore` implementation, allowing you to use Luma as a backend for RAG (Retrieval-Augmented Generation) pipelines, conversational memory, and more.

### Setup

```python
from langchain_luma.langchain.vectorstore import LumaVectorStore
from langchain_core.documents import Document
# Using FakeEmbeddings for demo purposes. In production, use OpenAIEmbeddings, HuggingFaceEmbeddings, etc.
from langchain_community.embeddings import FakeEmbeddings 

# Initialize embeddings (dimension must match your collection)
embeddings = FakeEmbeddings(size=384)

# Initialize the Luma Vector Store
vector_store = LumaVectorStore(
    client=luma,  # Pass the LumaClient instance created above
    collection_name="knowledge_base",
    embedding_function=embeddings
)

```

### Usage in a RAG Pipeline

```python
# 1. Add Documents
docs = [
    Document(page_content="Luma is a high-performance vector database written in Rust.", metadata={"id": 1}),
    Document(page_content="LangChain simplifies building LLM applications.", metadata={"id": 2}),
]

vector_store.add_documents(docs)

# 2. Perform a Similarity Search
query = "What is Luma?"
found_docs = vector_store.similarity_search(query, k=1)

print(f"\nQuery: {query}")
print(f"Answer: {found_docs[0].page_content}")

# 3. Use as a Retriever
retriever = vector_store.as_retriever()
# This retriever can now be passed to any LangChain chain (e.g., RetrievalQA)

```

## Features

* **⚡ High Performance:** Built on top of RustKissVDB for speed and efficiency.
* **🛠 Modular Design:** Clean separation between System management and Vector operations.
* **🔗 LangChain Ready:** Drop-in replacement for other vector stores in your AI pipelines.
* **🔍 Rich Metadata:** Full support for storing and retrieving metadata payloads.

