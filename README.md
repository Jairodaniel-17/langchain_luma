# langchain-luma

[![PyPI version](https://badge.fury.io/py/langchain-luma.svg)](https://badge.fury.io/py/langchain-luma)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Luma Backend](https://img.shields.io/badge/Luma_Backend-v0.2.2-orange)](https://github.com/Jairodaniel-17/rust-kiss-vdb/releases/tag/v0.2.2)

**langchain-luma** is a production-ready Python client SDK for **Luma (RustKissVDB)**, a high-performance, lightweight vector database written in Rust. This library provides both a direct HTTP client for low-level interaction and a fully compliant `VectorStore` implementation for seamless integration with the **LangChain** ecosystem.

## System Architecture

The following diagram illustrates how the Python client interacts with the Luma Rust backend and LangChain workflows.

```mermaid
graph TD
    subgraph "Python Environment"
        UserCode[User Application / RAG Pipeline]
        LC[LangChain Integration]
        SDK[LumaClient SDK]
    end

    subgraph "Infrastructure"
        Server[Luma Server - Rust Binary]
        Storage[(Vector Storage)]
    end

    UserCode -->|Direct API Calls| SDK
    UserCode -->|Via VectorStore| LC
    LC -->|Wraps| SDK
    SDK -->|HTTP/REST| Server
    Server -->|Read/Write| Storage


```

## Prerequisites: Luma Backend

This SDK acts as a client. To function, it requires a running instance of the Luma server (v0.2.2 or higher).

1. **Download the Server Binary:**
Download the appropriate executable for your operating system from the [Official Release Assets](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/Jairodaniel-17/rust-kiss-vdb/releases/tag/v0.2.2).
* **Linux:** `luma-linux-amd64`
* **Windows:** `luma-windows-amd64.exe`
* **macOS:** `luma-macos-amd64`


2. **Start the Server:**
Run the binary in a terminal window. By default, it listens on port `1234`.
```bash
# Linux/macOS
chmod +x luma-linux-amd64
./luma-linux-amd64

# Windows
.\luma-windows-amd64.exe

```



## Installation

Install the package directly from PyPI.

### Standard Client

For users who only need the direct API client without LangChain dependencies:

```bash
pip install langchain-luma

```

### With LangChain Support

For users building RAG pipelines or using LangChain abstractions:

```bash
pip install "langchain-luma[langchain]"

```

## Core SDK Usage

The `LumaClient` class provides granular control over the database operations. It is organized into namespaces (`system`, `vectors`) for clarity.

```python
import time
import logging
from langchain_luma import LumaClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LUMA_URL = "http://localhost:1234"
COLLECTION_NAME = "enterprise_docs"
VECTOR_DIMENSION = 384  # Must match your embedding model

def run_pipeline():
    # 1. Initialize Client
    client = LumaClient(url=LUMA_URL)

    # 2. Health Check
    try:
        health = client.system.health()
        logger.info(f"System Status: {health}")
    except Exception as e:
        logger.error(f"Failed to connect to Luma at {LUMA_URL}: {e}")
        return

    # 3. Create Collection
    # This defines the schema for the vector space.
    try:
        client.vectors.create_collection(
            name=COLLECTION_NAME, 
            dim=VECTOR_DIMENSION, 
            metric="cosine"
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        logger.warning(f"Collection creation skipped (might already exist): {e}")

    # 4. Upsert Data
    # Simulating a document vector with metadata
    vector_id = "doc_ref_1024"
    embedding_vector = [0.05] * VECTOR_DIMENSION 
    metadata = {
        "source": "internal_wiki",
        "author": "dev_ops",
        "created_at": time.time()
    }

    client.vectors.upsert(
        collection=COLLECTION_NAME,
        id=vector_id,
        vector=embedding_vector,
        meta=metadata
    )
    logger.info(f"Vector '{vector_id}' upserted.")

    # 5. Semantic Search
    search_results = client.vectors.search(
        collection=COLLECTION_NAME,
        vector=embedding_vector,
        k=5
    )

    logger.info("Search Results:")
    for result in search_results:
        print(f"ID: {result.id} | Score: {result.score:.4f} | Payload: {result.payload}")

if __name__ == "__main__":
    run_pipeline()

```

## LangChain Integration

`langchain-luma` implements the standard LangChain `VectorStore` interface. This allows Luma to be swapped seamlessly into existing RAG architectures.

### RAG Example

The following example demonstrates how to use Luma as a retrieval backend for a standard document search pipeline.

```python
from langchain_core.documents import Document
from langchain_luma.langchain.vectorstore import LumaVectorStore
# NOTE: In production, use standard embeddings (OpenAI, HuggingFace, etc.)
from langchain_community.embeddings import FakeEmbeddings 

# 1. Initialize Embeddings
# The dimension size must match the collection configuration in Luma
embeddings = FakeEmbeddings(size=384)

# 2. Connect to Vector Store
# If the collection does not exist, LumaVectorStore can create it automatically
# depending on the internal implementation of the 'add_documents' method.
client_config = {"url": "http://localhost:1234"} 
# Note: Alternatively, pass an instantiated LumaClient

vector_store = LumaVectorStore.from_connection(
    url="http://localhost:1234",
    collection_name="rag_knowledge_base",
    embedding=embeddings
)

# 3. Ingest Documents
documents = [
    Document(
        page_content="Luma is optimized for low-latency vector retrieval.",
        metadata={"category": "database", "priority": "high"}
    ),
    Document(
        page_content="LangChain provides the glue code for LLM applications.",
        metadata={"category": "framework", "priority": "medium"}
    )
]

vector_store.add_documents(documents)
print("Documents indexed.")

# 4. Retrieval
# Perform a similarity search
query = "latency optimization"
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
result = retriever.invoke(query)

print("-" * 40)
print(f"Query: {query}")
print(f"Retrieved Content: {result[0].page_content}")
print(f"Retrieved Metadata: {result[0].metadata}")
print("-" * 40)

```

## API Reference

### `LumaClient`

The main entry point for the SDK.

* `__init__(url: str, api_key: str = None)`
* Initializes the connection settings.



### `LumaClient.system`

* `health() -> dict`
* Returns the operational status of the Luma server.



### `LumaClient.vectors`

* `create_collection(name: str, dim: int, metric: str = "cosine")`
* Creates a new vector space. Metric options: `cosine`, `euclidean`, `dot`.


* `upsert(collection: str, id: str, vector: list[float], meta: dict = None)`
* Inserts or updates a vector with optional metadata payload.


* `search(collection: str, vector: list[float], k: int = 10) -> list[Hit]`
* Performs a k-Nearest Neighbors (k-NN) search. Returns a list of `Hit` objects containing `id`, `score`, and `payload`.



## License

This project is licensed under the MIT License.
