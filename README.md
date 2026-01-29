# langchain_luma

A Python client SDK for Luma (RustKissVDB), designed for simplicity and optional integration with LangChain.

## Installation

```bash
pip install .
```

To install with LangChain support:

```bash
pip install ".[langchain]"
```

## Quick Start

```python
from langchain_luma import LumaClient

# Initialize client (defaults to http://localhost:1234, api_key="dev")
luma = LumaClient(url="http://localhost:1234")

# Check system health
print(luma.system.health())

# Create a vector collection
luma.vectors.create_collection("docs", dim=384, metric="cosine")

# Upsert a vector
luma.vectors.upsert(
    collection="docs",
    id="vec1",
    vector=[0.1] * 384,
    meta={"category": "test"}
)

# Search
hits = luma.vectors.search("docs", vector=[0.1] * 384, k=3)
for hit in hits:
    print(f"ID: {hit.id}, Score: {hit.score}")
```

## LangChain Integration

If you have `langchain` installed, you can use `LumaVectorStore`:

```python
from langchain_luma.langchain.vectorstore import LumaVectorStore
# Ensure you have an embedding function compatible with LangChain
# from langchain_community.embeddings import FakeEmbeddings
# embeddings = FakeEmbeddings(size=384)

# vector_store = LumaVectorStore(
#     client=luma,
#     collection_name="docs",
#     embedding_function=embeddings
# )
```
