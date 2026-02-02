# vectorstores.py
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loguru import logger

from .client import LumaClient
from .vectors import VectorBatchItem, VectorSearchHit

_RESERVED_TEXT_KEY = "_luma_content"
_DEFAULT_BATCH_SIZE = 100


class LumaVectorStore(VectorStore):
    """LangChain VectorStore implementation for Luma."""

    def __init__(
        self,
        *,
        client: LumaClient,
        collection_name: str,
        embedding: Embeddings,
        dim: Optional[int] = None,
        metric: str = "cosine",
        create_if_not_exists: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self.client = client
        self.vectors = client.vectors
        self.collection_name = collection_name
        self.embedding = embedding
        self.dim = dim
        self.metric = metric
        self.batch_size = batch_size

        if create_if_not_exists:
            self._ensure_collection()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_params(
        cls,
        *,
        base_url: str,
        api_key: Optional[str],
        collection_name: str,
        embedding: Embeddings,
        dim: Optional[int] = None,
        metric: str = "cosine",
        create_if_not_exists: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> "LumaVectorStore":
        client = LumaClient(url=base_url, api_key=api_key)  # pyright: ignore[reportArgumentType]
        return cls(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            dim=dim,
            metric=metric,
            create_if_not_exists=create_if_not_exists,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        try:
            info = self.vectors.get_collection(self.collection_name)
            if self.dim and info.dim and self.dim != info.dim:
                raise ValueError(f"Collection dim mismatch: expected {self.dim}, got {info.dim}")
        except Exception:
            if not self.dim:
                raise ValueError("dim must be provided when creating a new collection")
            logger.info(f"Creating Luma collection '{self.collection_name}'")
            self.vectors.create_collection(
                name=self.collection_name,
                dim=self.dim,
                metric=self.metric,  # pyright: ignore[reportArgumentType]
            )

    def _batched(self, items: List[Any]) -> Iterable[List[Any]]:
        for i in range(0, len(items), self.batch_size):
            yield items[i : i + self.batch_size]

    # ------------------------------------------------------------------
    # LangChain required methods (sync)
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [uuid.uuid4().hex for _ in texts]

        if len(texts) != len(metadatas) or len(texts) != len(ids):
            raise ValueError("texts, metadatas and ids length mismatch")

        embeddings = self.embedding.embed_documents(texts)

        batch_items: List[VectorBatchItem] = []
        for text, meta, id_, vector in zip(texts, metadatas, ids, embeddings):
            if _RESERVED_TEXT_KEY in meta:
                raise ValueError(f"Metadata key '{_RESERVED_TEXT_KEY}' is reserved")
            meta = dict(meta)
            meta[_RESERVED_TEXT_KEY] = text
            batch_items.append(VectorBatchItem(id=id_, vector=vector, meta=meta))

        for batch in self._batched(batch_items):
            self.vectors.upsert_batch(self.collection_name, batch)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding.embed_query(query)
        hits: List[VectorSearchHit] = self.vectors.search(
            collection=self.collection_name,
            vector=embedding,
            k=k,
            include_meta=True,
        )

        documents: List[Document] = []
        for hit in hits:
            meta = hit.meta or {}
            text = meta.pop(_RESERVED_TEXT_KEY, "")
            documents.append(
                Document(
                    page_content=text,
                    metadata={**meta, "score": hit.score},
                )
            )
        return documents

    # ------------------------------------------------------------------
    # Async variants
    # ------------------------------------------------------------------

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return await asyncio.to_thread(self.add_texts, texts, metadatas, ids, **kwargs)

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        return await asyncio.to_thread(self.similarity_search, query, k, **kwargs)
    # ------------------------------------------------------------------
    # LangChain convenience methods
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        filters = kwargs.get("filter") or kwargs.get("filters")

        embedding = self.embedding.embed_query(query)
        hits: List[VectorSearchHit] = self.vectors.search(
            collection=self.collection_name,
            vector=embedding,
            k=k,
            filters=filters,
            include_meta=True,
        )

        results: List[tuple[Document, float]] = []
        for hit in hits:
            meta = hit.meta or {}
            text = meta.pop(_RESERVED_TEXT_KEY, "")
            doc = Document(
                page_content=text,
                metadata=meta,
            )
            results.append((doc, hit.score))

        return results

    # ------------------------------------------------------------------
    # Async variants (LangChain async compatibility)
    # ------------------------------------------------------------------

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return await asyncio.to_thread(self.add_documents, documents, ids, **kwargs)

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        return await asyncio.to_thread(
            self.similarity_search_with_score,
            query,
            k,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Optional LangChain hooks
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding
