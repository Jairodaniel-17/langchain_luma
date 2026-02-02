from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .vectors import (
    VectorsClient,
    VectorBatchItem,
    VectorSearchHit,
)

logger = logging.getLogger(__name__)


class LumaVectorStore(VectorStore):
    """
    LangChain-compatible VectorStore backed by Luma.

    Design guarantees:
    - Vector dimension is ALWAYS inferred from the embedding model
    - Collection creation is explicit and validated
    - Text content is persisted inside metadata
    - No silent failures on dimension mismatch
    """

    TEXT_META_KEY = "text"
    DEFAULT_BATCH_SIZE = 100

    def __init__(
        self,
        *,
        client: VectorsClient,
        collection_name: str,
        embedding: Embeddings,
        metric: str = "cosine",
        batch_size: int = DEFAULT_BATCH_SIZE,
        create_if_not_exists: bool = True,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding = embedding
        self.metric = metric
        self.batch_size = batch_size

        if create_if_not_exists:
            self._ensure_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """
        Ensure the collection exists and matches the embedding dimension.
        Dimension is inferred from the embedding model (single probe call).
        """
        probe = self.embedding.embed_query("dimension_probe")
        dim = len(probe)

        try:
            info = self.client.get_collection(self.collection_name)

            if info.dim != dim:
                raise ValueError(
                    f"Collection '{self.collection_name}' dimension mismatch: "
                    f"expected {dim}, got {info.dim}"
                )

        except Exception:
            logger.info(
                f"Creating Luma collection '{self.collection_name}' "
                f"(dim={dim}, metric={self.metric})"
            )
            self.client.create_collection(
                name=self.collection_name,
                dim=dim,
                metric=self.metric,
            )

    def _batched(self, items: List[Any]) -> Iterable[List[Any]]:
        for i in range(0, len(items), self.batch_size):
            yield items[i : i + self.batch_size]

    # ------------------------------------------------------------------
    # Required VectorStore API (sync)
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        if not texts:
            return []

        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]

        if len(texts) != len(metadatas) or len(texts) != len(ids):
            raise ValueError("texts, metadatas and ids must have the same length")

        embeddings = self.embedding.embed_documents(texts)

        items: List[VectorBatchItem] = []
        for text, meta, id_, vector in zip(texts, metadatas, ids, embeddings):
            if self.TEXT_META_KEY in meta:
                raise ValueError(
                    f"Metadata key '{self.TEXT_META_KEY}' is reserved by LumaVectorStore"
                )

            meta = dict(meta)
            meta[self.TEXT_META_KEY] = text

            items.append(
                VectorBatchItem(
                    id=id_,
                    vector=vector,
                    meta=meta,
                )
            )

        for batch in self._batched(items):
            self.client.upsert_batch(
                collection=self.collection_name,
                items=batch,
            )

        if logger.isEnabledFor(logging.DEBUG):
            try:
                info = self.client.get_collection(self.collection_name)
                logger.debug(
                    f"Collection '{self.collection_name}' now contains {info.count} vectors"
                )
            except Exception:
                pass

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k=k, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        hits: List[VectorSearchHit] = self.client.search(
            collection=self.collection_name,
            vector=embedding,
            k=k,
            include_meta=True,
            **kwargs,
        )

        documents: List[Document] = []
        for hit in hits:
            meta = dict(hit.meta or {})
            text = meta.pop(self.TEXT_META_KEY, "")

            documents.append(
                Document(
                    page_content=text,
                    metadata=meta,
                )
            )

        return documents

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
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        hits: List[VectorSearchHit] = self.client.search(
            collection=self.collection_name,
            vector=embedding,
            k=k,
            include_meta=True,
            **kwargs,
        )

        results: List[Tuple[Document, float]] = []
        for hit in hits:
            meta = dict(hit.meta or {})
            text = meta.pop(self.TEXT_META_KEY, "")

            doc = Document(
                page_content=text,
                metadata=meta,
            )
            results.append((doc, hit.score))

        return results

    # ------------------------------------------------------------------
    # Async variants (LangChain compatibility)
    # ------------------------------------------------------------------

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        import asyncio

        return await asyncio.to_thread(
            self.add_texts, texts, metadatas, ids, **kwargs
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        import asyncio

        return await asyncio.to_thread(
            self.similarity_search, query, k, **kwargs
        )

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        import asyncio

        return await asyncio.to_thread(
            self.add_documents, documents, ids, **kwargs
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        import asyncio

        return await asyncio.to_thread(
            self.similarity_search_with_score, query, k, **kwargs
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        client: VectorsClient,
        collection_name: str,
        **kwargs: Any,
    ) -> "LumaVectorStore":
        """
        Create a LumaVectorStore from raw texts.

        Required by LangChain's VectorStore abstract base class.
        """
        store = cls(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            **kwargs,
        )

        store.add_texts(
            texts=texts,
            metadatas=metadatas,
        )

        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        *,
        client: VectorsClient,
        collection_name: str,
        **kwargs: Any,
    ) -> "LumaVectorStore":
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]

        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embedding,
            client=client,
            collection_name=collection_name,
            **kwargs,
        )


    # ------------------------------------------------------------------
    # Required by LangChain
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

