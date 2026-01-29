import logging
from typing import Any, Iterable, List, Optional, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..client import LumaClient
from ..vectors import VectorBatchItem

logger = logging.getLogger(__name__)


class LumaVectorStore(VectorStore):
    """LangChain-compatible VectorStore backed by Luma."""

    def __init__(
        self,
        client: LumaClient,
        collection_name: str,
        embedding_function: Embeddings,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function

    # ---------------------------------------------------------------------
    # Required / expected LangChain API
    # ---------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed and add texts to the vector store."""
        import uuid

        texts = list(texts)
        if not texts:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")

        embeddings = self.embedding_function.embed_documents(texts)

        items: List[VectorBatchItem] = []
        for i, text in enumerate(texts):
            meta = dict(metadatas[i])
            meta["text"] = text  # persist text for reconstruction

            items.append(
                VectorBatchItem(
                    id=ids[i],
                    vector=embeddings[i],
                    meta=meta,
                )
            )

        self.client.vectors.upsert_batch(
            collection=self.collection_name,
            items=items,
        )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to a text query."""
        embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(embedding, k=k, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to an embedding vector."""
        hits = self.client.vectors.search(
            collection=self.collection_name,
            vector=embedding,
            k=k,
            include_meta=True,
            **kwargs,
        )

        documents: List[Document] = []
        for hit in hits:
            meta = dict(hit.meta or {})
            text = meta.pop("text", "")

            documents.append(
                Document(
                    page_content=text,
                    metadata=meta,
                )
            )

        return documents

    # ---------------------------------------------------------------------
    # Convenience constructor
    # ---------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls: Type["LumaVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Optional[LumaClient] = None,
        collection_name: str = "langchain",
        **kwargs: Any,
    ) -> "LumaVectorStore":
        """Create a LumaVectorStore from raw texts."""
        if client is None:
            client = LumaClient()

        # Infer embedding dimension safely
        test_vector = embedding.embed_query("dimension_probe")
        dim = len(test_vector)

        try:
            client.vectors.create_collection(
                name=collection_name,
                dim=dim,
                metric="cosine",
            )
        except Exception:
            # Collection already exists or backend handles idempotency
            pass

        store = cls(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding,
        )

        store.add_texts(
            texts=texts,
            metadatas=metadatas,
            **kwargs,
        )

        return store
