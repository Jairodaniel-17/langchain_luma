from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from .http import HttpTransport


@dataclass
class VectorCollectionInfo:
    collection: str
    dim: int
    metric: str
    live_count: int
    total_records: int
    upsert_count: int
    file_len: int
    applied_offset: int
    created_at_ms: Optional[int] = None
    updated_at_ms: Optional[int] = None


@dataclass
class VectorCollectionDetailResponse:
    collection: str
    dim: Optional[int] = None
    metric: Optional[str] = None
    count: Optional[int] = None
    created_at_ms: Optional[int] = None
    updated_at_ms: Optional[int] = None
    segments: Optional[int] = None
    deleted: Optional[int] = None
    manifest: Optional[Dict] = None
    notes: Optional[str] = None


@dataclass
class VectorSearchHit:
    id: str
    score: float
    meta: Optional[Dict[str, Any]] = None


@dataclass
class VectorBatchItem:
    id: str
    vector: List[float]
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "vector": self.vector}
        if self.meta is not None:
            d["meta"] = self.meta
        return d


@dataclass
class VectorBatchResult:
    status: str
    id: str
    error: Optional[Dict[str, Any]] = None


@dataclass
class VectorBatchResponse:
    results: List[VectorBatchResult]


class VectorsClient:
    def __init__(self, http: HttpTransport):
        self._http = http

    def create_collection(self, name: str, dim: int, metric: Literal["cosine", "dot"]) -> Dict[str, Any]:
        """Create a new vector collection."""
        payload = {"dim": dim, "metric": metric}
        return self._http._post(f"/v1/vector/{name}", json=payload)

    def list_collections(self) -> List[VectorCollectionInfo]:
        """List all vector collections."""
        data = self._http._get("/v1/vector")
        return [VectorCollectionInfo(**item) for item in data.get("collections", [])]

    def get_collection(self, name: str) -> VectorCollectionDetailResponse:
        """Get details of a vector collection."""
        data = self._http._get(f"/v1/vector/{name}")
        return VectorCollectionDetailResponse(**data)

    def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Upsert a single vector."""
        payload = {"id": id, "vector": vector, "meta": meta}
        return self._http._post(f"/v1/vector/{collection}/upsert", json=payload)

    def upsert_batch(self, collection: str, items: List[VectorBatchItem]) -> VectorBatchResponse:
        """Upsert multiple vectors in batch."""
        payload = {"items": [item.to_dict() for item in items]}
        data = self._http._post(f"/v1/vector/{collection}/upsert_batch", json=payload)
        results = [VectorBatchResult(**res) for res in data.get("results", [])]
        return VectorBatchResponse(results=results)

    def delete(self, collection: str, id: str) -> bool:
        """Delete a vector."""
        payload = {"id": id}
        data = self._http._post(f"/v1/vector/{collection}/delete", json=payload)
        return data.get("deleted", False)

    def search(
        self,
        collection: str,
        vector: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_meta: bool = False,
    ) -> List[VectorSearchHit]:
        """Search for vectors."""
        payload = {
            "vector": vector,
            "k": k,
            "filters": filters,
            "include_meta": include_meta,
        }
        data = self._http._post(f"/v1/vector/{collection}/search", json=payload)
        return [VectorSearchHit(**hit) for hit in data.get("hits", [])]
