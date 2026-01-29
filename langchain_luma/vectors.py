from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from .http import HttpTransport

# ---------- Helpers ----------


def _pick(data: Dict[str, Any], keys: set[str]) -> Dict[str, Any]:
    return {k: data[k] for k in keys if k in data}


# ---------- Models ----------


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

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "VectorCollectionInfo":
        return cls(
            **_pick(
                data,
                {
                    "collection",
                    "dim",
                    "metric",
                    "live_count",
                    "total_records",
                    "upsert_count",
                    "file_len",
                    "applied_offset",
                    "created_at_ms",
                    "updated_at_ms",
                },
            )
        )


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
    manifest: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "VectorCollectionDetailResponse":
        return cls(
            **_pick(
                data,
                {
                    "collection",
                    "dim",
                    "metric",
                    "count",
                    "created_at_ms",
                    "updated_at_ms",
                    "segments",
                    "deleted",
                    "manifest",
                    "notes",
                },
            )
        )


@dataclass
class VectorSearchHit:
    id: str
    score: float
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "VectorSearchHit":
        return cls(**_pick(data, {"id", "score", "meta"}))


@dataclass
class VectorBatchItem:
    id: str
    vector: List[float]
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {"id": self.id, "vector": self.vector}
        if self.meta is not None:
            payload["meta"] = self.meta
        return payload


@dataclass
class VectorBatchResult:
    status: str
    id: str
    error: Optional[Dict[str, Any]] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "VectorBatchResult":
        return cls(**_pick(data, {"status", "id", "error"}))


@dataclass
class VectorBatchResponse:
    results: List[VectorBatchResult]


# ---------- Client ----------


class VectorsClient:
    def __init__(self, http: HttpTransport):
        self._http = http

    # ---- Collections ----

    def create_collection(
        self,
        name: str,
        dim: int,
        metric: Literal["cosine", "dot"],
    ) -> Dict[str, Any]:
        payload = {"dim": dim, "metric": metric}
        return self._http._post(f"/v1/vector/{name}", json=payload)

    def list_collections(self) -> List[VectorCollectionInfo]:
        data = self._http._get("/v1/vector")
        return [VectorCollectionInfo.from_api(item) for item in data.get("collections", [])]

    def get_collection(self, name: str) -> VectorCollectionDetailResponse:
        data = self._http._get(f"/v1/vector/{name}")
        return VectorCollectionDetailResponse.from_api(data)

    # ---- Upsert / Delete ----

    def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {"id": id, "vector": vector}
        if meta is not None:
            payload["meta"] = meta
        return self._http._post(f"/v1/vector/{collection}/upsert", json=payload)

    def upsert_batch(
        self,
        collection: str,
        items: List[VectorBatchItem],
    ) -> VectorBatchResponse:
        payload = {"items": [item.to_dict() for item in items]}
        data = self._http._post(
            f"/v1/vector/{collection}/upsert_batch",
            json=payload,
        )
        return VectorBatchResponse(results=[VectorBatchResult.from_api(r) for r in data.get("results", [])])

    def delete(self, collection: str, id: str) -> bool:
        payload = {"id": id}
        data = self._http._post(
            f"/v1/vector/{collection}/delete",
            json=payload,
        )
        return bool(data.get("deleted", False))

    # ---- Search ----

    def search(
        self,
        collection: str,
        vector: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_meta: bool = False,
    ) -> List[VectorSearchHit]:
        payload = {
            "vector": vector,
            "k": k,
            "filters": filters,
            "include_meta": include_meta,
        }
        data = self._http._post(
            f"/v1/vector/{collection}/search",
            json=payload,
        )
        return [VectorSearchHit.from_api(hit) for hit in data.get("hits", [])]
