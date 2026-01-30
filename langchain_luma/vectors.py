from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from .errors import ValidationError
from .http import HttpTransport
from .validators import (
    validate_collection,
    validate_id,
    validate_json_size,
    validate_k,
    validate_vector,
)


def _pick(data: Dict[str, Any], keys: set[str]) -> Dict[str, Any]:
    return {k: data[k] for k in keys if k in data}


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


@dataclass
class VectorItemResponse:
    id: str
    vector: List[float]
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "VectorItemResponse":
        return cls(**_pick(data, {"id", "vector", "meta"}))


@dataclass
class DiskAnnBuildParams:
    max_degree: int
    build_threads: int
    search_list_size: int

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "DiskAnnBuildParams":
        return cls(**_pick(data, {"max_degree", "build_threads", "search_list_size"}))


@dataclass
class DiskAnnStatusResponse:
    available: bool
    last_built_ms: int
    graph_files: List[str]
    params: DiskAnnBuildParams

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "DiskAnnStatusResponse":
        params = DiskAnnBuildParams.from_api(data.get("params", {}))
        return cls(
            available=data.get("available", False),
            last_built_ms=data.get("last_built_ms", 0),
            graph_files=data.get("graph_files", []),
            params=params,
        )


@dataclass
class DiskAnnMutationResponse:
    ok: bool
    params: DiskAnnBuildParams
    status: DiskAnnStatusResponse

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "DiskAnnMutationResponse":
        return cls(
            ok=data.get("ok", False),
            params=DiskAnnBuildParams.from_api(data.get("params", {})),
            status=DiskAnnStatusResponse.from_api(data.get("status", {})),
        )


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
        validate_collection(name, self._http.config.max_collection_len)
        if dim <= 0 or dim > self._http.config.max_vector_dim:
            raise ValidationError(f"invalid dim (max {self._http.config.max_vector_dim})")

        payload = {"dim": dim, "metric": metric}
        return self._http._post(f"/v1/vector/{name}", json=payload)

    def list_collections(self) -> List[VectorCollectionInfo]:
        data = self._http._get("/v1/vector")
        return [VectorCollectionInfo.from_api(item) for item in data.get("collections", [])]

    def get_collection(self, name: str) -> VectorCollectionDetailResponse:
        validate_collection(name, self._http.config.max_collection_len)
        data = self._http._get(f"/v1/vector/{name}")
        return VectorCollectionDetailResponse.from_api(data)

    # ---- Operations ----

    def get_vector(self, collection: str, id: str) -> VectorItemResponse:
        validate_collection(collection, self._http.config.max_collection_len)
        validate_id(id, self._http.config.max_id_len)

        data = self._http._get(f"/v1/vector/{collection}/get", params={"id": id})
        return VectorItemResponse.from_api(data)

    def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        validate_collection(collection, self._http.config.max_collection_len)
        validate_id(id, self._http.config.max_id_len)
        validate_vector(vector, self._http.config.max_vector_dim)

        if meta is not None:
            validate_json_size(meta, self._http.config.max_json_bytes, "meta")

        payload = {"id": id, "vector": vector}
        if meta is not None:
            payload["meta"] = meta
        return self._http._post(f"/v1/vector/{collection}/upsert", json=payload)

    def update_vector(
        self,
        collection: str,
        id: str,
        vector: Optional[List[float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        validate_collection(collection, self._http.config.max_collection_len)
        validate_id(id, self._http.config.max_id_len)

        if vector is not None:
            validate_vector(vector, self._http.config.max_vector_dim)

        if meta is not None:
            validate_json_size(meta, self._http.config.max_json_bytes, "meta")

        # CORRECCIÓN AQUÍ: Definir explícitamente el tipo como Dict[str, Any]
        payload: Dict[str, Any] = {"id": id}

        if vector is not None:
            payload["vector"] = vector  # Ahora es válido
        if meta is not None:
            payload["meta"] = meta  # Ahora es válido

        data = self._http._post(f"/v1/vector/{collection}/update", json=payload)
        return bool(data.get("ok", False))

    def upsert_batch(
        self,
        collection: str,
        items: List[VectorBatchItem],
    ) -> VectorBatchResponse:
        validate_collection(collection, self._http.config.max_collection_len)
        if not items:
            raise ValidationError("items required")

        if len(items) > self._http.config.max_vector_batch:
            raise ValidationError("too many batch items")

        for item in items:
            validate_id(item.id, self._http.config.max_id_len)
            validate_vector(item.vector, self._http.config.max_vector_dim)
            if item.meta:
                validate_json_size(item.meta, self._http.config.max_json_bytes, "meta")

        payload = {"items": [item.to_dict() for item in items]}
        data = self._http._post(
            f"/v1/vector/{collection}/upsert_batch",
            json=payload,
        )
        return VectorBatchResponse(results=[VectorBatchResult.from_api(r) for r in data.get("results", [])])

    def delete(self, collection: str, id: str) -> bool:
        validate_collection(collection, self._http.config.max_collection_len)
        validate_id(id, self._http.config.max_id_len)

        payload = {"id": id}
        data = self._http._post(
            f"/v1/vector/{collection}/delete",
            json=payload,
        )
        return bool(data.get("deleted", False))

    def delete_batch(self, collection: str, ids: List[str]) -> VectorBatchResponse:
        validate_collection(collection, self._http.config.max_collection_len)
        if not ids:
            raise ValidationError("ids required")

        if len(ids) > self._http.config.max_vector_batch:
            raise ValidationError("too many batch ids")

        for id in ids:
            validate_id(id, self._http.config.max_id_len)

        payload = {"ids": ids}
        data = self._http._post(
            f"/v1/vector/{collection}/delete_batch",
            json=payload,
        )
        return VectorBatchResponse(results=[VectorBatchResult.from_api(r) for r in data.get("results", [])])

    # ---- Search ----

    def search(
        self,
        collection: str,
        vector: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_meta: bool = False,
    ) -> List[VectorSearchHit]:
        validate_collection(collection, self._http.config.max_collection_len)
        validate_vector(vector, self._http.config.max_vector_dim)
        validate_k(k, self._http.config.max_k)

        if filters is not None:
            validate_json_size(filters, self._http.config.max_json_bytes, "filters")

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

    # ---- DiskANN ----

    def diskann_build(
        self,
        collection: str,
        max_degree: Optional[int] = None,
        build_threads: Optional[int] = None,
        search_list_size: Optional[int] = None,
    ) -> DiskAnnMutationResponse:
        validate_collection(collection, self._http.config.max_collection_len)

        payload = {}
        if max_degree is not None:
            payload["max_degree"] = max_degree
        if build_threads is not None:
            payload["build_threads"] = build_threads
        if search_list_size is not None:
            payload["search_list_size"] = search_list_size

        data = self._http._post(f"/v1/vector/{collection}/diskann/build", json=payload)
        return DiskAnnMutationResponse.from_api(data)

    def diskann_tune(
        self,
        collection: str,
        max_degree: Optional[int] = None,
        build_threads: Optional[int] = None,
        search_list_size: Optional[int] = None,
    ) -> DiskAnnMutationResponse:
        validate_collection(collection, self._http.config.max_collection_len)

        payload = {}
        if max_degree is not None:
            payload["max_degree"] = max_degree
        if build_threads is not None:
            payload["build_threads"] = build_threads
        if search_list_size is not None:
            payload["search_list_size"] = search_list_size

        data = self._http._post(f"/v1/vector/{collection}/diskann/tune", json=payload)
        return DiskAnnMutationResponse.from_api(data)

    def diskann_status(self, collection: str) -> DiskAnnStatusResponse:
        validate_collection(collection, self._http.config.max_collection_len)
        data = self._http._get(f"/v1/vector/{collection}/diskann/status")
        return DiskAnnStatusResponse.from_api(data)
