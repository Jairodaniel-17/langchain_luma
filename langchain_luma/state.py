from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .http import HttpTransport

# =========================
# Data Models
# =========================


@dataclass
class StateItem:
    key: str
    value: Dict[str, Any]
    revision: int
    expires_at_ms: Optional[int] = None


@dataclass
class PutStateResponse:
    key: str
    revision: int
    expires_at_ms: Optional[int] = None


@dataclass
class DeleteStateResponse:
    deleted: bool


# =========================
# Batch Models
# =========================


@dataclass
class StateBatchOperation:
    key: str
    value: Dict[str, Any]
    ttl_ms: Optional[int] = None
    if_revision: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "key": self.key,
            "value": self.value,
        }

        if self.ttl_ms is not None:
            payload["ttl_ms"] = self.ttl_ms
        if self.if_revision is not None:
            payload["if_revision"] = self.if_revision

        return payload


@dataclass
class StateBatchResult:
    status: str  # "ok" | "error"
    key: str
    revision: Optional[int] = None
    expires_at_ms: Optional[int] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class StateBatchResponse:
    results: List[StateBatchResult]


# =========================
# Client
# =========================


class StateClient:
    def __init__(self, http: HttpTransport):
        self._http = http

    def get(self, key: str) -> StateItem:
        """Get a state item by key."""
        data = self._http._get(f"/v1/state/{key}")
        return StateItem(**data)

    def put(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_ms: Optional[int] = None,
        if_revision: Optional[int] = None,
    ) -> PutStateResponse:
        """Set a state item value."""

        payload: Dict[str, Any] = {
            "value": value,
        }

        if ttl_ms is not None:
            payload["ttl_ms"] = ttl_ms
        if if_revision is not None:
            payload["if_revision"] = if_revision

        data = self._http._put(f"/v1/state/{key}", json=payload)
        return PutStateResponse(**data)

    def delete(self, key: str) -> bool:
        """Delete a state item."""
        data = self._http._delete(f"/v1/state/{key}")
        return data.get("deleted", False)

    def list(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
    ) -> List[StateItem]:
        """List state items."""

        params: Dict[str, Any] = {
            "limit": limit,
        }

        if prefix is not None:
            params["prefix"] = prefix

        data = self._http._get("/v1/state", params=params)
        return [StateItem(**item) for item in data]

    def batch_put(
        self,
        operations: List[StateBatchOperation],
    ) -> StateBatchResponse:
        """Batch put state items."""

        payload: Dict[str, Any] = {
            "operations": [op.to_dict() for op in operations],
        }

        data = self._http._post("/v1/state/batch_put", json=payload)
        results = [StateBatchResult(**res) for res in data.get("results", [])]

        return StateBatchResponse(results=results)
