from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .http import HttpTransport


@dataclass
class DocRecord:
    id: str
    doc: Dict[str, Any]
    revision: int


class DocumentsClient:
    def __init__(self, http: HttpTransport):
        self._http = http

    def put(self, collection: str, id: str, document: Dict[str, Any]) -> DocRecord:
        """Store a document."""
        data = self._http._put(f"/v1/doc/{collection}/{id}", json=document)
        return DocRecord(**data)

    def get(self, collection: str, id: str) -> DocRecord:
        """Retrieve a document."""
        data = self._http._get(f"/v1/doc/{collection}/{id}")
        return DocRecord(**data)

    def delete(self, collection: str, id: str) -> bool:
        """Delete a document."""
        # DELETE /v1/doc returns {description: Deleted}, assuming 200 OK means True
        # but OpenAPI doesn't show response body for DELETE, only description.
        # However, http.py returns None for 204 or dict for JSON.
        # Let's assume successful call means deleted.
        try:
            self._http._delete(f"/v1/doc/{collection}/{id}")
            return True
        except Exception:
            # If 404, HttpTransport raises LumaNotFound
            # If user wants boolean, maybe we should catch 404?
            # AGENTS.md says "delete(collection: str, id: str) -> bool".
            # Usually returning False on not found is convenient.
            return False

    def find(self, collection: str, filter: Optional[Dict[str, Any]] = None, limit: int = 20) -> List[DocRecord]:
        """Find documents by metadata."""
        payload = {"filter": filter, "limit": limit}
        data = self._http._post(f"/v1/doc/{collection}/find", json=payload)
        return [DocRecord(**doc) for doc in data.get("documents", [])]
