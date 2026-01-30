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
        try:
            self._http._delete(f"/v1/doc/{collection}/{id}")
            return True
        except Exception:
            return False

    def find(self, collection: str, filter: Optional[Dict[str, Any]] = None, limit: int = 20) -> List[DocRecord]:
        """Find documents by metadata."""
        payload = {"limit": limit}
        if filter:
            payload["filter"] = filter

        data = self._http._post(f"/v1/doc/{collection}/find", json=payload)
        return [DocRecord(**doc) for doc in data.get("documents", [])]
