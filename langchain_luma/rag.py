from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .http import HttpTransport


@dataclass
class HighLevelDocumentMetadata:
    filename: Optional[str] = None
    processed_at: Optional[int] = None
    category: Optional[str] = None
    language: Optional[str] = None
    status: Optional[str] = None
    version: Optional[str] = None
    group_id: Optional[int] = None
    document_id: Optional[str] = None


@dataclass
class HighLevelDocumentResponse:
    id: int
    content: str
    metadata: Optional[HighLevelDocumentMetadata] = None

    def __post_init__(self):
        if isinstance(self.metadata, dict):
            self.metadata = HighLevelDocumentMetadata(**self.metadata)

    @property
    def metadata_safe(self) -> HighLevelDocumentMetadata:
        """Always return a metadata object (empty if missing)."""
        return self.metadata or HighLevelDocumentMetadata()


@dataclass
class HighLevelSearchResult:
    score: float
    document: HighLevelDocumentResponse

    def __post_init__(self):
        if isinstance(self.document, dict):
            self.document = HighLevelDocumentResponse(**self.document)


@dataclass
class HighLevelSearchResponse:
    query: str
    top_k: int
    results: List[HighLevelSearchResult]

    def __post_init__(self):
        if self.results:
            self.results = [HighLevelSearchResult(**res) if isinstance(res, dict) else res for res in self.results]


class RAGClient:
    def __init__(self, http: HttpTransport):
        self._http = http

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        group_by: Optional[str] = None,
        group_limit: int = 1,
    ) -> HighLevelSearchResponse:
        """Perform a high-level semantic search."""
        payload = {"query": query, "top_k": top_k, "group_limit": group_limit}
        if filters:
            payload["filters"] = filters
        if group_by:
            payload["group_by"] = group_by

        data = self._http._post("/search", json=payload)
        return HighLevelSearchResponse(**data)
