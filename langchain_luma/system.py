from typing import Any, Dict

from .http import HttpTransport


class SystemClient:
    def __init__(self, http: HttpTransport):
        self._http = http

    def health(self) -> Dict[str, Any]:
        """Check system health."""
        return self._http._get("/v1/health")

    def metrics(self) -> str:
        """Get Prometheus metrics."""
        return self._http._get("/v1/metrics")
