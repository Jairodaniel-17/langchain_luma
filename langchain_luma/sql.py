from typing import Any, Dict, List, Optional

from .http import HttpTransport


class SQLClient:
    def __init__(self, http: HttpTransport):
        self._http = http

    def query(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a SELECT query."""

        payload: Dict[str, Any] = {
            "sql": sql,
        }

        if params is not None:
            payload["params"] = params

        data = self._http._post("/v1/sql/query", json=payload)
        return data.get("rows", [])

    def exec(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a DDL/DML query."""

        payload: Dict[str, Any] = {
            "sql": sql,
        }

        if params is not None:
            payload["params"] = params

        data = self._http._post("/v1/sql/exec", json=payload)
        return data
