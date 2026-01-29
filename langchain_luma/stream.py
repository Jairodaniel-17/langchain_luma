from typing import Any, Generator, Optional

from .http import HttpTransport


class StreamClient:
    def __init__(self, http: HttpTransport):
        self._http = http

    def events(
        self,
        since: int = 0,
        types: Optional[str] = None,
        key_prefix: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Subscribe to database events via SSE."""

        params: dict[str, Any] = {"since": since}

        if types is not None:
            params["types"] = types
        if key_prefix is not None:
            params["key_prefix"] = key_prefix
        if collection is not None:
            params["collection"] = collection

        url = f"{self._http.base_url}/v1/stream"
        with self._http.session.get(
            url,
            params=params,
            stream=True,
            timeout=None,
        ) as response:
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8")
