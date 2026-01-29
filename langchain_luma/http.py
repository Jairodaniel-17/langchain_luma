import logging
from typing import Any, Dict, Optional

import requests

from .exceptions import (
    LumaAuthError,
    LumaConflict,
    LumaConnectionError,
    LumaError,
    LumaNotFound,
)

logger = logging.getLogger(__name__)


class HttpTransport:
    def __init__(self, base_url: str, api_key: str, timeout: int = 30, retries: int = 0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "langchain-luma-sdk/0.1.0",
            }
        )
        if retries > 0:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            adapter = HTTPAdapter(
                max_retries=Retry(
                    total=retries,
                    backoff_factor=0.1,
                    status_forcelist=[500, 502, 503, 504],
                )
            )
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.RequestException as e:
            logger.error(f"Connection error to {url}: {e}")
            raise LumaConnectionError(f"Failed to connect to Luma at {url}") from e

        self._handle_error(response)

        # Handle 204 No Content
        if response.status_code == 204:
            return None

        try:
            return response.json()
        except ValueError:
            return response.text

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(
        self,
        path: str,
        json: Optional[Any] = None,
        params: Optional[Dict] = None,
        stream: bool = False,
    ) -> Any:
        if stream:
            url = f"{self.base_url}{path}"
            return self.session.post(url, json=json, params=params, stream=True, timeout=self.timeout)
        return self._request("POST", path, json=json, params=params)

    def _put(self, path: str, json: Optional[Any] = None, params: Optional[Dict] = None) -> Any:
        return self._request("PUT", path, json=json, params=params)

    def _delete(self, path: str, json: Optional[Any] = None) -> Any:
        return self._request("DELETE", path, json=json)

    def _handle_error(self, response: requests.Response) -> None:
        if 200 <= response.status_code < 300:
            return

        err_msg = f"HTTP {response.status_code}: {response.text}"
        try:
            body = response.json()
            if isinstance(body, dict) and "message" in body:
                err_msg = body["message"]
            elif isinstance(body, dict) and "error" in body:
                err_msg = body.get("error")
        except ValueError:
            pass

        if response.status_code in (401, 403):
            raise LumaAuthError(err_msg)
        elif response.status_code == 404:
            raise LumaNotFound(err_msg)
        elif response.status_code == 409:
            raise LumaConflict(err_msg)
        else:
            raise LumaError(f"Request failed: {err_msg}")
