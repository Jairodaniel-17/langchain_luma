import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from langchain_luma.errors import AlreadyExists, DimMismatch, NotFound, PayloadTooLarge

from .exceptions import (
    LumaAuthError,
    LumaConflict,
    LumaConnectionError,
    LumaError,
    LumaNotFound,
)

logger = logging.getLogger(__name__)


@dataclass
class LumaConfig:
    """Client-side configuration for validation."""

    max_vector_dim: int = 4096
    max_vector_batch: int = 1000
    max_json_bytes: int = 1048576  # 1MB
    max_id_len: int = 256
    max_collection_len: int = 256
    max_k: int = 100


class HttpTransport:
    def __init__(self, base_url: str, api_key: str, timeout: int = 30, retries: int = 0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.config = LumaConfig()
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

        self._raise_for_error(response)

    def _raise_for_error(self, resp: requests.Response) -> None:
        if resp.status_code == 413:
            raise PayloadTooLarge(resp.text)

        err_msg = f"HTTP {resp.status_code}: {resp.text}"
        data = {}
        try:
            data = resp.json()
            if isinstance(data, dict):
                err_msg = data.get("message") or data.get("error") or err_msg
        except ValueError:
            pass

        err_code = data.get("error") if isinstance(data, dict) else None

        if err_code == "dim_mismatch":
            raise DimMismatch(err_msg)
        if err_code == "not_found" or resp.status_code == 404:
            if isinstance(data.get("message"), str):
                err_msg = data.get("message")
            raise (NotFound(err_msg) if err_code == "not_found" else LumaNotFound(err_msg))
        if err_code == "already_exists" or resp.status_code == 409:
            raise (AlreadyExists(err_msg) if err_code == "already_exists" else LumaConflict(err_msg))

        if resp.status_code in (401, 403):
            raise LumaAuthError(err_msg)

        raise LumaError(err_msg)
