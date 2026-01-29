from unittest.mock import MagicMock

import pytest

from langchain_luma import LumaClient


@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.headers = {}
    mock.status_code = 200
    mock.text = ""
    mock.json.return_value = {}
    return mock


@pytest.fixture
def client(monkeypatch, mock_response):
    # Patch requests.Session.request
    mock_session = MagicMock()
    mock_session.request.return_value = mock_response
    mock_session.headers = {}

    def mock_request(self, method, url, **kwargs):
        return mock_response

    monkeypatch.setattr("requests.Session.request", mock_request)
    return LumaClient()
