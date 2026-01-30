import pytest

from langchain_luma.client import LumaClient
from langchain_luma.errors import PayloadTooLarge, ValidationError


# Mock HttpTransport for testing without a real backend
class MockHttp:
    class Config:
        max_vector_dim = 10
        max_vector_batch = 5
        max_json_bytes = 20
        max_id_len = 5
        max_collection_len = 10
        max_k = 10

    def __init__(self):
        self.config = self.Config()
        self.last_request = None
        self.base_url = "http://test"

    def _post(self, path, json=None, params=None):
        self.last_request = ("POST", path, json, params)
        return {"ok": True, "results": [], "hits": [], "deleted": True}

    def _get(self, path, params=None):
        self.last_request = ("GET", path, params)
        if path.endswith("/get"):
            return {"id": params.get("id"), "vector": [0.1] * 10, "meta": None}
        return {"collections": []}


@pytest.fixture
def mock_client():
    client = LumaClient()
    client._http = MockHttp()
    client.vectors._http = client._http  # Inject mock into vectors client
    return client


def test_validate_collection_len(mock_client):
    with pytest.raises(ValidationError, match="collection name too long"):
        mock_client.vectors.create_collection("a" * 11, 4, "cosine")


def test_validate_dim(mock_client):
    with pytest.raises(ValidationError, match="invalid dim"):
        mock_client.vectors.create_collection("test", 11, "cosine")


def test_validate_id_len(mock_client):
    with pytest.raises(ValidationError, match="id too long"):
        mock_client.vectors.upsert("test", "a" * 6, [0.1] * 4)


def test_validate_vector_dim(mock_client):
    with pytest.raises(ValidationError, match="vector too large"):
        mock_client.vectors.upsert("test", "id1", [0.1] * 11)


def test_validate_json_size(mock_client):
    # {"a": "aaaaa..."} > 20 bytes
    meta = {"a": "a" * 20}
    with pytest.raises(PayloadTooLarge, match="meta too large"):
        mock_client.vectors.upsert("test", "id1", [0.1] * 4, meta=meta)


def test_validate_batch_size(mock_client):
    items = []
    from langchain_luma.vectors import VectorBatchItem

    for i in range(6):
        items.append(VectorBatchItem(id=f"id{i}", vector=[0.1] * 4))

    with pytest.raises(ValidationError, match="too many batch items"):
        mock_client.vectors.upsert_batch("test", items)


def test_validate_k(mock_client):
    with pytest.raises(ValidationError, match="k too large"):
        mock_client.vectors.search("test", [0.1] * 4, k=11)


def test_new_methods_exist(mock_client):
    # Just call them to ensure no AttributeErrors
    mock_client.vectors.delete_batch("test", ["id1"])
    assert mock_client._http.last_request[1] == "/v1/vector/test/delete_batch"

    mock_client.vectors.get_vector("test", "id1")
    assert mock_client._http.last_request[1] == "/v1/vector/test/get"

    mock_client.vectors.update_vector("test", "id1", vector=[0.1] * 4)
    assert mock_client._http.last_request[1] == "/v1/vector/test/update"
