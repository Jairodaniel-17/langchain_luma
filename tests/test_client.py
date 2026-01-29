def test_client_init():
    from langchain_luma import LumaClient

    client = LumaClient(url="http://test:1234", api_key="secret")
    assert client._http.base_url == "http://test:1234"
    assert client._http.session.headers["Authorization"] == "Bearer secret"


def test_system_health(client, mock_response, monkeypatch):
    mock_response.json.return_value = {"status": "ok"}

    res = client.system.health()
    assert res == {"status": "ok"}


def test_system_metrics(client, mock_response, monkeypatch):
    mock_response.text = "metrics"
    # We need to ensure json() raises ValueError for text response to work in our HttpTransport
    mock_response.json.side_effect = ValueError

    res = client.system.metrics()
    assert res == "metrics"
