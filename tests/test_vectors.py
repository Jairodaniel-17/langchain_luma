from langchain_luma.vectors import VectorSearchHit


def test_create_collection(client, mock_response):
    mock_response.json.return_value = {
        "collection": "test",
        "dim": 128,
        "metric": "cosine",
    }

    res = client.vectors.create_collection("test", 128, "cosine")
    assert res["collection"] == "test"


def test_search(client, mock_response):
    mock_response.json.return_value = {"hits": [{"id": "1", "score": 0.9, "meta": {"a": 1}}]}

    hits = client.vectors.search("test", [0.1] * 128)
    assert len(hits) == 1
    assert isinstance(hits[0], VectorSearchHit)
    assert hits[0].id == "1"
    assert hits[0].score == 0.9
    assert hits[0].meta == {"a": 1}


def test_upsert(client, mock_response):
    mock_response.json.return_value = {"description": "Upserted"}
    res = client.vectors.upsert("test", "1", [0.1])
    assert res["description"] == "Upserted"
