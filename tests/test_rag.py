from langchain_luma.rag import HighLevelSearchResponse, HighLevelSearchResult


def test_rag_search(client, mock_response):
    mock_response.json.return_value = {
        "query": "test",
        "top_k": 1,
        "results": [
            {
                "score": 0.8,
                "document": {
                    "id": 1,
                    "content": "hello",
                    "metadata": {"filename": "f.txt"},
                },
            }
        ],
    }

    res = client.rag.search("test")
    assert isinstance(res, HighLevelSearchResponse)
    assert len(res.results) == 1
    assert isinstance(res.results[0], HighLevelSearchResult)
    assert res.results[0].document.content == "hello"
    assert res.results[0].document.metadata_safe.filename == "f.txt"
