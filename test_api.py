
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health_endpoint():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["retrieval_method"] == "hybrid"


def test_valid_search():
    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "query": "I lost my bank card",
            "top_k": 5
        },
        timeout=60
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "I lost my bank card"
    assert data["result_count"] == 5
    assert len(data["results"]) == 5


def test_empty_query():
    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "query": "",
            "top_k": 5
        },
        timeout=10
    )

    assert response.status_code == 422


def test_whitespace_query():
    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "query": "   ",
            "top_k": 5
        },
        timeout=10
    )

    assert response.status_code == 422

    data = response.json()

    assert data["detail"] == (
        "Query cannot be empty or whitespace only."
    )


def test_invalid_top_k():
    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "query": "I lost my bank card",
            "top_k": 50
        },
        timeout=10
    )

    assert response.status_code == 422
