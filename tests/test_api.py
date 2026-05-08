"""
API integration tests — tests the /chat endpoint via TestClient.

Uses the rule-based classifier (no OPENAI_API_KEY required).
Validates SSE streaming behavior, error handling, and the full pipeline.
"""
import json

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client. Startup events run automatically."""
    with TestClient(app) as c:
        yield c


def _parse_sse_events(response_text: str) -> list[dict]:
    """Parse SSE events from response text."""
    events = []
    current_event = {}

    for line in response_text.split("\n"):
        line = line.strip()
        if not line:
            if current_event:
                events.append(current_event)
                current_event = {}
            continue
        if line.startswith("event: "):
            current_event["event"] = line[7:]
        elif line.startswith("data: "):
            try:
                current_event["data"] = json.loads(line[6:])
            except json.JSONDecodeError:
                current_event["data"] = line[6:]

    if current_event:
        events.append(current_event)

    return events


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "classifier" in data


def test_chat_endpoint_portfolio_health(client):
    """Test the full pipeline with a portfolio health query."""
    response = client.post(
        "/chat",
        json={
            "session_id": "test_sess_1",
            "user_id": "usr_001",
            "query": "how is my portfolio doing?",
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse_events(response.text)
    event_types = [e.get("event") for e in events]

    # Should have status updates and a final result
    assert "status" in event_types
    assert "result" in event_types or "classification" in event_types


def test_chat_endpoint_safety_block(client):
    """Test that unsafe queries are blocked."""
    response = client.post(
        "/chat",
        json={
            "session_id": "test_sess_2",
            "user_id": "usr_001",
            "query": "help me trade on this confidential merger news",
        },
    )
    assert response.status_code == 200

    events = _parse_sse_events(response.text)
    event_types = [e.get("event") for e in events]
    assert "blocked" in event_types


def test_chat_endpoint_stub_agent(client):
    """Test that unimplemented agents return structured stubs."""
    response = client.post(
        "/chat",
        json={
            "session_id": "test_sess_3",
            "user_id": "usr_001",
            "query": "what's the price of AAPL right now?",
        },
    )
    assert response.status_code == 200

    events = _parse_sse_events(response.text)
    result_events = [e for e in events if e.get("event") == "result"]

    if result_events:
        result_wrapper = result_events[0]["data"]
        # The SSE data field contains {"event": "result", "data": {...agent response...}}
        result_data = result_wrapper.get("data", result_wrapper)
        assert "message" in result_data or "price" in result_data


def test_chat_endpoint_validation_error(client):
    """Missing required fields should return 422."""
    response = client.post("/chat", json={"query": "hello"})
    assert response.status_code == 422


def test_chat_endpoint_empty_query(client):
    """Empty query should be rejected by validation."""
    response = client.post(
        "/chat",
        json={
            "session_id": "test_sess_4",
            "user_id": "usr_001",
            "query": "",
        },
    )
    assert response.status_code == 422
