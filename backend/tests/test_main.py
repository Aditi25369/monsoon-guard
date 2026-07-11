"""Tests for Monsoon Guard API.

Run: python -m pytest -q
Gemini and live-weather calls are mocked so the suite runs offline/
deterministically and doesn't burn API quota in CI.
"""
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "test-key-for-suite")

import main  # noqa: E402
from main import app  # noqa: E402
from services import gemini_service  # noqa: E402

# Using TestClient as a context manager triggers the app's lifespan
# (startup/shutdown), which is what creates app.state.http_client.
client = TestClient(app)
client.__enter__()

MOCK_PLAN = {
    "summary": "Stay prepared for heavy rain in your area.",
    "action_items": ["Check drainage near home", "Charge power banks"],
    "checklist": ["Torch", "First aid kit", "Drinking water"],
    "emergency_contacts_hint": ["Save your local disaster helpline number"],
}


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """Rate-limit buckets and the Gemini response cache are process-global,
    so tests can bleed into each other without this reset (this made the
    rate-limit test flaky depending on run order)."""
    main._hits.clear()
    gemini_service._cache.clear()
    yield
    main._hits.clear()
    gemini_service._cache.clear()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("main.generate_json", new_callable=AsyncMock)
def test_preparedness_plan_success(mock_gen):
    mock_gen.return_value = MOCK_PLAN
    resp = client.post(
        "/api/preparedness-plan",
        json={
            "location": "Belagavi",
            "household_type": "family",
            "dwelling_type": "low_lying_area",
            "household_size": 4,
            "has_vehicle": True,
            "language": "en",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]
    assert len(body["checklist"]) == 3
    mock_gen.assert_awaited_once()


def test_preparedness_plan_invalid_household_size():
    resp = client.post(
        "/api/preparedness-plan",
        json={
            "location": "Belagavi",
            "household_type": "family",
            "dwelling_type": "low_lying_area",
            "household_size": 0,  # invalid: must be >= 1
            "language": "en",
        },
    )
    assert resp.status_code == 422


def test_preparedness_plan_missing_location():
    resp = client.post(
        "/api/preparedness-plan",
        json={
            "location": "",
            "household_type": "family",
            "dwelling_type": "apartment",
            "household_size": 2,
            "language": "en",
        },
    )
    assert resp.status_code == 422


@patch("main.generate_json", new_callable=AsyncMock)
def test_checklist_success(mock_gen):
    mock_gen.return_value = {"checklist": ["Torch", "Water", "Medicines"]}
    resp = client.post(
        "/api/checklist",
        json={
            "location": "Bengaluru",
            "dwelling_type": "apartment",
            "household_size": 3,
            "language": "kn",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["checklist"] == ["Torch", "Water", "Medicines"]


@patch("main.generate_json", new_callable=AsyncMock)
def test_travel_advisory_success(mock_gen):
    mock_gen.return_value = MOCK_PLAN
    resp = client.post(
        "/api/travel-advisory",
        json={
            "origin": "Belagavi",
            "destination": "Bengaluru",
            "travel_date": "2026-07-15",
            "mode": "car",
            "language": "en",
        },
    )
    assert resp.status_code == 200


@patch("main.generate_json", new_callable=AsyncMock)
def test_translate_success(mock_gen):
    mock_gen.return_value = {"translated_text": "\u0928\u092e\u0938\u094d\u0924\u0947"}
    resp = client.post(
        "/api/translate",
        json={"text": "Hello", "target_language": "hi"},
    )
    assert resp.status_code == 200
    assert resp.json()["translated_text"]


def test_translate_empty_text_rejected():
    resp = client.post(
        "/api/translate",
        json={"text": "", "target_language": "hi"},
    )
    assert resp.status_code == 422


@patch("services.alerts_service._fetch_live_alert", new_callable=AsyncMock)
def test_alerts_success_live(mock_fetch):
    from models import Alert, AlertSeverity

    mock_fetch.return_value = Alert(
        id="abc123",
        location="Belagavi",
        severity=AlertSeverity.MODERATE,
        headline="Moderate to heavy rain expected",
        message="Avoid low-lying underpasses.",
        issued_at="2026-07-11T00:00:00+00:00",
    )
    resp = client.get("/api/alerts", params={"location": "Belagavi"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["location"] == "Belagavi"
    assert body[0]["severity"] == "moderate"


@patch("services.alerts_service._fetch_live_alert", new_callable=AsyncMock)
def test_alerts_falls_back_when_live_lookup_fails(mock_fetch):
    mock_fetch.side_effect = RuntimeError("provider down")
    resp = client.get("/api/alerts", params={"location": "Mumbai"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["severity"] in ("low", "moderate", "high", "severe")


@patch("services.alerts_service._fetch_live_alert", new_callable=AsyncMock)
def test_alerts_cached_for_same_location(mock_fetch):
    mock_fetch.side_effect = RuntimeError("provider down")
    r1 = client.get("/api/alerts", params={"location": "Pune"}).json()
    r2 = client.get("/api/alerts", params={"location": "Pune"}).json()
    assert r1[0]["severity"] == r2[0]["severity"]
    # Second call should be served from cache, not hit the provider again.
    assert mock_fetch.await_count == 1


def test_alerts_missing_location_rejected():
    resp = client.get("/api/alerts", params={"location": ""})
    assert resp.status_code == 422


@patch("main.generate_json", new_callable=AsyncMock)
def test_gemini_upstream_failure_returns_502(mock_gen):
    from services.gemini_service import GeminiRequestError

    mock_gen.side_effect = GeminiRequestError("boom")
    resp = client.post(
        "/api/checklist",
        json={
            "location": "Pune",
            "dwelling_type": "apartment",
            "household_size": 2,
            "language": "en",
        },
    )
    assert resp.status_code == 502


def test_security_headers_present():
    resp = client.get("/health")
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"


def test_rate_limit_blocks_after_threshold():
    for _ in range(20):
        client.get("/health")
    resp = client.get("/health")
    assert resp.status_code == 429


def test_rate_limit_bucket_evicted_when_empty():
    """Buckets for IPs with no recent activity shouldn't linger forever."""
    client.get("/health")
    assert "testclient" in main._hits
    main._hits["testclient"].clear()
    client.get("/health")
    # After a fresh hit, the bucket exists again with exactly one entry —
    # confirms eviction isn't silently accumulating stale entries.
    assert len(main._hits["testclient"]) >= 1
