"""CORS / session-cookie regression tests for the FastAPI app.

Regression: the Next.js dev server on http://localhost:3001 was missing from a
hard-coded allow-list, so the browser blocked every API response with
"TypeError: Failed to fetch" and the UI showed "Unable to start the RAG
session."
"""

import pytest
from fastapi.testclient import TestClient

from backend import main


@pytest.fixture
def client():
    return TestClient(main.app)


class TestAllowedOrigins:
    def test_local_dev_origins_included_by_default(self, monkeypatch):
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        monkeypatch.delenv("CORS_ALLOW_LOCALHOST", raising=False)
        origins = main.get_allowed_origins()
        for origin in ("http://localhost:3000", "http://localhost:3001",
                       "http://127.0.0.1:3001"):
            assert origin in origins

    def test_frontend_url_is_used_and_normalised(self, monkeypatch):
        monkeypatch.setenv("FRONTEND_URL", "https://my-frontend.example.com/")
        assert "https://my-frontend.example.com" in main.get_allowed_origins()

    def test_extra_origins_from_env(self, monkeypatch):
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:4000, https://a.example")
        origins = main.get_allowed_origins()
        assert "http://localhost:4000" in origins
        assert "https://a.example" in origins

    def test_localhost_can_be_disabled_for_production(self, monkeypatch):
        monkeypatch.setenv("CORS_ALLOW_LOCALHOST", "false")
        monkeypatch.setenv("FRONTEND_URL", "https://prod.example")
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        assert main.get_allowed_origins() == ["https://prod.example"]

    def test_wildcard_rejected_with_credentials(self, monkeypatch):
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
        with pytest.raises(RuntimeError):
            main.get_allowed_origins()


class TestSessionStartCors:
    def test_preflight_from_port_3001(self, client):
        response = client.options(
            "/api/session/start",
            headers={
                "Origin": "http://localhost:3001",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3001"
        assert response.headers["access-control-allow-credentials"] == "true"

    def test_start_sets_cookie_and_cors_headers(self, client):
        response = client.post(
            "/api/session/start", headers={"Origin": "http://localhost:3001"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True and body["session_id"]
        assert response.headers["access-control-allow-origin"] == "http://localhost:3001"
        cookie = response.headers["set-cookie"]
        assert f"rag_session_id={body['session_id']}" in cookie
        assert "HttpOnly" in cookie

    def test_unknown_origin_not_allowed(self, client):
        response = client.post(
            "/api/session/start", headers={"Origin": "https://evil.example"}
        )
        assert "access-control-allow-origin" not in response.headers

    def test_cookie_round_trip_and_isolation(self, client):
        start = client.post("/api/session/start", headers={"Origin": "http://localhost:3001"})
        session_id = start.json()["session_id"]
        status = client.get("/api/session/status")
        assert status.json()["active"] is True
        assert status.json()["session_id"] == session_id
        other_browser = TestClient(main.app)
        assert other_browser.get("/api/session/status").json()["active"] is False
        client.post("/api/session/end")
        assert client.get("/api/session/status").json()["active"] is False
