"""Tests for dashboard.py API endpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dashboard
from fastapi.testclient import TestClient

client = TestClient(dashboard.app)
TOKEN = dashboard._auth_token


class TestAuth:
    def test_data_no_token_returns_403(self):
        r = client.get("/api/data")
        assert r.status_code == 403

    def test_data_bad_token_returns_403(self):
        r = client.get("/api/data?token=wrong")
        assert r.status_code == 403

    def test_data_good_token_returns_200(self):
        r = client.get(f"/api/data?token={TOKEN}")
        assert r.status_code == 200

    def test_log_no_token_returns_403(self):
        r = client.get("/api/log")
        assert r.status_code == 403

    def test_remove_no_token_returns_403(self):
        r = client.post("/api/remove?key=test")
        assert r.status_code == 403


class TestHealthz:
    def test_healthz_no_auth_needed(self):
        r = client.get("/healthz")
        assert r.status_code == 200
        d = r.json()
        assert "version" in d
        assert d["version"] == "0.2.0"
        assert "uptime_seconds" in d
        assert "runs_loaded" in d

    def test_healthz_ssh_status(self):
        r = client.get("/healthz")
        d = r.json()
        assert "ssh_health" in d


class TestIndex:
    def test_index_returns_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "pgolf dashboard" in r.text
        # Token should be embedded (not the placeholder)
        assert "__PGOLF_TOKEN__" not in r.text
        assert TOKEN in r.text

    def test_baseline_placeholder_replaced(self):
        r = client.get("/")
        assert "__PGOLF_BASELINE__" not in r.text


class TestDataEndpoint:
    def test_returns_runs_and_config(self):
        r = client.get(f"/api/data?token={TOKEN}")
        d = r.json()
        assert "runs" in d
        assert "config" in d
        assert "refresh" in d["config"]

    def test_remove_nonexistent_key(self):
        r = client.post(f"/api/remove?key=nonexistent&token={TOKEN}")
        assert r.status_code == 200
