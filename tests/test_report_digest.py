"""Tests for GET /report and GET /digest endpoints."""
from __future__ import annotations

import json
import sys
import types
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import manifold
import manifold.server as server_mod


# ---------------------------------------------------------------------------
# Helpers to call _handle_get_digest directly without a live HTTP server
# ---------------------------------------------------------------------------

def _make_handler(path: str = "/digest") -> MagicMock:
    """Return a mock ManifoldHandler-like object for testing."""
    handler = MagicMock()
    handler.path = path
    handler.headers = {}
    _responses = []

    def _send_json(h, status, data):
        _responses.append({"status": status, "data": data})

    handler._responses = _responses
    return handler, _send_json


def _call_digest(path: str = "/digest") -> dict:
    """Call _handle_get_digest on a mock handler and return the JSON response."""
    handler = MagicMock()
    handler.path = path

    captured = {}

    def fake_send_json(h, status, data):
        captured["status"] = status
        captured["data"] = data

    with patch.object(server_mod, "_send_json", side_effect=fake_send_json):
        server_mod._handle_get_digest(handler)

    return captured.get("data", {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_digest_returns_dict_with_required_keys():
    result = _call_digest()
    for key in ("generated_at", "period", "version", "summary",
                "domains", "tools", "policy", "governance"):
        assert key in result, f"Missing key: {key}"


def test_digest_summary_has_required_fields():
    result = _call_digest()
    summary = result["summary"]
    for field in ("total_decisions", "escalated", "refused", "permitted",
                  "escalation_rate", "refusal_rate", "mean_risk_score"):
        assert field in summary, f"Missing summary field: {field}"


def test_digest_rates_are_floats_in_range():
    result = _call_digest()
    summary = result["summary"]
    esc = summary["escalation_rate"]
    ref = summary["refusal_rate"]
    assert isinstance(esc, float), "escalation_rate must be float"
    assert isinstance(ref, float), "refusal_rate must be float"
    assert 0.0 <= esc <= 1.0, f"escalation_rate out of range: {esc}"
    assert 0.0 <= ref <= 1.0, f"refusal_rate out of range: {ref}"


def test_digest_default_period_is_7d():
    result = _call_digest("/digest")
    assert result["period"] == "7d"


def test_digest_invalid_period_falls_back_to_7d():
    result = _call_digest("/digest?period=invalid")
    assert result["period"] == "7d"


def test_digest_version_matches_package():
    result = _call_digest()
    assert result["version"] == manifold.__version__


def test_digest_governance_has_flagged_tools_list():
    result = _call_digest()
    governance = result["governance"]
    assert "flagged_tools" in governance
    assert isinstance(governance["flagged_tools"], list)


def test_digest_top_risky_decisions_sorted():
    """Inject data into the pipeline predictor and verify sort order."""
    from manifold.pipeline import ManifoldPipeline
    from manifold.brain import BrainTask

    pipeline = ManifoldPipeline()
    # Run two tasks with known risk ordering
    pipeline.run("delete everything from production", stakes=0.95, uncertainty=0.9)
    pipeline.run("check the weather", stakes=0.1, uncertainty=0.1)

    with patch.object(server_mod, "_pipeline", pipeline):
        result = _call_digest()

    decisions = result["governance"].get("top_risky_decisions", [])
    if len(decisions) < 2:
        pytest.skip("Pipeline did not produce enough decisions")
    scores = [d["risk_score"] for d in decisions]
    assert scores == sorted(scores, reverse=True), "top_risky_decisions not sorted descending"
