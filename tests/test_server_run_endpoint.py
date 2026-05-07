"""Tests for POST /run endpoint handler."""
import io
import json
from unittest.mock import MagicMock

import pytest

from manifold.pipeline import ManifoldPipeline
from manifold.server import _handle_post_run


def _make_handler(response_bucket):
    """Build a minimal fake ManifoldHandler."""
    handler = MagicMock()
    handler.command = "POST"
    handler.path = "/run"

    # Capture _send_json calls
    def fake_send_json(h, status, body):
        response_bucket["status"] = status
        response_bucket["body"] = body

    # Patch module-level _send_json to capture output
    import manifold.server as srv
    srv._send_json = fake_send_json
    return handler


def _run(body: dict):
    """Call _handle_post_run directly and return (status, body)."""
    response = {}
    handler = _make_handler(response)
    _handle_post_run(handler, body)
    return response.get("status"), response.get("body")


def test_valid_prompt_returns_action():
    status, body = _run({"prompt": "what time is it"})
    assert status == 200
    assert "action" in body


def test_missing_prompt_returns_400():
    status, body = _run({})
    assert status == 400
    assert "error" in body


def test_tools_used_accepted():
    status, body = _run({"prompt": "test task", "tools_used": ["api_a", "api_b"]})
    assert status == 200


def test_explicit_domain_returned():
    status, body = _run({"prompt": "quarterly revenue forecast", "domain": "finance"})
    assert status == 200
    assert body.get("domain") == "finance"


def test_risk_score_bounded():
    status, body = _run({"prompt": "send email", "stakes": 0.4, "uncertainty": 0.3})
    assert status == 200
    assert 0.0 <= body["risk_score"] <= 1.0
