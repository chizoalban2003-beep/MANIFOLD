"""Tests for GET /learned endpoint handler."""
from unittest.mock import MagicMock

import pytest

from manifold.server import _handle_get_learned


def _run_learned():
    """Call _handle_get_learned and return (status, body)."""
    response = {}

    def fake_send_json(h, status, body):
        response["status"] = status
        response["body"] = body

    import manifold.server as srv
    srv._send_json = fake_send_json

    handler = MagicMock()
    handler.command = "GET"
    handler.path = "/learned"
    _handle_get_learned(handler)
    return response.get("status"), response.get("body")


def test_response_has_cognitive_map_key():
    status, body = _run_learned()
    assert status == 200
    assert "cognitive_map" in body


def test_response_has_cooccurrence_key():
    _, body = _run_learned()
    assert "cooccurrence" in body


def test_response_has_consolidation_key():
    _, body = _run_learned()
    assert "consolidation" in body


def test_response_has_prediction_key():
    _, body = _run_learned()
    assert "prediction" in body


def test_cognitive_map_total_outcomes_nonnegative_int():
    _, body = _run_learned()
    total = body["cognitive_map"]["total_outcomes"]
    assert isinstance(total, int)
    assert total >= 0
