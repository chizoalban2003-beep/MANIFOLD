"""tests/test_image_ingester.py — Tests for ImageIngester."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from manifold.ingestion.image_ingester import ImageIngester, ImageIngestionResult

FIXTURES = Path(__file__).parent / "fixtures"


def _make_ingester(vision_response: str = "") -> tuple[ImageIngester, MagicMock]:
    """Return an ImageIngester with _call_vision mocked."""
    ii = ImageIngester(org_id="test-org")
    mock_call = MagicMock(return_value=vision_response)
    ii._call_vision = mock_call
    return ii, mock_call


def test_policy_pipeline_extracts_rule():
    """Policy pipeline should extract a rule from a mock vision response."""
    vision_resp = (
        "I see a sign saying 'No finance operations after hours'.\n"
        "MANIFOLD_ACTION_START\n"
        "{\"type\": \"policy_rule\", \"name\": \"No after-hours finance\", "
        "\"conditions\": {\"domain\": \"finance\"}, \"action\": \"refuse\", \"priority\": 60}\n"
        "MANIFOLD_ACTION_END"
    )
    ii, _ = _make_ingester(vision_resp)
    result = ii.ingest(FIXTURES / "test_image.png", pipeline="policy")
    assert isinstance(result, ImageIngestionResult)
    assert result.pipeline_used == "policy"
    assert result.items_extracted >= 1


def test_spatial_pipeline_handles_json():
    """Spatial pipeline should parse room JSON from vision response."""
    rooms_json = json.dumps({
        "rooms": [
            {"name": "Kitchen", "type": "kitchen", "x": 0, "y": 0,
             "width": 3, "height": 3, "crna": {"c": 0.4, "r": 0.7, "n": 0.8, "a": 0.9}},
            {"name": "Bedroom", "type": "bedroom", "x": 3, "y": 0,
             "width": 3, "height": 3, "crna": {"c": 0.2, "r": 0.3, "n": 1.0, "a": 0.8}},
        ]
    })
    ii, _ = _make_ingester(rooms_json)
    result = ii.ingest(FIXTURES / "test_image.png", pipeline="spatial")
    assert result.pipeline_used == "spatial"
    assert result.items_extracted == 2


def test_ingest_bytes_policy():
    """ingest() with raw bytes should work in policy mode."""
    raw = (FIXTURES / "test_image.png").read_bytes()
    vision_resp = (
        "MANIFOLD_ACTION_START\n"
        "{\"type\": \"none\"}\n"
        "MANIFOLD_ACTION_END"
    )
    ii, _ = _make_ingester(vision_resp)
    result = ii.ingest(raw, pipeline="policy")
    assert isinstance(result, ImageIngestionResult)
    assert result.pipeline_used == "policy"
