"""tests/test_document_ingester.py — Tests for DocumentIngester."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from manifold.ingestion.document_ingester import DocumentIngester, IngestionResult

FIXTURES = Path(__file__).parent / "fixtures"


def test_ingest_txt_returns_result():
    """Ingesting a plain text file returns an IngestionResult."""
    di = DocumentIngester(org_id="test-org", apply_rules=False)
    # Mock the LLM so we don't need network
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.raw_response = "Some analysis.\nMANIFOLD_ACTION_START\n{\"type\": \"none\"}\nMANIFOLD_ACTION_END"
    mock_response.plain_text = "Some analysis."
    mock_response.apply_error = ""
    mock_llm.chat.return_value = mock_response
    di._llm = mock_llm

    result = di.ingest_file(FIXTURES / "sample_policy.txt")
    assert isinstance(result, IngestionResult)
    assert result.format_detected == "text"
    assert result.source.endswith("sample_policy.txt")
    assert len(result.raw_text_preview) <= 200


def test_ingest_csv_parses_rows():
    """Ingesting a CSV file parses rows into policies without LLM."""
    di = DocumentIngester(org_id="test-org", apply_rules=False)
    result = di.ingest_file(FIXTURES / "sample_policy.csv")
    assert isinstance(result, IngestionResult)
    assert result.format_detected == "csv"
    # Should have extracted at least 1 policy from the CSV rows
    assert result.policies_extracted >= 1


def test_ingest_text_direct():
    """ingest_text processes raw text and calls the LLM."""
    di = DocumentIngester(org_id="test-org", apply_rules=False)
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.raw_response = "MANIFOLD_ACTION_START\n{\"type\": \"policy_rule\", \"name\": \"Test\", \"action\": \"audit\", \"priority\": 50, \"conditions\": {}}\nMANIFOLD_ACTION_END"
    mock_response.plain_text = "Found a rule."
    mock_response.apply_error = ""
    mock_llm.chat.return_value = mock_response
    di._llm = mock_llm

    result = di.ingest_text("Audit all AI operations in finance.", source="test")
    assert isinstance(result, IngestionResult)
    # LLM returned one policy_rule block
    assert result.policies_extracted == 1


def test_strip_html_removes_tags():
    """_strip_html should remove all HTML tags."""
    html = "<html><body><h1>Policy</h1><p>Do <b>not</b> do this.</p></body></html>"
    text = DocumentIngester._strip_html(html)
    assert "<" not in text
    assert "Policy" in text
    assert "not" in text
