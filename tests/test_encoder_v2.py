"""Tests for manifold/encoder_v2.py"""
from __future__ import annotations
import pytest
from unittest.mock import patch
from manifold.encoder_v2 import encode_prompt, encoder_backend, EncodedTask


def test_keyword_encode_returns_valid_floats():
    result = encode_prompt("delete all data", force_keyword=True)
    assert isinstance(result, EncodedTask)
    for v in result.as_vector():
        assert 0.0 <= v <= 1.0

def test_encoder_backend_returns_valid_string():
    backend = encoder_backend()
    assert backend in ("semantic", "keyword")

def test_high_risk_scores_higher_than_low_risk():
    high = encode_prompt("delete override bypass password", force_keyword=True)
    low = encode_prompt("hello world nice day", force_keyword=True)
    assert high.risk > low.risk

def test_no_crash_when_sentence_transformers_missing():
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        result = encode_prompt("test prompt", force_keyword=True)
        assert isinstance(result, EncodedTask)
        for v in result.as_vector():
            assert 0.0 <= v <= 1.0

def test_cost_words_increase_cost():
    result = encode_prompt("verify check confirm retrieve", force_keyword=True)
    assert result.cost > 0.0

def test_asset_words_increase_asset():
    result = encode_prompt("resolve fix help complete", force_keyword=True)
    assert result.asset > 0.0
