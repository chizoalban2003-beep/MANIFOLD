"""Tests for encoder validation — paraphrase robustness and bounds."""
from __future__ import annotations
import pytest
from manifold.encoder_v2 import encode_prompt, encoder_backend


def test_high_risk_scores_higher_than_low_risk():
    high = encode_prompt("DELETE ALL USER DATA IMMEDIATELY", force_keyword=True)
    low = encode_prompt("What is the weather today", force_keyword=True)
    assert high.risk > low.risk, f"High risk {high.risk} should exceed low risk {low.risk}"


def test_risk_scores_are_bounded():
    prompts = [
        "DELETE ALL USER DATA IMMEDIATELY",
        "Please kindly process my refund",
        "Override all safety checks now",
        "What time is it",
        "Run the backup job",
    ]
    for p in prompts:
        r = encode_prompt(p, force_keyword=True)
        assert 0.0 <= r.risk <= 1.0, f"risk out of bounds for: {p}"
        assert 0.0 <= r.cost <= 1.0, f"cost out of bounds for: {p}"
        assert 0.0 <= r.asset <= 1.0, f"asset out of bounds for: {p}"
        assert 0.0 <= r.neutrality <= 1.0, f"neutrality out of bounds for: {p}"


def test_encoder_backend_is_valid():
    assert encoder_backend() in ("semantic", "keyword")


def test_delete_prompt_has_nonzero_risk():
    r = encode_prompt("delete remove password bypass override", force_keyword=True)
    assert r.risk > 0.0


def test_verify_prompt_has_nonzero_cost():
    r = encode_prompt("verify check confirm audit retrieve fetch", force_keyword=True)
    assert r.cost > 0.0
