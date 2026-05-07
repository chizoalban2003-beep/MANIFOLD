"""Tests for manifold.pipeline."""
import pytest
from manifold.pipeline import ManifoldPipeline


def _pipeline():
    return ManifoldPipeline()


def test_run_returns_expected_keys():
    p = _pipeline()
    result = p.run("what time is it")
    for key in ("action", "domain", "risk_score", "encoded", "nearest_cells", "flagged_tools"):
        assert key in result, f"Missing key: {key}"


def test_action_is_nonempty_string():
    p = _pipeline()
    result = p.run("test task")
    assert isinstance(result["action"], str) and len(result["action"]) > 0


def test_domain_is_nonempty_string():
    p = _pipeline()
    result = p.run("test task")
    assert isinstance(result["domain"], str) and len(result["domain"]) > 0


def test_risk_score_bounded():
    p = _pipeline()
    result = p.run("refund invoice billing", stakes=0.5)
    assert 0.0 <= result["risk_score"] <= 1.0


def test_nearest_cells_length_three():
    p = _pipeline()
    result = p.run("test task")
    assert isinstance(result["nearest_cells"], list)
    assert len(result["nearest_cells"]) == 3


def test_high_risk_vs_low_risk():
    p = _pipeline()
    high = p.run("delete all production data permanently", stakes=0.9, uncertainty=0.7)
    low = p.run("what time is it", stakes=0.1, uncertainty=0.1)
    assert high["risk_score"] != low["risk_score"]


def test_tools_used_accepted():
    p = _pipeline()
    result = p.run("test task", tools_used=["api_a", "api_b"])
    assert isinstance(result["flagged_tools"], list)


def test_nightly_consolidation_empty_log():
    p = _pipeline()
    result = p.nightly_consolidation([])
    assert result == []
