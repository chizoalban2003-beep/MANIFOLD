"""Tests for ManifoldWorker background learning loop."""
import pytest

from manifold.pipeline import ManifoldPipeline
from manifold.worker import ManifoldWorker


def _worker():
    return ManifoldWorker(pipeline=ManifoldPipeline())


def test_instantiate_without_db():
    w = _worker()
    assert w is not None


def test_status_has_expected_keys():
    w = _worker()
    s = w.status()
    for key in ("running", "last_run", "interval_seconds"):
        assert key in s


def test_status_running_false_before_start():
    w = _worker()
    assert w.status()["running"] is False


def test_run_once_returns_expected_keys():
    w = _worker()
    result = w._run_once()
    for key in ("calibration", "rules_promoted", "timestamp", "errors"):
        assert key in result


def test_run_once_empty_prediction_log():
    w = _worker()
    result = w._run_once()
    # Should not raise and errors should be empty or only non-fatal
    assert isinstance(result["errors"], list)


def test_run_once_rules_promoted_nonnegative():
    w = _worker()
    result = w._run_once()
    assert result["rules_promoted"] >= 0
