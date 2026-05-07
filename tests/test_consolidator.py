"""Tests for manifold.consolidator."""
import pytest
from manifold.consolidator import MemoryConsolidator, MIN_PATTERN_COUNT


def _log(domain, action, stakes, success, n):
    return [{"domain": domain, "action": action, "stakes": stakes, "success": success}] * n


def test_promotes_pattern_with_enough_successes():
    mc = MemoryConsolidator()
    log = _log("finance", "verify", 0.5, True, MIN_PATTERN_COUNT)
    promoted = mc.consolidate(log)
    assert len(promoted) == 1
    assert promoted[0].action == "verify"


def test_skips_fewer_than_min_count():
    mc = MemoryConsolidator()
    log = _log("finance", "verify", 0.5, True, MIN_PATTERN_COUNT - 1)
    promoted = mc.consolidate(log)
    assert promoted == []


def test_skips_low_confidence():
    mc = MemoryConsolidator()
    # 50% success rate — below 0.75 threshold
    log = (
        _log("finance", "verify", 0.5, True, MIN_PATTERN_COUNT)
        + _log("finance", "verify", 0.5, False, MIN_PATTERN_COUNT)
    )
    promoted = mc.consolidate(log)
    assert promoted == []


def test_bucket_stakes():
    assert MemoryConsolidator._bucket_stakes(0.82) == 0.75
    assert MemoryConsolidator._bucket_stakes(0.91) == 1.0


def test_accumulates_across_calls():
    mc = MemoryConsolidator()
    log1 = _log("finance", "verify", 0.5, True, MIN_PATTERN_COUNT)
    log2 = _log("legal", "answer", 0.25, True, MIN_PATTERN_COUNT)
    mc.consolidate(log1)
    mc.consolidate(log2)
    assert len(mc.promoted_rules()) == 2


def test_summary_non_empty_after_consolidation():
    mc = MemoryConsolidator()
    log = _log("finance", "verify", 0.5, True, MIN_PATTERN_COUNT)
    mc.consolidate(log)
    assert len(mc.summary()) > 0
