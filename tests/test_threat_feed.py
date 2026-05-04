"""Tests for Phase 34: Threat Intelligence Feed (manifold/threat_feed.py)."""

from __future__ import annotations

import json
import time

import pytest

from manifold.federation import FederatedGossipPacket
from manifold.probe import CanaryResult
from manifold.threat_feed import (
    ThreatFeedStreamer,
    ThreatIntelPayload,
    _canary_severity,
    _GOSSIP_SEVERITY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gossip(signal: str = "failing", tool: str = "gpt-4o") -> FederatedGossipPacket:
    return FederatedGossipPacket(
        tool_name=tool,
        signal=signal,  # type: ignore[arg-type]
        confidence=0.9,
        org_id="org-test",
        weight=1.0,
    )


def _canary(action: str = "fail", tool: str = "gpt-4o") -> CanaryResult:
    return CanaryResult(
        tool_name=tool,
        entropy_score_before=0.6,
        adversarial_suspect=(action == "suspect"),
        timestamp=time.time(),
        probe_action=action,
        penalty_applied=(action in {"fail", "suspect"}),
    )


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


class TestSeverityHelpers:
    def test_gossip_severity_failing(self) -> None:
        assert _GOSSIP_SEVERITY["failing"] == "high"

    def test_gossip_severity_degraded(self) -> None:
        assert _GOSSIP_SEVERITY["degraded"] == "medium"

    def test_gossip_severity_healthy(self) -> None:
        assert _GOSSIP_SEVERITY["healthy"] == "low"

    def test_canary_severity_suspect(self) -> None:
        result = _canary("suspect")
        assert _canary_severity(result) == "critical"

    def test_canary_severity_fail(self) -> None:
        result = _canary("fail")
        assert _canary_severity(result) == "high"

    def test_canary_severity_pass(self) -> None:
        result = _canary("pass")
        assert _canary_severity(result) == "low"


# ---------------------------------------------------------------------------
# ThreatIntelPayload
# ---------------------------------------------------------------------------


class TestThreatIntelPayload:
    def test_creation(self) -> None:
        p = ThreatIntelPayload(
            event_type="gossip_failing",
            tool_name="gpt-4o",
            severity="high",
            timestamp=1_000_000.0,
            details={"signal": "failing"},
        )
        assert p.event_type == "gossip_failing"
        assert p.severity == "high"
        assert p.source == "manifold"

    def test_custom_source(self) -> None:
        p = ThreatIntelPayload("t", "tool", "medium", 0.0, {}, source="canary_prober")
        assert p.source == "canary_prober"

    def test_frozen(self) -> None:
        p = ThreatIntelPayload("t", "tool", "high", 0.0, {})
        with pytest.raises((AttributeError, TypeError)):
            p.severity = "low"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        p = ThreatIntelPayload("e", "t", "high", 100.0, {"k": "v"})
        d = p.to_dict()
        assert d["event_type"] == "e"
        assert d["tool_name"] == "t"
        assert d["severity"] == "high"
        assert d["details"] == {"k": "v"}

    def test_to_json_valid(self) -> None:
        p = ThreatIntelPayload("e", "t", "high", 0.0, {})
        raw = p.to_json()
        parsed = json.loads(raw)
        assert parsed["event_type"] == "e"

    def test_to_sse_format(self) -> None:
        p = ThreatIntelPayload("e", "t", "high", 0.0, {})
        sse = p.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")

    def test_to_sse_json_parseable(self) -> None:
        p = ThreatIntelPayload("e", "t", "critical", 0.0, {})
        sse = p.to_sse()
        json_part = sse[len("data: "):]
        parsed = json.loads(json_part.strip())
        assert parsed["severity"] == "critical"


# ---------------------------------------------------------------------------
# ThreatFeedStreamer — configuration
# ---------------------------------------------------------------------------


class TestThreatFeedStreamerConfig:
    def test_defaults(self) -> None:
        streamer = ThreatFeedStreamer()
        assert streamer.max_events == 500
        assert streamer.include_low_severity is False

    def test_custom_max_events(self) -> None:
        streamer = ThreatFeedStreamer(max_events=10)
        assert streamer.max_events == 10

    def test_empty_event_count(self) -> None:
        streamer = ThreatFeedStreamer()
        assert streamer.event_count() == 0

    def test_clear_resets(self) -> None:
        streamer = ThreatFeedStreamer()
        streamer.ingest_gossip(_gossip("failing"))
        streamer.clear()
        assert streamer.event_count() == 0


# ---------------------------------------------------------------------------
# ThreatFeedStreamer — ingest_gossip()
# ---------------------------------------------------------------------------


class TestThreatFeedStreamerIngestGossip:
    def test_failing_signal_stored(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_gossip(_gossip("failing"))
        assert payload is not None
        assert streamer.event_count() == 1

    def test_degraded_signal_stored(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_gossip(_gossip("degraded"))
        assert payload is not None

    def test_healthy_signal_suppressed_by_default(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_gossip(_gossip("healthy"))
        assert payload is None
        assert streamer.event_count() == 0

    def test_healthy_signal_stored_when_include_low(self) -> None:
        streamer = ThreatFeedStreamer(include_low_severity=True)
        payload = streamer.ingest_gossip(_gossip("healthy"))
        assert payload is not None
        assert streamer.event_count() == 1

    def test_correct_event_type(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_gossip(_gossip("failing"))
        assert payload is not None
        assert payload.event_type == "gossip_failing"

    def test_correct_severity_failing(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_gossip(_gossip("failing"))
        assert payload is not None
        assert payload.severity == "high"

    def test_correct_severity_degraded(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_gossip(_gossip("degraded"))
        assert payload is not None
        assert payload.severity == "medium"

    def test_tool_name_preserved(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_gossip(_gossip("failing", tool="claude-3"))
        assert payload is not None
        assert payload.tool_name == "claude-3"

    def test_source_is_gossip_bridge(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_gossip(_gossip("failing"))
        assert payload is not None
        assert payload.source == "gossip_bridge"

    def test_details_contain_confidence(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_gossip(_gossip("failing"))
        assert payload is not None
        assert "confidence" in payload.details


# ---------------------------------------------------------------------------
# ThreatFeedStreamer — ingest_canary()
# ---------------------------------------------------------------------------


class TestThreatFeedStreamerIngestCanary:
    def test_fail_canary_stored(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_canary(_canary("fail"))
        assert payload is not None
        assert streamer.event_count() == 1

    def test_suspect_canary_stored(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_canary(_canary("suspect"))
        assert payload is not None
        assert payload.severity == "critical"

    def test_pass_canary_suppressed_by_default(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_canary(_canary("pass"))
        assert payload is None
        assert streamer.event_count() == 0

    def test_pass_canary_stored_when_include_low(self) -> None:
        streamer = ThreatFeedStreamer(include_low_severity=True)
        payload = streamer.ingest_canary(_canary("pass"))
        assert payload is not None

    def test_correct_event_type_fail(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_canary(_canary("fail"))
        assert payload is not None
        assert payload.event_type == "canary_fail"

    def test_correct_event_type_suspect(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_canary(_canary("suspect"))
        assert payload is not None
        assert payload.event_type == "canary_suspect"

    def test_source_is_canary_prober(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_canary(_canary("fail"))
        assert payload is not None
        assert payload.source == "canary_prober"

    def test_details_contain_probe_action(self) -> None:
        streamer = ThreatFeedStreamer()
        payload = streamer.ingest_canary(_canary("fail"))
        assert payload is not None
        assert payload.details["probe_action"] == "fail"


# ---------------------------------------------------------------------------
# ThreatFeedStreamer — retrieval
# ---------------------------------------------------------------------------


class TestThreatFeedStreamerRetrieval:
    def test_recent_events_empty(self) -> None:
        streamer = ThreatFeedStreamer()
        assert streamer.recent_events() == []

    def test_recent_events_newest_first(self) -> None:
        streamer = ThreatFeedStreamer(include_low_severity=True)
        streamer.ingest_gossip(_gossip("failing", "tool-1"))
        streamer.ingest_gossip(_gossip("failing", "tool-2"))
        events = streamer.recent_events(n=10)
        assert events[0].tool_name == "tool-2"

    def test_recent_events_limit(self) -> None:
        streamer = ThreatFeedStreamer()
        for i in range(10):
            streamer.ingest_gossip(_gossip("failing", f"tool-{i}"))
        events = streamer.recent_events(n=3)
        assert len(events) == 3

    def test_events_by_severity_high(self) -> None:
        streamer = ThreatFeedStreamer()
        streamer.ingest_gossip(_gossip("failing"))
        streamer.ingest_gossip(_gossip("degraded"))
        high = streamer.events_by_severity("high")
        assert all(e.severity == "high" for e in high)

    def test_events_by_severity_empty(self) -> None:
        streamer = ThreatFeedStreamer()
        assert streamer.events_by_severity("critical") == []

    def test_ring_buffer_overflow(self) -> None:
        streamer = ThreatFeedStreamer(max_events=5)
        for i in range(10):
            streamer.ingest_gossip(_gossip("failing", f"tool-{i}"))
        assert streamer.event_count() == 5

    def test_ring_buffer_keeps_newest(self) -> None:
        streamer = ThreatFeedStreamer(max_events=3)
        for i in range(5):
            streamer.ingest_gossip(_gossip("failing", f"tool-{i}"))
        events = list(reversed(streamer.recent_events(n=10)))  # oldest first
        # Should contain only tool-2, tool-3, tool-4
        names = [e.tool_name for e in events]
        assert "tool-0" not in names
        assert "tool-4" in names


# ---------------------------------------------------------------------------
# ThreatFeedStreamer — SSE stream
# ---------------------------------------------------------------------------


class TestThreatFeedStreamerSSE:
    def test_sse_stream_empty(self) -> None:
        streamer = ThreatFeedStreamer()
        chunks = list(streamer.sse_stream())
        assert chunks == []

    def test_sse_stream_yields_events(self) -> None:
        streamer = ThreatFeedStreamer()
        streamer.ingest_gossip(_gossip("failing"))
        chunks = list(streamer.sse_stream())
        assert len(chunks) == 1
        assert chunks[0].startswith("data: ")

    def test_sse_stream_max_events(self) -> None:
        streamer = ThreatFeedStreamer()
        for _ in range(10):
            streamer.ingest_gossip(_gossip("failing"))
        chunks = list(streamer.sse_stream(max_events=3))
        assert len(chunks) == 3

    def test_sse_events_parseable(self) -> None:
        streamer = ThreatFeedStreamer()
        streamer.ingest_gossip(_gossip("failing"))
        for chunk in streamer.sse_stream():
            json_part = chunk[len("data: "):]
            parsed = json.loads(json_part.strip())
            assert "event_type" in parsed


# ---------------------------------------------------------------------------
# ThreatFeedStreamer — summary
# ---------------------------------------------------------------------------


class TestThreatFeedStreamerSummary:
    def test_empty_summary(self) -> None:
        streamer = ThreatFeedStreamer()
        s = streamer.summary()
        assert s["total_events"] == 0
        assert s["critical"] == 0
        assert s["high"] == 0

    def test_summary_counts_by_severity(self) -> None:
        streamer = ThreatFeedStreamer()
        streamer.ingest_gossip(_gossip("failing"))  # high
        streamer.ingest_gossip(_gossip("degraded"))  # medium
        streamer.ingest_canary(_canary("suspect"))   # critical
        s = streamer.summary()
        assert s["total_events"] == 3
        assert s["critical"] == 1
        assert s["high"] == 1
        assert s["medium"] == 1

    def test_summary_keys_present(self) -> None:
        streamer = ThreatFeedStreamer()
        s = streamer.summary()
        for key in ("total_events", "critical", "high", "medium", "low"):
            assert key in s
