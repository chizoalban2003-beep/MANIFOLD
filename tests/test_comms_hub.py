"""tests/test_comms_hub.py — 5 tests for CommHub."""
from __future__ import annotations

import pytest

from manifold.comms_hub import ChannelConfig, CommChannel, CommHub


def _make_hub() -> CommHub:
    return CommHub()


def _make_config(
    channel: CommChannel = CommChannel.WEBHOOK,
    address: str = "http://example.com/hook",
    min_risk: float = 0.0,
    max_risk: float = 1.0,
    user_type: str = "executive",
) -> ChannelConfig:
    return ChannelConfig(
        channel=channel,
        address=address,
        min_risk=min_risk,
        max_risk=max_risk,
        user_type=user_type,
    )


# 1. register_channel stores config correctly
def test_register_channel_stores():
    hub = _make_hub()
    config = _make_config()
    hub.register_channel("owner-1", config)
    channels = hub.channels_for("owner-1")
    assert len(channels) == 1
    assert channels[0].channel == CommChannel.WEBHOOK


# 2. dispatch calls correct channels for given risk_score
def test_dispatch_calls_correct_channel(monkeypatch):
    hub = _make_hub()
    dispatched_from_send = []

    def fake_send(config: ChannelConfig, message: str, escalation: dict) -> bool:
        dispatched_from_send.append(config.channel.value)
        return True

    monkeypatch.setattr(hub, "_send", fake_send)

    hub.register_channel("owner-1", _make_config(channel=CommChannel.WEBHOOK, min_risk=0.0, max_risk=1.0))
    escalation = {"risk_score": 0.5, "action": "test", "domain": "general"}
    result = hub.dispatch("owner-1", escalation)
    assert "webhook" in result


# 3. dispatch skips channel when risk is outside min/max range
def test_dispatch_skips_when_risk_outside_range(monkeypatch):
    hub = _make_hub()

    def fake_send(config, message, escalation):  # noqa: ARG001
        return True

    monkeypatch.setattr(hub, "_send", fake_send)

    # Channel only for high-risk (0.8–1.0)
    hub.register_channel("owner-1", _make_config(channel=CommChannel.WEBHOOK, min_risk=0.8, max_risk=1.0))
    escalation = {"risk_score": 0.4, "action": "test", "domain": "general"}
    result = hub.dispatch("owner-1", escalation)
    assert len(result) == 0  # channel should be skipped


# 4. dispatch returns list of dispatched channel names
def test_dispatch_returns_channel_names(monkeypatch):
    hub = _make_hub()

    def fake_send(config, message, escalation):  # noqa: ARG001
        return True

    monkeypatch.setattr(hub, "_send", fake_send)

    hub.register_channel("owner-1", _make_config(channel=CommChannel.WEBHOOK))
    hub.register_channel("owner-1", _make_config(channel=CommChannel.WORLD_DASHBOARD))

    escalation = {"risk_score": 0.5, "action": "test", "domain": "general"}
    result = hub.dispatch("owner-1", escalation)
    assert isinstance(result, list)
    assert "webhook" in result
    assert "world_dashboard" in result


# 5. Unavailable channel (no env var) fails gracefully, others continue
def test_unavailable_channel_fails_gracefully(monkeypatch):
    hub = _make_hub()

    # Patch SMS to simulate env var missing → returns False
    original_send = hub._send_sms

    def patched_sms(config, message):  # noqa: ARG001
        return False  # simulate no MANIFOLD_SMS_URL

    monkeypatch.setattr(hub, "_send_sms", patched_sms)

    # Patch webhook to succeed
    def patched_webhook(config, message, escalation):  # noqa: ARG001
        return True

    monkeypatch.setattr(hub, "_send_webhook", patched_webhook)

    hub.register_channel("owner-1", _make_config(channel=CommChannel.SMS, address="+447700900123"))
    hub.register_channel("owner-1", _make_config(channel=CommChannel.WEBHOOK, address="http://x.com"))

    escalation = {"risk_score": 0.5, "action": "test", "domain": "general"}
    result = hub.dispatch("owner-1", escalation)
    # Webhook should succeed; SMS should fail gracefully
    assert "webhook" in result
    assert "sms" not in result
