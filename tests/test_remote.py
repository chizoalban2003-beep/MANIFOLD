"""Tests for Phase 58: Remote Sovereign Authorization (remote.py)."""

from __future__ import annotations

import hashlib
import hmac
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from manifold.remote import (
    DeliveryResult,
    GatewayLog,
    MobileAlertGateway,
    RemoteDecision,
    RemoteSigner,
    SigningRequest,
    WebhookConfig,
)


# ---------------------------------------------------------------------------
# WebhookConfig
# ---------------------------------------------------------------------------


class TestWebhookConfig:
    def test_to_dict_redacts_secret(self) -> None:
        wh = WebhookConfig(url="https://example.com/hook", secret="supersecret")
        d = wh.to_dict()
        assert d["secret"] == "<redacted>"

    def test_to_dict_empty_secret(self) -> None:
        wh = WebhookConfig(url="https://example.com/hook")
        d = wh.to_dict()
        assert d["secret"] == ""

    def test_to_dict_url(self) -> None:
        wh = WebhookConfig(url="https://hooks.slack.com/xyz")
        assert wh.to_dict()["url"] == "https://hooks.slack.com/xyz"

    def test_default_topics(self) -> None:
        wh = WebhookConfig(url="https://x.com/")
        assert "*" in wh.topics

    def test_custom_topics(self) -> None:
        wh = WebhookConfig(url="https://x.com/", topics=("sandbox.violation",))
        assert "sandbox.violation" in wh.topics


# ---------------------------------------------------------------------------
# DeliveryResult
# ---------------------------------------------------------------------------


class TestDeliveryResult:
    def test_to_dict(self) -> None:
        dr = DeliveryResult(
            webhook_name="slack",
            topic="sandbox.violation",
            status_code=200,
            success=True,
            elapsed_ms=45.2,
            error="",
            timestamp=time.time(),
        )
        d = dr.to_dict()
        assert d["success"] is True
        assert d["status_code"] == 200
        assert d["webhook_name"] == "slack"


# ---------------------------------------------------------------------------
# GatewayLog
# ---------------------------------------------------------------------------


class TestGatewayLog:
    def _result(self, success: bool = True) -> DeliveryResult:
        return DeliveryResult(
            webhook_name="test",
            topic="t",
            status_code=200 if success else 500,
            success=success,
            elapsed_ms=1.0,
            error="",
            timestamp=time.time(),
        )

    def test_record_and_recent(self) -> None:
        log = GatewayLog()
        log.record(self._result())
        assert len(log.recent()) == 1

    def test_recent_limit(self) -> None:
        log = GatewayLog()
        for _ in range(10):
            log.record(self._result())
        assert len(log.recent(5)) == 5

    def test_success_rate_all_success(self) -> None:
        log = GatewayLog()
        for _ in range(5):
            log.record(self._result(True))
        assert log.success_rate() == 1.0

    def test_success_rate_all_fail(self) -> None:
        log = GatewayLog()
        for _ in range(5):
            log.record(self._result(False))
        assert log.success_rate() == 0.0

    def test_success_rate_mixed(self) -> None:
        log = GatewayLog()
        log.record(self._result(True))
        log.record(self._result(False))
        assert abs(log.success_rate() - 0.5) < 1e-9

    def test_empty_success_rate(self) -> None:
        log = GatewayLog()
        assert log.success_rate() == 1.0

    def test_max_entries_eviction(self) -> None:
        log = GatewayLog(max_entries=3)
        for _ in range(5):
            log.record(self._result())
        assert len(log.recent(100)) == 3

    def test_to_dict(self) -> None:
        log = GatewayLog()
        log.record(self._result(True))
        log.record(self._result(False))
        d = log.to_dict()
        assert d["total_deliveries"] == 2
        assert d["successful"] == 1


# ---------------------------------------------------------------------------
# MobileAlertGateway — topic matching & sync delivery
# ---------------------------------------------------------------------------


class TestMobileAlertGateway:
    def _mock_event(self, topic: str, payload: dict | None = None) -> MagicMock:
        evt = MagicMock()
        evt.topic = topic
        evt.payload = payload or {}
        evt.timestamp = time.time()
        return evt

    def test_deliver_sync_wildcard(self) -> None:
        called = []

        class FakeWebhookConfig(WebhookConfig):
            pass

        gw = MobileAlertGateway(
            webhooks=[WebhookConfig(url="https://example.com/", topics=("*",), name="all")],
            async_delivery=False,
        )

        # Patch _deliver to avoid real HTTP
        original_deliver = gw._deliver

        def fake_deliver(wh, topic, payload, timestamp):
            called.append(topic)
            return DeliveryResult(
                webhook_name=wh.name,
                topic=topic,
                status_code=200,
                success=True,
                elapsed_ms=1.0,
                error="",
                timestamp=time.time(),
            )

        gw._deliver = fake_deliver  # type: ignore[method-assign]
        results = gw.deliver_sync("sandbox.violation", {"x": 1})
        assert len(results) == 1
        assert called == ["sandbox.violation"]

    def test_deliver_sync_specific_topic_match(self) -> None:
        gw = MobileAlertGateway(
            webhooks=[
                WebhookConfig(url="https://a.com/", topics=("sandbox.violation",), name="sv")
            ],
            async_delivery=False,
        )

        def fake_deliver(wh, topic, payload, timestamp):
            return DeliveryResult(
                webhook_name=wh.name, topic=topic, status_code=200,
                success=True, elapsed_ms=1.0, error="", timestamp=time.time()
            )

        gw._deliver = fake_deliver  # type: ignore[method-assign]
        results = gw.deliver_sync("sandbox.violation", {})
        assert len(results) == 1

    def test_deliver_sync_topic_mismatch(self) -> None:
        gw = MobileAlertGateway(
            webhooks=[
                WebhookConfig(url="https://a.com/", topics=("sandbox.violation",), name="sv")
            ],
            async_delivery=False,
        )
        gw._deliver = MagicMock()  # type: ignore[method-assign]
        results = gw.deliver_sync("dht.peer.joined", {})
        assert results == []
        gw._deliver.assert_not_called()

    def test_deliver_sync_wildcard_prefix_match(self) -> None:
        gw = MobileAlertGateway(
            webhooks=[
                WebhookConfig(url="https://a.com/", topics=("sandbox.*",), name="sb")
            ],
            async_delivery=False,
        )
        called = []

        def fake_deliver(wh, topic, payload, timestamp):
            called.append(topic)
            return DeliveryResult(
                webhook_name=wh.name, topic=topic, status_code=200,
                success=True, elapsed_ms=1.0, error="", timestamp=time.time()
            )

        gw._deliver = fake_deliver  # type: ignore[method-assign]
        gw.deliver_sync("sandbox.timeout", {})
        assert called == ["sandbox.timeout"]

    def test_on_event_async_fires_thread(self) -> None:
        gw = MobileAlertGateway(
            webhooks=[WebhookConfig(url="https://a.com/", name="t")],
            async_delivery=True,
        )
        fired = threading.Event()

        def fake_deliver(wh, topic, payload, timestamp):
            fired.set()
            return DeliveryResult(
                webhook_name=wh.name, topic=topic, status_code=200,
                success=True, elapsed_ms=1.0, error="", timestamp=time.time()
            )

        gw._deliver = fake_deliver  # type: ignore[method-assign]
        evt = self._mock_event("sandbox.violation")
        gw.on_event(evt)
        assert fired.wait(timeout=2.0)

    def test_gateway_log_populated_on_deliver_sync(self) -> None:
        gw = MobileAlertGateway(
            webhooks=[WebhookConfig(url="https://a.com/", name="t")],
            async_delivery=False,
        )

        def fake_deliver(wh, topic, payload, timestamp):
            result = DeliveryResult(
                webhook_name=wh.name, topic=topic, status_code=200,
                success=True, elapsed_ms=1.0, error="", timestamp=time.time()
            )
            gw.log.record(result)
            return result

        gw._deliver = fake_deliver  # type: ignore[method-assign]
        gw.deliver_sync("test.event", {})
        assert len(gw.log.recent()) == 1

    def test_multiple_webhooks_all_notified(self) -> None:
        notified = []
        gw = MobileAlertGateway(
            webhooks=[
                WebhookConfig(url="https://a.com/", name="a"),
                WebhookConfig(url="https://b.com/", name="b"),
            ],
            async_delivery=False,
        )

        def fake_deliver(wh, topic, payload, timestamp):
            notified.append(wh.name)
            return DeliveryResult(
                webhook_name=wh.name, topic=topic, status_code=200,
                success=True, elapsed_ms=1.0, error="", timestamp=time.time()
            )

        gw._deliver = fake_deliver  # type: ignore[method-assign]
        gw.deliver_sync("any.event", {})
        assert set(notified) == {"a", "b"}


# ---------------------------------------------------------------------------
# SigningRequest
# ---------------------------------------------------------------------------


class TestSigningRequest:
    def _req(self, expired: bool = False) -> SigningRequest:
        now = time.time()
        return SigningRequest(
            request_id="req-001",
            task_id="task-001",
            challenge_token="abc123",
            created_at=now,
            expires_at=now + (-1 if expired else 300),
            context="Transfer funds",
        )

    def test_not_expired(self) -> None:
        assert not self._req().is_expired

    def test_is_expired(self) -> None:
        assert self._req(expired=True).is_expired

    def test_to_dict(self) -> None:
        d = self._req().to_dict()
        assert d["task_id"] == "task-001"
        assert d["challenge_token"] == "abc123"
        assert "expires_at" in d


# ---------------------------------------------------------------------------
# RemoteDecision
# ---------------------------------------------------------------------------


class TestRemoteDecision:
    def test_to_dict(self) -> None:
        d = RemoteDecision(
            request_id="r-1",
            task_id="t-1",
            approved=True,
            reason="approved",
            decided_at=time.time(),
        ).to_dict()
        assert d["approved"] is True
        assert d["reason"] == "approved"


# ---------------------------------------------------------------------------
# RemoteSigner
# ---------------------------------------------------------------------------


class TestRemoteSigner:
    def test_open_request_creates_request(self) -> None:
        signer = RemoteSigner()
        req = signer.open_request("task-001")
        assert isinstance(req, SigningRequest)
        assert req.task_id == "task-001"

    def test_challenge_token_non_empty(self) -> None:
        signer = RemoteSigner()
        req = signer.open_request("task-001")
        assert len(req.challenge_token) == 64  # SHA-256 hex

    def test_get_request_returns_request(self) -> None:
        signer = RemoteSigner()
        req = signer.open_request("task-001")
        fetched = signer.get_request(req.request_id)
        assert fetched is not None
        assert fetched.request_id == req.request_id

    def test_get_decision_before_submission(self) -> None:
        signer = RemoteSigner()
        req = signer.open_request("task-001")
        assert signer.get_decision(req.request_id) is None

    def test_submit_without_secret_approves(self) -> None:
        signer = RemoteSigner(shared_secret="")
        req = signer.open_request("task-001")
        decision = signer.submit_signature(req.request_id, "", approved=True)
        assert decision.approved

    def test_submit_without_secret_rejects(self) -> None:
        signer = RemoteSigner(shared_secret="")
        req = signer.open_request("task-002")
        decision = signer.submit_signature(req.request_id, "", approved=False)
        assert not decision.approved
        assert decision.reason == "rejected"

    def test_submit_with_valid_signature(self) -> None:
        secret = "mysecret"
        signer = RemoteSigner(shared_secret=secret)
        req = signer.open_request("task-003")
        sig = hmac.new(
            secret.encode(), req.challenge_token.encode(), hashlib.sha256
        ).hexdigest()
        decision = signer.submit_signature(req.request_id, sig, approved=True)
        assert decision.approved

    def test_submit_with_invalid_signature(self) -> None:
        signer = RemoteSigner(shared_secret="mysecret")
        req = signer.open_request("task-004")
        decision = signer.submit_signature(req.request_id, "badsig", approved=True)
        assert not decision.approved
        assert decision.reason == "invalid_signature"

    def test_submit_unknown_request(self) -> None:
        signer = RemoteSigner()
        decision = signer.submit_signature("nonexistent-req-id", "")
        assert not decision.approved
        assert decision.reason == "unknown_request"

    def test_await_decision_immediate(self) -> None:
        signer = RemoteSigner(shared_secret="")
        req = signer.open_request("task-005")
        # Approve in background thread
        def approve():
            time.sleep(0.05)
            signer.submit_signature(req.request_id, "", approved=True)

        t = threading.Thread(target=approve)
        t.start()
        decision = signer.await_decision(req.request_id, timeout_seconds=2.0)
        t.join()
        assert decision.approved

    def test_await_decision_timeout(self) -> None:
        signer = RemoteSigner(default_ttl_seconds=300.0)
        req = signer.open_request("task-timeout")
        decision = signer.await_decision(req.request_id, timeout_seconds=0.2)
        assert not decision.approved
        assert decision.reason == "expired"

    def test_await_decision_expired_request(self) -> None:
        signer = RemoteSigner(default_ttl_seconds=0.01)
        req = signer.open_request("task-exp")
        time.sleep(0.05)  # Let it expire
        decision = signer.await_decision(req.request_id, timeout_seconds=0.5)
        assert not decision.approved
        assert decision.reason == "expired"

    def test_pending_count(self) -> None:
        signer = RemoteSigner()
        signer.open_request("p1")
        signer.open_request("p2")
        assert signer.pending_count() == 2

    def test_pending_count_decreases_after_decision(self) -> None:
        signer = RemoteSigner(shared_secret="")
        req = signer.open_request("p3")
        assert signer.pending_count() == 1
        signer.submit_signature(req.request_id, "")
        assert signer.pending_count() == 0

    def test_context_stored_in_request(self) -> None:
        signer = RemoteSigner()
        req = signer.open_request("task-ctx", context="Approve wire transfer")
        assert "wire transfer" in req.context

    def test_custom_ttl(self) -> None:
        signer = RemoteSigner(default_ttl_seconds=3600.0)
        req = signer.open_request("ttl-task", ttl_seconds=10.0)
        assert req.expires_at < time.time() + 11

    def test_multiple_requests_independent(self) -> None:
        signer = RemoteSigner(shared_secret="")
        r1 = signer.open_request("t1")
        r2 = signer.open_request("t2")
        signer.submit_signature(r1.request_id, "", approved=True)
        # r2 still pending
        assert signer.get_decision(r2.request_id) is None
        d1 = signer.get_decision(r1.request_id)
        assert d1 is not None
        assert d1.approved

    def test_get_decision_after_submit(self) -> None:
        signer = RemoteSigner(shared_secret="")
        req = signer.open_request("t99")
        signer.submit_signature(req.request_id, "", approved=False)
        decision = signer.get_decision(req.request_id)
        assert decision is not None
        assert not decision.approved

    def test_request_ids_are_unique(self) -> None:
        signer = RemoteSigner()
        ids = {signer.open_request(f"task-{i}").request_id for i in range(10)}
        assert len(ids) == 10
