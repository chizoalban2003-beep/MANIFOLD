"""Phase 58: Remote Sovereign Authorization — Human-in-the-Loop for High-Stakes Decisions.

Provides two components for keeping humans in control of high-stakes MANIFOLD
decisions when they are not sitting at the terminal:

1. **MobileAlertGateway** — hooks into the :class:`~manifold.ipc.EventBus` to
   fire HTTP webhook callbacks (Slack/Discord/Pushover-compatible) whenever a
   high-stakes event arrives.
2. **RemoteSigner** — a secure in-process token store that lets a human inject
   a cryptographic signature to break a :class:`~manifold.consensus.Braintrust`
   deadlock from a remote device.

Architecture
------------
Both components are pure stdlib — they use only :mod:`urllib.request`,
:mod:`hashlib`, :mod:`hmac`, :mod:`threading`, and :mod:`time`.

Webhook delivery
~~~~~~~~~~~~~~~~
The :class:`MobileAlertGateway` subscribes to configurable topics on the
:class:`~manifold.ipc.EventBus`.  When a matching event arrives it serialises
the payload to JSON and fires a ``POST`` request to each registered webhook
URL.  Delivery is **best-effort** and non-blocking (failures are logged to the
:class:`GatewayLog` but do not propagate to the caller).

Remote signing
~~~~~~~~~~~~~~
The :class:`RemoteSigner` works as a secure, short-lived token vault:

1. A *deadlock* event causes the system to call
   :meth:`RemoteSigner.open_request` — this creates a :class:`SigningRequest`
   with a one-time challenge token and a short TTL.
2. A human sends their HMAC-SHA256 signature of the challenge back via the
   :meth:`RemoteSigner.submit_signature` endpoint.
3. Any waiter can call :meth:`RemoteSigner.await_decision` to block until the
   human either approves, rejects, or the request times out.

Key classes
-----------
``WebhookConfig``
    Configuration for one webhook destination.
``DeliveryResult``
    Outcome of one webhook POST attempt.
``GatewayLog``
    In-memory log of all :class:`DeliveryResult` objects.
``MobileAlertGateway``
    EventBus subscriber that fires webhook callbacks.
``SigningRequest``
    A pending remote-authorisation request.
``RemoteDecision``
    Outcome of a remote signing attempt.
``RemoteSigner``
    Secure, TTL-bounded in-process token vault for remote authorisation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# WebhookConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebhookConfig:
    """Configuration for one webhook destination.

    Attributes
    ----------
    url:
        The HTTP(S) endpoint to POST to.
    secret:
        Optional HMAC-SHA256 secret for request signing.  If non-empty an
        ``X-MANIFOLD-Signature`` header is added to every delivery.
    topics:
        Tuple of EventBus topic strings (or ``"*"`` for all) that should
        trigger a delivery to this webhook.
    name:
        Human-readable label for this webhook (used in logs).
    timeout_seconds:
        HTTP request timeout.  Default: ``5.0``.
    """

    url: str
    secret: str = ""
    topics: tuple[str, ...] = ("*",)
    name: str = "webhook"
    timeout_seconds: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (secret redacted)."""
        return {
            "url": self.url,
            "secret": "<redacted>" if self.secret else "",
            "topics": list(self.topics),
            "name": self.name,
            "timeout_seconds": self.timeout_seconds,
        }


# ---------------------------------------------------------------------------
# DeliveryResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeliveryResult:
    """Outcome of one webhook POST attempt.

    Attributes
    ----------
    webhook_name:
        Name of the destination webhook.
    topic:
        EventBus topic that triggered the delivery.
    status_code:
        HTTP response status code, or ``-1`` on network error.
    success:
        ``True`` if status code is 2xx.
    elapsed_ms:
        Round-trip time in milliseconds.
    error:
        Exception message if a network error occurred, otherwise ``""``.
    timestamp:
        POSIX timestamp of the delivery attempt.
    """

    webhook_name: str
    topic: str
    status_code: int
    success: bool
    elapsed_ms: float
    error: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "webhook_name": self.webhook_name,
            "topic": self.topic,
            "status_code": self.status_code,
            "success": self.success,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# GatewayLog
# ---------------------------------------------------------------------------


@dataclass
class GatewayLog:
    """In-memory log of webhook delivery attempts.

    Attributes
    ----------
    max_entries:
        Maximum entries to retain.  Older entries are dropped when the cap is
        reached.  Default: ``1000``.
    """

    max_entries: int = 1_000

    _entries: list[DeliveryResult] = field(
        default_factory=list, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def record(self, result: DeliveryResult) -> None:
        """Append *result* to the log, evicting old entries if needed."""
        with self._lock:
            self._entries.append(result)
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]

    def recent(self, n: int = 20) -> list[DeliveryResult]:
        """Return the *n* most recent entries (newest last)."""
        with self._lock:
            return list(self._entries[-n:])

    def success_rate(self) -> float:
        """Fraction of successful deliveries (0.0–1.0)."""
        with self._lock:
            if not self._entries:
                return 1.0
            return sum(1 for e in self._entries if e.success) / len(self._entries)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        with self._lock:
            total = len(self._entries)
            successes = sum(1 for e in self._entries if e.success)
        return {
            "total_deliveries": total,
            "successful": successes,
            "success_rate": round(self.success_rate(), 4),
        }


# ---------------------------------------------------------------------------
# MobileAlertGateway
# ---------------------------------------------------------------------------


@dataclass
class MobileAlertGateway:
    """EventBus subscriber that fires webhook callbacks on high-stakes events.

    Parameters
    ----------
    webhooks:
        List of :class:`WebhookConfig` objects to notify.
    log:
        Optional :class:`GatewayLog` for recording delivery outcomes.
        A fresh one is created if not provided.
    async_delivery:
        When ``True`` (default), each delivery is made in a daemon thread so
        the EventBus callback returns immediately.

    Example
    -------
    ::

        gw = MobileAlertGateway(
            webhooks=[WebhookConfig(url="https://hooks.slack.com/…", name="slack")],
        )
        # Register with EventBus
        bus.subscribe("*", gw.on_event)
    """

    webhooks: list[WebhookConfig] = field(default_factory=list)
    log: GatewayLog = field(default_factory=GatewayLog)
    async_delivery: bool = True

    def on_event(self, event: Any) -> None:
        """Handle an EventBus :class:`~manifold.ipc.Event`.

        Called by the EventBus whenever a subscribed topic fires.

        Parameters
        ----------
        event:
            An :class:`~manifold.ipc.Event` (duck-typed to avoid a hard
            import cycle).
        """
        topic = getattr(event, "topic", str(event))
        payload = getattr(event, "payload", {})
        timestamp = getattr(event, "timestamp", time.time())

        matching = [
            wh for wh in self.webhooks
            if "*" in wh.topics or topic in wh.topics
            or any(
                t.endswith("*") and topic.startswith(t[:-1])
                for t in wh.topics
            )
        ]

        for wh in matching:
            if self.async_delivery:
                t = threading.Thread(
                    target=self._deliver,
                    args=(wh, topic, payload, timestamp),
                    daemon=True,
                )
                t.start()
            else:
                self._deliver(wh, topic, payload, timestamp)

    def deliver_sync(
        self,
        topic: str,
        payload: dict[str, Any],
        timestamp: float | None = None,
    ) -> list[DeliveryResult]:
        """Deliver a notification synchronously to all matching webhooks.

        Parameters
        ----------
        topic:
            Event topic string.
        payload:
            Event payload dict.
        timestamp:
            POSIX timestamp.  Defaults to ``time.time()``.

        Returns
        -------
        list[DeliveryResult]
            One result per matching webhook.
        """
        ts = timestamp or time.time()
        results = []
        for wh in self.webhooks:
            if (
                "*" in wh.topics
                or topic in wh.topics
                or any(
                    t.endswith("*") and topic.startswith(t[:-1])
                    for t in wh.topics
                )
            ):
                results.append(self._deliver(wh, topic, payload, ts))
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _deliver(
        self,
        wh: WebhookConfig,
        topic: str,
        payload: dict[str, Any],
        timestamp: float,
    ) -> DeliveryResult:
        """Make one HTTP POST to *wh* and record the result."""
        body = json.dumps(
            {"topic": topic, "payload": payload, "timestamp": timestamp},
            default=str,
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MANIFOLD-MobileAlertGateway/58",
        }
        if wh.secret:
            sig = hmac.new(
                wh.secret.encode("utf-8"), body, hashlib.sha256
            ).hexdigest()
            headers["X-MANIFOLD-Signature"] = f"sha256={sig}"

        t0 = time.monotonic()
        try:
            req = urllib.request.Request(
                wh.url, data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=wh.timeout_seconds) as resp:
                status = resp.status
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = DeliveryResult(
                webhook_name=wh.name,
                topic=topic,
                status_code=status,
                success=200 <= status < 300,
                elapsed_ms=elapsed_ms,
                error="",
                timestamp=time.time(),
            )
        except urllib.error.HTTPError as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = DeliveryResult(
                webhook_name=wh.name,
                topic=topic,
                status_code=exc.code,
                success=False,
                elapsed_ms=elapsed_ms,
                error=str(exc),
                timestamp=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result = DeliveryResult(
                webhook_name=wh.name,
                topic=topic,
                status_code=-1,
                success=False,
                elapsed_ms=elapsed_ms,
                error=str(exc),
                timestamp=time.time(),
            )

        self.log.record(result)
        return result


# ---------------------------------------------------------------------------
# SigningRequest
# ---------------------------------------------------------------------------

_REQUEST_COUNTER = 0
_COUNTER_LOCK = threading.Lock()


def _next_request_id() -> str:
    global _REQUEST_COUNTER
    with _COUNTER_LOCK:
        _REQUEST_COUNTER += 1
        return f"req-{_REQUEST_COUNTER:06d}"


@dataclass(frozen=True)
class SigningRequest:
    """A pending remote-authorisation request.

    Attributes
    ----------
    request_id:
        Unique identifier for this request.
    task_id:
        The MANIFOLD task awaiting human authorisation.
    challenge_token:
        Random hex string the human must HMAC-sign with their private key.
    created_at:
        POSIX timestamp when the request was opened.
    expires_at:
        POSIX timestamp after which the request auto-rejects.
    context:
        Human-readable description of what is being authorised.
    """

    request_id: str
    task_id: str
    challenge_token: str
    created_at: float
    expires_at: float
    context: str

    @property
    def is_expired(self) -> bool:
        """``True`` if the request TTL has elapsed."""
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "challenge_token": self.challenge_token,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "context": self.context,
        }


# ---------------------------------------------------------------------------
# RemoteDecision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteDecision:
    """Outcome of a remote signing attempt.

    Attributes
    ----------
    request_id:
        Matches :attr:`SigningRequest.request_id`.
    task_id:
        Matches :attr:`SigningRequest.task_id`.
    approved:
        ``True`` if the human approved (valid signature received in time).
    reason:
        Short explanation: ``"approved"``, ``"rejected"``, ``"expired"``, or
        ``"invalid_signature"``.
    decided_at:
        POSIX timestamp of the decision.
    """

    request_id: str
    task_id: str
    approved: bool
    reason: str
    decided_at: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "approved": self.approved,
            "reason": self.reason,
            "decided_at": self.decided_at,
        }


# ---------------------------------------------------------------------------
# RemoteSigner
# ---------------------------------------------------------------------------


@dataclass
class RemoteSigner:
    """Secure, TTL-bounded in-process vault for remote human authorisation.

    The :class:`RemoteSigner` maintains a registry of open
    :class:`SigningRequest` objects.  It exposes three public operations:

    1. :meth:`open_request` — create a new signing request and return it.
    2. :meth:`submit_signature` — accept a human's HMAC-SHA256 signature.
    3. :meth:`await_decision` — block (with timeout) until decided.

    Parameters
    ----------
    shared_secret:
        HMAC-SHA256 key shared between the MANIFOLD node and the human's
        authorisation device (e.g. mobile app).  If empty, signature
        verification is skipped (useful for testing / open authorisation).
    default_ttl_seconds:
        Default time-to-live for signing requests.  Default: ``300`` (5 min).
    poll_interval_seconds:
        How often :meth:`await_decision` polls for a decision.  Default: 0.1.

    Example
    -------
    ::

        signer = RemoteSigner(shared_secret="mysecret")
        req = signer.open_request("task-001", context="Transfer $5000")
        # Human sends: HMAC-SHA256(secret, challenge_token) + ":approve"
        signer.submit_signature(req.request_id, signature, approved=True)
        decision = signer.await_decision(req.request_id, timeout_seconds=60.0)
        assert decision.approved
    """

    shared_secret: str = ""
    default_ttl_seconds: float = 300.0
    poll_interval_seconds: float = 0.1

    _requests: dict[str, SigningRequest] = field(
        default_factory=dict, init=False, repr=False
    )
    _decisions: dict[str, RemoteDecision] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def open_request(
        self,
        task_id: str,
        *,
        context: str = "",
        ttl_seconds: float | None = None,
    ) -> SigningRequest:
        """Create a new :class:`SigningRequest` and register it.

        Parameters
        ----------
        task_id:
            Identifier of the MANIFOLD task awaiting authorisation.
        context:
            Human-readable description of what is being authorised.
        ttl_seconds:
            Override the default TTL.

        Returns
        -------
        SigningRequest
            The newly created request.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        now = time.time()
        token_raw = f"{task_id}:{now}:{id(self)}".encode("utf-8")
        challenge_token = hashlib.sha256(token_raw).hexdigest()
        req = SigningRequest(
            request_id=_next_request_id(),
            task_id=task_id,
            challenge_token=challenge_token,
            created_at=now,
            expires_at=now + ttl,
            context=context,
        )
        with self._lock:
            self._requests[req.request_id] = req
        return req

    def submit_signature(
        self,
        request_id: str,
        signature: str,
        *,
        approved: bool = True,
    ) -> RemoteDecision:
        """Accept a human's signature and record a :class:`RemoteDecision`.

        If :attr:`shared_secret` is set, *signature* must be the HMAC-SHA256
        hex digest of the request's ``challenge_token`` keyed with the shared
        secret.

        Parameters
        ----------
        request_id:
            Identifies which :class:`SigningRequest` to resolve.
        signature:
            HMAC-SHA256 hex digest of the challenge token (or empty string
            when no ``shared_secret`` is configured).
        approved:
            Whether the human is approving or rejecting.

        Returns
        -------
        RemoteDecision
            The resulting decision.
        """
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                return RemoteDecision(
                    request_id=request_id,
                    task_id="",
                    approved=False,
                    reason="unknown_request",
                    decided_at=time.time(),
                )

        # Check expiry
        if req.is_expired:
            decision = RemoteDecision(
                request_id=request_id,
                task_id=req.task_id,
                approved=False,
                reason="expired",
                decided_at=time.time(),
            )
            with self._lock:
                self._decisions[request_id] = decision
            return decision

        # Verify signature if a shared secret is configured
        if self.shared_secret:
            expected = hmac.new(
                self.shared_secret.encode("utf-8"),
                req.challenge_token.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected, signature):
                decision = RemoteDecision(
                    request_id=request_id,
                    task_id=req.task_id,
                    approved=False,
                    reason="invalid_signature",
                    decided_at=time.time(),
                )
                with self._lock:
                    self._decisions[request_id] = decision
                return decision

        decision = RemoteDecision(
            request_id=request_id,
            task_id=req.task_id,
            approved=approved,
            reason="approved" if approved else "rejected",
            decided_at=time.time(),
        )
        with self._lock:
            self._decisions[request_id] = decision
        return decision

    def await_decision(
        self,
        request_id: str,
        timeout_seconds: float = 60.0,
    ) -> RemoteDecision:
        """Block until a decision is recorded or the timeout elapses.

        Parameters
        ----------
        request_id:
            Which request to wait for.
        timeout_seconds:
            Maximum seconds to wait before returning an auto-expired decision.

        Returns
        -------
        RemoteDecision
            The human's decision, or a timeout/expired decision.
        """
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            with self._lock:
                decision = self._decisions.get(request_id)
                if decision is not None:
                    return decision
                req = self._requests.get(request_id)
                if req is not None and req.is_expired:
                    exp_decision = RemoteDecision(
                        request_id=request_id,
                        task_id=req.task_id,
                        approved=False,
                        reason="expired",
                        decided_at=time.time(),
                    )
                    self._decisions[request_id] = exp_decision
                    return exp_decision
            time.sleep(self.poll_interval_seconds)

        # Timed out
        with self._lock:
            req = self._requests.get(request_id)
            task_id = req.task_id if req else ""
            decision = RemoteDecision(
                request_id=request_id,
                task_id=task_id,
                approved=False,
                reason="expired",
                decided_at=time.time(),
            )
            self._decisions[request_id] = decision
        return decision

    def get_request(self, request_id: str) -> SigningRequest | None:
        """Return the :class:`SigningRequest` for *request_id*, or ``None``."""
        with self._lock:
            return self._requests.get(request_id)

    def get_decision(self, request_id: str) -> RemoteDecision | None:
        """Return the :class:`RemoteDecision` for *request_id*, or ``None``."""
        with self._lock:
            return self._decisions.get(request_id)

    def pending_count(self) -> int:
        """Number of open requests awaiting a decision."""
        with self._lock:
            decided = set(self._decisions.keys())
            return sum(
                1 for rid in self._requests if rid not in decided
            )
