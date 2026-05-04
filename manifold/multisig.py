"""Phase 40: Multi-Sig Execution — M-of-N Cryptographic Consensus.

For high-stakes tasks (``BrainTask.stakes > 0.9``), MANIFOLD requires
M-of-N peer endorsements before the task proceeds.

``MultiSigVault`` holds tasks in a ``PENDING_MULTISIG`` state while
:class:`PeerEndorsement` objects arrive from swarm peers (using
Phase 21's :mod:`~manifold.crypto` HMAC-SHA256 infrastructure).  Once
at least *M* valid signatures are collected the task is *approved*.

Key classes
-----------
``MultiSigConfig``
    M-of-N threshold and stakes cutoff configuration.
``PeerEndorsement``
    A signed endorsement from a single swarm peer.
``MultiSigEntry``
    In-flight pending task awaiting endorsements.
``MultiSigResult``
    Outcome of an endorsement attempt or approval check.
``MultiSigVault``
    Thread-safe vault tracking pending high-stakes tasks.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .brain import BrainTask
from .crypto import PolicySigningKey


# ---------------------------------------------------------------------------
# MultiSigConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiSigConfig:
    """Configuration for the Multi-Sig Execution layer.

    Parameters
    ----------
    required_signatures:
        M — number of endorsements required to approve a task.  Default: ``2``.
    total_peers:
        N — expected total peers in the endorsement pool.  Default: ``3``.
    high_stakes_threshold:
        Tasks with ``stakes >= high_stakes_threshold`` enter
        ``PENDING_MULTISIG`` state.  Default: ``0.9``.
    signature_ttl_seconds:
        How long (seconds) a pending task remains valid before it expires.
        Default: ``300`` (5 minutes).
    """

    required_signatures: int = 2
    total_peers: int = 3
    high_stakes_threshold: float = 0.9
    signature_ttl_seconds: float = 300.0


# ---------------------------------------------------------------------------
# PeerEndorsement
# ---------------------------------------------------------------------------


def _task_canonical(task: BrainTask) -> bytes:
    """Return deterministic canonical bytes for a BrainTask."""
    payload: dict[str, Any] = {
        "collaboration_value": task.collaboration_value,
        "complexity": task.complexity,
        "domain": task.domain,
        "dynamic_goal": task.dynamic_goal,
        "prompt": task.prompt,
        "safety_sensitivity": task.safety_sensitivity,
        "source_confidence": task.source_confidence,
        "stakes": task.stakes,
        "time_pressure": task.time_pressure,
        "tool_relevance": task.tool_relevance,
        "uncertainty": task.uncertainty,
        "user_patience": task.user_patience,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class PeerEndorsement:
    """A cryptographic endorsement from a single swarm peer.

    Attributes
    ----------
    peer_org_id:
        Identifier of the endorsing peer.
    task_hash:
        SHA-256 hex digest of the canonical task payload.
    signature:
        HMAC-SHA256 signature over the canonical task payload.
    key_id:
        Key ID used to sign the endorsement.
    timestamp:
        POSIX time when the endorsement was issued.
    """

    peer_org_id: str
    task_hash: str
    signature: str
    key_id: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_org_id": self.peer_org_id,
            "task_hash": self.task_hash,
            "signature": self.signature,
            "key_id": self.key_id,
            "timestamp": self.timestamp,
        }


def sign_endorsement(
    task: BrainTask,
    peer_org_id: str,
    key: PolicySigningKey,
) -> PeerEndorsement:
    """Create a :class:`PeerEndorsement` for *task* using *key*.

    Parameters
    ----------
    task:
        The task to endorse.
    peer_org_id:
        The endorsing peer's org ID.
    key:
        The signing key.

    Returns
    -------
    PeerEndorsement
    """
    payload = _task_canonical(task)
    task_hash = hashlib.sha256(payload).hexdigest()
    sig = key.sign(payload)
    return PeerEndorsement(
        peer_org_id=peer_org_id,
        task_hash=task_hash,
        signature=sig,
        key_id=key.key_id,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# MultiSigEntry — in-flight pending task
# ---------------------------------------------------------------------------

MultiSigStatus = Any  # Literal["pending", "approved", "expired", "rejected"]


@dataclass
class MultiSigEntry:
    """A pending task awaiting M-of-N endorsements.

    Attributes
    ----------
    entry_id:
        Unique identifier for this pending entry (SHA-256 of task hash +
        submission timestamp).
    task:
        The :class:`~manifold.brain.BrainTask` awaiting endorsement.
    task_hash:
        SHA-256 of the canonical task payload.
    submitted_at:
        POSIX timestamp when the task was submitted for multi-sig.
    endorsements:
        List of :class:`PeerEndorsement` objects received so far.
    status:
        One of ``"pending"``, ``"approved"``, ``"expired"``, ``"rejected"``.
    """

    entry_id: str
    task: BrainTask
    task_hash: str
    submitted_at: float
    endorsements: list[PeerEndorsement] = field(default_factory=list)
    status: str = "pending"

    def endorsement_count(self) -> int:
        return len(self.endorsements)

    def endorsing_peers(self) -> set[str]:
        return {e.peer_org_id for e in self.endorsements}

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.time() - self.submitted_at) > ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "task_hash": self.task_hash,
            "submitted_at": self.submitted_at,
            "status": self.status,
            "endorsement_count": self.endorsement_count(),
            "endorsing_peers": sorted(self.endorsing_peers()),
            "endorsements": [e.to_dict() for e in self.endorsements],
        }


# ---------------------------------------------------------------------------
# MultiSigResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiSigResult:
    """Outcome of a multi-sig operation.

    Attributes
    ----------
    accepted:
        ``True`` if the endorsement was accepted (or the task was approved).
    entry_id:
        The affected ``MultiSigEntry`` ID.
    status:
        Updated status of the entry.
    endorsement_count:
        Total valid endorsements collected so far.
    required:
        M — endorsements required.
    approved:
        ``True`` if the task has reached M-of-N approval.
    reason:
        Human-readable description.
    """

    accepted: bool
    entry_id: str
    status: str
    endorsement_count: int
    required: int
    approved: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "entry_id": self.entry_id,
            "status": self.status,
            "endorsement_count": self.endorsement_count,
            "required": self.required,
            "approved": self.approved,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# MultiSigVault
# ---------------------------------------------------------------------------


@dataclass
class MultiSigVault:
    """Thread-safe vault for pending high-stakes tasks.

    Workflow
    --------
    1. The ``ActiveInterceptor`` calls :meth:`submit` when
       ``task.stakes >= config.high_stakes_threshold``.
    2. Swarm peers call :meth:`endorse` to submit their
       :class:`PeerEndorsement`.
    3. Once ``M`` valid endorsements arrive, :meth:`endorse` returns a
       :class:`MultiSigResult` with ``approved=True`` and the entry is
       marked ``"approved"``.
    4. The interceptor can call :meth:`is_approved` before executing.

    Parameters
    ----------
    config:
        Multi-sig threshold configuration.
    trusted_keys:
        Mapping of ``key_id → PolicySigningKey`` for all trusted endorsing
        peers.  Only signatures whose ``key_id`` appears here are accepted.
        When empty all endorsements are rejected on signature verification.

    Example
    -------
    ::

        vault = MultiSigVault()
        entry_id = vault.submit(high_stakes_task)
        key = PolicySigningKey.generate("peer-a")
        endorsement = sign_endorsement(high_stakes_task, "peer-a", key)
        vault.add_trusted_key(key)
        result = vault.endorse(entry_id, endorsement)
        if result.approved:
            print("Task approved — executing!")
    """

    config: MultiSigConfig = field(default_factory=MultiSigConfig)
    trusted_keys: dict[str, PolicySigningKey] = field(default_factory=dict)

    _entries: dict[str, MultiSigEntry] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def add_trusted_key(self, key: PolicySigningKey) -> None:
        """Register a trusted peer signing key.

        Parameters
        ----------
        key:
            The key to trust.
        """
        with self._lock:
            self.trusted_keys[key.key_id] = key

    def requires_multisig(self, task: BrainTask) -> bool:
        """Return ``True`` if *task* exceeds the high-stakes threshold."""
        return task.stakes >= self.config.high_stakes_threshold

    def submit(self, task: BrainTask) -> str:
        """Submit *task* for multi-sig approval.

        Parameters
        ----------
        task:
            The task to pend.

        Returns
        -------
        str
            The ``entry_id`` for subsequent :meth:`endorse` calls.

        Raises
        ------
        ValueError
            If the task does not meet the high-stakes threshold.
        """
        if not self.requires_multisig(task):
            raise ValueError(
                f"Task stakes={task.stakes} is below "
                f"high_stakes_threshold={self.config.high_stakes_threshold}"
            )
        payload = _task_canonical(task)
        task_hash = hashlib.sha256(payload).hexdigest()
        now = time.time()
        entry_id = hashlib.sha256(
            f"{task_hash}:{now}".encode()
        ).hexdigest()[:16]

        entry = MultiSigEntry(
            entry_id=entry_id,
            task=task,
            task_hash=task_hash,
            submitted_at=now,
        )
        with self._lock:
            self._entries[entry_id] = entry
        return entry_id

    def endorse(
        self, entry_id: str, endorsement: PeerEndorsement
    ) -> MultiSigResult:
        """Submit a :class:`PeerEndorsement` for a pending task.

        Parameters
        ----------
        entry_id:
            The pending entry to endorse.
        endorsement:
            The peer's cryptographic endorsement.

        Returns
        -------
        MultiSigResult
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return MultiSigResult(
                    accepted=False,
                    entry_id=entry_id,
                    status="not_found",
                    endorsement_count=0,
                    required=self.config.required_signatures,
                    approved=False,
                    reason=f"entry_id={entry_id!r} not found",
                )

            if entry.status != "pending":
                return MultiSigResult(
                    accepted=False,
                    entry_id=entry_id,
                    status=entry.status,
                    endorsement_count=entry.endorsement_count(),
                    required=self.config.required_signatures,
                    approved=entry.status == "approved",
                    reason=f"entry is already {entry.status!r}",
                )

            # Check TTL
            if entry.is_expired(self.config.signature_ttl_seconds):
                entry.status = "expired"
                return MultiSigResult(
                    accepted=False,
                    entry_id=entry_id,
                    status="expired",
                    endorsement_count=entry.endorsement_count(),
                    required=self.config.required_signatures,
                    approved=False,
                    reason="entry has expired",
                )

            # Reject duplicate peer endorsements
            if endorsement.peer_org_id in entry.endorsing_peers():
                return MultiSigResult(
                    accepted=False,
                    entry_id=entry_id,
                    status=entry.status,
                    endorsement_count=entry.endorsement_count(),
                    required=self.config.required_signatures,
                    approved=False,
                    reason=f"peer {endorsement.peer_org_id!r} already endorsed",
                )

            # Verify task hash
            if endorsement.task_hash != entry.task_hash:
                return MultiSigResult(
                    accepted=False,
                    entry_id=entry_id,
                    status=entry.status,
                    endorsement_count=entry.endorsement_count(),
                    required=self.config.required_signatures,
                    approved=False,
                    reason="task_hash mismatch",
                )

            # Verify cryptographic signature
            key = self.trusted_keys.get(endorsement.key_id)
            if key is None:
                return MultiSigResult(
                    accepted=False,
                    entry_id=entry_id,
                    status=entry.status,
                    endorsement_count=entry.endorsement_count(),
                    required=self.config.required_signatures,
                    approved=False,
                    reason=f"unknown key_id={endorsement.key_id!r}",
                )

            payload = _task_canonical(entry.task)
            if not key.verify(payload, endorsement.signature):
                return MultiSigResult(
                    accepted=False,
                    entry_id=entry_id,
                    status=entry.status,
                    endorsement_count=entry.endorsement_count(),
                    required=self.config.required_signatures,
                    approved=False,
                    reason="signature verification failed",
                )

            # Accept the endorsement
            entry.endorsements.append(endorsement)
            count = entry.endorsement_count()
            approved = count >= self.config.required_signatures

            if approved:
                entry.status = "approved"

            return MultiSigResult(
                accepted=True,
                entry_id=entry_id,
                status=entry.status,
                endorsement_count=count,
                required=self.config.required_signatures,
                approved=approved,
                reason="approved" if approved else f"{count}/{self.config.required_signatures} endorsements",
            )

    def is_approved(self, entry_id: str) -> bool:
        """Return ``True`` if *entry_id* has reached M-of-N approval."""
        with self._lock:
            entry = self._entries.get(entry_id)
            return entry is not None and entry.status == "approved"

    def get_entry(self, entry_id: str) -> MultiSigEntry | None:
        """Return the :class:`MultiSigEntry` for *entry_id*, or ``None``."""
        with self._lock:
            return self._entries.get(entry_id)

    def pending_count(self) -> int:
        """Return the number of entries currently in ``"pending"`` status."""
        with self._lock:
            return sum(1 for e in self._entries.values() if e.status == "pending")

    def summary(self) -> dict[str, Any]:
        """Return a summary of vault activity."""
        with self._lock:
            total = len(self._entries)
            by_status: dict[str, int] = {}
            for entry in self._entries.values():
                by_status[entry.status] = by_status.get(entry.status, 0) + 1
        return {
            "total_entries": total,
            "by_status": by_status,
            "required_signatures": self.config.required_signatures,
            "total_peers": self.config.total_peers,
            "high_stakes_threshold": self.config.high_stakes_threshold,
        }

    def purge_expired(self) -> int:
        """Mark all expired pending entries and return the count."""
        expired = 0
        with self._lock:
            for entry in self._entries.values():
                if (
                    entry.status == "pending"
                    and entry.is_expired(self.config.signature_ttl_seconds)
                ):
                    entry.status = "expired"
                    expired += 1
        return expired
