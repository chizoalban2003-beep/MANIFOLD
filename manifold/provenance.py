"""Phase 29: Decision Provenance — Cryptographic Audit Trails.

Every MANIFOLD decision is recorded as a :class:`DecisionReceipt` — a frozen
snapshot that captures the task context, Braintrust vote breakdown, active
policy hash, and final action.  Receipts are chained into a
:class:`ProvenanceLedger` using SHA-256 so that any tampering with a past
entry is detectable (Merkle-chain property).

Key classes
-----------
``DecisionReceipt``
    Frozen audit record for a single decision.
``ProvenanceLedger``
    Append-only SHA-256 Merkle chain of ``DecisionReceipt`` objects.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# DecisionReceipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionReceipt:
    """Cryptographic audit record for a single MANIFOLD decision.

    Attributes
    ----------
    timestamp:
        POSIX timestamp (seconds since epoch) when the decision was made.
    task_id:
        Unique identifier for the task (typically a deterministic hash of the
        prompt + domain).
    grid_state_summary:
        Lightweight dict summarising the ``BrainDecision`` grid state (action,
        risk_score, confidence).
    braintrust_votes:
        Tuple of per-genome vote dicts when Braintrust consensus was used.
        Empty tuple if single-brain mode.
    policy_hash:
        SHA-256 hex digest of the serialised ``ManifoldPolicy`` active at
        decision time, or ``""`` if no policy was provided.
    final_decision:
        The action string that was ultimately executed (e.g. ``"use_tool"``).
    previous_hash:
        Hash of the immediately preceding receipt in the ledger, forming the
        Merkle chain.  For the first receipt this is the genesis hash.
    """

    timestamp: float
    task_id: str
    grid_state_summary: dict[str, Any]
    braintrust_votes: tuple[dict[str, Any], ...]
    policy_hash: str
    final_decision: str
    previous_hash: str

    @property
    def receipt_hash(self) -> str:
        """SHA-256 hex digest of this receipt's canonical JSON representation."""
        content: dict[str, Any] = {
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "grid_state_summary": self.grid_state_summary,
            "braintrust_votes": list(self.braintrust_votes),
            "policy_hash": self.policy_hash,
            "final_decision": self.final_decision,
            "previous_hash": self.previous_hash,
        }
        raw = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation of this receipt."""
        return {
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "grid_state_summary": self.grid_state_summary,
            "braintrust_votes": list(self.braintrust_votes),
            "policy_hash": self.policy_hash,
            "final_decision": self.final_decision,
            "previous_hash": self.previous_hash,
            "receipt_hash": self.receipt_hash,
        }


# ---------------------------------------------------------------------------
# ProvenanceLedger
# ---------------------------------------------------------------------------

# SHA-256 of a fixed "genesis" sentinel — the chain anchor.
_GENESIS_HASH: str = hashlib.sha256(b"MANIFOLD_PROVENANCE_GENESIS_v1").hexdigest()


@dataclass
class ProvenanceLedger:
    """Append-only Merkle chain of :class:`DecisionReceipt` objects.

    Each new receipt includes the hash of the previous one so that any
    retroactive modification of a past receipt will break all subsequent
    hashes — making tampering detectable via :meth:`verify_chain`.

    Example
    -------
    ::

        ledger = ProvenanceLedger()
        receipt = ledger.record(
            task_id="task-001",
            final_decision="use_tool",
            grid_state_summary={"action": "use_tool", "risk_score": 0.2},
        )
        assert ledger.verify_chain()
        same_receipt = ledger.get("task-001")
    """

    _receipts: list[DecisionReceipt] = field(
        default_factory=list, init=False, repr=False
    )
    _index: dict[str, DecisionReceipt] = field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def record(
        self,
        task_id: str,
        final_decision: str,
        *,
        grid_state_summary: dict[str, Any] | None = None,
        braintrust_votes: tuple[dict[str, Any], ...] = (),
        policy_hash: str = "",
        clock: Any = None,
    ) -> DecisionReceipt:
        """Create and append a new :class:`DecisionReceipt` to the chain.

        Parameters
        ----------
        task_id:
            Unique identifier for this decision (caller-supplied).
        final_decision:
            The action string that was ultimately taken.
        grid_state_summary:
            Dict summarising the brain's decision state.
        braintrust_votes:
            Per-genome vote dicts from Phase 27 consensus (or empty).
        policy_hash:
            SHA-256 digest of the active ``ManifoldPolicy`` (or ``""``).
        clock:
            Optional callable returning current POSIX time.  Defaults to
            ``time.time``.

        Returns
        -------
        DecisionReceipt
            The newly created and appended receipt.
        """
        ts: float = clock() if callable(clock) else time.time()
        receipt = DecisionReceipt(
            timestamp=ts,
            task_id=task_id,
            grid_state_summary=dict(grid_state_summary or {}),
            braintrust_votes=tuple(braintrust_votes),
            policy_hash=policy_hash,
            final_decision=final_decision,
            previous_hash=self._head_hash(),
        )
        self._receipts.append(receipt)
        # Last write wins for duplicate task_ids (re-evaluations)
        self._index[task_id] = receipt
        return receipt

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get(self, task_id: str) -> DecisionReceipt | None:
        """Return the :class:`DecisionReceipt` for *task_id*, or ``None``."""
        return self._index.get(task_id)

    def all_receipts(self) -> list[DecisionReceipt]:
        """Return all receipts in chain order (oldest first)."""
        return list(self._receipts)

    def receipt_count(self) -> int:
        """Return the number of receipts in the ledger."""
        return len(self._receipts)

    def head_hash(self) -> str:
        """Return the hash of the most recent receipt (or the genesis hash)."""
        return self._head_hash()

    # ------------------------------------------------------------------
    # Chain verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire Merkle chain.

        Returns
        -------
        bool
            ``True`` if every receipt's ``previous_hash`` equals the hash of
            its predecessor (or the genesis hash for the first receipt), and
            each stored hash is consistent with the receipt's content.
        """
        prev = _GENESIS_HASH
        for receipt in self._receipts:
            if receipt.previous_hash != prev:
                return False
            prev = receipt.receipt_hash
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _head_hash(self) -> str:
        """Return the hash of the last receipt, or the genesis hash."""
        if not self._receipts:
            return _GENESIS_HASH
        return self._receipts[-1].receipt_hash


# ---------------------------------------------------------------------------
# Utility: build a task_id from a BrainTask
# ---------------------------------------------------------------------------


def make_task_id(prompt: str, domain: str = "general") -> str:
    """Return a deterministic task ID from *prompt* and *domain*.

    Parameters
    ----------
    prompt:
        The task prompt string.
    domain:
        The task domain.

    Returns
    -------
    str
        A 16-character hex prefix of SHA-256(prompt + domain).
    """
    raw = f"{prompt}::{domain}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Utility: hash a ManifoldPolicy
# ---------------------------------------------------------------------------


def hash_policy(policy: object) -> str:
    """Return a SHA-256 hex digest of a ``ManifoldPolicy`` snapshot.

    Serialises via the policy's ``to_dict()`` method if available, otherwise
    falls back to ``str()``.

    Parameters
    ----------
    policy:
        Any object with an optional ``to_dict()`` method.

    Returns
    -------
    str
        64-character hex SHA-256 digest.
    """
    try:
        raw = json.dumps(
            policy.to_dict(),  # type: ignore[attr-defined]
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    except (AttributeError, TypeError):
        raw = str(policy).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
