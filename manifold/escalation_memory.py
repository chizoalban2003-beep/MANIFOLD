"""manifold/escalation_memory.py — Human-decision memory for PolicyLearner."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EscalationRecord:
    """A single human decision on an escalated action."""

    escalation_id: str
    agent_id: str
    action: str
    domain: str
    risk_score: float
    context_hash: str
    human_decision: str  # "approve" or "deny"
    org_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


class EscalationMemory:
    """Stores human decisions and detects promotable patterns.

    Parameters
    ----------
    confidence_threshold:
        Minimum approval (or denial) rate to consider a pattern
        promotable by PolicyLearner.
    min_decisions:
        Global minimum number of decisions before any pattern can
        be considered for promotion.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        min_decisions: int = 10,
    ) -> None:
        self._confidence = confidence_threshold
        self._min = min_decisions
        self._index: dict[str, list[EscalationRecord]] = {}

    def record(self, rec: EscalationRecord) -> None:
        """Store a human decision record."""
        self._index.setdefault(rec.context_hash, []).append(rec)

    def should_auto_decide(self, context_hash: str) -> tuple[bool, str]:
        """Return (can_auto_decide, decision) based on stored history."""
        bucket = self._index.get(context_hash, [])
        n = len(bucket)
        if n < self._min:
            return False, ""
        approvals = sum(1 for r in bucket if r.human_decision == "approve")
        rate = approvals / n
        if rate >= self._confidence:
            return True, "approve"
        if (1.0 - rate) >= self._confidence:
            return True, "deny"
        return False, ""

    def get_history(self, context_hash: str) -> list[EscalationRecord]:
        return list(self._index.get(context_hash, []))

    def clear(self) -> None:
        self._index.clear()

    @staticmethod
    def make_context_hash(agent_id: str, domain: str, action: str) -> str:
        """Deterministic hash for (agent_id, domain, action) triples."""
        key = f"{agent_id}|{domain}|{action}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
