"""manifold/escalation_memory.py — Human-decision memory for PolicyLearner."""

from __future__ import annotations

import hashlib
import time
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
    timestamp: float = field(default_factory=time.time)
    org_id: str = "default"
    auto_decided: bool = False
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

    def should_auto_decide(
        self,
        agent_id: str,
        action: str,
        domain: str,
        risk_score: float,
    ) -> tuple[bool, str, float]:
        """Return (can_auto_decide, decision, confidence) based on stored history."""
        context_hash = self.make_context_hash(agent_id, domain, action)
        bucket = self._index.get(context_hash, [])
        n = len(bucket)
        if n < self._min:
            return False, "", 0.0
        approvals = sum(1 for r in bucket if r.human_decision == "approve")
        rate = approvals / n
        if rate >= self._confidence:
            return True, "approve", rate
        if (1.0 - rate) >= self._confidence:
            return True, "deny", 1.0 - rate
        return False, "", max(rate, 1.0 - rate)

    def get_history(self, context_hash: str) -> list[EscalationRecord]:
        return list(self._index.get(context_hash, []))

    def clear(self) -> None:
        self._index.clear()

    def weekly_summary(self) -> dict:
        all_records = [r for bucket in self._index.values() for r in bucket]
        total = len(all_records)
        auto = sum(1 for r in all_records if getattr(r, "auto_decided", False))
        return {
            "total_escalations": total,
            "auto_decided": auto,
            "manual_decisions": total - auto,
            "decisions_saved": auto,
        }

    @staticmethod
    def make_context_hash(agent_id: str, domain: str, action: str) -> str:
        """Deterministic hash for (agent_id, domain, action) triples."""
        key = f"{agent_id}|{domain}|{action}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
