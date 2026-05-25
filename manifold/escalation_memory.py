"""manifold/escalation_memory.py — EscalationMemory + EscalationRecord.

Learns from human approve/deny/delegate decisions and stops MANIFOLD
asking the same question twice.  After min_decisions consistent decisions
the system auto-decides with a logged notification.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EscalationRecord:
    """One human (or auto) decision on an escalation event."""
    escalation_id: str
    agent_id: str
    action: str
    domain: str
    risk_score: float
    context_hash: str       # hash of (agent_type, domain, action_category)
    human_decision: str     # "approve" | "deny" | "delegate"
    delegated_to: str = ""  # if delegated: who received it
    timestamp: float = field(default_factory=time.time)
    auto_decided: bool = False  # True if MANIFOLD decided without asking


# ---------------------------------------------------------------------------
# EscalationMemory
# ---------------------------------------------------------------------------

class EscalationMemory:
    """Stores and learns from human escalation decisions.

    Parameters
    ----------
    vault_path:
        Optional path to persist records as JSON-lines.  Pass ``None``
        (the default) for in-memory only.
    confidence_threshold:
        Fraction of consistent decisions required to trigger auto-decide.
        Default 0.85.
    min_decisions:
        Minimum number of stored decisions before auto-decide activates.
        Default 3.
    """

    def __init__(
        self,
        vault_path: str | None = None,
        confidence_threshold: float = 0.85,
        min_decisions: int = 3,
    ) -> None:
        self._vault_path = vault_path
        self._threshold = confidence_threshold
        self._min = min_decisions
        # context_hash → list of EscalationRecord
        self._index: dict[str, list[EscalationRecord]] = {}
        self._all: list[EscalationRecord] = []
        if vault_path:
            self._load_from_vault(vault_path)

    # ------------------------------------------------------------------
    # Context hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_category(action: str) -> str:
        """Coarsen action to a broad category for hash grouping."""
        action = action.lower()
        if any(k in action for k in ("delete", "remove", "destroy", "terminate")):
            return "destructive"
        if any(k in action for k in ("send", "email", "notify", "publish", "post")):
            return "communicate"
        if any(k in action for k in ("read", "view", "list", "get", "fetch", "query")):
            return "read"
        if any(k in action for k in ("write", "update", "create", "insert", "add")):
            return "write"
        if any(k in action for k in ("execute", "run", "deploy", "trade", "transact")):
            return "execute"
        return "other"

    @staticmethod
    def _agent_type(agent_id: str) -> str:
        """Coarsen agent_id to a broad type."""
        agent_id = agent_id.lower()
        for marker in ("finance", "legal", "devops", "floor", "aerial", "device", "llm", "framework"):
            if marker in agent_id:
                return marker
        return "generic"

    @classmethod
    def make_context_hash(cls, agent_id: str, domain: str, action: str) -> str:
        """Return a stable short hash for (agent_type, domain, action_category)."""
        parts = f"{cls._agent_type(agent_id)}|{domain.lower()}|{cls._action_category(action)}"
        return hashlib.sha256(parts.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(self, rec: EscalationRecord) -> None:
        """Store a decision record in memory and optionally persist it."""
        self._all.append(rec)
        bucket = self._index.setdefault(rec.context_hash, [])
        bucket.append(rec)
        if self._vault_path:
            self._append_to_vault(rec)

    def should_auto_decide(
        self,
        agent_id: str,
        action: str,
        domain: str,
        risk_score: float,  # noqa: ARG002  (reserved for future risk-banding)
    ) -> tuple[bool, str, float]:
        """Check whether past decisions justify an automatic decision.

        Returns
        -------
        (should_auto, decision, confidence)
            ``should_auto`` is True when enough consistent decisions exist.
            ``decision`` is "approve" or "deny".
            ``confidence`` is the fraction of decisions that agreed.
        """
        context_hash = self.make_context_hash(agent_id, domain, action)
        bucket = self._index.get(context_hash, [])
        if len(bucket) < self._min:
            return False, "", 0.0

        approvals = sum(1 for r in bucket if r.human_decision == "approve")
        total = len(bucket)
        approval_rate = approvals / total

        if approval_rate >= self._threshold:
            return True, "approve", approval_rate
        if approval_rate <= 1.0 - self._threshold:
            return True, "deny", 1.0 - approval_rate
        return False, "", 0.0

    def notify_mode(self, context_hash: str) -> bool:
        """Return True when 2 consistent decisions exist but confidence < threshold.

        In notify mode MANIFOLD decides AND sends a non-blocking notification.
        The CEO is informed but not required to act.
        """
        bucket = self._index.get(context_hash, [])
        if len(bucket) < 2:
            return False
        approvals = sum(1 for r in bucket if r.human_decision == "approve")
        total = len(bucket)
        approval_rate = approvals / total
        # Consistent means all approve or all deny
        consistent = approval_rate >= 1.0 or approval_rate <= 0.0
        return consistent and total < self._min

    def weekly_summary(self) -> dict[str, Any]:
        """Return a summary dict of the last 7 days of escalation activity."""
        cutoff = time.time() - 7 * 24 * 3600
        recent = [r for r in self._all if r.timestamp >= cutoff]
        auto = [r for r in recent if r.auto_decided]
        manual = [r for r in recent if not r.auto_decided]

        # Top promoted patterns
        hash_counts: dict[str, int] = {}
        for r in recent:
            hash_counts[r.context_hash] = hash_counts.get(r.context_hash, 0) + 1
        top_hashes = sorted(hash_counts, key=lambda h: hash_counts[h], reverse=True)[:5]

        # Average confidence across auto-decided contexts
        confidences = []
        for h in self._index:
            can, _, conf = self.should_auto_decide.__func__(  # type: ignore[attr-defined]
                self, "", "", "", 0.0
            ) if False else (None, None, None)
            bucket = self._index[h]
            if len(bucket) >= self._min:
                approvals = sum(1 for r in bucket if r.human_decision == "approve")
                rate = approvals / len(bucket)
                confidences.append(max(rate, 1.0 - rate))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "total_escalations": len(recent),
            "auto_decided": len(auto),
            "manual_decisions": len(manual),
            "top_promoted_rules": top_hashes,
            "decisions_saved": len(auto),
            "avg_confidence": round(avg_confidence, 4),
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _append_to_vault(self, rec: EscalationRecord) -> None:
        try:
            path = Path(self._vault_path)  # type: ignore[arg-type]
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec)) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _load_from_vault(self, vault_path: str) -> None:
        try:
            path = Path(vault_path)
            if not path.exists():
                return
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        rec = EscalationRecord(**d)
                        self._all.append(rec)
                        self._index.setdefault(rec.context_hash, []).append(rec)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
