"""MemoryConsolidator — promotes recurring high-confidence outcome patterns."""
from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass


MIN_PATTERN_COUNT: int = 5
MIN_CONFIDENCE: float = 0.75


@dataclass
class ConsolidatedRule:
    domain: str
    action: str
    stakes_min: float
    confidence: float
    sample_count: int


class MemoryConsolidator:
    """Groups outcome logs and promotes stable patterns into rules."""

    def __init__(self) -> None:
        self._promoted_rules: list[ConsolidatedRule] = []
        # Track which (domain, action, bucket) have already been promoted
        self._promoted_keys: set[tuple] = set()

    # ------------------------------------------------------------------
    @staticmethod
    def _bucket_stakes(stakes: float) -> float:
        """Round to nearest 0.25."""
        return round(round(stakes / 0.25) * 0.25, 10)

    # ------------------------------------------------------------------
    def consolidate(self, outcome_log: list[dict]) -> list[ConsolidatedRule]:
        """Promote patterns from outcome_log. Returns newly promoted rules."""
        buckets: dict[tuple, list[bool]] = defaultdict(list)
        for entry in outcome_log:
            key = (entry["domain"], entry["action"], self._bucket_stakes(entry["stakes"]))
            buckets[key].append(entry["success"])

        newly_promoted: list[ConsolidatedRule] = []
        for (domain, action, bucket), outcomes in buckets.items():
            if len(outcomes) < MIN_PATTERN_COUNT:
                continue
            confidence = sum(outcomes) / len(outcomes)
            if confidence < MIN_CONFIDENCE:
                continue
            pkey = (domain, action, bucket)
            if pkey in self._promoted_keys:
                continue
            rule = ConsolidatedRule(
                domain=domain,
                action=action,
                stakes_min=bucket,
                confidence=round(confidence, 4),
                sample_count=len(outcomes),
            )
            self._promoted_rules.append(rule)
            self._promoted_keys.add(pkey)
            newly_promoted.append(rule)
        return newly_promoted

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise state to a JSON file at path."""
        rules = [dataclasses.asdict(r) for r in self._promoted_rules]
        keys = [list(k) for k in self._promoted_keys]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"promoted_rules": rules, "promoted_keys": keys}, fh)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "MemoryConsolidator":
        """Deserialise from a JSON file at path.

        Returns a new instance with restored state.
        Returns a fresh instance if the file does not exist.
        """
        instance = cls()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for rule_dict in data.get("promoted_rules", []):
                instance._promoted_rules.append(ConsolidatedRule(**rule_dict))
            for key in data.get("promoted_keys", []):
                instance._promoted_keys.add(tuple(key))
        except FileNotFoundError:
            pass
        return instance

    # ------------------------------------------------------------------
    def promoted_rules(self) -> list[ConsolidatedRule]:
        return list(self._promoted_rules)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        if not self._promoted_rules:
            return "No promoted rules yet."
        lines = ["Promoted rules:"]
        for r in self._promoted_rules:
            lines.append(
                f"  [{r.domain}] action={r.action} stakes>={r.stakes_min} "
                f"conf={r.confidence:.2f} n={r.sample_count}"
            )
        return "\n".join(lines)
