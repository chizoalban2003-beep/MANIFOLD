"""manifold/world_model.py — Vault-learned transition model for MPC world simulator.

Replaces the hardcoded 20% hazard heuristic in WorldModelSimulator with a
data-driven model learned from vault outcome records.

Research: VaultTransitionModel learns p_obstacle per (domain, r_bucket) from
observed vault events.  Falls back to 0.20 (old heuristic) when data is
insufficient (< min_samples events).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict


class VaultTransitionModel:
    """Learns obstacle transition probabilities from vault outcome records.

    Parameters
    ----------
    vault_path:
        Path to the directory containing the vault ``*.jsonl`` WAL files.
        If ``None``, defaults to ``manifold_data/`` in the current directory.
    """

    FALLBACK_P = 0.20  # original heuristic — used when data is insufficient
    OBSTACLE_OUTCOMES = frozenset({"blocked", "escalated", "obstacle"})

    def __init__(self, vault_path: str | None = None) -> None:
        self._vault_path = vault_path or os.path.join(os.getcwd(), "manifold_data")
        # {domain: {r_bucket: float}}
        self._table: dict[str, dict[float, float]] = {}
        self._status: str = "not_learned"
        self._total_events: int = 0
        self._domains_learned: int = 0

    def learn(self, min_samples: int = 50) -> dict:
        """Read vault records and build the transition table.

        Groups by ``(domain, r_bucket)`` where ``r_bucket = round(risk_score, 1)``.
        Computes ``p_obstacle = count(outcome == "blocked" | "escalated") / total``
        for each group.

        Returns
        -------
        dict with keys:
            ``domains_learned``, ``total_events``, ``status``
        """
        # Collect events from all *.jsonl WAL files in vault_path
        events: list[dict] = []
        if os.path.isdir(self._vault_path):
            for fname in os.listdir(self._vault_path):
                if not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(self._vault_path, fname)
                try:
                    with open(fpath, encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            # Only care about records that have outcome + risk_score
                            if "risk_score" in rec and "outcome" in rec:
                                events.append(rec)
                except OSError:
                    continue

        self._total_events = len(events)

        if self._total_events < min_samples:
            self._status = "insufficient_data"
            self._table = {}
            return {
                "domains_learned": 0,
                "total_events": self._total_events,
                "status": self._status,
            }

        # Group by (domain, r_bucket)
        groups: dict[str, dict[float, list[str]]] = defaultdict(lambda: defaultdict(list))
        for rec in events:
            domain = rec.get("domain", "general")
            r_val = float(rec.get("risk_score", 0.5))
            r_bucket = round(r_val, 1)
            outcome = str(rec.get("outcome", ""))
            groups[domain][r_bucket].append(outcome)

        # Compute p_obstacle per group
        table: dict[str, dict[float, float]] = {}
        for domain, bucket_map in groups.items():
            table[domain] = {}
            for r_bucket, outcomes in bucket_map.items():
                n = len(outcomes)
                blocked = sum(
                    1 for o in outcomes if o in self.OBSTACLE_OUTCOMES
                )
                table[domain][r_bucket] = blocked / n if n > 0 else self.FALLBACK_P

        self._table = table
        self._domains_learned = len(table)
        self._status = "learned"

        return {
            "domains_learned": self._domains_learned,
            "total_events": self._total_events,
            "status": self._status,
        }

    def predict(self, domain: str, r_value: float) -> float:
        """Return predicted obstacle probability for *(domain, r_value)*.

        Fallback chain: exact domain → "general" → FALLBACK_P (0.20).

        Parameters
        ----------
        domain:
            Agent or task domain (e.g. "physical", "finance", "general").
        r_value:
            Current cell risk score [0, 1].
        """
        r_bucket = round(float(r_value), 1)
        if domain in self._table:
            if r_bucket in self._table[domain]:
                return self._table[domain][r_bucket]
        # Fallback to "general"
        if "general" in self._table:
            if r_bucket in self._table["general"]:
                return self._table["general"][r_bucket]
        return self.FALLBACK_P

    @property
    def status(self) -> str:
        """Return the learning status: ``"not_learned"``, ``"insufficient_data"``, or ``"learned"``."""
        return self._status

    def stats(self) -> dict:
        """Return summary statistics about the learned model."""
        return {
            "status": self._status,
            "total_events": self._total_events,
            "domains_learned": self._domains_learned,
            "fallback_p": self.FALLBACK_P,
            "vault_path": self._vault_path,
        }
