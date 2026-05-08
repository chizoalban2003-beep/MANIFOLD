"""ToolCooccurrenceGraph — tracks tool co-usage and propagates trust signals."""
from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean


class ToolCooccurrenceGraph:
    """Tracks which tools are used together and how often they succeed."""

    def __init__(self, min_correlation: float = 0.5) -> None:
        self.min_correlation = min_correlation
        self._cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
        self._tool_counts: dict[str, int] = defaultdict(int)
        self._tool_outcomes: dict[str, list[bool]] = defaultdict(list)

    # ------------------------------------------------------------------
    def record_task(self, tools_used: list[str], success: bool) -> None:
        """Record tool counts, outcomes, and pairwise co-occurrence."""
        for tool in tools_used:
            self._tool_counts[tool] += 1
            self._tool_outcomes[tool].append(success)
        sorted_tools = sorted(set(tools_used))
        for i, a in enumerate(sorted_tools):
            for b in sorted_tools[i + 1:]:
                self._cooccurrence[(a, b)] += 1

    # ------------------------------------------------------------------
    def correlation(self, tool_a: str, tool_b: str) -> float:
        """Jaccard correlation between two tools."""
        a, b = tuple(sorted([tool_a, tool_b]))
        pair_count = self._cooccurrence.get((a, b), 0)
        count_a = self._tool_counts.get(tool_a, 0)
        count_b = self._tool_counts.get(tool_b, 0)
        denom = min(count_a, count_b)
        if denom == 0:
            return 0.0
        return pair_count / denom

    # ------------------------------------------------------------------
    def correlated_partners(
        self, tool: str, min_corr: float | None = None
    ) -> list[tuple[str, float]]:
        """All tools above threshold, sorted by correlation descending."""
        threshold = min_corr if min_corr is not None else self.min_correlation
        all_tools = set(self._tool_counts.keys()) - {tool}
        partners = []
        for other in all_tools:
            c = self.correlation(tool, other)
            if c >= threshold:
                partners.append((other, c))
        partners.sort(key=lambda x: x[1], reverse=True)
        return partners

    # ------------------------------------------------------------------
    def propagate_flag(self, degraded_tool: str) -> list[str]:
        """Return tool names correlated with the degraded tool."""
        return [t for t, _ in self.correlated_partners(degraded_tool)]

    # ------------------------------------------------------------------
    def success_rate(self, tool: str) -> float:
        """Mean success rate. Returns 1.0 for unseen tools (optimistic)."""
        outcomes = self._tool_outcomes.get(tool)
        if not outcomes:
            return 1.0
        return mean(float(o) for o in outcomes)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise state to a JSON file at path."""
        cooc = [[a, b, cnt] for (a, b), cnt in self._cooccurrence.items()]
        tool_outcomes = {k: list(v) for k, v in self._tool_outcomes.items()}
        tool_counts = dict(self._tool_counts)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "cooccurrence": cooc,
                    "tool_outcomes": tool_outcomes,
                    "tool_counts": tool_counts,
                },
                fh,
            )

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "ToolCooccurrenceGraph":
        """Deserialise from a JSON file at path.

        Returns a new instance with restored state.
        Returns a fresh instance if the file does not exist.
        """
        instance = cls()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for entry in data.get("cooccurrence", []):
                a, b, cnt = entry
                instance._cooccurrence[(a, b)] = cnt
            for tool, outcomes in data.get("tool_outcomes", {}).items():
                instance._tool_outcomes[tool] = [bool(o) for o in outcomes]
            for tool, cnt in data.get("tool_counts", {}).items():
                instance._tool_counts[tool] = cnt
        except FileNotFoundError:
            pass
        return instance

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Per-tool success_rate, total_tasks, top 3 partners."""
        result = {}
        for tool in self._tool_counts:
            top3 = self.correlated_partners(tool, min_corr=0.0)[:3]
            result[tool] = {
                "success_rate": self.success_rate(tool),
                "total_tasks": self._tool_counts[tool],
                "top_partners": top3,
            }
        return result
