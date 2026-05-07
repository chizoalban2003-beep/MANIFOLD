"""Phase 70: ATSRegistry — in-memory Agent Trust Score registry."""
from __future__ import annotations

from manifold.trust_network.models import (
    AgentTrustScore,
    ToolRegistration,
    TrustSignal,
    TrustTier,
)


class ATSRegistry:
    """In-memory Agent Trust Score registry.

    In production this persists to ManifoldDB.
    All methods are synchronous for simplicity.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolRegistration] = {}
        self._signals: dict[str, list[TrustSignal]] = {}

    def register_tool(self, tool: ToolRegistration) -> None:
        """Register a tool in the ATS network."""
        self._tools[tool.tool_id] = tool
        if tool.tool_id not in self._signals:
            self._signals[tool.tool_id] = []

    def submit_signal(self, signal: TrustSignal) -> None:
        """Submit a trust signal for a tool."""
        if signal.tool_id not in self._signals:
            self._signals[signal.tool_id] = []
        self._signals[signal.tool_id].append(signal)

    def get_score(self, tool_id: str) -> AgentTrustScore:
        """Return the current AgentTrustScore for a tool."""
        signals = self._signals.get(tool_id, [])
        return AgentTrustScore.from_signals(tool_id, signals)

    def get_all_scores(self) -> list[AgentTrustScore]:
        """Return scores for all registered tools."""
        return [self.get_score(tid) for tid in self._tools]

    def get_tools_by_tier(self, tier: TrustTier) -> list[AgentTrustScore]:
        """Return all tools in a given trust tier."""
        return [s for s in self.get_all_scores() if s.tier == tier]

    def leaderboard(self, limit: int = 10) -> list[AgentTrustScore]:
        """Return top tools by score, excluding banned tools with < 5 signals."""
        scores = [
            s
            for s in self.get_all_scores()
            if s.tier not in ("banned",) and s.total_signals >= 5
        ]
        return sorted(scores, key=lambda s: s.score, reverse=True)[:limit]

    def to_dict(self, tool_id: str) -> dict:
        """Return a JSON-serialisable dict for a tool's ATS."""
        ats = self.get_score(tool_id)
        reg = self._tools.get(tool_id)
        return {
            "tool_id": ats.tool_id,
            "score": ats.score,
            "tier": ats.tier,
            "total_signals": ats.total_signals,
            "success_rate": ats.success_rate,
            "adversarial_rate": ats.adversarial_rate,
            "contributing_orgs": ats.contributing_orgs,
            "display_name": reg.display_name if reg else tool_id,
            "domain": reg.domain if reg else "unknown",
        }
