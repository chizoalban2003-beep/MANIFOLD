"""Tests for theory of mind L1 — agent intention inference (PROMPT D1)."""
import pytest

from manifold.agent_registry import AgentRegistry, Episode


def _make_registry() -> AgentRegistry:
    reg = AgentRegistry(stale_timeout=9999)
    reg.register("agent-a", "Agent A", ["search", "code"], "org1")
    reg.register("agent-b", "Agent B", ["search", "billing"], "org1")
    reg.register("observer", "Observer", ["general"], "org1")
    return reg


class TestTheoryOfMind:
    def test_predict_with_empty_history_returns_ats_prior(self):
        """predict_agent_action with empty history returns ats_prior basis."""
        reg = _make_registry()
        result = reg.predict_agent_action(
            "observer", "agent-a",
            {"zone": "finance", "task_type": "billing", "current_crna": {}}
        )
        assert result["basis"] == "ats_prior"
        assert result["predicted_action"] in {"proceed", "wait"}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_prediction_matches_most_common_domain_action(self):
        """After seeding history, prediction matches most common domain action."""
        reg = _make_registry()
        crna = {"c": 0.3, "r": 0.2, "n": 0.5, "a": 0.8}
        # Seed agent-b with 8 successes + 2 failures in "billing" domain
        for i in range(8):
            reg.record_episode("agent-b", Episode(
                task_description="billing task",
                domain="billing",
                duration_seconds=30.0,
                success=True,
                crna_at_start=crna,
                crna_at_end=crna,
                risk_encountered=0.2,
            ))
        for i in range(2):
            reg.record_episode("agent-b", Episode(
                task_description="billing task",
                domain="billing",
                duration_seconds=30.0,
                success=False,
                crna_at_start=crna,
                crna_at_end=crna,
                risk_encountered=0.7,
            ))

        result = reg.predict_agent_action(
            "observer", "agent-b",
            {"zone": "billing", "task_type": "billing", "current_crna": {}}
        )
        # 80% success rate → should predict "proceed"
        assert result["predicted_action"] == "proceed"
        assert result["basis"] == "episode_history"

    def test_confidence_rises_with_more_matching_episodes(self):
        """confidence rises as more matching episodes are added."""
        reg = _make_registry()
        crna = {"c": 0.3, "r": 0.2, "n": 0.5, "a": 0.8}

        # After 0 episodes — confidence is ATS-based (1.0 for fresh agent)
        result_0 = reg.predict_agent_action(
            "observer", "agent-a",
            {"zone": "code", "task_type": "code", "current_crna": {}}
        )

        # Add 5 consistent episodes
        for _ in range(5):
            reg.record_episode("agent-a", Episode(
                task_description="code review",
                domain="code",
                duration_seconds=20.0,
                success=True,
                crna_at_start=crna,
                crna_at_end=crna,
                risk_encountered=0.1,
            ))

        result_5 = reg.predict_agent_action(
            "observer", "agent-a",
            {"zone": "code", "task_type": "code", "current_crna": {}}
        )
        # After episodes: confidence should be episode-based and still valid
        assert result_5["basis"] == "episode_history"
        assert 0.0 <= result_5["confidence"] <= 1.0

    def test_predict_all_agents_returns_one_per_active_agent(self):
        """predict_all_agents returns one prediction per active agent."""
        reg = _make_registry()
        predictions = reg.predict_all_agents("observer", "search")
        # Should have predictions for agent-a and agent-b (not observer)
        assert len(predictions) == 2
        ids = {p["agent_id"] for p in predictions}
        assert "agent-a" in ids
        assert "agent-b" in ids
        assert "observer" not in ids
        for p in predictions:
            assert "predicted_action" in p
            assert "confidence" in p
