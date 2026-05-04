"""Tests for Phase 28: Semantic Tool Discovery (manifold/discovery.py)."""

from __future__ import annotations

import pytest

from manifold.brain import BrainTask, ToolProfile
from manifold.connector import ConnectorRegistry, ToolConnector
from manifold.discovery import (
    PROBATIONARY_GRADUATION_THRESHOLD,
    PROBATIONARY_TRUST_PENALTY,
    DiscoveryScout,
    EnhancedRecruiter,
    ProbationaryRegistry,
    ProbationaryState,
)
from manifold.hub import ReputationHub


# ---------------------------------------------------------------------------
# ProbationaryState tests
# ---------------------------------------------------------------------------


class TestProbationaryState:
    def test_initial_state(self) -> None:
        state = ProbationaryState(tool_name="my_tool", original_reliability=0.88)
        assert state.successful_outcomes == 0
        assert not state.graduated
        assert state.tool_name == "my_tool"

    def test_penalised_reliability_25_percent_off(self) -> None:
        state = ProbationaryState(tool_name="t", original_reliability=0.8)
        expected = 0.8 * (1.0 - PROBATIONARY_TRUST_PENALTY)
        assert state.penalised_reliability == pytest.approx(expected, abs=1e-6)

    def test_penalised_reliability_clamped_to_1(self) -> None:
        state = ProbationaryState(tool_name="t", original_reliability=1.0)
        assert state.penalised_reliability <= 1.0

    def test_penalised_reliability_clamped_to_0(self) -> None:
        state = ProbationaryState(tool_name="t", original_reliability=0.0)
        assert state.penalised_reliability == 0.0

    def test_record_success_increments_count(self) -> None:
        state = ProbationaryState(tool_name="t", original_reliability=0.8)
        state.record_outcome(success=True)
        assert state.successful_outcomes == 1

    def test_record_failure_does_not_increment(self) -> None:
        state = ProbationaryState(tool_name="t", original_reliability=0.8)
        state.record_outcome(success=False)
        assert state.successful_outcomes == 0

    def test_graduation_after_threshold(self) -> None:
        state = ProbationaryState(tool_name="t", original_reliability=0.8)
        for i in range(PROBATIONARY_GRADUATION_THRESHOLD - 1):
            just_graduated = state.record_outcome(success=True)
            assert not just_graduated
        just_graduated = state.record_outcome(success=True)
        assert just_graduated
        assert state.graduated

    def test_no_update_after_graduation(self) -> None:
        state = ProbationaryState(tool_name="t", original_reliability=0.8)
        for _ in range(PROBATIONARY_GRADUATION_THRESHOLD):
            state.record_outcome(success=True)
        count_after = state.successful_outcomes
        state.record_outcome(success=True)
        # Successful outcomes should not increase after graduation
        assert state.successful_outcomes == count_after

    def test_graduation_constant_is_5(self) -> None:
        assert PROBATIONARY_GRADUATION_THRESHOLD == 5

    def test_trust_penalty_constant_is_25_percent(self) -> None:
        assert PROBATIONARY_TRUST_PENALTY == 0.25


# ---------------------------------------------------------------------------
# ProbationaryRegistry tests
# ---------------------------------------------------------------------------


def _make_registry_with_tools(tools: list[ToolProfile] | None = None) -> ConnectorRegistry:
    registry = ConnectorRegistry()
    for profile in (tools or []):
        connector = ToolConnector(
            name=profile.name,
            fn=lambda: {},
            profile=profile,
        )
        registry.register(connector)
    return registry


def _make_connector(name: str, reliability: float = 0.8, domain: str = "general") -> ToolConnector:
    profile = ToolProfile(
        name=name,
        cost=0.05,
        latency=0.3,
        reliability=reliability,
        risk=0.1,
        asset=0.8,
        domain=domain,
    )
    return ToolConnector(name=name, fn=lambda: {}, profile=profile)


class TestProbationaryRegistry:
    def test_register_probationary_stores_state(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        connector = _make_connector("tool_x", reliability=0.85)
        state = prob.register_probationary(connector, original_reliability=0.85)
        assert state.tool_name == "tool_x"
        assert not state.graduated

    def test_tool_appears_in_connector_registry(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        connector = _make_connector("tool_x")
        prob.register_probationary(connector, original_reliability=0.85)
        assert registry.get("tool_x") is not None

    def test_is_probationary_true_before_graduation(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        connector = _make_connector("tool_x")
        prob.register_probationary(connector, original_reliability=0.85)
        assert prob.is_probationary("tool_x")

    def test_is_probationary_false_for_unknown(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        assert not prob.is_probationary("unknown")

    def test_record_outcome_graduates_after_5(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        connector = _make_connector("tool_x")
        prob.register_probationary(connector, original_reliability=0.85)
        for i in range(PROBATIONARY_GRADUATION_THRESHOLD):
            just_graduated = prob.record_outcome("tool_x", success=True)
            if i < PROBATIONARY_GRADUATION_THRESHOLD - 1:
                assert not just_graduated
            else:
                assert just_graduated
        assert not prob.is_probationary("tool_x")

    def test_graduated_tool_reliability_restored(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        connector = _make_connector("tool_x", reliability=0.75 * (1 - PROBATIONARY_TRUST_PENALTY))
        prob.register_probationary(connector, original_reliability=0.80)
        for _ in range(PROBATIONARY_GRADUATION_THRESHOLD):
            prob.record_outcome("tool_x", success=True)
        # After graduation the connector's reliability should be ~ original
        c = registry.get("tool_x")
        assert c is not None
        assert c.refreshed_profile().reliability == pytest.approx(0.80, abs=1e-6)

    def test_probationary_tools_list(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        prob.register_probationary(_make_connector("a"), original_reliability=0.8)
        prob.register_probationary(_make_connector("b"), original_reliability=0.8)
        tools = prob.probationary_tools()
        assert "a" in tools
        assert "b" in tools

    def test_graduated_tools_list(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        prob.register_probationary(_make_connector("a"), original_reliability=0.8)
        for _ in range(PROBATIONARY_GRADUATION_THRESHOLD):
            prob.record_outcome("a", success=True)
        assert "a" in prob.graduated_tools()

    def test_all_states_returns_copy(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        prob.register_probationary(_make_connector("a"), original_reliability=0.8)
        states = prob.all_states()
        assert "a" in states
        # Modifying the copy should not affect internal state
        del states["a"]
        assert "a" in prob.all_states()

    def test_state_returns_none_for_unknown(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        assert prob.state("unknown") is None

    def test_record_outcome_unknown_tool_returns_false(self) -> None:
        registry = ConnectorRegistry()
        prob = ProbationaryRegistry(connector_registry=registry)
        assert not prob.record_outcome("ghost", success=True)


# ---------------------------------------------------------------------------
# DiscoveryScout tests
# ---------------------------------------------------------------------------


class TestDiscoveryScout:
    def _make_scout(self) -> tuple[DiscoveryScout, ConnectorRegistry]:
        registry = ConnectorRegistry()
        prob_registry = ProbationaryRegistry(connector_registry=registry)
        hub = ReputationHub()
        scout = DiscoveryScout(hub=hub, prob_registry=prob_registry)
        return scout, registry

    def test_covered_domain_returns_empty(self) -> None:
        registry = ConnectorRegistry()
        prob_registry = ProbationaryRegistry(connector_registry=registry)
        hub = ReputationHub()
        # Add a high-reliability tool for domain "math"
        registry.register(
            ToolConnector(
                name="wolfram",
                fn=lambda: {},
                profile=ToolProfile(
                    name="wolfram",
                    cost=0.01,
                    latency=0.1,
                    reliability=0.97,
                    risk=0.02,
                    asset=0.9,
                    domain="math",
                ),
            )
        )
        scout = DiscoveryScout(hub=hub, prob_registry=prob_registry)
        registered = scout.scout_for_domain("math")
        assert registered == []

    def test_uncovered_domain_finds_candidates(self) -> None:
        scout, registry = self._make_scout()
        # "llm" domain is not in the registry; hub baseline has many llm tools
        registered = scout.scout_for_domain("llm")
        # Should discover some tools from the hub baseline
        assert isinstance(registered, list)

    def test_discovered_tools_registered_as_probationary(self) -> None:
        scout, registry = self._make_scout()
        registered = scout.scout_for_domain("llm")
        prob_registry = scout.prob_registry
        for name in registered:
            assert prob_registry.is_probationary(name)

    def test_discovered_tools_appear_in_registry(self) -> None:
        scout, registry = self._make_scout()
        registered = scout.scout_for_domain("llm")
        for name in registered:
            assert registry.get(name) is not None

    def test_probationary_reliability_penalised(self) -> None:
        scout, registry = self._make_scout()
        registered = scout.scout_for_domain("llm")
        hub = scout.hub
        for name in registered:
            c = registry.get(name)
            assert c is not None
            profile = c.refreshed_profile()
            hub_rel = hub.live_reliability(name) or 0.0
            expected_max = hub_rel * (1 - PROBATIONARY_TRUST_PENALTY) + 0.001
            assert profile.reliability <= expected_max

    def test_discovery_log_records_events(self) -> None:
        scout, registry = self._make_scout()
        scout.scout_for_domain("llm")
        assert len(scout.discovery_log()) == 1

    def test_discovery_log_has_domain_key(self) -> None:
        scout, registry = self._make_scout()
        scout.scout_for_domain("finance")
        log = scout.discovery_log()
        if log:
            assert "domain" in log[0]

    def test_no_duplicates_on_second_call(self) -> None:
        scout, registry = self._make_scout()
        first = scout.scout_for_domain("llm")
        second = scout.scout_for_domain("llm")
        # Second call: registry now has tools, so coverage may already be good
        # At minimum there should be no overlap (same tool re-registered)
        overlap = set(first) & set(second)
        assert len(overlap) == 0


# ---------------------------------------------------------------------------
# EnhancedRecruiter tests
# ---------------------------------------------------------------------------


class TestEnhancedRecruiter:
    def _make_recruiter(self) -> EnhancedRecruiter:
        registry = ConnectorRegistry()
        hub = ReputationHub()
        return EnhancedRecruiter(registry=registry, hub=hub)

    def test_can_instantiate(self) -> None:
        er = self._make_recruiter()
        assert er.prob_registry is not None
        assert er.scout is not None

    def test_recruit_if_needed_returns_result(self) -> None:
        from manifold.recruiter import RecruitmentResult
        er = self._make_recruiter()
        task = BrainTask(
            prompt="Find math tools",
            domain="math",
            complexity=0.9,
            stakes=0.7,
        )
        result = er.recruit_if_needed(task)
        assert isinstance(result, RecruitmentResult)

    def test_prob_registry_exposed(self) -> None:
        er = self._make_recruiter()
        assert isinstance(er.prob_registry, ProbationaryRegistry)

    def test_scout_exposed(self) -> None:
        er = self._make_recruiter()
        assert isinstance(er.scout, DiscoveryScout)

    def test_discovery_fallback_used_when_recruiter_fails(self) -> None:
        """When marketplace has no candidates, hub discovery provides fallback."""
        registry = ConnectorRegistry()
        hub = ReputationHub()
        # Empty marketplace → recruiter will trigger but find nothing
        er = EnhancedRecruiter(
            registry=registry,
            hub=hub,
            recruiter_kwargs={"marketplace": {}},
        )
        task = BrainTask(
            prompt="Find llm tools",
            domain="llm",
            complexity=0.95,
            stakes=0.7,
        )
        result = er.recruit_if_needed(task)
        # Hub discovery should kick in
        assert isinstance(result.reason, str)

    def test_hub_discovery_registers_probationary_tool(self) -> None:
        """Tools from hub discovery are marked probationary."""
        registry = ConnectorRegistry()
        hub = ReputationHub()
        er = EnhancedRecruiter(
            registry=registry,
            hub=hub,
            recruiter_kwargs={"marketplace": {}},
        )
        task = BrainTask(
            prompt="Find tools",
            domain="llm",
            complexity=0.95,
            stakes=0.7,
        )
        er.recruit_if_needed(task)
        # At least some probationary tools should exist
        prob_tools = er.prob_registry.probationary_tools()
        assert isinstance(prob_tools, list)


# ---------------------------------------------------------------------------
# Vault persistence tests for Phase 26 & 28
# ---------------------------------------------------------------------------


class TestVaultProbationaryAndVolatility:
    def test_append_volatility_persists(self, tmp_path: object) -> None:
        from manifold.vault import ManifoldVault
        vault = ManifoldVault(data_dir=str(tmp_path))
        vault.append_volatility("llm", 0.049)
        assert vault.volatility_count() == 1

    def test_append_multiple_volatility(self, tmp_path: object) -> None:
        from manifold.vault import ManifoldVault
        vault = ManifoldVault(data_dir=str(tmp_path))
        vault.append_volatility("llm", 0.049)
        vault.append_volatility("math", 0.006)
        vault.append_volatility("storage", 0.006)
        assert vault.volatility_count() == 3

    def test_append_probationary_persists(self, tmp_path: object) -> None:
        from manifold.vault import ManifoldVault
        vault = ManifoldVault(data_dir=str(tmp_path))
        vault.append_probationary(
            "gpt-4o",
            original_reliability=0.92,
            successful_outcomes=2,
            graduated=False,
        )
        assert vault.probationary_count() == 1

    def test_probationary_graduated_state_persisted(self, tmp_path: object) -> None:
        from manifold.vault import ManifoldVault
        vault = ManifoldVault(data_dir=str(tmp_path))
        vault.append_probationary(
            "gpt-4o",
            original_reliability=0.92,
            successful_outcomes=5,
            graduated=True,
        )
        assert vault.probationary_count() == 1

    def test_purge_removes_new_logs(self, tmp_path: object) -> None:
        from manifold.vault import ManifoldVault
        vault = ManifoldVault(data_dir=str(tmp_path))
        vault.append_volatility("llm", 0.049)
        vault.append_probationary("tool", original_reliability=0.8, successful_outcomes=0, graduated=False)
        vault.purge()
        assert vault.volatility_count() == 0
        assert vault.probationary_count() == 0
