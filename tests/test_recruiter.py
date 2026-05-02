"""Tests for Phase 17: Sovereign Recruiter."""

from __future__ import annotations

import pytest

from manifold import (
    BrainTask,
    ConnectorRegistry,
    ToolConnector,
    ToolProfile,
)
from manifold.recruiter import (
    MarketplaceListing,
    RecruitmentResult,
    SovereignRecruiter,
    _DEFAULT_MARKETPLACE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry_with_low_reliability(domain: str = "finance") -> ConnectorRegistry:
    """Registry where all tools have reliability < 0.6."""
    registry = ConnectorRegistry()
    profile = ToolProfile(
        name="flaky_tool",
        cost=0.05,
        latency=0.3,
        reliability=0.40,
        risk=0.30,
        asset=0.60,
        domain=domain,
    )
    registry.register(ToolConnector(name="flaky_tool", fn=lambda: None, profile=profile))
    return registry


def _registry_healthy(domain: str = "finance") -> ConnectorRegistry:
    """Registry with a reliable tool."""
    registry = ConnectorRegistry()
    profile = ToolProfile(
        name="good_tool",
        cost=0.05,
        latency=0.2,
        reliability=0.90,
        risk=0.10,
        asset=0.80,
        domain=domain,
    )
    registry.register(ToolConnector(name="good_tool", fn=lambda: None, profile=profile))
    return registry


def _empty_registry() -> ConnectorRegistry:
    return ConnectorRegistry()


def _high_complexity_task(domain: str = "finance") -> BrainTask:
    return BrainTask(prompt="complex task", domain=domain, complexity=0.95, stakes=0.8)


def _low_complexity_task(domain: str = "finance") -> BrainTask:
    return BrainTask(prompt="simple task", domain=domain, complexity=0.3, stakes=0.3)


# ---------------------------------------------------------------------------
# MarketplaceListing tests
# ---------------------------------------------------------------------------


class TestMarketplaceListing:
    def test_to_tool_profile_probationary(self):
        listing = MarketplaceListing(
            name="test-tool",
            domain="finance",
            description="A test tool",
            estimated_reliability=0.90,
            estimated_risk=0.10,
            estimated_cost=0.05,
            estimated_latency=0.5,
            estimated_asset=0.80,
        )
        profile = listing.to_tool_profile(probationary=True)
        # Probationary discount: reliability * 0.75
        assert profile.reliability < 0.90
        assert profile.reliability == pytest.approx(0.90 * 0.75, rel=0.01)
        assert profile.name == "test-tool"
        assert profile.domain == "finance"

    def test_to_tool_profile_non_probationary(self):
        listing = MarketplaceListing(
            name="test-tool",
            domain="finance",
            description="A test tool",
            estimated_reliability=0.85,
            estimated_risk=0.10,
            estimated_cost=0.05,
            estimated_latency=0.5,
            estimated_asset=0.80,
        )
        profile = listing.to_tool_profile(probationary=False)
        assert profile.reliability == pytest.approx(0.85, rel=0.01)

    def test_to_tool_profile_clamps_at_one(self):
        listing = MarketplaceListing(
            name="perfect-tool",
            domain="general",
            description="",
            estimated_reliability=1.0,
            estimated_risk=0.0,
            estimated_cost=0.0,
            estimated_latency=0.0,
            estimated_asset=1.0,
        )
        profile = listing.to_tool_profile(probationary=False)
        assert profile.reliability <= 1.0
        assert profile.risk >= 0.0


# ---------------------------------------------------------------------------
# SovereignRecruiter.search_marketplace tests
# ---------------------------------------------------------------------------


class TestSearchMarketplace:
    def test_returns_domain_matches(self):
        recruiter = SovereignRecruiter(registry=_empty_registry())
        listings = recruiter.search_marketplace("finance")
        names = {l.name for l in listings}
        assert "finance-data-api" in names

    def test_returns_math_domain(self):
        recruiter = SovereignRecruiter(registry=_empty_registry())
        listings = recruiter.search_marketplace("math")
        names = {l.name for l in listings}
        assert "wolfram-alpha" in names

    def test_excludes_already_registered_tools(self):
        registry = ConnectorRegistry()
        profile = ToolProfile("wolfram-alpha", 0.01, 0.5, 0.97, 0.02, 0.90, "math")
        registry.register(ToolConnector("wolfram-alpha", fn=lambda: None, profile=profile))
        recruiter = SovereignRecruiter(registry=registry)
        listings = recruiter.search_marketplace("math")
        names = {l.name for l in listings}
        assert "wolfram-alpha" not in names

    def test_sorted_by_utility_descending(self):
        recruiter = SovereignRecruiter(registry=_empty_registry())
        listings = recruiter.search_marketplace("language")
        if len(listings) >= 2:
            # Utility = asset * reliability - cost - latency - risk
            def utility(l: MarketplaceListing) -> float:
                return (
                    l.estimated_asset * l.estimated_reliability
                    - l.estimated_cost
                    - l.estimated_latency
                    - l.estimated_risk
                )
            utils = [utility(l) for l in listings]
            assert utils == sorted(utils, reverse=True)

    def test_empty_registry_returns_candidates(self):
        recruiter = SovereignRecruiter(registry=_empty_registry())
        listings = recruiter.search_marketplace("search")
        assert len(listings) > 0

    def test_custom_marketplace(self):
        custom = {
            "my-custom-tool": {
                "domain": "custom",
                "reliability": 0.88,
                "risk": 0.05,
                "cost": 0.02,
                "latency": 0.2,
                "asset": 0.80,
                "description": "Custom tool",
            }
        }
        recruiter = SovereignRecruiter(registry=_empty_registry(), marketplace=custom)
        listings = recruiter.search_marketplace("custom")
        assert len(listings) == 1
        assert listings[0].name == "my-custom-tool"


# ---------------------------------------------------------------------------
# SovereignRecruiter.run_scout_tasks tests
# ---------------------------------------------------------------------------


class TestRunScoutTasks:
    def test_all_pass_high_reliability(self):
        listing = MarketplaceListing(
            name="perfect", domain="test", description="",
            estimated_reliability=1.0, estimated_risk=0.0,
            estimated_cost=0.0, estimated_latency=0.0, estimated_asset=1.0,
        )
        recruiter = SovereignRecruiter(registry=_empty_registry(), seed=0)
        rate = recruiter.run_scout_tasks(listing)
        assert rate == 1.0

    def test_zero_scout_tasks_returns_one(self):
        listing = MarketplaceListing(
            name="t", domain="test", description="",
            estimated_reliability=0.5, estimated_risk=0.1,
            estimated_cost=0.1, estimated_latency=0.1, estimated_asset=0.7,
        )
        recruiter = SovereignRecruiter(
            registry=_empty_registry(), n_scout_tasks=0, seed=0
        )
        assert recruiter.run_scout_tasks(listing) == 1.0

    def test_custom_probe_fn(self):
        listing = MarketplaceListing(
            name="custom", domain="test", description="",
            estimated_reliability=0.5, estimated_risk=0.1,
            estimated_cost=0.1, estimated_latency=0.1, estimated_asset=0.7,
        )
        recruiter = SovereignRecruiter(
            registry=_empty_registry(),
            n_scout_tasks=4,
            probe_fn=lambda name, domain: True,  # always succeeds
        )
        assert recruiter.run_scout_tasks(listing) == 1.0

    def test_custom_probe_fn_always_fails(self):
        listing = MarketplaceListing(
            name="broken", domain="test", description="",
            estimated_reliability=0.9, estimated_risk=0.1,
            estimated_cost=0.1, estimated_latency=0.1, estimated_asset=0.7,
        )
        recruiter = SovereignRecruiter(
            registry=_empty_registry(),
            n_scout_tasks=4,
            probe_fn=lambda name, domain: False,
        )
        assert recruiter.run_scout_tasks(listing) == 0.0

    def test_rate_between_zero_and_one(self):
        listing = MarketplaceListing(
            name="t", domain="test", description="",
            estimated_reliability=0.7, estimated_risk=0.1,
            estimated_cost=0.1, estimated_latency=0.1, estimated_asset=0.7,
        )
        recruiter = SovereignRecruiter(
            registry=_empty_registry(), n_scout_tasks=10, seed=99
        )
        rate = recruiter.run_scout_tasks(listing)
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# SovereignRecruiter.recruit_if_needed tests
# ---------------------------------------------------------------------------


class TestRecruitIfNeeded:
    def test_not_triggered_low_complexity(self):
        recruiter = SovereignRecruiter(registry=_registry_with_low_reliability())
        task = _low_complexity_task()
        result = recruiter.recruit_if_needed(task)
        assert not result.triggered
        assert result.registered is False

    def test_not_triggered_healthy_registry(self):
        recruiter = SovereignRecruiter(registry=_registry_healthy("finance"))
        task = _high_complexity_task("finance")
        result = recruiter.recruit_if_needed(task)
        assert not result.triggered

    def test_triggered_on_high_complexity_low_reliability(self):
        recruiter = SovereignRecruiter(
            registry=_registry_with_low_reliability("finance"),
            probe_fn=lambda n, d: True,  # scouts always pass
            seed=42,
        )
        task = _high_complexity_task("finance")
        result = recruiter.recruit_if_needed(task)
        assert result.triggered

    def test_registers_new_tool_when_triggered(self):
        recruiter = SovereignRecruiter(
            registry=_registry_with_low_reliability("finance"),
            probe_fn=lambda n, d: True,
            seed=42,
        )
        task = _high_complexity_task("finance")
        result = recruiter.recruit_if_needed(task)
        if result.triggered:
            # May or may not register depending on candidates found
            if result.registered:
                assert result.registered_tool_name is not None
                # Tool should now be in registry
                assert recruiter.registry.get(result.registered_tool_name) is not None

    def test_empty_registry_triggers_recruitment(self):
        recruiter = SovereignRecruiter(
            registry=_empty_registry(),
            probe_fn=lambda n, d: True,
            seed=0,
        )
        task = _high_complexity_task("math")
        result = recruiter.recruit_if_needed(task)
        assert result.triggered

    def test_result_has_correct_attributes(self):
        recruiter = SovereignRecruiter(
            registry=_registry_with_low_reliability("code"),
            probe_fn=lambda n, d: True,
            seed=1,
        )
        task = BrainTask(prompt="write compiler", domain="code", complexity=0.9)
        result = recruiter.recruit_if_needed(task)
        assert isinstance(result, RecruitmentResult)
        assert isinstance(result.triggered, bool)
        assert isinstance(result.candidates_found, int)
        assert 0.0 <= result.scout_pass_rate <= 1.0

    def test_scouts_fail_means_not_registered(self):
        recruiter = SovereignRecruiter(
            registry=_registry_with_low_reliability("finance"),
            probe_fn=lambda n, d: False,  # scouts always fail
            min_scout_pass_rate=0.5,
            seed=42,
        )
        task = _high_complexity_task("finance")
        result = recruiter.recruit_if_needed(task)
        if result.triggered:
            assert not result.registered

    def test_candidates_found_is_nonnegative(self):
        recruiter = SovereignRecruiter(registry=_empty_registry(), seed=0)
        task = _high_complexity_task()
        result = recruiter.recruit_if_needed(task)
        assert result.candidates_found >= 0


# ---------------------------------------------------------------------------
# SovereignRecruiter log and summary tests
# ---------------------------------------------------------------------------


class TestRecruiterSummary:
    def test_initial_summary(self):
        recruiter = SovereignRecruiter(registry=_empty_registry())
        s = recruiter.summary()
        assert s["total_checks"] == 0
        assert s["triggers"] == 0
        assert s["registered"] == 0

    def test_log_grows_with_calls(self):
        recruiter = SovereignRecruiter(registry=_empty_registry(), seed=0)
        t = BrainTask(prompt="x", domain="math", complexity=0.9)
        recruiter.recruit_if_needed(t)
        recruiter.recruit_if_needed(t)
        assert len(recruiter.recruitment_log()) == 2

    def test_hired_count_correct(self):
        recruiter = SovereignRecruiter(
            registry=_registry_with_low_reliability("math"),
            probe_fn=lambda n, d: True,
            seed=0,
        )
        task = BrainTask(prompt="prove theorem", domain="math", complexity=0.95)
        result = recruiter.recruit_if_needed(task)
        expected = 1 if result.registered else 0
        assert recruiter.hired_count() == expected

    def test_summary_hire_rate_zero_if_no_triggers(self):
        recruiter = SovereignRecruiter(registry=_registry_healthy(), seed=0)
        task = _high_complexity_task()
        recruiter.recruit_if_needed(task)
        s = recruiter.summary()
        # Not triggered → hire_rate should be 0.0
        assert s["hire_rate"] == 0.0

    def test_summary_contains_expected_keys(self):
        recruiter = SovereignRecruiter(registry=_empty_registry(), seed=0)
        recruiter.recruit_if_needed(_high_complexity_task("search"))
        s = recruiter.summary()
        for key in ("total_checks", "triggers", "registered", "not_triggered", "hire_rate"):
            assert key in s


# ---------------------------------------------------------------------------
# Default marketplace catalog sanity tests
# ---------------------------------------------------------------------------


class TestDefaultMarketplace:
    def test_catalog_not_empty(self):
        assert len(_DEFAULT_MARKETPLACE) > 0

    def test_all_entries_have_required_keys(self):
        for name, meta in _DEFAULT_MARKETPLACE.items():
            for key in ("domain", "reliability", "risk", "cost", "latency", "asset"):
                assert key in meta, f"{name!r} missing key {key!r}"

    def test_reliabilities_in_range(self):
        for name, meta in _DEFAULT_MARKETPLACE.items():
            r = float(meta["reliability"])  # type: ignore[arg-type]
            assert 0.0 <= r <= 1.0, f"{name!r} reliability out of range: {r}"
