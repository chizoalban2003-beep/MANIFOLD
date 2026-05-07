"""Tests for multi-tenant org isolation — different orgs, different governance."""

from __future__ import annotations

import pytest

from manifold.brain import BrainConfig, BrainTask, ManifoldBrain
from manifold.orgs import OrgConfig, OrgRole


def _make_brain() -> ManifoldBrain:
    return ManifoldBrain(
        config=BrainConfig(grid_size=7, generations=10, population_size=20),
        tools=[],
    )


def _medium_risk_task() -> BrainTask:
    return BrainTask(
        prompt="process quarterly settlement batch",
        domain="finance",
        stakes=0.65,
        uncertainty=0.55,
        complexity=0.60,
    )


def _is_vetoed(brain: ManifoldBrain, task: BrainTask, veto_threshold: float) -> bool:
    """Return True if the brain decision action + risk exceed veto_threshold."""
    decision = brain.decide(task)
    if decision.action in ("refuse", "stop"):
        return True
    return decision.risk_score >= veto_threshold


def test_strict_org_vetoes_medium_risk():
    """A very strict threshold (0.25) should veto a medium-risk task."""
    brain = _make_brain()
    task = _medium_risk_task()
    vetoed = _is_vetoed(brain, task, veto_threshold=0.25)
    assert vetoed is True


def test_lenient_org_permits_medium_risk():
    """A lenient threshold (0.95) should permit a medium-risk task."""
    brain = _make_brain()
    task = _medium_risk_task()
    vetoed = _is_vetoed(brain, task, veto_threshold=0.95)
    assert vetoed is False


def test_same_task_different_orgs_different_outcome():
    """Strict and lenient thresholds should produce different outcomes."""
    brain = _make_brain()
    task = _medium_risk_task()
    strict_vetoed = _is_vetoed(brain, task, veto_threshold=0.25)
    lenient_vetoed = _is_vetoed(brain, task, veto_threshold=0.95)
    assert strict_vetoed != lenient_vetoed


def test_org_registry_isolates_configs():
    org_a = OrgConfig(
        org_id="org-a",
        display_name="Org A",
        role=OrgRole.AGENT,
        api_key_hash="aaa",
        risk_tolerance=0.30,
    )
    org_b = OrgConfig(
        org_id="org-b",
        display_name="Org B",
        role=OrgRole.AGENT,
        api_key_hash="bbb",
        risk_tolerance=0.70,
    )
    org_a.notes = "modified"
    assert org_b.notes == ""
    assert org_a.org_id != org_b.org_id


def test_org_tool_restriction():
    org = OrgConfig(
        org_id="restricted",
        display_name="Restricted Org",
        role=OrgRole.AGENT,
        api_key_hash="xyz",
        allowed_tools=["safe_api"],
    )
    assert "safe_api" in org.allowed_tools
    assert "payment_api" not in org.allowed_tools
