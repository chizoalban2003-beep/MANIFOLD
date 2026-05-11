"""tests/test_governance_reporter.py — Tests for GovernanceReporter."""
from __future__ import annotations

import pytest

from manifold.governance_reporter import GovernanceReporter


def test_daily_summary_returns_string():
    """daily_summary should return a non-empty string."""
    reporter = GovernanceReporter()
    summary = reporter.daily_summary(org_id="test-org")
    assert isinstance(summary, str)
    assert "test-org" in summary
    assert "Tasks processed" in summary


def test_weekly_digest_returns_string():
    """weekly_digest should return a multi-line string with key metrics."""
    reporter = GovernanceReporter()
    digest = reporter.weekly_digest(org_id="test-org")
    assert isinstance(digest, str)
    assert "Weekly Governance Digest" in digest
    assert "Total tasks" in digest


def test_explain_escalation_returns_message():
    """explain_escalation returns a string for any event_id."""
    reporter = GovernanceReporter()
    result = reporter.explain_escalation("event-999")
    assert isinstance(result, str)
    # Should mention the event id or state it was not found
    assert "event-999" in result or "not found" in result.lower()


def test_simulate_validates_policy():
    """simulate returns a plain English string describing impact."""
    reporter = GovernanceReporter()
    result = reporter.simulate(
        org_id="test-org",
        policy_change_dict={
            "name": "Test sandbox finance",
            "conditions": {"domain": "finance", "risk_gt": 0.5},
            "action": "sandbox",
            "priority": 60,
        },
    )
    assert isinstance(result, str)
    assert "Simulation" in result or "Analysed" in result
