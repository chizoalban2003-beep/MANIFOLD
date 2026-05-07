"""Tests for manifold.workspace."""
import pytest
from manifold.workspace import GlobalWorkspace


def test_finance_routing():
    ws = GlobalWorkspace()
    domain = ws.route_task("refund invoice billing payment")
    assert domain == "finance"


def test_medical_routing():
    ws = GlobalWorkspace()
    domain = ws.route_task("patient diagnosis drug treatment")
    assert domain == "medical"


def test_legal_routing():
    ws = GlobalWorkspace()
    domain = ws.route_task("GDPR contract compliance clause regulation")
    assert domain == "legal"


def test_vague_routes_to_general():
    ws = GlobalWorkspace()
    domain = ws.route_task("what time is the meeting")
    assert domain == "general"


def test_explicit_domain_overrides_auto():
    ws = GlobalWorkspace()
    # medical text but explicit finance domain
    domain = ws.route_task("patient diagnosis drug treatment", explicit_domain="finance")
    assert domain == "finance"


def test_competition_scores_contains_finance_and_general():
    ws = GlobalWorkspace()
    scores = ws.competition_scores("refund invoice billing")
    assert "finance" in scores
    assert "general" in scores


def test_explicit_general_falls_through_to_auto():
    ws = GlobalWorkspace()
    # "general" as explicit domain should fall through
    domain = ws.route_task("refund invoice billing", explicit_domain="general")
    assert domain == "finance"


def test_route_task_no_domain_returns_nonempty():
    ws = GlobalWorkspace()
    result = ws.route_task("hello world run task")
    assert isinstance(result, str) and len(result) > 0
