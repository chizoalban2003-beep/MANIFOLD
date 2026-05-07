"""Tests for RBAC role-based access control (manifold/orgs.py)."""

from __future__ import annotations

from manifold.orgs import OrgConfig, OrgRole, has_permission


def _org(role: OrgRole) -> OrgConfig:
    return OrgConfig(
        org_id=f"test-{role.value}",
        display_name=f"Test {role.value.title()}",
        role=role,
        api_key_hash="deadbeef",
    )


def test_admin_can_access_everything():
    org = _org(OrgRole.ADMIN)
    assert has_permission(org, "POST", "/run") is True
    assert has_permission(org, "POST", "/orgs") is True
    assert has_permission(org, "GET", "/metrics") is True
    assert has_permission(org, "GET", "/dashboard") is True


def test_agent_can_use_run_and_shield():
    org = _org(OrgRole.AGENT)
    assert has_permission(org, "POST", "/run") is True
    assert has_permission(org, "POST", "/shield") is True


def test_agent_cannot_manage_orgs():
    org = _org(OrgRole.AGENT)
    assert has_permission(org, "POST", "/orgs") is False


def test_readonly_can_get_learned():
    org = _org(OrgRole.READONLY)
    assert has_permission(org, "GET", "/learned") is True


def test_readonly_cannot_post_run():
    org = _org(OrgRole.READONLY)
    assert has_permission(org, "POST", "/run") is False


def test_viewer_can_only_see_dashboard():
    org = _org(OrgRole.VIEWER)
    assert has_permission(org, "GET", "/dashboard") is True
    assert has_permission(org, "GET", "/metrics") is True
    assert has_permission(org, "GET", "/learned") is False
    assert has_permission(org, "POST", "/run") is False
    assert has_permission(org, "POST", "/orgs") is False
