"""Tests for manifold/orgs.py — OrgRegistry, OrgConfig, OrgRole, has_permission."""

from __future__ import annotations

import os

import pytest

from manifold.orgs import OrgConfig, OrgRegistry, OrgRole, has_permission


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(tmp_path) -> OrgRegistry:
    """Create a fresh OrgRegistry backed by a temp file (no env-key seeding)."""
    orgs_file = str(tmp_path / "orgs.json")
    # Ensure no MANIFOLD_API_KEY leaks into the registry
    old = os.environ.pop("MANIFOLD_API_KEY", None)
    reg = OrgRegistry(orgs_file)
    if old is not None:
        os.environ["MANIFOLD_API_KEY"] = old
    return reg


def _make_org(role: OrgRole = OrgRole.AGENT) -> OrgConfig:
    return OrgConfig(
        org_id="test-org",
        display_name="Test Org",
        role=role,
        api_key_hash="deadbeef",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_and_lookup(tmp_path):
    reg = _make_registry(tmp_path)
    reg.register("my-secret-key", "acme", "Acme Corp", OrgRole.AGENT)
    found = reg.lookup("my-secret-key")
    assert found is not None
    assert found.org_id == "acme"


def test_lookup_unknown_key_returns_none(tmp_path):
    reg = _make_registry(tmp_path)
    result = reg.lookup("unknown-key-xyz")
    assert result is None


def test_generate_key_returns_raw_and_config(tmp_path):
    reg = _make_registry(tmp_path)
    raw_key, config = reg.generate_key("beta-org", "Beta Corp", OrgRole.AGENT)
    assert isinstance(raw_key, str)
    assert len(raw_key) == 64  # secrets.token_hex(32) = 64 hex chars
    assert isinstance(config, OrgConfig)
    assert config.org_id == "beta-org"


def test_update_policy_changes_fields(tmp_path):
    reg = _make_registry(tmp_path)
    raw_key, _ = reg.generate_key("gamma", "Gamma Inc", OrgRole.AGENT)
    reg.update_policy(raw_key, risk_tolerance=0.30)
    updated = reg.lookup(raw_key)
    assert updated is not None
    assert updated.risk_tolerance == pytest.approx(0.30)


def test_delete_org(tmp_path):
    reg = _make_registry(tmp_path)
    raw_key, config = reg.generate_key("delta", "Delta LLC", OrgRole.AGENT)
    deleted = reg.delete("delta")
    assert deleted is True
    assert reg.lookup(raw_key) is None


def test_all_orgs_returns_list(tmp_path):
    orgs_file = str(tmp_path / "orgs.json")
    old = os.environ.pop("MANIFOLD_API_KEY", None)
    reg = OrgRegistry(orgs_file)
    if old is not None:
        os.environ["MANIFOLD_API_KEY"] = old

    reg.register("key1", "org-one", "Org One", OrgRole.AGENT)
    reg.register("key2", "org-two", "Org Two", OrgRole.READONLY)
    all_orgs = reg.all_orgs()
    # 2 explicitly registered (no env-key admin was seeded above)
    assert len(all_orgs) >= 2


def test_admin_has_all_permissions():
    org = _make_org(OrgRole.ADMIN)
    assert has_permission(org, "POST", "/run") is True
    assert has_permission(org, "GET", "/metrics") is True
    assert has_permission(org, "POST", "/orgs") is True
    assert has_permission(org, "DELETE", "/orgs/x") is True


def test_agent_can_post_run():
    org = _make_org(OrgRole.AGENT)
    assert has_permission(org, "POST", "/run") is True


def test_viewer_cannot_post_run():
    org = _make_org(OrgRole.VIEWER)
    assert has_permission(org, "POST", "/run") is False
