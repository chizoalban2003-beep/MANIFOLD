"""tests/test_delegation.py — 5 tests for DelegationManager."""
from __future__ import annotations

import time

import pytest

from manifold.delegation import DelegationManager, DelegationProfile


def _make_profile(
    owner: str = "ceo-1",
    delegate: str = "mgr-1",
    domains: list | None = None,
    risk_max: float = 0.75,
    valid_until: float = 0.0,
    active: bool = True,
) -> DelegationProfile:
    return DelegationProfile(
        owner_id=owner,
        delegate_id=delegate,
        delegate_contact="mgr@example.com",
        domains=domains or ["finance", "legal"],
        risk_max=risk_max,
        valid_from=time.time() - 1,  # started 1 second ago
        valid_until=valid_until,
        active=active,
    )


# 1. add_profile stores correctly
def test_add_profile_stores():
    mgr = DelegationManager()
    p = _make_profile()
    mgr.add_profile(p)
    assert len(mgr._profiles) == 1
    assert mgr._profiles[0].delegate_id == "mgr-1"


# 2. get_delegate returns delegate for matching domain + risk
def test_get_delegate_matching():
    mgr = DelegationManager()
    mgr.add_profile(_make_profile(risk_max=0.75))
    result = mgr.get_delegate("ceo-1", "finance", 0.5)
    assert result == "mgr-1"


# 3. get_delegate returns None when risk exceeds risk_max
def test_get_delegate_none_when_risk_too_high():
    mgr = DelegationManager()
    mgr.add_profile(_make_profile(risk_max=0.75))
    result = mgr.get_delegate("ceo-1", "finance", 0.9)
    assert result is None


# 4. get_delegate returns None when profile is expired
def test_get_delegate_none_when_expired():
    mgr = DelegationManager()
    p = _make_profile(valid_until=time.time() - 3600)  # expired 1 hour ago
    mgr.add_profile(p)
    result = mgr.get_delegate("ceo-1", "finance", 0.5)
    assert result is None


# 5. active_profiles returns only active, non-expired profiles
def test_active_profiles_filters_correctly():
    mgr = DelegationManager()
    # Active + not expired
    mgr.add_profile(_make_profile(delegate="mgr-active"))
    # Active but expired
    expired = _make_profile(delegate="mgr-expired", valid_until=time.time() - 100)
    mgr.add_profile(expired)
    # Inactive
    inactive = _make_profile(delegate="mgr-inactive", active=False)
    mgr.add_profile(inactive)

    profiles = mgr.active_profiles("ceo-1")
    delegate_ids = [p.delegate_id for p in profiles]
    assert "mgr-active" in delegate_ids
    assert "mgr-expired" not in delegate_ids
    assert "mgr-inactive" not in delegate_ids
