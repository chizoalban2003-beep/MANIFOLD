"""manifold/delegation.py — Delegation profiles for MANIFOLD.

Allows a CEO / owner to delegate routine escalation decisions to named
people for specific domains, risk levels, and time windows.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DelegationProfile:
    """A single delegation rule from an owner to a delegate.

    Parameters
    ----------
    owner_id:
        Identifier of the person who set this profile (the CEO).
    delegate_id:
        Identifier of the person who receives delegated escalations.
    delegate_contact:
        Contact address (email / phone / Slack ID) of the delegate.
    domains:
        Which domains to delegate.  ``["*"]`` means all domains.
    risk_max:
        Only delegate escalations where ``risk_score <= risk_max``.
    valid_from:
        Start of the delegation window (Unix timestamp).
    valid_until:
        End of the delegation window (0 = no expiry).
    active:
        Whether this profile is currently enabled.
    """
    owner_id: str
    delegate_id: str
    delegate_contact: str
    domains: list[str]
    risk_max: float = 0.75
    valid_from: float = field(default_factory=time.time)
    valid_until: float = 0.0  # 0 = no expiry
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "delegate_id": self.delegate_id,
            "delegate_contact": self.delegate_contact,
            "domains": self.domains,
            "risk_max": self.risk_max,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "active": self.active,
        }


class DelegationManager:
    """Manages delegation profiles.

    Thread-safe (simple list with a re-scan on each lookup).
    """

    def __init__(self) -> None:
        self._profiles: list[DelegationProfile] = []

    # ------------------------------------------------------------------

    def add_profile(self, profile: DelegationProfile) -> None:
        """Add or replace a delegation profile (keyed on owner+delegate)."""
        self._profiles = [
            p for p in self._profiles
            if not (p.owner_id == profile.owner_id and p.delegate_id == profile.delegate_id)
        ]
        self._profiles.append(profile)

    def remove_profile(self, owner_id: str, delegate_id: str) -> bool:
        """Remove a delegation profile.  Returns True if found."""
        before = len(self._profiles)
        self._profiles = [
            p for p in self._profiles
            if not (p.owner_id == owner_id and p.delegate_id == delegate_id)
        ]
        return len(self._profiles) < before

    def get_delegate(
        self,
        owner_id: str,
        domain: str,
        risk_score: float,
    ) -> str | None:
        """Return the delegate_id for a matching active profile, or None.

        A profile matches when:
        * ``profile.owner_id == owner_id``
        * ``domain`` is in ``profile.domains`` or ``"*"`` is in it
        * ``risk_score <= profile.risk_max``
        * current time is within ``valid_from .. valid_until`` (or no expiry)
        * ``profile.active`` is True
        """
        now = time.time()
        for p in self._profiles:
            if not p.active:
                continue
            if p.owner_id != owner_id:
                continue
            # Domain check
            if "*" not in p.domains and domain not in p.domains:
                continue
            # Risk check
            if risk_score > p.risk_max:
                continue
            # Time window check
            if now < p.valid_from:
                continue
            if p.valid_until != 0.0 and now > p.valid_until:
                continue
            return p.delegate_id
        return None

    def active_profiles(self, owner_id: str) -> list[DelegationProfile]:
        """Return active, non-expired profiles for owner_id."""
        now = time.time()
        return [
            p for p in self._profiles
            if p.owner_id == owner_id
            and p.active
            and now >= p.valid_from
            and (p.valid_until == 0.0 or now <= p.valid_until)
        ]

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of all profiles."""
        return {
            "total_profiles": len(self._profiles),
            "active_profiles": sum(1 for p in self._profiles if p.active),
            "profiles": [p.to_dict() for p in self._profiles],
        }
