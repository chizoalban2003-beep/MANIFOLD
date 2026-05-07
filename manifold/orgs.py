"""MANIFOLD Multi-Tenancy — Org Registry and RBAC.

Every API request is resolved to an OrgConfig via its bearer token.
Each org carries its own policy thresholds and role-based permissions.

Key classes
-----------
``OrgRole``
    Enum of four access levels: admin / agent / readonly / viewer.
``OrgConfig``
    Per-org configuration (policy thresholds, role, API key hash).
``OrgRegistry``
    Persistent registry of all orgs, keyed by SHA-256 of the API key.
``ROLE_PERMISSIONS``
    Mapping from role to the set of allowed method:path strings.
``has_permission``
    Check whether an org role permits a given HTTP method + path.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# OrgRole
# ---------------------------------------------------------------------------


class OrgRole(str, Enum):
    ADMIN    = "admin"     # full access including org management
    AGENT    = "agent"     # can use /run, /shield, /v1/chat/completions
    READONLY = "readonly"  # GET endpoints only, plus /learned
    VIEWER   = "viewer"    # /dashboard and /metrics only


# ---------------------------------------------------------------------------
# OrgConfig
# ---------------------------------------------------------------------------


@dataclass
class OrgConfig:
    """Per-organisation configuration record.

    Attributes
    ----------
    org_id:
        Unique identifier for this organisation.
    display_name:
        Human-readable name.
    role:
        The :class:`OrgRole` governing what this org can access.
    api_key_hash:
        SHA-256 hex digest of the raw API key (never store the raw key).
    domain:
        Default task domain for this org (e.g. ``"finance"``).
    risk_tolerance:
        Risk score below which tasks are automatically permitted.
    veto_threshold:
        Combined risk × stakes above which tasks are vetoed.
    min_reliability:
        Minimum acceptable tool reliability score.
    fallback:
        Action on veto: ``"hitl"`` or ``"refuse"``.
    allowed_tools:
        Whitelist of tool names.  Empty list means all tools allowed.
    created_at:
        Unix timestamp of creation.
    notes:
        Free-text notes for operators.
    """

    org_id:           str
    display_name:     str
    role:             OrgRole
    api_key_hash:     str
    domain:           str   = "general"
    risk_tolerance:   float = 0.45
    veto_threshold:   float = 0.45
    min_reliability:  float = 0.70
    fallback:         str   = "hitl"
    allowed_tools:    list  = field(default_factory=list)
    created_at:       float = field(default_factory=time.time)
    notes:            str   = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        d = self.__dict__.copy()
        d["role"] = self.role.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OrgConfig":
        """Reconstruct from a plain dict (e.g. loaded from JSON)."""
        d = d.copy()
        d["role"] = OrgRole(d["role"])
        return cls(**d)


# ---------------------------------------------------------------------------
# OrgRegistry
# ---------------------------------------------------------------------------


class OrgRegistry:
    """Persistent, thread-safe registry of all orgs.

    Orgs are stored in a JSON file at ``orgs_file`` (default: ``orgs.json``).
    The file is read on construction and written on every mutating operation.

    A default admin org is seeded automatically if the file is empty and
    ``MANIFOLD_API_KEY`` is set in the environment.
    """

    def __init__(self, orgs_file: str = "orgs.json") -> None:
        self._file = orgs_file
        self._orgs: dict[str, OrgConfig] = {}   # key_hash → OrgConfig
        self._load()
        if not self._orgs:
            self._seed_default_admin()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self._file):
            with open(self._file) as f:
                data = json.load(f)
            self._orgs = {k: OrgConfig.from_dict(v) for k, v in data.items()}

    def _save(self) -> None:
        with open(self._file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self._orgs.items()}, f, indent=2)

    def _seed_default_admin(self) -> None:
        admin_key = os.environ.get("MANIFOLD_API_KEY", "")
        if admin_key:
            self.register(
                api_key=admin_key,
                org_id="default-admin",
                display_name="Default Admin",
                role=OrgRole.ADMIN,
            )

    def _hash_key(self, api_key: str) -> str:
        # HMAC-SHA256 with a server-specific salt is appropriate here because
        # API keys are randomly-generated 64-char hex tokens (not passwords).
        # The entropy of the raw key (256 bits) makes brute-force infeasible;
        # HMAC adds a keyed layer so the hash file alone is insufficient to
        # attempt an attack.  SHA-256 is not used to hash a low-entropy secret.
        secret = os.environ.get("MANIFOLD_KEY_SALT", "manifold-default-salt").encode()
        return hmac.new(secret, api_key.encode(), hashlib.sha256).hexdigest()  # lgtm[py/weak-sensitive-data-hashing]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        api_key: str,
        org_id: str,
        display_name: str,
        role: OrgRole = OrgRole.AGENT,
        **kwargs: Any,
    ) -> OrgConfig:
        """Register a new org.  Returns the :class:`OrgConfig`."""
        config = OrgConfig(
            org_id=org_id,
            display_name=display_name,
            role=role,
            api_key_hash=self._hash_key(api_key),
            **kwargs,
        )
        self._orgs[config.api_key_hash] = config
        self._save()
        return config

    def lookup(self, api_key: str) -> OrgConfig | None:
        """Return :class:`OrgConfig` for this API key, or ``None``."""
        key_hash = self._hash_key(api_key)
        return self._orgs.get(key_hash)

    def generate_key(
        self,
        org_id: str,
        display_name: str,
        role: OrgRole = OrgRole.AGENT,
        **kwargs: Any,
    ) -> tuple[str, OrgConfig]:
        """Generate a random API key for an org.

        Returns ``(raw_key, config)``.  The raw key is shown only once —
        callers must store it immediately.
        """
        raw_key = secrets.token_hex(32)
        config = self.register(raw_key, org_id, display_name, role, **kwargs)
        return raw_key, config

    def update_policy(self, api_key: str, **policy_fields: Any) -> OrgConfig | None:
        """Update policy fields for the org identified by *api_key*.

        Returns the updated :class:`OrgConfig`, or ``None`` if not found.
        """
        config = self.lookup(api_key)
        if config is None:
            return None
        for field_name, value in policy_fields.items():
            if hasattr(config, field_name):
                setattr(config, field_name, value)
        self._save()
        return config

    def all_orgs(self) -> list[OrgConfig]:
        """Return all registered :class:`OrgConfig` objects."""
        return list(self._orgs.values())

    def save(self) -> None:
        """Persist current state to the backing JSON file."""
        self._save()

    def delete(self, org_id: str) -> bool:
        """Delete an org by ``org_id``.  Returns ``True`` if found."""
        key = next(
            (k for k, v in self._orgs.items() if v.org_id == org_id), None
        )
        if key:
            del self._orgs[key]
            self._save()
            return True
        return False


# ---------------------------------------------------------------------------
# RBAC permission table
# ---------------------------------------------------------------------------


ROLE_PERMISSIONS: dict[OrgRole, set[str]] = {
    OrgRole.ADMIN: {"GET", "POST", "DELETE", "admin", "orgs"},
    OrgRole.AGENT: {
        "POST:/run",
        "POST:/shield",
        "POST:/v1/chat/completions",
        "POST:/ats/signal",
        "POST:/ats/register",
        "GET:/v1/models",
        "GET:/learned",
        "GET:/metrics",
        "GET:/policy",
        "GET:/ats/score",
        "GET:/ats/leaderboard",
    },
    OrgRole.READONLY: {
        "GET:/learned",
        "GET:/metrics",
        "GET:/policy",
        "GET:/ats/score",
        "GET:/ats/leaderboard",
        "GET:/v1/models",
    },
    OrgRole.VIEWER: {
        "GET:/dashboard",
        "GET:/metrics",
    },
}


def has_permission(org: OrgConfig, method: str, path: str) -> bool:
    """Return ``True`` if *org*'s role permits *method* + *path*.

    Admins are always permitted.  For other roles the check uses exact
    match or prefix match against the ``"METHOD:/path"`` permission strings.
    """
    if org.role == OrgRole.ADMIN:
        return True
    perms = ROLE_PERMISSIONS.get(org.role, set())
    key = f"{method}:{path}"
    return key in perms or any(
        key.startswith(p) for p in perms if ":" in p
    )
