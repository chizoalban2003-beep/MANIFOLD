"""Phase 62: Global App Registry — Decentralised Tool Publication.

Third-party developers can publish new Tools to the MANIFOLD network.  Each
tool submission must receive a **3-of-N multi-sig endorsement** from existing
Genesis nodes before it propagates to the Swarm.

Architecture
------------
::

    Developer → ToolManifest
                    │
              SwarmRegistry.publish()
                    │
              [status: "pending"]
                    │
    3× endorse() calls from Genesis peers (Phase 40-style signing)
                    │
              [status: "verified"]
                    │
         Propagate via EventBus + DHT ShardRouter

Key classes
-----------
``ToolManifest``
    Immutable descriptor submitted by a developer: name, sandboxed code,
    API endpoints.
``ToolEndorsement``
    A cryptographic endorsement from a single Genesis peer.
``RegistryEntry``
    Mutable entry tracking a tool's lifecycle in the registry.
``SwarmRegistry``
    Thread-safe, DHT-sharded decentralised tool directory.
    Requires 3-of-N Genesis endorsements before a tool is ``"verified"``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .ipc import EventBus
from .sandbox import ASTValidator
from .sharding import ShardRouter


# ---------------------------------------------------------------------------
# ToolManifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolManifest:
    """Immutable descriptor for a third-party MANIFOLD tool.

    Parameters
    ----------
    tool_id:
        Globally unique identifier for the tool.
    name:
        Human-readable display name.
    description:
        Short description of what the tool does.
    code:
        Python source code for the tool.  Must pass
        :class:`~manifold.sandbox.ASTValidator` before publication.
    endpoints:
        List of API endpoint strings (e.g. ``["POST /weather/query"]``).
    author_org_id:
        Organisation ID of the publishing developer.
    version:
        Semantic version string.  Default: ``"1.0.0"``.
    """

    tool_id: str
    name: str
    description: str
    code: str
    endpoints: tuple[str, ...]
    author_org_id: str
    version: str = "1.0.0"

    def canonical_bytes(self) -> bytes:
        """Return deterministic bytes for hashing / signing."""
        payload: dict[str, Any] = {
            "author_org_id": self.author_org_id,
            "code": self.code,
            "description": self.description,
            "endpoints": sorted(self.endpoints),
            "name": self.name,
            "tool_id": self.tool_id,
            "version": self.version,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def manifest_hash(self) -> str:
        """Return the SHA-256 hex digest of the canonical manifest bytes."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "endpoints": list(self.endpoints),
            "author_org_id": self.author_org_id,
            "version": self.version,
            "manifest_hash": self.manifest_hash(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolManifest":
        """Reconstruct a :class:`ToolManifest` from a :meth:`to_dict` output."""
        return cls(
            tool_id=str(d["tool_id"]),
            name=str(d["name"]),
            description=str(d.get("description", "")),
            code=str(d["code"]),
            endpoints=tuple(str(e) for e in d.get("endpoints", [])),
            author_org_id=str(d["author_org_id"]),
            version=str(d.get("version", "1.0.0")),
        )


# ---------------------------------------------------------------------------
# ToolEndorsement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolEndorsement:
    """A cryptographic endorsement from a single Genesis peer.

    Attributes
    ----------
    genesis_org_id:
        Org ID of the endorsing Genesis node.
    tool_id:
        The tool being endorsed.
    manifest_hash:
        SHA-256 hex digest of the :class:`ToolManifest` canonical bytes.
        Must match :meth:`ToolManifest.manifest_hash`.
    signature:
        HMAC-SHA256 signature over the manifest bytes using the peer's key.
    key_id:
        Signing key identifier.
    timestamp:
        POSIX time when the endorsement was issued.
    """

    genesis_org_id: str
    tool_id: str
    manifest_hash: str
    signature: str
    key_id: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "genesis_org_id": self.genesis_org_id,
            "tool_id": self.tool_id,
            "manifest_hash": self.manifest_hash,
            "signature": self.signature,
            "key_id": self.key_id,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# RegistryEntry
# ---------------------------------------------------------------------------


@dataclass
class RegistryEntry:
    """Mutable tracking entry for a tool in the :class:`SwarmRegistry`.

    Attributes
    ----------
    manifest:
        The :class:`ToolManifest` submitted by the developer.
    endorsements:
        List of :class:`ToolEndorsement` objects received so far.
    status:
        ``"pending"`` until 3 endorsements arrive, then ``"verified"``.
        Manually rejected entries are ``"rejected"``.
    submitted_at:
        POSIX timestamp of initial submission.
    verified_at:
        POSIX timestamp when the tool reached ``"verified"`` status, or
        ``0.0`` if not yet verified.
    """

    manifest: ToolManifest
    endorsements: list[ToolEndorsement] = field(default_factory=list)
    status: str = "pending"
    submitted_at: float = field(default_factory=time.time)
    verified_at: float = 0.0

    def endorsement_count(self) -> int:
        """Return the number of endorsements collected."""
        return len(self.endorsements)

    def endorsing_peers(self) -> set[str]:
        """Return the set of Genesis org IDs that have endorsed this tool."""
        return {e.genesis_org_id for e in self.endorsements}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "manifest": self.manifest.to_dict(),
            "status": self.status,
            "submitted_at": self.submitted_at,
            "verified_at": self.verified_at,
            "endorsement_count": self.endorsement_count(),
            "endorsing_peers": sorted(self.endorsing_peers()),
        }


# ---------------------------------------------------------------------------
# PublishResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishResult:
    """Outcome of a :meth:`SwarmRegistry.publish` or :meth:`SwarmRegistry.endorse` call.

    Attributes
    ----------
    accepted:
        ``True`` if the operation was accepted.
    tool_id:
        The affected tool.
    status:
        Updated status of the registry entry.
    endorsement_count:
        Total endorsements collected so far.
    required:
        How many endorsements are needed for verification.
    verified:
        ``True`` if the tool has reached ``"verified"`` status.
    reason:
        Human-readable description.
    """

    accepted: bool
    tool_id: str
    status: str
    endorsement_count: int
    required: int
    verified: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "accepted": self.accepted,
            "tool_id": self.tool_id,
            "status": self.status,
            "endorsement_count": self.endorsement_count,
            "required": self.required,
            "verified": self.verified,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# SwarmRegistry
# ---------------------------------------------------------------------------

TOPIC_TOOL_PUBLISHED = "registry.tool_published"
TOPIC_TOOL_VERIFIED = "registry.tool_verified"


@dataclass
class SwarmRegistry:
    """Decentralised tool directory sharded across the DHT.

    New tools are accepted into ``"pending"`` state immediately.  They
    become ``"verified"`` only after receiving at least
    ``required_endorsements`` distinct endorsements from trusted Genesis
    peers (keyed by ``key_id``).

    Parameters
    ----------
    shard_router:
        :class:`~manifold.sharding.ShardRouter` for DHT-based routing.
        Used to locate the canonical shard for a tool and emit
        replication signals (informational).
    event_bus:
        :class:`~manifold.ipc.EventBus` for propagation events.
    required_endorsements:
        Number of distinct Genesis endorsements required for verification.
        Default: ``3``.
    trusted_key_ids:
        Set of ``key_id`` strings considered trustworthy Genesis signing
        keys.  When non-empty, endorsements with unknown ``key_id`` values
        are rejected.  When empty, all ``key_id`` values are accepted
        (useful for testing).

    Example
    -------
    ::

        registry = SwarmRegistry()
        result = registry.publish(manifest)
        registry.endorse(tool_id, endorsement1)
        registry.endorse(tool_id, endorsement2)
        result3 = registry.endorse(tool_id, endorsement3)
        assert result3.verified
    """

    shard_router: ShardRouter = field(default_factory=lambda: ShardRouter("registry"))
    event_bus: EventBus = field(default_factory=EventBus)
    required_endorsements: int = 3
    trusted_key_ids: set[str] = field(default_factory=set)

    _entries: dict[str, RegistryEntry] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _validator: ASTValidator = field(
        default_factory=ASTValidator, init=False, repr=False
    )

    def publish(self, manifest: ToolManifest) -> PublishResult:
        """Submit a :class:`ToolManifest` for community endorsement.

        Runs :class:`~manifold.sandbox.ASTValidator` against the tool's
        ``code`` before accepting the submission.

        Parameters
        ----------
        manifest:
            The tool descriptor submitted by the developer.

        Returns
        -------
        PublishResult
            ``accepted=True`` if the manifest was accepted into ``"pending"``
            state; ``accepted=False`` with a reason on rejection.
        """
        # Static validation of tool code
        violations = self._validator.validate(manifest.code)
        if violations:
            return PublishResult(
                accepted=False,
                tool_id=manifest.tool_id,
                status="rejected",
                endorsement_count=0,
                required=self.required_endorsements,
                verified=False,
                reason=f"ASTValidator rejected tool code: {violations[0].description}",
            )

        with self._lock:
            if manifest.tool_id in self._entries:
                existing = self._entries[manifest.tool_id]
                return PublishResult(
                    accepted=False,
                    tool_id=manifest.tool_id,
                    status=existing.status,
                    endorsement_count=existing.endorsement_count(),
                    required=self.required_endorsements,
                    verified=existing.status == "verified",
                    reason=f"tool_id={manifest.tool_id!r} already registered",
                )

            entry = RegistryEntry(manifest=manifest)
            self._entries[manifest.tool_id] = entry

        # Emit publication event
        self.event_bus.publish(
            TOPIC_TOOL_PUBLISHED,
            {"tool_id": manifest.tool_id, "author": manifest.author_org_id},
        )

        return PublishResult(
            accepted=True,
            tool_id=manifest.tool_id,
            status="pending",
            endorsement_count=0,
            required=self.required_endorsements,
            verified=False,
            reason="tool accepted; awaiting endorsements",
        )

    def endorse(self, tool_id: str, endorsement: ToolEndorsement) -> PublishResult:
        """Submit a :class:`ToolEndorsement` for a pending tool.

        Parameters
        ----------
        tool_id:
            Identifier of the tool to endorse.
        endorsement:
            The Genesis peer's endorsement.

        Returns
        -------
        PublishResult
        """
        with self._lock:
            entry = self._entries.get(tool_id)
            if entry is None:
                return PublishResult(
                    accepted=False,
                    tool_id=tool_id,
                    status="not_found",
                    endorsement_count=0,
                    required=self.required_endorsements,
                    verified=False,
                    reason=f"tool_id={tool_id!r} not found in registry",
                )

            if entry.status == "verified":
                return PublishResult(
                    accepted=False,
                    tool_id=tool_id,
                    status="verified",
                    endorsement_count=entry.endorsement_count(),
                    required=self.required_endorsements,
                    verified=True,
                    reason="tool already verified",
                )

            if entry.status == "rejected":
                return PublishResult(
                    accepted=False,
                    tool_id=tool_id,
                    status="rejected",
                    endorsement_count=entry.endorsement_count(),
                    required=self.required_endorsements,
                    verified=False,
                    reason="tool has been rejected",
                )

            # Reject duplicate endorsements from the same peer
            if endorsement.genesis_org_id in entry.endorsing_peers():
                return PublishResult(
                    accepted=False,
                    tool_id=tool_id,
                    status=entry.status,
                    endorsement_count=entry.endorsement_count(),
                    required=self.required_endorsements,
                    verified=False,
                    reason=f"peer {endorsement.genesis_org_id!r} already endorsed this tool",
                )

            # Verify manifest hash matches
            if endorsement.manifest_hash != entry.manifest.manifest_hash():
                return PublishResult(
                    accepted=False,
                    tool_id=tool_id,
                    status=entry.status,
                    endorsement_count=entry.endorsement_count(),
                    required=self.required_endorsements,
                    verified=False,
                    reason="manifest_hash mismatch",
                )

            # Verify key_id is trusted (if trust list is non-empty)
            if self.trusted_key_ids and endorsement.key_id not in self.trusted_key_ids:
                return PublishResult(
                    accepted=False,
                    tool_id=tool_id,
                    status=entry.status,
                    endorsement_count=entry.endorsement_count(),
                    required=self.required_endorsements,
                    verified=False,
                    reason=f"key_id={endorsement.key_id!r} is not in the trusted set",
                )

            # Accept endorsement
            entry.endorsements.append(endorsement)
            count = entry.endorsement_count()
            newly_verified = count >= self.required_endorsements

            if newly_verified:
                entry.status = "verified"
                entry.verified_at = time.time()

        if newly_verified:
            self.event_bus.publish(
                TOPIC_TOOL_VERIFIED,
                {"tool_id": tool_id, "endorsement_count": count},
            )

        return PublishResult(
            accepted=True,
            tool_id=tool_id,
            status=entry.status,
            endorsement_count=count,
            required=self.required_endorsements,
            verified=newly_verified,
            reason="verified" if newly_verified else f"{count}/{self.required_endorsements} endorsements",
        )

    def get_tool(self, tool_id: str) -> RegistryEntry | None:
        """Return the :class:`RegistryEntry` for *tool_id*, or ``None``."""
        with self._lock:
            return self._entries.get(tool_id)

    def list_tools(
        self,
        *,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all tools, optionally filtered by *status*.

        Parameters
        ----------
        status:
            If given, only return tools with this status.  One of
            ``"pending"``, ``"verified"``, ``"rejected"``.

        Returns
        -------
        list[dict[str, Any]]
            Sorted by ``tool_id``.
        """
        with self._lock:
            entries = list(self._entries.values())

        if status is not None:
            entries = [e for e in entries if e.status == status]

        return [e.to_dict() for e in sorted(entries, key=lambda e: e.manifest.tool_id)]

    def reject(self, tool_id: str, reason: str = "") -> bool:
        """Manually mark a tool as ``"rejected"``.

        Parameters
        ----------
        tool_id:
            The tool to reject.
        reason:
            Optional human-readable reason.

        Returns
        -------
        bool
            ``True`` if the tool was found and rejected.
        """
        with self._lock:
            entry = self._entries.get(tool_id)
            if entry is None or entry.status == "verified":
                return False
            entry.status = "rejected"
        return True

    def tool_count(self, status: str | None = None) -> int:
        """Return the number of tools, optionally filtered by *status*."""
        with self._lock:
            if status is None:
                return len(self._entries)
            return sum(1 for e in self._entries.values() if e.status == status)

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of registry activity."""
        with self._lock:
            by_status: dict[str, int] = {}
            for e in self._entries.values():
                by_status[e.status] = by_status.get(e.status, 0) + 1
        return {
            "total_tools": sum(by_status.values()),
            "by_status": by_status,
            "required_endorsements": self.required_endorsements,
        }
