"""Tests for Phase 62: Global App Registry (manifold/registry.py)."""

from __future__ import annotations

import pytest

from manifold.registry import (
    PublishResult,
    RegistryEntry,
    SwarmRegistry,
    ToolEndorsement,
    ToolManifest,
    TOPIC_TOOL_PUBLISHED,
    TOPIC_TOOL_VERIFIED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SAFE_CODE = "def run(x):\n    return x * 2\n"
UNSAFE_CODE = "import os\ndef run(x):\n    return os.getcwd()\n"


def _manifest(
    tool_id: str = "tool-001",
    name: str = "Test Tool",
    code: str = SAFE_CODE,
    author: str = "dev-org",
) -> ToolManifest:
    return ToolManifest(
        tool_id=tool_id,
        name=name,
        description="A test tool",
        code=code,
        endpoints=("POST /test/run",),
        author_org_id=author,
    )


def _endorsement(
    manifest: ToolManifest,
    genesis_org: str = "genesis-1",
    key_id: str = "key-001",
) -> ToolEndorsement:
    return ToolEndorsement(
        genesis_org_id=genesis_org,
        tool_id=manifest.tool_id,
        manifest_hash=manifest.manifest_hash(),
        signature="fake-sig",
        key_id=key_id,
        timestamp=1_000_000.0,
    )


def _registry(**kwargs) -> SwarmRegistry:  # type: ignore[return]
    return SwarmRegistry(**kwargs)


# ---------------------------------------------------------------------------
# ToolManifest
# ---------------------------------------------------------------------------


class TestToolManifest:
    def test_creation(self) -> None:
        m = _manifest()
        assert m.tool_id == "tool-001"
        assert m.version == "1.0.0"

    def test_frozen(self) -> None:
        m = _manifest()
        with pytest.raises((AttributeError, TypeError)):
            m.name = "other"  # type: ignore[misc]

    def test_manifest_hash_is_deterministic(self) -> None:
        m = _manifest()
        assert m.manifest_hash() == m.manifest_hash()

    def test_different_manifests_have_different_hashes(self) -> None:
        m1 = _manifest(tool_id="t1")
        m2 = _manifest(tool_id="t2")
        assert m1.manifest_hash() != m2.manifest_hash()

    def test_to_dict_contains_hash(self) -> None:
        m = _manifest()
        d = m.to_dict()
        assert "manifest_hash" in d
        assert d["manifest_hash"] == m.manifest_hash()

    def test_from_dict_roundtrip(self) -> None:
        m = _manifest()
        restored = ToolManifest.from_dict(m.to_dict())
        assert restored.tool_id == m.tool_id
        assert restored.code == m.code
        assert restored.manifest_hash() == m.manifest_hash()

    def test_endpoints_are_tuple(self) -> None:
        m = ToolManifest(
            tool_id="t", name="n", description="d", code=SAFE_CODE,
            endpoints=("POST /foo", "GET /bar"), author_org_id="org",
        )
        assert isinstance(m.endpoints, tuple)

    def test_canonical_bytes_are_stable(self) -> None:
        m = _manifest()
        assert m.canonical_bytes() == m.canonical_bytes()


# ---------------------------------------------------------------------------
# ToolEndorsement
# ---------------------------------------------------------------------------


class TestToolEndorsement:
    def test_creation(self) -> None:
        m = _manifest()
        e = _endorsement(m)
        assert e.genesis_org_id == "genesis-1"
        assert e.tool_id == "tool-001"

    def test_frozen(self) -> None:
        m = _manifest()
        e = _endorsement(m)
        with pytest.raises((AttributeError, TypeError)):
            e.signature = "new"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        m = _manifest()
        e = _endorsement(m)
        d = e.to_dict()
        assert d["genesis_org_id"] == "genesis-1"
        assert d["manifest_hash"] == m.manifest_hash()


# ---------------------------------------------------------------------------
# SwarmRegistry — publish
# ---------------------------------------------------------------------------


class TestSwarmRegistryPublish:
    def test_valid_publish_accepted(self) -> None:
        reg = _registry()
        result = reg.publish(_manifest())
        assert result.accepted
        assert result.status == "pending"
        assert result.verified is False
        assert result.required == 3

    def test_duplicate_publish_rejected(self) -> None:
        reg = _registry()
        reg.publish(_manifest())
        result = reg.publish(_manifest())
        assert not result.accepted
        assert "already registered" in result.reason

    def test_unsafe_code_rejected(self) -> None:
        reg = _registry()
        m = _manifest(code=UNSAFE_CODE)
        result = reg.publish(m)
        assert not result.accepted
        assert "ASTValidator" in result.reason

    def test_publish_emits_event(self) -> None:
        from manifold.ipc import EventBus
        bus = EventBus()
        q = bus.subscribe(TOPIC_TOOL_PUBLISHED)
        reg = SwarmRegistry(event_bus=bus)
        reg.publish(_manifest())
        event = q.get_nowait()
        assert event.payload["tool_id"] == "tool-001"

    def test_tool_count_increments(self) -> None:
        reg = _registry()
        assert reg.tool_count() == 0
        reg.publish(_manifest("t1"))
        reg.publish(_manifest("t2"))
        assert reg.tool_count() == 2


# ---------------------------------------------------------------------------
# SwarmRegistry — endorse
# ---------------------------------------------------------------------------


class TestSwarmRegistryEndorse:
    def test_single_endorsement_not_verified(self) -> None:
        reg = _registry()
        m = _manifest()
        reg.publish(m)
        result = reg.endorse(m.tool_id, _endorsement(m, "g1"))
        assert result.accepted
        assert result.verified is False
        assert result.endorsement_count == 1

    def test_three_endorsements_verified(self) -> None:
        reg = _registry()
        m = _manifest()
        reg.publish(m)
        reg.endorse(m.tool_id, _endorsement(m, "g1", "k1"))
        reg.endorse(m.tool_id, _endorsement(m, "g2", "k2"))
        result = reg.endorse(m.tool_id, _endorsement(m, "g3", "k3"))
        assert result.verified
        assert result.status == "verified"
        assert result.endorsement_count == 3

    def test_duplicate_peer_endorsement_rejected(self) -> None:
        reg = _registry()
        m = _manifest()
        reg.publish(m)
        reg.endorse(m.tool_id, _endorsement(m, "g1"))
        result = reg.endorse(m.tool_id, _endorsement(m, "g1"))
        assert not result.accepted
        assert "already endorsed" in result.reason

    def test_endorse_unknown_tool_rejected(self) -> None:
        reg = _registry()
        m = _manifest()
        result = reg.endorse("unknown-tool", _endorsement(m, "g1"))
        assert not result.accepted
        assert "not found" in result.reason

    def test_endorse_already_verified_returns_idempotent(self) -> None:
        reg = _registry()
        m = _manifest()
        reg.publish(m)
        reg.endorse(m.tool_id, _endorsement(m, "g1", "k1"))
        reg.endorse(m.tool_id, _endorsement(m, "g2", "k2"))
        reg.endorse(m.tool_id, _endorsement(m, "g3", "k3"))
        result = reg.endorse(m.tool_id, _endorsement(m, "g4", "k4"))
        assert not result.accepted
        assert result.verified is True

    def test_manifest_hash_mismatch_rejected(self) -> None:
        reg = _registry()
        m = _manifest()
        reg.publish(m)
        bad_endorsement = ToolEndorsement(
            genesis_org_id="g1",
            tool_id=m.tool_id,
            manifest_hash="deadbeef" * 8,  # wrong hash
            signature="sig",
            key_id="k1",
            timestamp=1.0,
        )
        result = reg.endorse(m.tool_id, bad_endorsement)
        assert not result.accepted
        assert "mismatch" in result.reason

    def test_trusted_key_ids_enforced(self) -> None:
        reg = SwarmRegistry(trusted_key_ids={"trusted-key"})
        m = _manifest()
        reg.publish(m)
        bad = _endorsement(m, "g1", "untrusted-key")
        result = reg.endorse(m.tool_id, bad)
        assert not result.accepted
        assert "trusted" in result.reason

    def test_trusted_key_ids_allows_known_key(self) -> None:
        reg = SwarmRegistry(trusted_key_ids={"k1"})
        m = _manifest()
        reg.publish(m)
        result = reg.endorse(m.tool_id, _endorsement(m, "g1", "k1"))
        assert result.accepted

    def test_verification_emits_event(self) -> None:
        from manifold.ipc import EventBus
        bus = EventBus()
        verified_q = bus.subscribe(TOPIC_TOOL_VERIFIED)
        reg = SwarmRegistry(event_bus=bus)
        m = _manifest()
        reg.publish(m)
        reg.endorse(m.tool_id, _endorsement(m, "g1", "k1"))
        reg.endorse(m.tool_id, _endorsement(m, "g2", "k2"))
        reg.endorse(m.tool_id, _endorsement(m, "g3", "k3"))
        event = verified_q.get_nowait()
        assert event.payload["tool_id"] == m.tool_id

    def test_endorse_rejected_tool_fails(self) -> None:
        reg = _registry()
        m = _manifest()
        reg.publish(m)
        reg.reject(m.tool_id)
        result = reg.endorse(m.tool_id, _endorsement(m, "g1"))
        assert not result.accepted


# ---------------------------------------------------------------------------
# SwarmRegistry — listing and querying
# ---------------------------------------------------------------------------


class TestSwarmRegistryQuery:
    def test_get_tool_returns_entry(self) -> None:
        reg = _registry()
        m = _manifest()
        reg.publish(m)
        entry = reg.get_tool(m.tool_id)
        assert entry is not None
        assert entry.manifest.tool_id == "tool-001"

    def test_get_unknown_tool_returns_none(self) -> None:
        reg = _registry()
        assert reg.get_tool("ghost") is None

    def test_list_tools_empty(self) -> None:
        reg = _registry()
        assert reg.list_tools() == []

    def test_list_tools_all(self) -> None:
        reg = _registry()
        reg.publish(_manifest("t1"))
        reg.publish(_manifest("t2"))
        tools = reg.list_tools()
        assert len(tools) == 2

    def test_list_tools_filter_by_status(self) -> None:
        reg = _registry()
        m1 = _manifest("t1")
        m2 = _manifest("t2")
        reg.publish(m1)
        reg.publish(m2)
        # Verify m1
        reg.endorse(m1.tool_id, _endorsement(m1, "g1", "k1"))
        reg.endorse(m1.tool_id, _endorsement(m1, "g2", "k2"))
        reg.endorse(m1.tool_id, _endorsement(m1, "g3", "k3"))
        verified = reg.list_tools(status="verified")
        assert len(verified) == 1
        assert verified[0]["manifest"]["tool_id"] == "t1"
        pending = reg.list_tools(status="pending")
        assert len(pending) == 1
        assert pending[0]["manifest"]["tool_id"] == "t2"

    def test_list_tools_sorted_by_tool_id(self) -> None:
        reg = _registry()
        reg.publish(_manifest("zzz"))
        reg.publish(_manifest("aaa"))
        reg.publish(_manifest("mmm"))
        ids = [t["manifest"]["tool_id"] for t in reg.list_tools()]
        assert ids == ["aaa", "mmm", "zzz"]

    def test_tool_count_by_status(self) -> None:
        reg = _registry()
        m = _manifest()
        reg.publish(m)
        reg.endorse(m.tool_id, _endorsement(m, "g1", "k1"))
        reg.endorse(m.tool_id, _endorsement(m, "g2", "k2"))
        reg.endorse(m.tool_id, _endorsement(m, "g3", "k3"))
        assert reg.tool_count("verified") == 1
        assert reg.tool_count("pending") == 0


# ---------------------------------------------------------------------------
# SwarmRegistry — reject
# ---------------------------------------------------------------------------


class TestSwarmRegistryReject:
    def test_reject_pending_tool(self) -> None:
        reg = _registry()
        m = _manifest()
        reg.publish(m)
        assert reg.reject(m.tool_id)
        entry = reg.get_tool(m.tool_id)
        assert entry is not None
        assert entry.status == "rejected"

    def test_reject_unknown_tool_returns_false(self) -> None:
        reg = _registry()
        assert not reg.reject("ghost")

    def test_reject_verified_tool_returns_false(self) -> None:
        reg = _registry()
        m = _manifest()
        reg.publish(m)
        reg.endorse(m.tool_id, _endorsement(m, "g1", "k1"))
        reg.endorse(m.tool_id, _endorsement(m, "g2", "k2"))
        reg.endorse(m.tool_id, _endorsement(m, "g3", "k3"))
        assert not reg.reject(m.tool_id)


# ---------------------------------------------------------------------------
# SwarmRegistry — summary
# ---------------------------------------------------------------------------


class TestSwarmRegistrySummary:
    def test_empty_summary(self) -> None:
        reg = _registry()
        s = reg.summary()
        assert s["total_tools"] == 0
        assert s["required_endorsements"] == 3

    def test_summary_counts_by_status(self) -> None:
        reg = _registry()
        m1 = _manifest("t1")
        m2 = _manifest("t2")
        m3 = _manifest("t3")
        reg.publish(m1)
        reg.publish(m2)
        reg.publish(m3)
        # Verify t1
        reg.endorse(m1.tool_id, _endorsement(m1, "g1", "k1"))
        reg.endorse(m1.tool_id, _endorsement(m1, "g2", "k2"))
        reg.endorse(m1.tool_id, _endorsement(m1, "g3", "k3"))
        # Reject t2
        reg.reject(m2.tool_id)
        s = reg.summary()
        assert s["total_tools"] == 3
        assert s["by_status"]["verified"] == 1
        assert s["by_status"]["pending"] == 1
        assert s["by_status"]["rejected"] == 1


# ---------------------------------------------------------------------------
# RegistryEntry
# ---------------------------------------------------------------------------


class TestRegistryEntry:
    def test_endorsement_count(self) -> None:
        m = _manifest()
        entry = RegistryEntry(manifest=m)
        assert entry.endorsement_count() == 0
        entry.endorsements.append(_endorsement(m, "g1"))
        assert entry.endorsement_count() == 1

    def test_endorsing_peers_set(self) -> None:
        m = _manifest()
        entry = RegistryEntry(manifest=m)
        entry.endorsements.append(_endorsement(m, "g1"))
        entry.endorsements.append(_endorsement(m, "g2"))
        assert entry.endorsing_peers() == {"g1", "g2"}

    def test_to_dict(self) -> None:
        m = _manifest()
        entry = RegistryEntry(manifest=m)
        d = entry.to_dict()
        assert d["status"] == "pending"
        assert "manifest" in d
        assert "endorsement_count" in d


# ---------------------------------------------------------------------------
# PublishResult
# ---------------------------------------------------------------------------


class TestPublishResult:
    def test_to_dict(self) -> None:
        r = PublishResult(
            accepted=True,
            tool_id="t1",
            status="pending",
            endorsement_count=0,
            required=3,
            verified=False,
            reason="ok",
        )
        d = r.to_dict()
        assert d["accepted"] is True
        assert d["tool_id"] == "t1"
        assert d["required"] == 3

    def test_frozen(self) -> None:
        r = PublishResult(True, "t", "p", 0, 3, False, "ok")
        with pytest.raises((AttributeError, TypeError)):
            r.accepted = False  # type: ignore[misc]
