"""Tests for Phase 24: ManifoldVault (Immutable Persistence Layer)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manifold.b2b import AgentEconomyLedger, EconomyEntry
from manifold.federation import FederatedGossipPacket
from manifold.hub import ReputationHub
from manifold.vault import ManifoldVault, VaultLoadResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_vault(tmp_path: Path) -> ManifoldVault:
    """Return a fresh vault backed by a temporary directory."""
    return ManifoldVault(data_dir=tmp_path)


@pytest.fixture
def gossip_packet() -> FederatedGossipPacket:
    return FederatedGossipPacket(
        tool_name="gpt-4o",
        signal="healthy",
        confidence=0.9,
        org_id="test-org",
        weight=1.0,
    )


@pytest.fixture
def failing_packet() -> FederatedGossipPacket:
    return FederatedGossipPacket(
        tool_name="flaky-tool",
        signal="failing",
        confidence=0.8,
        org_id="test-org",
        weight=1.0,
    )


@pytest.fixture
def economy_entry() -> EconomyEntry:
    return EconomyEntry(
        local_org_id="local",
        remote_org_id="partner-corp",
        allowed=True,
        reputation_score=0.85,
        surcharge=0.02,
        net_trust_cost=1.02,
        block_reason="",
    )


# ---------------------------------------------------------------------------
# ManifoldVault construction
# ---------------------------------------------------------------------------


def test_vault_creates_data_dir(tmp_path: Path) -> None:
    new_dir = tmp_path / "nested" / "vault"
    vault = ManifoldVault(data_dir=new_dir)
    assert new_dir.exists()
    assert vault.gossip_count() == 0
    assert vault.economy_count() == 0


# ---------------------------------------------------------------------------
# append_gossip — WAL write
# ---------------------------------------------------------------------------


def test_append_gossip_creates_file(
    tmp_vault: ManifoldVault,
    gossip_packet: FederatedGossipPacket,
) -> None:
    tmp_vault.append_gossip(gossip_packet)
    assert tmp_vault._gossip_path.exists()
    assert tmp_vault.gossip_count() == 1


def test_append_gossip_multiple(
    tmp_vault: ManifoldVault,
    gossip_packet: FederatedGossipPacket,
    failing_packet: FederatedGossipPacket,
) -> None:
    tmp_vault.append_gossip(gossip_packet)
    tmp_vault.append_gossip(failing_packet)
    assert tmp_vault.gossip_count() == 2


def test_append_gossip_jsonl_format(
    tmp_vault: ManifoldVault,
    gossip_packet: FederatedGossipPacket,
) -> None:
    tmp_vault.append_gossip(gossip_packet)
    lines = tmp_vault._gossip_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["_type"] == "gossip"
    assert record["tool_name"] == "gpt-4o"
    assert record["signal"] == "healthy"
    assert record["confidence"] == pytest.approx(0.9)
    assert record["org_id"] == "test-org"


# ---------------------------------------------------------------------------
# append_economy — WAL write
# ---------------------------------------------------------------------------


def test_append_economy_creates_file(
    tmp_vault: ManifoldVault,
    economy_entry: EconomyEntry,
) -> None:
    tmp_vault.append_economy(economy_entry)
    assert tmp_vault._economy_path.exists()
    assert tmp_vault.economy_count() == 1


def test_append_economy_jsonl_format(
    tmp_vault: ManifoldVault,
    economy_entry: EconomyEntry,
) -> None:
    tmp_vault.append_economy(economy_entry)
    lines = tmp_vault._economy_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["_type"] == "economy"
    assert record["local_org_id"] == "local"
    assert record["remote_org_id"] == "partner-corp"
    assert record["allowed"] is True
    assert record["reputation_score"] == pytest.approx(0.85)
    assert record["surcharge"] == pytest.approx(0.02)
    assert record["net_trust_cost"] == pytest.approx(1.02)


# ---------------------------------------------------------------------------
# load_state — gossip replay
# ---------------------------------------------------------------------------


def test_load_state_gossip_replays_into_hub(
    tmp_vault: ManifoldVault,
    gossip_packet: FederatedGossipPacket,
) -> None:
    tmp_vault.append_gossip(gossip_packet)

    hub = ReputationHub()
    result = tmp_vault.load_state(hub=hub)

    assert isinstance(result, VaultLoadResult)
    assert result.gossip_loaded == 1
    assert result.economy_loaded == 0
    assert result.skipped == 0
    assert result.total_loaded == 1

    # After replay the hub should know about the tool
    reliability = hub.live_reliability("gpt-4o")
    assert reliability is not None
    assert 0.0 <= reliability <= 1.0


def test_load_state_gossip_failing_lowers_reliability(
    tmp_vault: ManifoldVault,
    failing_packet: FederatedGossipPacket,
) -> None:
    # Write enough failing packets to influence the hub
    for _ in range(10):
        tmp_vault.append_gossip(failing_packet)

    hub = ReputationHub()
    result = tmp_vault.load_state(hub=hub)

    assert result.gossip_loaded == 10
    reliability = hub.live_reliability("flaky-tool")
    assert reliability is not None
    assert reliability < 1.0


# ---------------------------------------------------------------------------
# load_state — economy replay
# ---------------------------------------------------------------------------


def test_load_state_economy_replays_into_ledger(
    tmp_vault: ManifoldVault,
    economy_entry: EconomyEntry,
) -> None:
    tmp_vault.append_economy(economy_entry)

    ledger = AgentEconomyLedger()
    result = tmp_vault.load_state(ledger=ledger)

    assert result.economy_loaded == 1
    assert result.gossip_loaded == 0
    entries = ledger.entries()
    assert len(entries) == 1
    assert entries[0].remote_org_id == "partner-corp"
    assert entries[0].allowed is True
    assert entries[0].net_trust_cost == pytest.approx(1.02)


def test_load_state_multiple_economy_entries(
    tmp_vault: ManifoldVault,
    economy_entry: EconomyEntry,
) -> None:
    for _ in range(5):
        tmp_vault.append_economy(economy_entry)

    ledger = AgentEconomyLedger()
    result = tmp_vault.load_state(ledger=ledger)
    assert result.economy_loaded == 5
    assert len(ledger.entries()) == 5


# ---------------------------------------------------------------------------
# load_state — None targets are skipped silently
# ---------------------------------------------------------------------------


def test_load_state_skips_hub_when_none(
    tmp_vault: ManifoldVault,
    gossip_packet: FederatedGossipPacket,
) -> None:
    tmp_vault.append_gossip(gossip_packet)
    result = tmp_vault.load_state(hub=None, ledger=None)
    assert result.gossip_loaded == 0
    assert result.economy_loaded == 0


# ---------------------------------------------------------------------------
# load_state — graceful handling of corrupt WAL lines
# ---------------------------------------------------------------------------


def test_load_state_skips_corrupt_lines(tmp_vault: ManifoldVault) -> None:
    # Write a corrupt gossip line
    with tmp_vault._gossip_path.open("w") as fh:
        fh.write("not-valid-json\n")
        fh.write('{"_type": "gossip", "tool_name": "ok-tool", "signal": "healthy"}\n')

    hub = ReputationHub()
    result = tmp_vault.load_state(hub=hub)
    assert result.gossip_loaded == 1
    assert result.skipped == 1


def test_load_state_skips_unknown_type_tag(tmp_vault: ManifoldVault) -> None:
    with tmp_vault._gossip_path.open("w") as fh:
        fh.write('{"_type": "unknown_tag", "tool_name": "x"}\n')

    hub = ReputationHub()
    result = tmp_vault.load_state(hub=hub)
    assert result.gossip_loaded == 0
    assert result.skipped == 1


def test_load_state_invalid_signal_defaults_to_healthy(tmp_vault: ManifoldVault) -> None:
    with tmp_vault._gossip_path.open("w") as fh:
        fh.write(
            '{"_type": "gossip", "tool_name": "tool-x", "signal": "INVALID_SIGNAL", '
            '"confidence": 0.5, "org_id": "org", "weight": 1.0}\n'
        )
    hub = ReputationHub()
    result = tmp_vault.load_state(hub=hub)
    # Should load without raising — invalid signal is normalised to "healthy"
    assert result.gossip_loaded == 1
    assert result.skipped == 0


# ---------------------------------------------------------------------------
# VaultLoadResult properties
# ---------------------------------------------------------------------------


def test_vault_load_result_total_loaded() -> None:
    r = VaultLoadResult(gossip_loaded=3, economy_loaded=7, skipped=1)
    assert r.total_loaded == 10


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------


def test_purge_removes_wal_files(
    tmp_vault: ManifoldVault,
    gossip_packet: FederatedGossipPacket,
    economy_entry: EconomyEntry,
) -> None:
    tmp_vault.append_gossip(gossip_packet)
    tmp_vault.append_economy(economy_entry)
    assert tmp_vault.gossip_count() == 1
    assert tmp_vault.economy_count() == 1

    tmp_vault.purge()

    assert tmp_vault.gossip_count() == 0
    assert tmp_vault.economy_count() == 0
    assert not tmp_vault._gossip_path.exists()
    assert not tmp_vault._economy_path.exists()


def test_purge_is_idempotent(tmp_vault: ManifoldVault) -> None:
    tmp_vault.purge()  # no files exist — should not raise
    tmp_vault.purge()


# ---------------------------------------------------------------------------
# Thread safety — concurrent appends
# ---------------------------------------------------------------------------


def test_concurrent_appends_are_thread_safe(
    tmp_vault: ManifoldVault,
    gossip_packet: FederatedGossipPacket,
) -> None:
    import threading

    errors: list[Exception] = []

    def write_gossip() -> None:
        try:
            for _ in range(50):
                tmp_vault.append_gossip(gossip_packet)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write_gossip) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert tmp_vault.gossip_count() == 200
