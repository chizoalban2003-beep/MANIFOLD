"""Tests for Phase 41: Genesis Node Bootstrapping (manifold/genesis.py)."""

from __future__ import annotations

import math

import pytest

from manifold.genesis import GenesisAllocation, GenesisConfig, GenesisMint


# ---------------------------------------------------------------------------
# GenesisConfig
# ---------------------------------------------------------------------------


class TestGenesisConfig:
    def test_defaults(self) -> None:
        cfg = GenesisConfig()
        assert cfg.total_tokens == 1000.0
        assert cfg.gamma == 1.0
        assert cfg.genesis_node_id == "genesis-0"
        assert cfg.initial_threshold == 0.45

    def test_custom_values(self) -> None:
        cfg = GenesisConfig(total_tokens=500.0, gamma=2.0, genesis_node_id="root", initial_threshold=0.3)
        assert cfg.total_tokens == 500.0
        assert cfg.gamma == 2.0
        assert cfg.genesis_node_id == "root"
        assert cfg.initial_threshold == 0.3

    def test_to_dict(self) -> None:
        cfg = GenesisConfig()
        d = cfg.to_dict()
        assert d["total_tokens"] == 1000.0
        assert d["gamma"] == 1.0
        assert d["genesis_node_id"] == "genesis-0"
        assert d["initial_threshold"] == 0.45

    def test_frozen(self) -> None:
        cfg = GenesisConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.gamma = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GenesisAllocation
# ---------------------------------------------------------------------------


class TestGenesisAllocation:
    def test_to_dict(self) -> None:
        alloc = GenesisAllocation(peer_id="peer-a", distance=1.0, weight=0.5, tokens=100.0)
        d = alloc.to_dict()
        assert d["peer_id"] == "peer-a"
        assert d["distance"] == 1.0
        assert d["weight"] == 0.5
        assert d["tokens"] == 100.0

    def test_frozen(self) -> None:
        alloc = GenesisAllocation(peer_id="x", distance=0.0, weight=1.0, tokens=500.0)
        with pytest.raises((AttributeError, TypeError)):
            alloc.tokens = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GenesisMint — basic operation
# ---------------------------------------------------------------------------


class TestGenesisMint:
    def test_empty_peers_returns_empty(self) -> None:
        mint = GenesisMint()
        assert mint.mint({}) == []

    def test_single_peer_gets_all_tokens(self) -> None:
        mint = GenesisMint(GenesisConfig(total_tokens=200.0))
        allocs = mint.mint({"only": 1.0})
        assert len(allocs) == 1
        assert allocs[0].tokens == pytest.approx(200.0)

    def test_two_peers_same_distance(self) -> None:
        mint = GenesisMint(GenesisConfig(total_tokens=100.0))
        allocs = mint.mint({"a": 1.0, "b": 1.0})
        assert len(allocs) == 2
        tokens = sorted(a.tokens for a in allocs)
        assert tokens[0] == pytest.approx(50.0)
        assert tokens[1] == pytest.approx(50.0)

    def test_closer_peer_gets_more(self) -> None:
        mint = GenesisMint(GenesisConfig(total_tokens=1000.0, gamma=1.0))
        allocs = mint.mint({"near": 0.5, "far": 5.0})
        near = next(a for a in allocs if a.peer_id == "near")
        far = next(a for a in allocs if a.peer_id == "far")
        assert near.tokens > far.tokens

    def test_token_sum_equals_total(self) -> None:
        mint = GenesisMint(GenesisConfig(total_tokens=500.0))
        allocs = mint.mint({"a": 1.0, "b": 2.0, "c": 3.0})
        assert sum(a.tokens for a in allocs) == pytest.approx(500.0)

    def test_sorted_descending(self) -> None:
        mint = GenesisMint()
        allocs = mint.mint({"close": 0.1, "mid": 1.0, "far": 5.0})
        for i in range(len(allocs) - 1):
            assert allocs[i].tokens >= allocs[i + 1].tokens

    def test_negative_distance_raises(self) -> None:
        mint = GenesisMint()
        with pytest.raises(ValueError, match="non-negative"):
            mint.mint({"bad": -1.0})

    def test_gamma_zero_equal_distribution(self) -> None:
        # gamma=0 → all weights = e^0 = 1 → equal distribution
        mint = GenesisMint(GenesisConfig(total_tokens=300.0, gamma=0.0))
        allocs = mint.mint({"a": 1.0, "b": 2.0, "c": 3.0})
        tokens = [a.tokens for a in allocs]
        assert all(t == pytest.approx(100.0) for t in tokens)

    def test_spatial_decay_formula(self) -> None:
        gamma = 1.0
        distances = {"a": 0.0, "b": 1.0}
        cfg = GenesisConfig(total_tokens=1000.0, gamma=gamma)
        mint = GenesisMint(cfg)
        allocs = mint.mint(distances)
        wa = math.exp(-gamma * 0.0)
        wb = math.exp(-gamma * 1.0)
        total_w = wa + wb
        expected_a = 1000.0 * wa / total_w
        expected_b = 1000.0 * wb / total_w
        a_alloc = next(a for a in allocs if a.peer_id == "a")
        b_alloc = next(a for a in allocs if a.peer_id == "b")
        assert a_alloc.tokens == pytest.approx(expected_a)
        assert b_alloc.tokens == pytest.approx(expected_b)

    def test_history_recorded(self) -> None:
        mint = GenesisMint()
        assert len(mint.history()) == 0
        mint.mint({"p": 1.0})
        mint.mint({"q": 2.0})
        assert len(mint.history()) == 2

    def test_history_returns_copy(self) -> None:
        mint = GenesisMint()
        mint.mint({"p": 0.5})
        h1 = mint.history()
        h1.clear()
        assert len(mint.history()) == 1

    def test_summary_keys(self) -> None:
        mint = GenesisMint()
        s = mint.summary()
        assert "genesis_node_id" in s
        assert "total_tokens" in s
        assert "gamma" in s
        assert "mint_events" in s

    def test_summary_mint_events_count(self) -> None:
        mint = GenesisMint()
        mint.mint({"a": 1.0})
        mint.mint({"b": 2.0})
        assert mint.summary()["mint_events"] == 2

    def test_history_record_structure(self) -> None:
        mint = GenesisMint()
        mint.mint({"x": 1.0, "y": 2.0})
        rec = mint.history()[0]
        assert "timestamp" in rec
        assert rec["peer_count"] == 2
        assert len(rec["allocations"]) == 2

    def test_weight_values_stored(self) -> None:
        gamma = 0.5
        mint = GenesisMint(GenesisConfig(gamma=gamma))
        allocs = mint.mint({"p": 2.0})
        assert allocs[0].weight == pytest.approx(math.exp(-gamma * 2.0))

    def test_multiple_peers_five(self) -> None:
        mint = GenesisMint(GenesisConfig(total_tokens=1000.0))
        peers = {f"peer-{i}": float(i) for i in range(1, 6)}
        allocs = mint.mint(peers)
        assert len(allocs) == 5
        assert sum(a.tokens for a in allocs) == pytest.approx(1000.0)

    def test_default_config_used(self) -> None:
        mint = GenesisMint()
        assert mint.config.total_tokens == 1000.0
        assert mint.config.gamma == 1.0
