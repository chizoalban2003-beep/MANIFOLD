"""Tests for VaultTransitionModel (PROMPT B1)."""
import json
import os
import tempfile

import pytest

from manifold.world_model import VaultTransitionModel


class TestVaultTransitionModel:
    def test_learn_empty_vault_returns_insufficient_data(self):
        """VaultTransitionModel.learn() with empty vault returns insufficient_data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = VaultTransitionModel(vault_path=tmpdir)
            result = model.learn(min_samples=50)
        assert result["status"] == "insufficient_data"
        assert result["total_events"] == 0
        assert result["domains_learned"] == 0

    def test_predict_returns_fallback_for_unknown_domain(self):
        """predict() returns 0.20 fallback for unknown domain."""
        model = VaultTransitionModel()
        # Not learned yet — always uses fallback
        p = model.predict("unknown_domain_xyz", 0.8)
        assert p == VaultTransitionModel.FALLBACK_P

    def test_learn_with_sufficient_events(self, tmp_path):
        """After seeding vault with 60 events: learn() returns status=learned."""
        # Write 60 events to a jsonl file
        events = []
        for i in range(60):
            outcome = "blocked" if i % 3 == 0 else "success"
            events.append(json.dumps({
                "domain": "physical",
                "risk_score": 0.7,
                "outcome": outcome,
            }))
        wal_file = tmp_path / "test_events.jsonl"
        wal_file.write_text("\n".join(events) + "\n")

        model = VaultTransitionModel(vault_path=str(tmp_path))
        result = model.learn(min_samples=50)
        assert result["status"] == "learned"
        assert result["total_events"] == 60
        assert result["domains_learned"] >= 1

    def test_predict_with_learned_data_differs_from_fallback(self, tmp_path):
        """predict() with learned data differs from 0.20 fallback."""
        # Write events where all outcomes are "blocked" for domain=test_domain
        events = [
            json.dumps({"domain": "test_domain", "risk_score": 0.8, "outcome": "blocked"})
            for _ in range(60)
        ]
        wal_file = tmp_path / "events.jsonl"
        wal_file.write_text("\n".join(events) + "\n")

        model = VaultTransitionModel(vault_path=str(tmp_path))
        model.learn(min_samples=50)
        p = model.predict("test_domain", 0.8)
        # p should be 1.0 (all blocked) — not 0.20
        assert p != VaultTransitionModel.FALLBACK_P
        assert p > 0.5
