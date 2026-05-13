"""Tests for simplified BFT federation trust scores (PROMPT D4)."""
import pytest

from manifold.federation import FederatedGossipBridge
from manifold.crypto import PolicySigningKey


def _sign(org_id: str, agent_id: str, score: float) -> str:
    """Sign a (agent_id, score) pair using the org's deterministic key."""
    key = PolicySigningKey.from_passphrase(org_id)
    payload = f"{agent_id}:{score:.8f}".encode()
    return key.sign(payload)


class TestBFTFederation:
    def test_single_signature_update_not_applied(self):
        """Single signature: update NOT applied (below quorum of 2)."""
        bridge = FederatedGossipBridge(bft_enabled=True, quorum=2)
        bridge.register("org-a")
        bridge.register("org-b")

        sig = _sign("org-a", "agent-1", 0.85)
        result = bridge.receive_signed_update("agent-1", 0.85, sig, "org-a")
        # Only 1 signature — should not be applied
        assert result is False, "Update should not be applied with only 1 signature"

    def test_two_matching_signatures_apply_update(self):
        """Two matching signatures from different orgs: update applied."""
        bridge = FederatedGossipBridge(bft_enabled=True, quorum=2)
        bridge.register("org-a")
        bridge.register("org-b")
        bridge.register("org-c")

        agent_id = "agent-2"
        score = 0.90

        # First signature (org-a)
        sig_a = _sign("org-a", agent_id, score)
        result1 = bridge.receive_signed_update(agent_id, score, sig_a, "org-a")
        assert result1 is False, "First signature should not reach quorum"

        # Second signature (org-b)
        sig_b = _sign("org-b", agent_id, score)
        result2 = bridge.receive_signed_update(agent_id, score, sig_b, "org-b")
        assert result2 is True, "Two valid signatures should reach quorum and apply update"

    def test_invalid_signature_not_counted(self):
        """One valid + one invalid signature: update NOT applied."""
        bridge = FederatedGossipBridge(bft_enabled=True, quorum=2)
        bridge.register("org-a")

        agent_id = "agent-3"
        score = 0.80

        # First valid signature
        sig_valid = _sign("org-a", agent_id, score)
        bridge.receive_signed_update(agent_id, score, sig_valid, "org-a")

        # Invalid signature (wrong payload)
        invalid_sig = "000000000000000000000000000000deadbeef"
        result = bridge.receive_signed_update(agent_id, score, invalid_sig, "org-a")
        # Invalid sig should be rejected — quorum not reached
        assert result is False

    def test_mismatched_score_in_second_signature(self):
        """Mismatched score in second signature: quorum not reached."""
        bridge = FederatedGossipBridge(bft_enabled=True, quorum=2)
        bridge.register("org-a")
        bridge.register("org-b")

        agent_id = "agent-4"
        score_a = 0.75
        score_b = 0.50  # Different score — different hash bucket

        sig_a = _sign("org-a", agent_id, score_a)
        bridge.receive_signed_update(agent_id, score_a, sig_a, "org-a")

        sig_b = _sign("org-b", agent_id, score_b)
        result = bridge.receive_signed_update(agent_id, score_b, sig_b, "org-b")
        # Different score hashes → separate vote buckets → quorum not reached
        assert result is False, "Mismatched scores should not reach quorum"

    def test_broadcast_produces_valid_signature(self):
        """broadcast_signed_update produces a signature that can be verified."""
        bridge = FederatedGossipBridge(bft_enabled=True, quorum=2)
        bridge.register("org-x")

        sig = bridge.broadcast_signed_update("agent-5", 0.95, "org-x")
        assert sig is not None
        # Verify the signature is valid
        key = PolicySigningKey.from_passphrase("org-x")
        payload = "agent-5:0.95000000".encode()
        assert key.verify(payload, sig)
