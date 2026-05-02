"""Phase 21: Zero-Trust Cryptography ("TLS for Agents").

In a real-world B2B marketplace, adversarial agents may attempt to forge an
``OrgPolicy`` or spoof a high ``ReputationHub`` score.  Phase 21 closes this
attack surface by attaching cryptographic signatures to every cross-org
artefact.

Key ideas
---------
* **HMAC-SHA256 signing** — built on Python's standard-library ``hmac`` and
  ``hashlib`` modules, so no third-party dependency is required.
* **Signed OrgPolicy** — ``OrgPolicy`` objects are serialised to a
  deterministic canonical form, then HMAC-signed with the issuing
  organisation's secret key.  The ``SignedOrgPolicy`` wrapper carries both
  the original data and the hex-encoded signature.
* **Signed GossipNote** — ``GossipNote`` (Phase 10) objects are signed the
  same way, so that reputation-inflation attacks ("Sybil attacks") require
  forging an HMAC secret known only to the reporting agent.
* **VerifiedPolicyHandshake** — extends ``PolicyHandshake`` (Phase 20) to
  *require* a verified ``SignedOrgPolicy``.  An unverified or tampered policy
  is rejected *before* the risk/reliability gates are evaluated.

Key classes
-----------
``PolicySigningKey``
    Wraps a raw secret key, providing ``sign`` and ``verify`` helpers.
``SignedOrgPolicy``
    An ``OrgPolicy`` with an HMAC-SHA256 signature and the issuing key ID.
``OrgPolicySigner``
    Signs and verifies ``OrgPolicy`` objects.
``SignedGossipNote``
    A ``GossipNote`` with an HMAC-SHA256 signature.
``GossipSigner``
    Signs and verifies ``GossipNote`` objects.
``VerifiedPolicyHandshake``
    Extends ``PolicyHandshake`` to require a verified ``SignedOrgPolicy``.
``SignatureVerificationError``
    Raised when a signature is invalid or missing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from typing import Any

from .b2b import HandshakeResult, OrgPolicy, PolicyHandshake
from .brain import GossipNote
from .policy import ManifoldPolicy


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SignatureVerificationError(ValueError):
    """Raised when an HMAC signature cannot be verified.

    Attributes
    ----------
    reason:
        Human-readable explanation.
    key_id:
        The key ID that was used (or attempted) during verification.
    """

    def __init__(self, reason: str, key_id: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.key_id = key_id


# ---------------------------------------------------------------------------
# PolicySigningKey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicySigningKey:
    """An HMAC-SHA256 signing key for MANIFOLD artefacts.

    Parameters
    ----------
    key_id:
        Human-readable identifier for this key (e.g. ``"org-a-v1"``).
    secret:
        Raw secret bytes.  Must be kept confidential.  Use
        ``PolicySigningKey.generate()`` to create a cryptographically
        random key.

    Example
    -------
    ::

        key = PolicySigningKey.generate("org-a")
        sig = key.sign(b"hello")
        assert key.verify(b"hello", sig)
    """

    key_id: str
    secret: bytes

    @classmethod
    def generate(cls, key_id: str = "default", length: int = 32) -> "PolicySigningKey":
        """Generate a cryptographically random signing key.

        Parameters
        ----------
        key_id:
            Identifier for the key.
        length:
            Number of random bytes.  Default 32 (256 bits).

        Returns
        -------
        PolicySigningKey
        """
        return cls(key_id=key_id, secret=secrets.token_bytes(length))

    @classmethod
    def from_passphrase(cls, passphrase: str, key_id: str = "default") -> "PolicySigningKey":
        """Derive a deterministic key from a passphrase via SHA-256.

        Parameters
        ----------
        passphrase:
            Human-readable passphrase.  Keep this secret.
        key_id:
            Identifier for the derived key.

        Returns
        -------
        PolicySigningKey
        """
        digest = hashlib.sha256(passphrase.encode()).digest()
        return cls(key_id=key_id, secret=digest)

    def sign(self, data: bytes) -> str:
        """Compute an HMAC-SHA256 signature over *data*.

        Parameters
        ----------
        data:
            Bytes to sign.

        Returns
        -------
        str
            Hex-encoded HMAC-SHA256 digest.
        """
        return hmac.new(self.secret, data, hashlib.sha256).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        """Verify an HMAC-SHA256 *signature* over *data*.

        Uses ``hmac.compare_digest`` to prevent timing attacks.

        Parameters
        ----------
        data:
            The original bytes that were signed.
        signature:
            Hex-encoded HMAC-SHA256 digest to verify.

        Returns
        -------
        bool
        """
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Canonical serialisation helpers
# ---------------------------------------------------------------------------


def _canonical_bytes(obj: dict[str, Any]) -> bytes:
    """Serialise *obj* to canonical JSON bytes (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


# ---------------------------------------------------------------------------
# SignedOrgPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedOrgPolicy:
    """An ``OrgPolicy`` with an attached HMAC-SHA256 signature.

    Attributes
    ----------
    policy:
        The original ``OrgPolicy``.
    signature:
        Hex-encoded HMAC-SHA256 signature over the canonical policy payload.
    key_id:
        Identifier of the signing key.

    Example
    -------
    ::

        key = PolicySigningKey.generate("org-a")
        signer = OrgPolicySigner(key)
        signed = signer.sign(remote_policy)
        signer.verify(signed)   # returns True
    """

    policy: OrgPolicy
    signature: str
    key_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (suitable for JSON/wire transfer)."""
        return {
            "policy": self.policy.to_dict(),
            "signature": self.signature,
            "key_id": self.key_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignedOrgPolicy":
        """Deserialise from a plain dict."""
        return cls(
            policy=OrgPolicy.from_dict(data["policy"]),
            signature=str(data["signature"]),
            key_id=str(data["key_id"]),
        )


# ---------------------------------------------------------------------------
# OrgPolicySigner
# ---------------------------------------------------------------------------


class OrgPolicySigner:
    """Signs and verifies ``OrgPolicy`` objects.

    Parameters
    ----------
    key:
        The ``PolicySigningKey`` used for signing/verification.

    Example
    -------
    ::

        key = PolicySigningKey.generate("org-a")
        signer = OrgPolicySigner(key)
        signed = signer.sign(my_policy)
        verified_policy = signer.verify(signed)
    """

    def __init__(self, key: PolicySigningKey) -> None:
        self.key = key

    def sign(self, policy: OrgPolicy) -> SignedOrgPolicy:
        """Sign an ``OrgPolicy``.

        Parameters
        ----------
        policy:
            The policy to sign.

        Returns
        -------
        SignedOrgPolicy
        """
        payload = _canonical_bytes(policy.to_dict())
        sig = self.key.sign(payload)
        return SignedOrgPolicy(policy=policy, signature=sig, key_id=self.key.key_id)

    def verify(self, signed: SignedOrgPolicy) -> OrgPolicy:
        """Verify a ``SignedOrgPolicy`` and return the inner ``OrgPolicy``.

        Parameters
        ----------
        signed:
            The signed policy to verify.

        Returns
        -------
        OrgPolicy
            The inner policy if the signature is valid.

        Raises
        ------
        SignatureVerificationError
            If the signature does not match.
        """
        payload = _canonical_bytes(signed.policy.to_dict())
        if not self.key.verify(payload, signed.signature):
            raise SignatureVerificationError(
                f"OrgPolicy signature verification failed for key_id={signed.key_id!r}",
                key_id=signed.key_id,
            )
        return signed.policy

    def verify_bool(self, signed: SignedOrgPolicy) -> bool:
        """Return ``True`` if the signature is valid, ``False`` otherwise."""
        try:
            self.verify(signed)
            return True
        except SignatureVerificationError:
            return False


# ---------------------------------------------------------------------------
# SignedGossipNote
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedGossipNote:
    """A ``GossipNote`` with an attached HMAC-SHA256 signature.

    Signing gossip notes prevents Sybil attacks where adversarial agents
    submit fabricated failure reports to tank a competitor's reputation.

    Attributes
    ----------
    note:
        The original ``GossipNote``.
    signature:
        Hex-encoded HMAC-SHA256 signature over the canonical note payload.
    key_id:
        Identifier of the signing key.

    Example
    -------
    ::

        key = PolicySigningKey.generate("agent-x")
        signer = GossipSigner(key)
        signed = signer.sign(gossip_note)
        signer.verify(signed)
    """

    note: GossipNote
    signature: str
    key_id: str

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "tool": self.note.tool,
            "claim": self.note.claim,
            "source_id": self.note.source_id,
            "source_reputation": self.note.source_reputation,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "note": self._payload_dict(),
            "signature": self.signature,
            "key_id": self.key_id,
        }


# ---------------------------------------------------------------------------
# GossipSigner
# ---------------------------------------------------------------------------


class GossipSigner:
    """Signs and verifies ``GossipNote`` objects.

    Parameters
    ----------
    key:
        The ``PolicySigningKey`` used for signing/verification.

    Example
    -------
    ::

        key = PolicySigningKey.generate("agent-x")
        signer = GossipSigner(key)
        signed = signer.sign(note)
        verified = signer.verify(signed)
    """

    def __init__(self, key: PolicySigningKey) -> None:
        self.key = key

    @staticmethod
    def _note_to_dict(note: GossipNote) -> dict[str, Any]:
        return {
            "tool": note.tool,
            "claim": note.claim,
            "source_id": note.source_id,
            "source_reputation": note.source_reputation,
        }

    def sign(self, note: GossipNote) -> SignedGossipNote:
        """Sign a ``GossipNote``.

        Parameters
        ----------
        note:
            The gossip note to sign.

        Returns
        -------
        SignedGossipNote
        """
        payload = _canonical_bytes(self._note_to_dict(note))
        sig = self.key.sign(payload)
        return SignedGossipNote(note=note, signature=sig, key_id=self.key.key_id)

    def verify(self, signed: SignedGossipNote) -> GossipNote:
        """Verify a ``SignedGossipNote`` and return the inner ``GossipNote``.

        Parameters
        ----------
        signed:
            The signed gossip note.

        Returns
        -------
        GossipNote

        Raises
        ------
        SignatureVerificationError
            If the signature does not match.
        """
        payload = _canonical_bytes(self._note_to_dict(signed.note))
        if not self.key.verify(payload, signed.signature):
            raise SignatureVerificationError(
                f"GossipNote signature verification failed for key_id={signed.key_id!r}",
                key_id=signed.key_id,
            )
        return signed.note

    def verify_bool(self, signed: SignedGossipNote) -> bool:
        """Return ``True`` if the signature is valid, ``False`` otherwise."""
        try:
            self.verify(signed)
            return True
        except SignatureVerificationError:
            return False


# ---------------------------------------------------------------------------
# VerifiedPolicyHandshake
# ---------------------------------------------------------------------------


class VerifiedPolicyHandshake(PolicyHandshake):
    """Extends ``PolicyHandshake`` to require a verified ``SignedOrgPolicy``.

    Before evaluating risk/reliability gates, this handshake verifies the
    HMAC signature on the remote org's policy.  An unverified or tampered
    policy causes an immediate ``SignatureVerificationError`` — the risk gate
    is never reached.

    Parameters
    ----------
    local_policy:
        The calling organisation's ``ManifoldPolicy``.
    local_org_id:
        Identifier for the calling organisation.
    trusted_keys:
        Mapping of ``key_id`` → ``PolicySigningKey`` for all orgs the local
        org trusts.  Only keys present in this mapping are accepted.

    Example
    -------
    ::

        key = PolicySigningKey.generate("org-b-key")
        signer = OrgPolicySigner(key)
        signed = signer.sign(remote_policy)

        vhs = VerifiedPolicyHandshake(
            local_policy=local_policy,
            local_org_id="org-a",
            trusted_keys={"org-b-key": key},
        )
        result = vhs.check_signed(signed)
    """

    def __init__(
        self,
        local_policy: ManifoldPolicy,
        local_org_id: str = "local",
        trusted_keys: dict[str, PolicySigningKey] | None = None,
    ) -> None:
        super().__init__(local_policy=local_policy, local_org_id=local_org_id)
        self.trusted_keys: dict[str, PolicySigningKey] = trusted_keys or {}

    def check_signed(
        self, signed: SignedOrgPolicy, domain: str | None = None
    ) -> HandshakeResult:
        """Verify the signature and then run the standard policy handshake.

        Parameters
        ----------
        signed:
            The signed remote policy to evaluate.
        domain:
            Domain context override.

        Returns
        -------
        HandshakeResult

        Raises
        ------
        SignatureVerificationError
            If the key_id is unknown or the signature is invalid.
        """
        key_id = signed.key_id
        if key_id not in self.trusted_keys:
            raise SignatureVerificationError(
                f"Unknown key_id={key_id!r} — not in trusted key registry",
                key_id=key_id,
            )
        signer = OrgPolicySigner(self.trusted_keys[key_id])
        signer.verify(signed)  # raises on failure
        return self.check(signed.policy, domain=domain)

    def add_trusted_key(self, key: PolicySigningKey) -> None:
        """Add a trusted key to the registry.

        Parameters
        ----------
        key:
            Key to trust.
        """
        self.trusted_keys[key.key_id] = key
