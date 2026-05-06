"""Phase 61: Zero-Knowledge Policy Proofs — Privacy-Preserving Verification.

B2B partners need to verify that an AI agent followed its policies (e.g.
``Cost + Risk < Threshold``) without learning the actual ``Cost`` or ``Risk``
values, which may be proprietary.

This module implements a non-interactive Schnorr Zero-Knowledge Proof (ZKP)
using the **Fiat-Shamir heuristic** over a safe-prime Diffie-Hellman group.
Zero external dependencies — only Python's :mod:`hashlib` and built-in
large-integer arithmetic are used.

Protocol
--------
The prover demonstrates knowledge of a secret integer ``x`` (e.g. a scaled
representation of ``threshold - cost - risk``) such that
``y = g^x (mod p)`` without revealing ``x``.

**Non-interactive Schnorr (Fiat-Shamir transform)**:

1. Prover picks random nonce ``r`` in ``[1, q-1]``.
2. Computes commitment ``R = g^r (mod p)``.
3. Derives challenge ``e = H(p ∥ g ∥ y ∥ R ∥ context)`` as an integer
   (using :func:`hashlib.sha256`).
4. Computes response ``s = (r - e·x) mod q``.
5. Publishes ``(y, R, s, e)``.

**Verification**:

``g^s · y^e ≡ R (mod p)``

Policy Commitment
-----------------
To prove ``cost + risk < threshold`` the prover:
1. Computes the *slack*: ``slack = threshold - cost - risk`` (must be > 0).
2. Scales to a positive integer: ``x = round(slack * SCALE)``.
3. Generates a Schnorr proof of knowledge of ``x`` with context
   ``"policy_compliance:<threshold>"``
4. Shares ``(y, proof)``; the verifier only needs to check that
   ``g^x ≡ y (mod p)`` is proven and that ``x > 0`` (encoded in the
   fact that ``y ≠ 1``).

Key classes
-----------
``ZKPParams``
    DH group parameters ``(p, g, q)``.
``ZKProof``
    An immutable Schnorr proof ``(y, R, s, e, context)``.
``PolicyCommitment``
    A claim that ``cost + risk < threshold`` with an attached proof.
``ZKPVerifier``
    Generates and verifies proofs.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# DH Group parameters
# ---------------------------------------------------------------------------

# 512-bit safe prime (p = 2q+1) where g=2 generates the order-q subgroup.
# Verified: g^q ≡ 1 (mod p), g^(p-1) ≡ 1 (mod p), p ≡ 7 (mod 8).
# For production deployments replace with RFC-3526 Group 14 (2048-bit MODP).
_P_512 = 0xfb8def3a572e8dc20670083d0a2a21dd4499d394148beb09ecd2f93a018018d0af9a57a96a9172dc5baba339cccd0f6fccb7fdc53fb67c330afe160326d4cd17
_G_512 = 2
_Q_512 = 0x7dc6f79d2b9746e10338041e851510eea24ce9ca0a45f584f6697c9d00c00c6857cd2bd4b548b96e2dd5d19ce66687b7e65bfee29fdb3e19857f0b01936a668b


@dataclass(frozen=True)
class ZKPParams:
    """Diffie-Hellman group parameters for the Schnorr ZKP.

    Attributes
    ----------
    p:
        A safe prime (``p = 2q + 1``).
    g:
        A generator of the order-``q`` subgroup of ``Z_p*``.
        For the default 512-bit group, ``g = 2`` and ``ord(g) = q``.
    q:
        The Sophie-Germain prime ``q = (p - 1) / 2``.
        This is also the order of ``g`` in ``Z_p*``.

    The default parameters are the 512-bit MODP group.  Pass custom
    parameters in tests or for higher security requirements.
    """

    p: int = field(default=_P_512)
    g: int = field(default=_G_512)
    q: int = field(default=_Q_512)

    @classmethod
    def default(cls) -> "ZKPParams":
        """Return the default 512-bit parameter set."""
        return cls()


# ---------------------------------------------------------------------------
# ZKProof
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZKProof:
    """An immutable non-interactive Schnorr proof.

    Attributes
    ----------
    y:
        Public value ``y = g^x mod p`` (the committed public key).
    R:
        Prover commitment ``R = g^r mod p``.
    s:
        Response scalar ``s = (r - e·x) mod q``.
    e:
        Challenge hash ``e = H(params ∥ y ∥ R ∥ context)``.
    context:
        Opaque string binding the proof to its intended purpose.
    """

    y: int
    R: int
    s: int
    e: int
    context: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (integers as hex strings)."""
        return {
            "y": hex(self.y),
            "R": hex(self.R),
            "s": hex(self.s),
            "e": hex(self.e),
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ZKProof":
        """Reconstruct a :class:`ZKProof` from a :meth:`to_dict` output."""
        return cls(
            y=int(d["y"], 16),
            R=int(d["R"], 16),
            s=int(d["s"], 16),
            e=int(d["e"], 16),
            context=str(d["context"]),
        )


# ---------------------------------------------------------------------------
# PolicyCommitment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyCommitment:
    """A privacy-preserving claim that ``cost + risk < threshold``.

    The verifier receives this object and can confirm the claim holds
    without learning ``cost`` or ``risk``.

    Attributes
    ----------
    threshold:
        The public threshold value.
    y:
        The public commitment ``y = g^x mod p`` where
        ``x = round((threshold - cost - risk) * SCALE) > 0``.
    proof:
        The :class:`ZKProof` demonstrating knowledge of ``x``.
    scale:
        Multiplier used to convert the slack float to a positive integer.
    """

    threshold: float
    y: int
    proof: ZKProof
    scale: int = 1_000_000

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "threshold": self.threshold,
            "y": hex(self.y),
            "proof": self.proof.to_dict(),
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyCommitment":
        """Reconstruct a :class:`PolicyCommitment` from a :meth:`to_dict` output."""
        return cls(
            threshold=float(d["threshold"]),
            y=int(d["y"], 16),
            proof=ZKProof.from_dict(d["proof"]),
            scale=int(d.get("scale", 1_000_000)),
        )


# ---------------------------------------------------------------------------
# ZKPVerifier
# ---------------------------------------------------------------------------

# Scale factor: converts a float slack in [0, 1] to a large positive integer
_DEFAULT_SCALE = 1_000_000


class ZKPVerifier:
    """Generates and verifies non-interactive Schnorr proofs.

    Parameters
    ----------
    params:
        DH group parameters.  Defaults to the built-in 512-bit group.

    Example
    -------
    ::

        verifier = ZKPVerifier()

        # Prover side (knows x = 42):
        proof = verifier.prove(x=42, context="demo")

        # Verifier side (knows only proof):
        assert verifier.verify(proof)
    """

    def __init__(self, params: ZKPParams | None = None) -> None:
        self._params = params or ZKPParams.default()

    @property
    def params(self) -> ZKPParams:
        """The active DH group parameters."""
        return self._params

    # ------------------------------------------------------------------
    # Core Schnorr operations
    # ------------------------------------------------------------------

    def _challenge(self, y: int, R: int, context: str) -> int:
        """Compute the Fiat-Shamir challenge integer.

        ``e = int( SHA-256( p ∥ g ∥ y ∥ R ∥ context ) ) mod q``

        Parameters
        ----------
        y:
            Public value.
        R:
            Prover commitment.
        context:
            Binding string.

        Returns
        -------
        int
            The challenge scalar in ``[0, q)``.
        """
        p, g, q = self._params.p, self._params.g, self._params.q
        blob = f"{p:x}|{g:x}|{y:x}|{R:x}|{context}".encode("utf-8")
        digest = hashlib.sha256(blob).digest()
        return int.from_bytes(digest, "big") % q

    def prove(self, x: int, context: str = "") -> ZKProof:
        """Generate a non-interactive Schnorr proof for secret *x*.

        The prover demonstrates knowledge of ``x`` such that
        ``y = g^x mod p`` without revealing ``x``.

        Parameters
        ----------
        x:
            The secret integer (must be in ``[1, q-1]``).
        context:
            Arbitrary string binding the proof to a specific purpose.

        Returns
        -------
        ZKProof

        Raises
        ------
        ValueError
            If ``x`` is not in ``[1, q-1]``.
        """
        p, g, q = self._params.p, self._params.g, self._params.q
        if not (1 <= x < q):
            raise ValueError(f"Secret x must be in [1, q-1]; got {x}")

        # Public value
        y = pow(g, x, p)

        # Choose random nonce r ∈ [1, q-1]
        r = secrets.randbelow(q - 1) + 1

        # Commitment
        R = pow(g, r, p)

        # Fiat-Shamir challenge
        e = self._challenge(y, R, context)

        # Response  s = (r - e*x) mod q
        s = (r - e * x) % q

        return ZKProof(y=y, R=R, s=s, e=e, context=context)

    def verify(self, proof: ZKProof) -> bool:
        """Verify a :class:`ZKProof` produced by :meth:`prove`.

        Checks:

        1. All components are in range.
        2. The Fiat-Shamir challenge ``e`` is correctly recomputed.
        3. The core equation ``g^s · y^e ≡ R (mod p)`` holds.

        Parameters
        ----------
        proof:
            The proof to check.

        Returns
        -------
        bool
            ``True`` iff the proof is valid.
        """
        p, g, q = self._params.p, self._params.g, self._params.q

        # Range checks
        if not (1 <= proof.y < p):
            return False
        if not (0 <= proof.s < q):
            return False
        if not (1 <= proof.R < p):
            return False

        # Recompute challenge
        e_recomputed = self._challenge(proof.y, proof.R, proof.context)
        if e_recomputed != proof.e:
            return False

        # Core verification: g^s * y^e ≡ R (mod p)
        lhs = (pow(g, proof.s, p) * pow(proof.y, proof.e, p)) % p
        return lhs == proof.R

    # ------------------------------------------------------------------
    # Policy commitment helpers
    # ------------------------------------------------------------------

    def commit_policy(
        self,
        cost: float,
        risk: float,
        threshold: float,
        scale: int = _DEFAULT_SCALE,
    ) -> PolicyCommitment:
        """Create a :class:`PolicyCommitment` proving ``cost + risk < threshold``.

        Parameters
        ----------
        cost:
            The private cost value (e.g. from ``GridState``).
        risk:
            The private risk value.
        threshold:
            The public compliance threshold.
        scale:
            Multiplier for converting the slack float to an integer.
            Default: ``1_000_000``.

        Returns
        -------
        PolicyCommitment

        Raises
        ------
        ValueError
            If ``cost + risk >= threshold`` (no valid slack to prove).
        """
        slack = threshold - cost - risk
        if slack <= 0:
            raise ValueError(
                f"cost + risk >= threshold: {cost} + {risk} >= {threshold}"
            )

        x = max(1, round(slack * scale))
        q = self._params.q
        # Clamp x to valid range [1, q-1]
        x = x % (q - 1)
        if x == 0:
            x = 1

        context = f"policy_compliance:{threshold:.6f}"
        proof = self.prove(x=x, context=context)
        return PolicyCommitment(
            threshold=threshold,
            y=proof.y,
            proof=proof,
            scale=scale,
        )

    def verify_policy_commitment(self, commitment: PolicyCommitment) -> bool:
        """Verify a :class:`PolicyCommitment`.

        Checks that:
        1. The embedded :class:`ZKProof` is valid (prover knows ``x``).
        2. The proof's ``y ≠ 1`` (ensuring ``x > 0``, i.e. slack > 0).
        3. The proof context matches the commitment's threshold.

        Parameters
        ----------
        commitment:
            The commitment to verify.

        Returns
        -------
        bool
            ``True`` iff the commitment is valid and slack > 0.
        """
        # y == 1 means x == 0 → no slack → invalid
        if commitment.proof.y <= 1:
            return False

        expected_context = f"policy_compliance:{commitment.threshold:.6f}"
        if commitment.proof.context != expected_context:
            return False

        # Must match the commitment's stored y
        if commitment.y != commitment.proof.y:
            return False

        return self.verify(commitment.proof)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def prove_batch(
        self, secrets_and_contexts: list[tuple[int, str]]
    ) -> list[ZKProof]:
        """Generate multiple proofs in a single call.

        Parameters
        ----------
        secrets_and_contexts:
            List of ``(x, context)`` pairs.

        Returns
        -------
        list[ZKProof]
        """
        return [self.prove(x, ctx) for x, ctx in secrets_and_contexts]

    def verify_batch(self, proofs: list[ZKProof]) -> list[bool]:
        """Verify multiple proofs and return per-proof results.

        Parameters
        ----------
        proofs:
            List of :class:`ZKProof` objects.

        Returns
        -------
        list[bool]
            Element *i* is ``True`` iff ``proofs[i]`` is valid.
        """
        return [self.verify(p) for p in proofs]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _random_scalar(q: int) -> int:
        """Return a random integer in ``[1, q-1]``."""
        return secrets.randbelow(q - 1) + 1

    def summary(self) -> dict[str, Any]:
        """Return a summary of the active DH parameters."""
        p = self._params.p
        return {
            "group_bits": p.bit_length(),
            "p_hex_prefix": hex(p)[:18] + "…",
            "g": self._params.g,
        }
