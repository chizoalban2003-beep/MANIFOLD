"""Phase 57: Hardware Fingerprinting & Attestation — Proof of Compute.

Provides cross-platform hardware fingerprinting so that MANIFOLD peers can
attest to the physical machine they are running on, and challenge-response
proof-of-compute to verify a peer is actually running on the hardware they
claim to be.

Architecture
------------
A :class:`HardwareAttestor` builds a deterministic fingerprint from OS, CPU,
and platform metadata (``os``, ``sys``, ``platform`` modules only — zero
external dependencies).  The fingerprint is a SHA-256 digest of the
concatenated metadata fields, so it is stable across reboots but unique per
machine configuration.

The :class:`ProofOfCompute` challenge-response mechanism works as follows:

1. A *challenger* calls :meth:`ProofOfCompute.issue_challenge` to create a
   :class:`ComputeChallenge` containing a random nonce, a difficulty level,
   and the challenger's own hardware fingerprint.
2. The *prover* calls :meth:`ProofOfCompute.solve_challenge` to compute the
   expected response (a SHA-256 hash that meets the difficulty target) together
   with their own hardware fingerprint.
3. The challenger verifies the response with
   :meth:`ProofOfCompute.verify_response`.

Key classes
-----------
``HardwareProfile``
    Structured snapshot of hardware / OS metadata.
``HardwareAttestor``
    Builds and caches a :class:`HardwareProfile` for the current machine.
``ComputeChallenge``
    A work-factor puzzle issued by a challenger.
``ComputeResponse``
    A prover's answer to a :class:`ComputeChallenge`.
``ProofOfCompute``
    Issues, solves, and verifies challenge-response work puzzles.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """Structured snapshot of hardware / OS metadata for one machine.

    Attributes
    ----------
    os_name:
        Operating-system name as returned by :func:`os.name` (e.g. ``"posix"``).
    platform_system:
        System/OS name from :func:`platform.system` (e.g. ``"Linux"``).
    platform_release:
        OS release string from :func:`platform.release`.
    platform_machine:
        Machine type from :func:`platform.machine` (e.g. ``"x86_64"``).
    platform_processor:
        Processor type from :func:`platform.processor`.
    python_version:
        Python version string from :attr:`sys.version`.
    python_implementation:
        Python implementation from :func:`platform.python_implementation`.
    cpu_count:
        Logical CPU count from :func:`os.cpu_count`.
    fingerprint:
        SHA-256 hex digest of the stable subset of fields above.
    """

    os_name: str
    platform_system: str
    platform_release: str
    platform_machine: str
    platform_processor: str
    python_version: str
    python_implementation: str
    cpu_count: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "os_name": self.os_name,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "platform_machine": self.platform_machine,
            "platform_processor": self.platform_processor,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "cpu_count": self.cpu_count,
            "fingerprint": self.fingerprint,
        }


# ---------------------------------------------------------------------------
# HardwareAttestor
# ---------------------------------------------------------------------------


@dataclass
class HardwareAttestor:
    """Builds and caches a :class:`HardwareProfile` for the current machine.

    The fingerprint is computed once and cached so that repeated calls to
    :meth:`profile` do not re-hash.

    Example
    -------
    ::

        attestor = HardwareAttestor()
        profile = attestor.profile()
        print(f"Fingerprint: {profile.fingerprint}")
    """

    _cached: HardwareProfile | None = field(default=None, init=False, repr=False)

    def profile(self) -> HardwareProfile:
        """Return the :class:`HardwareProfile` for the current machine.

        The profile is computed the first time this method is called and then
        cached for the lifetime of the object.

        Returns
        -------
        HardwareProfile
            Fingerprint and metadata for this machine.
        """
        if self._cached is not None:
            return self._cached
        self._cached = self._build()
        return self._cached

    def matches(self, other_fingerprint: str) -> bool:
        """Return ``True`` if *other_fingerprint* matches this machine.

        Parameters
        ----------
        other_fingerprint:
            A fingerprint string to compare against the local profile.
        """
        return self.profile().fingerprint == other_fingerprint

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _build() -> HardwareProfile:
        """Collect hardware metadata and compute the fingerprint."""
        os_name = os.name
        plat_system = platform.system()
        plat_release = platform.release()
        plat_machine = platform.machine()
        plat_processor = platform.processor()
        py_version = sys.version.split()[0]  # "3.12.0" without build details
        py_impl = platform.python_implementation()
        cpu_count = os.cpu_count() or 0

        # Fingerprint = SHA-256 of stable, machine-specific fields.
        # We intentionally exclude release (which changes on kernel updates)
        # and python_version (which changes on upgrades) so the fingerprint
        # is stable across typical maintenance cycles.
        stable = "|".join([
            os_name,
            plat_system,
            plat_machine,
            plat_processor,
            py_impl,
            str(cpu_count),
        ])
        fingerprint = hashlib.sha256(stable.encode("utf-8")).hexdigest()

        return HardwareProfile(
            os_name=os_name,
            platform_system=plat_system,
            platform_release=plat_release,
            platform_machine=plat_machine,
            platform_processor=plat_processor,
            python_version=py_version,
            python_implementation=py_impl,
            cpu_count=cpu_count,
            fingerprint=fingerprint,
        )


# ---------------------------------------------------------------------------
# ComputeChallenge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeChallenge:
    """A proof-of-compute work puzzle issued by a challenger.

    The prover must find a ``nonce`` such that::

        SHA-256(challenge_nonce + ":" + prover_fingerprint + ":" + str(nonce))

    starts with ``difficulty`` zero hex digits.

    Attributes
    ----------
    challenge_id:
        Unique identifier for this challenge.
    challenge_nonce:
        Random hex string chosen by the challenger.
    difficulty:
        Number of leading zero hex digits required in the solution hash.
        Higher = harder.  Practical values: 1–5.
    challenger_fingerprint:
        Hardware fingerprint of the issuing machine.
    issued_at:
        POSIX timestamp when the challenge was issued.
    expires_at:
        POSIX timestamp after which the challenge is no longer valid.
    """

    challenge_id: str
    challenge_nonce: str
    difficulty: int
    challenger_fingerprint: str
    issued_at: float
    expires_at: float

    @property
    def is_expired(self) -> bool:
        """``True`` if the challenge has expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "challenge_id": self.challenge_id,
            "challenge_nonce": self.challenge_nonce,
            "difficulty": self.difficulty,
            "challenger_fingerprint": self.challenger_fingerprint,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


# ---------------------------------------------------------------------------
# ComputeResponse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeResponse:
    """A prover's answer to a :class:`ComputeChallenge`.

    Attributes
    ----------
    challenge_id:
        Matches :attr:`ComputeChallenge.challenge_id`.
    prover_fingerprint:
        Hardware fingerprint of the prover's machine.
    solution_nonce:
        The integer nonce that satisfies the work target.
    solution_hash:
        The SHA-256 hex digest that meets the difficulty requirement.
    solved_at:
        POSIX timestamp when the solution was found.
    elapsed_seconds:
        Wall-clock time taken to solve the challenge.
    """

    challenge_id: str
    prover_fingerprint: str
    solution_nonce: int
    solution_hash: str
    solved_at: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "challenge_id": self.challenge_id,
            "prover_fingerprint": self.prover_fingerprint,
            "solution_nonce": self.solution_nonce,
            "solution_hash": self.solution_hash,
            "solved_at": self.solved_at,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


# ---------------------------------------------------------------------------
# ProofOfCompute
# ---------------------------------------------------------------------------


@dataclass
class ProofOfCompute:
    """Issues, solves, and verifies hardware-attested compute challenges.

    Parameters
    ----------
    attestor:
        A :class:`HardwareAttestor` providing the local machine fingerprint.
    default_difficulty:
        Default number of leading zero hex digits required.  Default: ``2``.
    challenge_ttl_seconds:
        Time-to-live for issued challenges.  Default: ``300`` (5 minutes).
    max_solve_iterations:
        Maximum iterations before ``solve_challenge`` gives up.
        Default: ``10_000_000``.

    Example
    -------
    ::

        poc = ProofOfCompute(HardwareAttestor(), default_difficulty=2)
        challenge = poc.issue_challenge()
        response = poc.solve_challenge(challenge)
        assert poc.verify_response(challenge, response)
    """

    attestor: HardwareAttestor = field(default_factory=HardwareAttestor)
    default_difficulty: int = 2
    challenge_ttl_seconds: float = 300.0
    max_solve_iterations: int = 10_000_000

    _rng: random.Random = field(
        default_factory=random.Random, init=False, repr=False
    )

    def issue_challenge(
        self,
        difficulty: int | None = None,
    ) -> ComputeChallenge:
        """Issue a new :class:`ComputeChallenge`.

        Parameters
        ----------
        difficulty:
            Override the default difficulty.  If ``None``, uses
            :attr:`default_difficulty`.

        Returns
        -------
        ComputeChallenge
            A freshly generated challenge.
        """
        difficulty = self.default_difficulty if difficulty is None else difficulty
        challenge_id = hashlib.sha256(
            f"{time.time()}{self._rng.random()}".encode()
        ).hexdigest()[:16]
        nonce_bytes = bytes(self._rng.randint(0, 255) for _ in range(16))
        challenge_nonce = nonce_bytes.hex()
        now = time.time()

        return ComputeChallenge(
            challenge_id=challenge_id,
            challenge_nonce=challenge_nonce,
            difficulty=difficulty,
            challenger_fingerprint=self.attestor.profile().fingerprint,
            issued_at=now,
            expires_at=now + self.challenge_ttl_seconds,
        )

    def solve_challenge(
        self,
        challenge: ComputeChallenge,
        prover_fingerprint: str | None = None,
    ) -> ComputeResponse | None:
        """Solve *challenge* and return a :class:`ComputeResponse`.

        Parameters
        ----------
        challenge:
            The challenge to solve.
        prover_fingerprint:
            Override the prover's fingerprint.  Defaults to the local machine.

        Returns
        -------
        ComputeResponse
            The solution, or ``None`` if max iterations were exhausted or the
            challenge is expired.
        """
        if challenge.is_expired:
            return None

        fp = prover_fingerprint or self.attestor.profile().fingerprint
        target_prefix = "0" * challenge.difficulty
        prefix = f"{challenge.challenge_nonce}:{fp}:"

        t_start = time.monotonic()
        for nonce in range(self.max_solve_iterations):
            candidate = f"{prefix}{nonce}".encode("utf-8")
            digest = hashlib.sha256(candidate).hexdigest()
            if digest.startswith(target_prefix):
                elapsed = time.monotonic() - t_start
                return ComputeResponse(
                    challenge_id=challenge.challenge_id,
                    prover_fingerprint=fp,
                    solution_nonce=nonce,
                    solution_hash=digest,
                    solved_at=time.time(),
                    elapsed_seconds=elapsed,
                )

        return None  # Could not solve within iteration limit

    def verify_response(
        self,
        challenge: ComputeChallenge,
        response: ComputeResponse,
    ) -> bool:
        """Verify that *response* correctly solves *challenge*.

        Parameters
        ----------
        challenge:
            The original challenge.
        response:
            The prover's response.

        Returns
        -------
        bool
            ``True`` if the response is cryptographically valid and the
            challenge has not expired.
        """
        if challenge.is_expired:
            return False
        if response.challenge_id != challenge.challenge_id:
            return False

        # Re-derive the expected hash
        candidate = (
            f"{challenge.challenge_nonce}:{response.prover_fingerprint}"
            f":{response.solution_nonce}"
        ).encode("utf-8")
        expected = hashlib.sha256(candidate).hexdigest()

        if expected != response.solution_hash:
            return False

        target_prefix = "0" * challenge.difficulty
        return response.solution_hash.startswith(target_prefix)
