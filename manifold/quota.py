"""Phase 30: Trust-Based Rate Limiting — Dynamic Reputation-Aware Traffic Shaping.

Implements the Token Bucket algorithm with a refill rate that is mathematically
tied to the tool's live trust score from the ``ReputationHub``.  Tools with
high entropy (Phase 26) or low reliability suffer severe throttling.
Probationary tools (Phase 28) receive a hard-capped burst limit.

Key classes
-----------
``QuotaExhaustedError``
    Raised when a tool or org has insufficient tokens.
``TrustTokenBucket``
    Per-entity token bucket with reputation-aware refill.
``QuotaManager``
    Registry of ``TrustTokenBucket`` instances; integrates with
    ``ReputationHub`` and ``B2BRouter``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# QuotaExhaustedError
# ---------------------------------------------------------------------------


class QuotaExhaustedError(RuntimeError):
    """Raised when a tool or org has no tokens in their quota bucket.

    Attributes
    ----------
    entity_id:
        The tool or org identifier that was throttled.
    available:
        Number of tokens currently available (< requested amount).
    capacity:
        Maximum capacity of the bucket.
    """

    def __init__(self, entity_id: str, available: float, capacity: float) -> None:
        super().__init__(
            f"Quota exhausted for {entity_id!r}: "
            f"{available:.3f}/{capacity:.3f} tokens available"
        )
        self.entity_id = entity_id
        self.available = available
        self.capacity = capacity


# ---------------------------------------------------------------------------
# TrustTokenBucket
# ---------------------------------------------------------------------------


@dataclass
class TrustTokenBucket:
    """Token bucket with reputation-aware refill rate.

    The Token Bucket algorithm: tokens accumulate at a rate up to *capacity*.
    Each call consumes one (or more) tokens.  The effective refill rate is
    penalised by:

    * **Low trust score** — multiplied linearly (0 trust → near-zero rate).
    * **High entropy** — a high Phase 26 entropy score further slows refill,
      down to 10 % of the nominal rate at entropy = 1.0.

    Parameters
    ----------
    entity_id:
        Unique identifier (tool name or org ID).
    capacity:
        Maximum number of tokens the bucket can hold.
    refill_rate:
        Nominal token refill rate (tokens per second) at full trust.
    """

    entity_id: str
    capacity: float = 10.0
    refill_rate: float = 1.0  # tokens/second at full trust

    _tokens: float = field(init=False, repr=False)
    _last_refill: float = field(init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last_refill = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tokens(self) -> float:
        """Return the current number of available tokens (snapshot; no refill)."""
        return self._tokens

    def refill(self, trust_score: float = 1.0, entropy: float = 0.0) -> None:
        """Refill the bucket based on elapsed time and effective rate.

        Parameters
        ----------
        trust_score:
            Current reliability/trust score [0, 1].
        entropy:
            Current entropy score [0, 1] from Phase 26.
        """
        now = time.time()
        with self._lock:
            elapsed = max(0.0, now - self._last_refill)
            rate = self._effective_refill_rate(trust_score, entropy)
            self._tokens = min(self.capacity, self._tokens + elapsed * rate)
            self._last_refill = now

    def consume(self, amount: float = 1.0) -> bool:
        """Consume *amount* tokens.

        Parameters
        ----------
        amount:
            Number of tokens to consume.

        Returns
        -------
        bool
            ``True`` if the tokens were available and consumed, ``False`` if
            the bucket did not have enough tokens (no tokens are consumed).
        """
        with self._lock:
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

    def available_tokens(self) -> float:
        """Return the current token level."""
        return self._tokens

    def utilisation(self) -> float:
        """Return the fraction of capacity currently used [0, 1]."""
        if self.capacity <= 0:
            return 1.0
        used = self.capacity - self._tokens
        return max(0.0, min(1.0, used / self.capacity))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effective_refill_rate(self, trust_score: float, entropy: float) -> float:
        """Compute the penalised refill rate.

        Parameters
        ----------
        trust_score:
            [0, 1] — higher trust → faster refill.
        entropy:
            [0, 1] — higher entropy → slower refill (down to 10 % of nominal
            at entropy = 1.0).

        Returns
        -------
        float
            Actual refill rate, always ≥ 0.01 tokens/second.
        """
        trust_clamped = max(0.0, min(1.0, trust_score))
        entropy_clamped = max(0.0, min(1.0, entropy))
        # entropy penalty: linearly maps [0, 1] → [1.0, 0.1]
        entropy_penalty = 1.0 - entropy_clamped * 0.9
        return max(0.01, self.refill_rate * entropy_penalty * trust_clamped)


# ---------------------------------------------------------------------------
# QuotaManager
# ---------------------------------------------------------------------------


@dataclass
class QuotaManager:
    """Registry of :class:`TrustTokenBucket` instances per tool or org.

    Integrates with :class:`~manifold.hub.ReputationHub` to automatically
    derive trust and entropy scores when refilling buckets.

    Parameters
    ----------
    hub:
        Optional ``ReputationHub`` for live trust/entropy data.  When
        ``None``, trust defaults to ``1.0`` and entropy to ``0.0``.
    default_capacity:
        Nominal bucket capacity for standard (non-probationary) entities.
    default_refill_rate:
        Nominal token refill rate (tokens/second) for standard entities.
    probationary_burst_limit:
        Hard-capped bucket capacity for probationary tools (Phase 28).

    Example
    -------
    ::

        qm = QuotaManager(hub=hub, default_capacity=10.0)
        qm.register("gpt-4o")
        qm.check_and_consume("gpt-4o")  # raises QuotaExhaustedError if empty
    """

    hub: Any = None  # ReputationHub | None — avoid circular import at module level
    default_capacity: float = 10.0
    default_refill_rate: float = 1.0
    probationary_burst_limit: float = 3.0

    _buckets: dict[str, TrustTokenBucket] = field(
        default_factory=dict, init=False, repr=False
    )
    _probationary: set[str] = field(default_factory=set, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        entity_id: str,
        *,
        probationary: bool = False,
        capacity: float | None = None,
        refill_rate: float | None = None,
    ) -> TrustTokenBucket:
        """Register *entity_id* and return its :class:`TrustTokenBucket`.

        If already registered, returns the existing bucket.

        Parameters
        ----------
        entity_id:
            Unique tool or org identifier.
        probationary:
            When ``True``, cap capacity at ``probationary_burst_limit``.
        capacity:
            Override the default capacity.
        refill_rate:
            Override the default refill rate.
        """
        with self._lock:
            if entity_id in self._buckets:
                if probationary:
                    self._probationary.add(entity_id)
                return self._buckets[entity_id]
            effective_capacity = (
                self.probationary_burst_limit
                if probationary
                else (capacity if capacity is not None else self.default_capacity)
            )
            effective_rate = refill_rate if refill_rate is not None else self.default_refill_rate
            bucket = TrustTokenBucket(
                entity_id=entity_id,
                capacity=effective_capacity,
                refill_rate=effective_rate,
            )
            self._buckets[entity_id] = bucket
            if probationary:
                self._probationary.add(entity_id)
            return bucket

    def mark_probationary(self, entity_id: str) -> None:
        """Mark an already-registered entity as probationary and cap its bucket.

        Parameters
        ----------
        entity_id:
            The entity to demote to probationary status.
        """
        with self._lock:
            self._probationary.add(entity_id)
            bucket = self._buckets.get(entity_id)
            if bucket is not None and bucket.capacity > self.probationary_burst_limit:
                # Rebuild with capped capacity; preserve token level
                new_tokens = min(bucket.tokens, self.probationary_burst_limit)
                bucket.capacity = self.probationary_burst_limit
                bucket._tokens = new_tokens  # noqa: SLF001

    # ------------------------------------------------------------------
    # Core consume / refill
    # ------------------------------------------------------------------

    def check_and_consume(
        self,
        entity_id: str,
        *,
        amount: float = 1.0,
    ) -> None:
        """Refill then attempt to consume *amount* tokens.

        Raises :class:`QuotaExhaustedError` if the bucket is empty.

        Parameters
        ----------
        entity_id:
            The tool or org to check.
        amount:
            Number of tokens to consume.

        Raises
        ------
        QuotaExhaustedError
            If the entity has insufficient tokens after refill.
        """
        bucket = self._get_or_create(entity_id)
        trust, entropy = self._trust_and_entropy(entity_id)
        bucket.refill(trust_score=trust, entropy=entropy)
        if not bucket.consume(amount):
            raise QuotaExhaustedError(
                entity_id, bucket.available_tokens(), bucket.capacity
            )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def is_probationary(self, entity_id: str) -> bool:
        """Return ``True`` if *entity_id* is in probationary status."""
        return entity_id in self._probationary

    def bucket(self, entity_id: str) -> TrustTokenBucket | None:
        """Return the bucket for *entity_id*, or ``None`` if not registered."""
        return self._buckets.get(entity_id)

    def quota_summary(self) -> dict[str, dict[str, float]]:
        """Return ``{entity_id: {tokens, capacity, utilisation}}`` for all entities."""
        return {
            eid: {
                "tokens": b.tokens,
                "capacity": b.capacity,
                "utilisation": b.utilisation(),
            }
            for eid, b in self._buckets.items()
        }

    def all_entity_ids(self) -> list[str]:
        """Return a sorted list of all registered entity IDs."""
        return sorted(self._buckets)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, entity_id: str) -> TrustTokenBucket:
        """Return the bucket for *entity_id*, creating one if absent."""
        # Check without lock first (fast path)
        if entity_id in self._buckets:
            return self._buckets[entity_id]
        # Create without holding the lock to avoid deadlock with register()
        is_prob = entity_id in self._probationary
        effective_capacity = (
            self.probationary_burst_limit if is_prob else self.default_capacity
        )
        new_bucket = TrustTokenBucket(
            entity_id=entity_id,
            capacity=effective_capacity,
            refill_rate=self.default_refill_rate,
        )
        with self._lock:
            # Double-check after acquiring lock
            if entity_id not in self._buckets:
                self._buckets[entity_id] = new_bucket
        return self._buckets[entity_id]

    def _trust_and_entropy(self, entity_id: str) -> tuple[float, float]:
        """Return ``(trust_score, entropy)`` from the hub for *entity_id*."""
        if self.hub is None:
            return 1.0, 0.0
        try:
            rel = self.hub.live_reliability(entity_id)
            trust = rel if rel is not None else 0.5
            entropy = self.hub.tool_entropy(entity_id)
        except Exception:  # noqa: BLE001
            return 0.5, 0.0
        return trust, entropy
