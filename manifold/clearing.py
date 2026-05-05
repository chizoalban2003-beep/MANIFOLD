"""Phase 32: Trust Clearinghouse — Economic Settlement for the B2B Trust Ledger.

The ``ClearingEngine`` periodically reads the ``AgentEconomyLedger`` and
calculates the **net trust debt** between all organisations using a bilateral
netting equation:

.. math::

    Net_{A \\to B} = \\sum Tax_{A \\to B} - \\sum Tax_{B \\to A}

If a node's cumulative debt exceeds ``SystemConfig.max_trust_debt``, a
``BankruptcyFreeze`` event is emitted, blocking their B2B handshake.  Nodes
that successfully identify tool failures via canary probes (Phase 31) receive
minted "Trust Tokens" as a reward.

Key classes
-----------
``SystemConfig``
    Clearinghouse system-level configuration parameters.
``SettlementEvent``
    Frozen dataclass recording a bilateral netting operation.
``BankruptcyFreeze``
    Frozen event emitted when an org's net debt exceeds the cap.
``ClearingEngine``
    Runs netting calculations, detects bankruptcy, and mints rewards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .b2b import AgentEconomyLedger


# ---------------------------------------------------------------------------
# SystemConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemConfig:
    """Clearinghouse system-level parameters.

    Attributes
    ----------
    max_trust_debt:
        Maximum net outbound debt allowed before ``BankruptcyFreeze`` is
        emitted.  Default: ``50.0``.
    mint_reward:
        Number of "Trust Tokens" minted for an org when a canary probe
        successfully identifies a tool failure.  Default: ``5.0``.
    """

    max_trust_debt: float = 50.0
    mint_reward: float = 5.0


# ---------------------------------------------------------------------------
# SettlementEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettlementEvent:
    """Immutable record of a bilateral trust netting operation.

    Attributes
    ----------
    timestamp:
        POSIX timestamp when the settlement was calculated.
    from_org:
        Debtor organisation ID (net payer).
    to_org:
        Creditor organisation ID (net receiver).
    gross_forward:
        Sum of all ``from_org → to_org`` call costs.
    gross_reverse:
        Sum of all ``to_org → from_org`` call costs.
    net_amount:
        ``gross_forward - gross_reverse`` — positive means ``from_org`` owes
        ``to_org`` this amount after netting.
    settled:
        ``True`` if the net amount was within the bankruptcy threshold.
    """

    timestamp: float
    from_org: str
    to_org: str
    gross_forward: float
    gross_reverse: float
    net_amount: float
    settled: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "timestamp": self.timestamp,
            "from_org": self.from_org,
            "to_org": self.to_org,
            "gross_forward": self.gross_forward,
            "gross_reverse": self.gross_reverse,
            "net_amount": self.net_amount,
            "settled": self.settled,
        }


# ---------------------------------------------------------------------------
# BankruptcyFreeze
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankruptcyFreeze:
    """Event emitted when an org's net outbound debt exceeds the cap.

    Attributes
    ----------
    org_id:
        The organisation whose debt exceeded the threshold.
    debt:
        The total net outbound debt at the time of the freeze.
    timestamp:
        POSIX timestamp of the freeze decision.
    """

    org_id: str
    debt: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "org_id": self.org_id,
            "debt": self.debt,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# ClearingEngine
# ---------------------------------------------------------------------------


@dataclass
class ClearingEngine:
    """Economic settlement engine for the B2B Trust Ledger.

    Reads all :class:`~manifold.b2b.EconomyEntry` records from the provided
    ``AgentEconomyLedger`` and computes bilateral netting between every pair
    of organisations.

    Parameters
    ----------
    ledger:
        The ``AgentEconomyLedger`` containing all cross-org call records.
    config:
        System configuration controlling debt caps and mint rewards.
    clock:
        Callable returning the current POSIX timestamp.  Override in tests.

    Example
    -------
    ::

        engine = ClearingEngine(ledger=router.ledger)
        events = engine.settle()
        freezes = engine.check_bankruptcy()
    """

    ledger: AgentEconomyLedger
    config: SystemConfig = field(default_factory=SystemConfig)
    clock: Any = None  # callable | None

    # Internal mutable state
    _trust_balances: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _settlements: list[SettlementEvent] = field(
        default_factory=list, init=False, repr=False
    )
    _freezes: list[BankruptcyFreeze] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.clock is None:
            self.clock = time.time

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self) -> list[SettlementEvent]:
        """Run a netting pass over all ledger entries.

        For each ordered org pair ``(A, B)`` with ``A < B`` (lexicographic),
        calculates:

        * ``gross_forward`` = sum of all ``A → B`` call costs
        * ``gross_reverse`` = sum of all ``B → A`` call costs
        * ``net_amount``    = ``gross_forward - gross_reverse``

        A :class:`SettlementEvent` is emitted for the pair only if any
        cross-org activity exists.

        Returns
        -------
        list[SettlementEvent]
            Newly generated settlement events (may be empty).
        """
        # Build a flow matrix: flows[(from_org, to_org)] = total cost
        flows: dict[tuple[str, str], float] = {}
        for entry in self.ledger.entries():
            key = (entry.local_org_id, entry.remote_org_id)
            flows[key] = flows.get(key, 0.0) + entry.net_trust_cost

        # Collect all unique org pairs
        all_orgs: set[str] = set()
        for (a, b) in flows:
            all_orgs.add(a)
            all_orgs.add(b)

        sorted_orgs = sorted(all_orgs)
        new_events: list[SettlementEvent] = []
        now: float = self.clock()  # type: ignore[operator]

        for i, org_a in enumerate(sorted_orgs):
            for org_b in sorted_orgs[i + 1:]:
                forward = flows.get((org_a, org_b), 0.0)
                reverse = flows.get((org_b, org_a), 0.0)
                if forward == 0.0 and reverse == 0.0:
                    continue
                net = forward - reverse
                settled = abs(net) <= self.config.max_trust_debt
                event = SettlementEvent(
                    timestamp=now,
                    from_org=org_a,
                    to_org=org_b,
                    gross_forward=forward,
                    gross_reverse=reverse,
                    net_amount=net,
                    settled=settled,
                )
                new_events.append(event)
                self._settlements.append(event)

        return new_events

    # ------------------------------------------------------------------
    # Bankruptcy detection
    # ------------------------------------------------------------------

    def check_bankruptcy(self) -> list[BankruptcyFreeze]:
        """Emit a :class:`BankruptcyFreeze` for each org whose total net
        outbound debt exceeds ``config.max_trust_debt``.

        The outbound debt for org ``A`` is computed as:

        ``sum(entry.net_trust_cost for entry where local_org_id == A)``

        Returns
        -------
        list[BankruptcyFreeze]
            Newly generated freeze events (may be empty).
        """
        # Accumulate outbound costs per org
        outbound: dict[str, float] = {}
        for entry in self.ledger.entries():
            org = entry.local_org_id
            outbound[org] = outbound.get(org, 0.0) + entry.net_trust_cost

        now: float = self.clock()  # type: ignore[operator]
        new_freezes: list[BankruptcyFreeze] = []
        for org, debt in outbound.items():
            if debt > self.config.max_trust_debt:
                freeze = BankruptcyFreeze(org_id=org, debt=debt, timestamp=now)
                new_freezes.append(freeze)
                self._freezes.append(freeze)

        return new_freezes

    # ------------------------------------------------------------------
    # Trust token minting
    # ------------------------------------------------------------------

    def mint_for_canary_success(self, org_id: str) -> float:
        """Add ``config.mint_reward`` trust tokens to *org_id*'s balance.

        Called when a canary probe (Phase 31) run by *org_id* successfully
        identifies a tool failure.

        Parameters
        ----------
        org_id:
            Organisation that identified the failing tool.

        Returns
        -------
        float
            New cumulative trust balance for *org_id*.
        """
        current = self._trust_balances.get(org_id, 0.0)
        new_balance = current + self.config.mint_reward
        self._trust_balances[org_id] = new_balance
        return new_balance

    # ------------------------------------------------------------------
    # Net debt helpers
    # ------------------------------------------------------------------

    def net_debt(self, org_id: str) -> float:
        """Return the net debt for *org_id*.

        Defined as ``outbound_cost - inbound_cost`` across all ledger entries.
        A positive value means the org has paid more than it has received.

        Parameters
        ----------
        org_id:
            Organisation identifier.

        Returns
        -------
        float
            Net debt (positive = net payer, negative = net receiver).
        """
        outbound = sum(
            e.net_trust_cost
            for e in self.ledger.entries()
            if e.local_org_id == org_id
        )
        inbound = sum(
            e.net_trust_cost
            for e in self.ledger.entries()
            if e.remote_org_id == org_id
        )
        return outbound - inbound

    def all_net_debts(self) -> dict[str, float]:
        """Return ``{org_id: net_debt}`` for every org in the ledger.

        Returns
        -------
        dict[str, float]
        """
        orgs: set[str] = set()
        for entry in self.ledger.entries():
            orgs.add(entry.local_org_id)
            orgs.add(entry.remote_org_id)
        return {org: self.net_debt(org) for org in sorted(orgs)}

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def trust_balance(self, org_id: str) -> float:
        """Return the current minted trust-token balance for *org_id*.

        Returns ``0.0`` if no tokens have been minted yet.

        Parameters
        ----------
        org_id:
            Organisation identifier.

        Returns
        -------
        float
        """
        return self._trust_balances.get(org_id, 0.0)

    def settlements(self) -> list[SettlementEvent]:
        """Return all recorded :class:`SettlementEvent` objects."""
        return list(self._settlements)

    def freezes(self) -> list[BankruptcyFreeze]:
        """Return all recorded :class:`BankruptcyFreeze` events."""
        return list(self._freezes)

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of clearinghouse activity.

        Returns
        -------
        dict
            Keys: ``total_settlements``, ``total_freezes``,
            ``net_debts``, ``trust_balances``.
        """
        return {
            "total_settlements": len(self._settlements),
            "total_freezes": len(self._freezes),
            "net_debts": self.all_net_debts(),
            "trust_balances": dict(self._trust_balances),
        }
