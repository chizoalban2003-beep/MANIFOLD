"""Phase 24: The Immutable Vault — Zero-Dependency Persistence Layer.

``ManifoldVault`` provides durable, file-backed storage for MANIFOLD's
runtime state.  It satisfies two requirements:

1. **Write-Ahead Log (WAL)** — every :class:`~manifold.federation.FederatedGossipPacket`
   (Phase 10) and :class:`~manifold.b2b.EconomyEntry` (Phase 20) is
   *immediately* appended to a ``.jsonl`` file so that no event is lost on a
   server restart.

2. **State Recovery** — on server startup, :meth:`ManifoldVault.load_state`
   replays the WAL to repopulate the :class:`~manifold.hub.ReputationHub` and
   :class:`~manifold.b2b.AgentEconomyLedger`.

Design constraints
------------------
* **Zero external dependencies** — uses only :mod:`json`, :mod:`os`,
  :mod:`pathlib`, and :mod:`threading` from the standard library.
* **Thread-safe** — all public write operations are guarded by a
  :class:`threading.Lock`.
* **Append-only WAL** — records are never modified after they are written;
  recovery replays them in order.

Key classes
-----------
``ManifoldVault``
    File-backed WAL + state recovery for gossip packets and economy entries.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .b2b import AgentEconomyLedger, EconomyEntry
from .federation import FederatedGossipPacket
from .hub import ReputationHub

# Accepted signal values for FederatedGossipPacket
_VALID_SIGNALS = frozenset({"failing", "healthy", "degraded"})


# ---------------------------------------------------------------------------
# Record-type tags written to the WAL
# ---------------------------------------------------------------------------

_TAG_GOSSIP = "gossip"
_TAG_ECONOMY = "economy"
_TAG_VOLATILITY = "volatility"
_TAG_PROBATIONARY = "probationary"
_TAG_PROVENANCE = "provenance"
_TAG_TOKEN_BUCKET = "token_bucket"
_TAG_SETTLEMENT = "settlement"


# ---------------------------------------------------------------------------
# ManifoldVault
# ---------------------------------------------------------------------------


@dataclass
class ManifoldVault:
    """File-backed Write-Ahead Log and state-recovery layer.

    Parameters
    ----------
    data_dir:
        Directory where WAL files will be stored.  Created automatically if
        it does not exist.  Defaults to the current working directory.
    gossip_log:
        Filename for the gossip-packet WAL (relative to *data_dir*).
        Default: ``"gossip.jsonl"``.
    economy_log:
        Filename for the economy-entry WAL (relative to *data_dir*).
        Default: ``"economy.jsonl"``.

    Example
    -------
    ::

        vault = ManifoldVault(data_dir="/var/manifold")
        vault.append_gossip(packet)
        vault.append_economy(entry)

        hub = ReputationHub()
        ledger = AgentEconomyLedger()
        vault.load_state(hub=hub, ledger=ledger)
    """

    data_dir: str | os.PathLike[str] = field(default_factory=os.getcwd)
    gossip_log: str = "gossip.jsonl"
    economy_log: str = "economy.jsonl"
    volatility_log: str = "volatility.jsonl"
    probationary_log: str = "probationary.jsonl"
    provenance_log: str = "provenance.jsonl"
    token_bucket_log: str = "token_buckets.jsonl"
    settlements_log: str = "settlements.jsonl"

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write-Ahead Log helpers
    # ------------------------------------------------------------------

    @property
    def _gossip_path(self) -> Path:
        return Path(self.data_dir) / self.gossip_log

    @property
    def _economy_path(self) -> Path:
        return Path(self.data_dir) / self.economy_log

    @property
    def _volatility_path(self) -> Path:
        return Path(self.data_dir) / self.volatility_log

    @property
    def _probationary_path(self) -> Path:
        return Path(self.data_dir) / self.probationary_log

    @property
    def _provenance_path(self) -> Path:
        return Path(self.data_dir) / self.provenance_log

    @property
    def _token_bucket_path(self) -> Path:
        return Path(self.data_dir) / self.token_bucket_log

    @property
    def _settlements_path(self) -> Path:
        return Path(self.data_dir) / self.settlements_log

    def _append_line(self, path: Path, record: dict[str, Any]) -> None:
        """Append a single JSON record to *path* (thread-safe)."""
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def append_gossip(self, packet: FederatedGossipPacket) -> None:
        """Append a :class:`~manifold.federation.FederatedGossipPacket` to the gossip WAL.

        Parameters
        ----------
        packet:
            The gossip packet to persist.
        """
        record: dict[str, Any] = {
            "_type": _TAG_GOSSIP,
            "tool_name": packet.tool_name,
            "signal": packet.signal,
            "confidence": packet.confidence,
            "org_id": packet.org_id,
            "weight": packet.weight,
        }
        self._append_line(self._gossip_path, record)

    def append_economy(self, entry: EconomyEntry) -> None:
        """Append an :class:`~manifold.b2b.EconomyEntry` to the economy WAL.

        Parameters
        ----------
        entry:
            The economy entry to persist.
        """
        record: dict[str, Any] = {
            "_type": _TAG_ECONOMY,
            "local_org_id": entry.local_org_id,
            "remote_org_id": entry.remote_org_id,
            "allowed": entry.allowed,
            "reputation_score": entry.reputation_score,
            "surcharge": entry.surcharge,
            "net_trust_cost": entry.net_trust_cost,
            "block_reason": entry.block_reason,
        }
        self._append_line(self._economy_path, record)

    def append_volatility(self, domain: str, lambda_value: float) -> None:
        """Persist a Volatility Coefficient override to the WAL (Phase 26).

        Parameters
        ----------
        domain:
            The domain whose λ (decay rate) is being persisted.
        lambda_value:
            The λ value to store.
        """
        record: dict[str, Any] = {
            "_type": _TAG_VOLATILITY,
            "domain": domain,
            "lambda": lambda_value,
        }
        self._append_line(self._volatility_path, record)

    def append_probationary(
        self,
        tool_name: str,
        *,
        original_reliability: float,
        successful_outcomes: int,
        graduated: bool,
    ) -> None:
        """Persist a Probationary tool state snapshot to the WAL (Phase 28).

        Parameters
        ----------
        tool_name:
            The probationary tool's name.
        original_reliability:
            The un-penalised reliability score.
        successful_outcomes:
            Number of successful outcomes recorded so far.
        graduated:
            Whether the tool has graduated from probationary status.
        """
        record: dict[str, Any] = {
            "_type": _TAG_PROBATIONARY,
            "tool_name": tool_name,
            "original_reliability": original_reliability,
            "successful_outcomes": successful_outcomes,
            "graduated": graduated,
        }
        self._append_line(self._probationary_path, record)

    def append_provenance(
        self,
        task_id: str,
        final_decision: str,
        *,
        timestamp: float,
        grid_state_summary: dict[str, Any] | None = None,
        braintrust_votes: list[dict[str, Any]] | None = None,
        policy_hash: str = "",
        receipt_hash: str = "",
        previous_hash: str = "",
    ) -> None:
        """Append a :class:`~manifold.provenance.DecisionReceipt` to the provenance WAL (Phase 29).

        Parameters
        ----------
        task_id:
            Unique decision identifier.
        final_decision:
            The action that was taken.
        timestamp:
            POSIX timestamp of the decision.
        grid_state_summary:
            Lightweight brain-decision summary dict.
        braintrust_votes:
            List of per-genome vote dicts (or empty).
        policy_hash:
            SHA-256 hex digest of the active policy.
        receipt_hash:
            SHA-256 hex digest of the full receipt.
        previous_hash:
            Hash of the preceding receipt in the Merkle chain.
        """
        record: dict[str, Any] = {
            "_type": _TAG_PROVENANCE,
            "task_id": task_id,
            "final_decision": final_decision,
            "timestamp": timestamp,
            "grid_state_summary": grid_state_summary or {},
            "braintrust_votes": braintrust_votes or [],
            "policy_hash": policy_hash,
            "receipt_hash": receipt_hash,
            "previous_hash": previous_hash,
        }
        self._append_line(self._provenance_path, record)

    def append_token_bucket(
        self,
        entity_id: str,
        *,
        tokens: float,
        capacity: float,
        probationary: bool = False,
    ) -> None:
        """Persist a token bucket state snapshot to the WAL (Phase 30).

        Parameters
        ----------
        entity_id:
            The tool or org identifier.
        tokens:
            Current token level.
        capacity:
            Maximum bucket capacity.
        probationary:
            Whether this entity is in probationary status.
        """
        record: dict[str, Any] = {
            "_type": _TAG_TOKEN_BUCKET,
            "entity_id": entity_id,
            "tokens": tokens,
            "capacity": capacity,
            "probationary": probationary,
        }
        self._append_line(self._token_bucket_path, record)

    def append_settlement(
        self,
        from_org: str,
        to_org: str,
        *,
        timestamp: float,
        gross_forward: float,
        gross_reverse: float,
        net_amount: float,
        settled: bool,
    ) -> None:
        """Append a :class:`~manifold.clearing.SettlementEvent` to the settlements WAL (Phase 32).

        Parameters
        ----------
        from_org:
            Debtor organisation ID.
        to_org:
            Creditor organisation ID.
        timestamp:
            POSIX timestamp of the settlement.
        gross_forward:
            Sum of from_org → to_org call costs.
        gross_reverse:
            Sum of to_org → from_org call costs.
        net_amount:
            Net amount after bilateral netting.
        settled:
            Whether the net amount was within the bankruptcy threshold.
        """
        record: dict[str, Any] = {
            "_type": _TAG_SETTLEMENT,
            "from_org": from_org,
            "to_org": to_org,
            "timestamp": timestamp,
            "gross_forward": gross_forward,
            "gross_reverse": gross_reverse,
            "net_amount": net_amount,
            "settled": settled,
        }
        self._append_line(self._settlements_path, record)

    # ------------------------------------------------------------------
    # State recovery
    # ------------------------------------------------------------------

    def load_state(
        self,
        hub: ReputationHub | None = None,
        ledger: AgentEconomyLedger | None = None,
    ) -> "VaultLoadResult":
        """Replay WAL files to repopulate *hub* and *ledger*.

        Any line that cannot be parsed is silently skipped and counted in
        :attr:`VaultLoadResult.skipped`.

        Parameters
        ----------
        hub:
            :class:`~manifold.hub.ReputationHub` to repopulate with gossip.
            If ``None``, gossip WAL replay is skipped.
        ledger:
            :class:`~manifold.b2b.AgentEconomyLedger` to repopulate with
            economy entries.  If ``None``, economy WAL replay is skipped.

        Returns
        -------
        VaultLoadResult
            Summary of the replay: counts of loaded and skipped records.
        """
        gossip_loaded = 0
        economy_loaded = 0
        skipped = 0

        # -- Gossip replay -----------------------------------------------
        if hub is not None and self._gossip_path.exists():
            for line in self._read_lines(self._gossip_path):
                try:
                    rec = json.loads(line)
                    if rec.get("_type") != _TAG_GOSSIP:
                        skipped += 1
                        continue
                    raw_signal = str(rec.get("signal", "healthy"))
                    signal = raw_signal if raw_signal in _VALID_SIGNALS else "healthy"
                    packet = FederatedGossipPacket(
                        tool_name=str(rec["tool_name"]),
                        signal=signal,  # type: ignore[arg-type]
                        confidence=float(rec.get("confidence", 0.5)),
                        org_id=str(rec.get("org_id", "anonymous")),
                        weight=float(rec.get("weight", 1.0)),
                    )
                    # Contribute the replayed packet into the hub (anonymize=False
                    # so the original org_id is preserved for ledger purposes).
                    hub.contribute(packet, anonymize=False)
                    gossip_loaded += 1
                except (KeyError, ValueError, TypeError):
                    skipped += 1

        # -- Economy replay -----------------------------------------------
        if ledger is not None and self._economy_path.exists():
            for line in self._read_lines(self._economy_path):
                try:
                    rec = json.loads(line)
                    if rec.get("_type") != _TAG_ECONOMY:
                        skipped += 1
                        continue
                    entry = EconomyEntry(
                        local_org_id=str(rec["local_org_id"]),
                        remote_org_id=str(rec["remote_org_id"]),
                        allowed=bool(rec["allowed"]),
                        reputation_score=float(rec["reputation_score"]),
                        surcharge=float(rec["surcharge"]),
                        net_trust_cost=float(rec["net_trust_cost"]),
                        block_reason=str(rec.get("block_reason", "")),
                    )
                    # Append directly to the ledger's internal list; the public
                    # record() API requires a B2BRouteResult which we don't have
                    # during WAL replay.
                    ledger._entries.append(entry)  # noqa: SLF001
                    economy_loaded += 1
                except (KeyError, ValueError, TypeError):
                    skipped += 1

        return VaultLoadResult(
            gossip_loaded=gossip_loaded,
            economy_loaded=economy_loaded,
            skipped=skipped,
        )

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def gossip_count(self) -> int:
        """Return the number of gossip records in the WAL."""
        return self._count_lines(self._gossip_path)

    def economy_count(self) -> int:
        """Return the number of economy records in the WAL."""
        return self._count_lines(self._economy_path)

    def volatility_count(self) -> int:
        """Return the number of volatility records in the WAL (Phase 26)."""
        return self._count_lines(self._volatility_path)

    def probationary_count(self) -> int:
        """Return the number of probationary records in the WAL (Phase 28)."""
        return self._count_lines(self._probationary_path)

    def provenance_count(self) -> int:
        """Return the number of provenance records in the WAL (Phase 29)."""
        return self._count_lines(self._provenance_path)

    def token_bucket_count(self) -> int:
        """Return the number of token-bucket state records in the WAL (Phase 30)."""
        return self._count_lines(self._token_bucket_path)

    def settlements_count(self) -> int:
        """Return the number of settlement records in the WAL (Phase 32)."""
        return self._count_lines(self._settlements_path)

    def purge(self) -> None:
        """Delete all WAL files (irreversible).  Useful in tests."""
        with self._lock:
            for path in (
                self._gossip_path,
                self._economy_path,
                self._volatility_path,
                self._probationary_path,
                self._provenance_path,
                self._token_bucket_path,
                self._settlements_path,
            ):
                if path.exists():
                    path.unlink()

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        """Read all non-empty lines from *path*."""
        with path.open("r", encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]

    @staticmethod
    def _count_lines(path: Path) -> int:
        """Count non-empty lines in *path*; return 0 if file doesn't exist."""
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for ln in fh if ln.strip())


# ---------------------------------------------------------------------------
# VaultLoadResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VaultLoadResult:
    """Summary of a :meth:`ManifoldVault.load_state` replay.

    Attributes
    ----------
    gossip_loaded:
        Number of gossip packets successfully replayed into the hub.
    economy_loaded:
        Number of economy entries successfully replayed into the ledger.
    skipped:
        Number of lines that could not be parsed or had an unexpected type
        tag and were therefore ignored.
    """

    gossip_loaded: int
    economy_loaded: int
    skipped: int

    @property
    def total_loaded(self) -> int:
        """Total number of records successfully loaded."""
        return self.gossip_loaded + self.economy_loaded
