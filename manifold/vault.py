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

    def purge(self) -> None:
        """Delete both WAL files (irreversible).  Useful in tests."""
        with self._lock:
            for path in (self._gossip_path, self._economy_path):
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
