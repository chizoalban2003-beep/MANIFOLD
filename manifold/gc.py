"""Phase 50: Garbage Collector & Log Compaction — Self-Maintaining WAL Pruner.

Solves the infinite-log-growth problem by providing atomic, safe compaction of
the ``ManifoldVault`` Write-Ahead Log (``.jsonl``) files and selective pruning of
orphaned in-memory state (``VectorIndex`` entries, decayed reputations).

Architecture
------------
* **LogCompactor** — reads a ``.jsonl`` WAL, keeps only the *N* most-recent
  records or drops records whose ``timestamp`` field is older than *ttl_seconds*,
  then atomically rewrites the file using ``os.rename`` so that no data is ever
  lost during a compaction.
* **StatePruner** — iterates through ``VectorIndex`` and ``ReputationHub`` to
  delete entries that have not been accessed for *max_age_days* days.
* **GCRunReport** — immutable summary of a single GC cycle.
* **ManifoldGC** — orchestrates ``LogCompactor`` + ``StatePruner`` and exposes
  a ``run()`` method; can be wired into the ``EventBus`` / ``Watchdog`` to run
  silently every 24 hours.

Safety guarantees
-----------------
* Every WAL rewrite uses a ``<target>.tmp`` scratch file followed by
  ``os.rename`` — an atomic operation on POSIX-compliant file systems.
* The compaction lock prevents concurrent rewrites of the same file.

Key classes
-----------
``CompactionResult``
    Stats from compacting one WAL file.
``LogCompactor``
    Prunes oversized or stale ``.jsonl`` WAL files.
``StatePruner``
    Removes orphaned vectors and decayed reputation entries.
``GCRunReport``
    Immutable summary of a complete GC cycle.
``ManifoldGC``
    High-level orchestrator that runs compaction + pruning.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CompactionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionResult:
    """Stats from compacting one WAL file.

    Attributes
    ----------
    path:
        Absolute path of the compacted file.
    records_before:
        Number of non-empty lines before compaction.
    records_after:
        Number of non-empty lines after compaction.
    bytes_saved:
        Approximate number of bytes freed.
    dropped_by_count:
        Records dropped because they exceeded the *keep_last_n* limit.
    dropped_by_ttl:
        Records dropped because their ``timestamp`` was older than the TTL.
    """

    path: str
    records_before: int
    records_after: int
    bytes_saved: int
    dropped_by_count: int
    dropped_by_ttl: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "path": self.path,
            "records_before": self.records_before,
            "records_after": self.records_after,
            "bytes_saved": self.bytes_saved,
            "dropped_by_count": self.dropped_by_count,
            "dropped_by_ttl": self.dropped_by_ttl,
        }


# ---------------------------------------------------------------------------
# LogCompactor
# ---------------------------------------------------------------------------


@dataclass
class LogCompactor:
    """Safe, atomic ``.jsonl`` WAL compactor.

    Parameters
    ----------
    keep_last_n:
        Maximum number of records to keep per file.  Older excess records are
        dropped first.  ``0`` disables the count-based cutoff.
    ttl_seconds:
        Records whose ``timestamp`` field (POSIX float) is older than this
        many seconds are dropped.  ``0.0`` disables TTL pruning.

    Example
    -------
    ::

        compactor = LogCompactor(keep_last_n=1000, ttl_seconds=86400 * 7)
        result = compactor.compact(Path("/var/manifold/gossip.jsonl"))
        print(f"Freed {result.bytes_saved} bytes")
    """

    keep_last_n: int = 10_000
    ttl_seconds: float = 0.0

    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def compact(self, path: Path) -> CompactionResult:
        """Compact the WAL file at *path*.

        Steps
        -----
        1. Read all non-empty lines.
        2. Apply TTL filter (if enabled).
        3. Apply keep-last-N filter (if enabled).
        4. Atomically rewrite via a temporary file + ``os.rename``.

        Parameters
        ----------
        path:
            Path to the ``.jsonl`` file to compact.  If the file does not
            exist, a no-op ``CompactionResult`` with zeros is returned.

        Returns
        -------
        CompactionResult
            Stats about what was kept and dropped.
        """
        if not path.exists():
            return CompactionResult(
                path=str(path),
                records_before=0,
                records_after=0,
                bytes_saved=0,
                dropped_by_count=0,
                dropped_by_ttl=0,
            )

        with self._lock:
            size_before = path.stat().st_size

            # Read all valid lines
            raw_lines: list[str] = []
            with path.open("r", encoding="utf-8") as fh:
                for ln in fh:
                    stripped = ln.strip()
                    if stripped:
                        raw_lines.append(stripped)

            records_before = len(raw_lines)

            # Parse what we can; keep malformed lines as opaque strings
            parsed: list[tuple[float, str]] = []  # (timestamp, raw_line)
            now = time.time()
            for ln in raw_lines:
                try:
                    obj = json.loads(ln)
                    ts = float(obj.get("timestamp", now))
                except (json.JSONDecodeError, ValueError, TypeError):
                    ts = now  # keep malformed lines (WALRepair handles them)
                parsed.append((ts, ln))

            # TTL filter
            dropped_by_ttl = 0
            if self.ttl_seconds > 0.0:
                cutoff = now - self.ttl_seconds
                keep: list[tuple[float, str]] = []
                for ts, ln in parsed:
                    if ts >= cutoff:
                        keep.append((ts, ln))
                    else:
                        dropped_by_ttl += 1
                parsed = keep

            # Count-based filter (keep newest N)
            dropped_by_count = 0
            if self.keep_last_n > 0 and len(parsed) > self.keep_last_n:
                dropped_by_count = len(parsed) - self.keep_last_n
                parsed = parsed[-self.keep_last_n:]

            records_after = len(parsed)
            kept_lines = [ln for _, ln in parsed]

            # Atomic rewrite via temp file + rename
            tmp_path = path.with_suffix(".tmp")
            try:
                with tmp_path.open("w", encoding="utf-8") as fh:
                    for ln in kept_lines:
                        fh.write(ln + "\n")
                os.rename(str(tmp_path), str(path))
            except Exception:  # noqa: BLE001
                # If rename fails, leave original intact and clean up temp
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise

            size_after = path.stat().st_size if path.exists() else 0
            bytes_saved = max(0, size_before - size_after)

            return CompactionResult(
                path=str(path),
                records_before=records_before,
                records_after=records_after,
                bytes_saved=bytes_saved,
                dropped_by_count=dropped_by_count,
                dropped_by_ttl=dropped_by_ttl,
            )

    def compact_directory(
        self,
        directory: Path,
        *,
        glob_pattern: str = "*.jsonl",
    ) -> list[CompactionResult]:
        """Compact all matching WAL files in *directory*.

        Parameters
        ----------
        directory:
            Directory to scan.
        glob_pattern:
            Filename glob (default: ``"*.jsonl"``).

        Returns
        -------
        list[CompactionResult]
            One result per file, in lexicographic path order.
        """
        results: list[CompactionResult] = []
        for p in sorted(directory.glob(glob_pattern)):
            results.append(self.compact(p))
        return results


# ---------------------------------------------------------------------------
# StatePruner
# ---------------------------------------------------------------------------


@dataclass
class StatePruner:
    """Removes orphaned or stale in-memory state.

    Parameters
    ----------
    max_age_days:
        Entries not accessed in this many days are eligible for pruning.
        Default: ``30``.

    Notes
    -----
    ``VectorIndex`` entries are pruned if their metadata contains an
    ``"accessed_at"`` field (POSIX timestamp) that is older than
    *max_age_days*.  Entries without ``"accessed_at"`` are kept.
    """

    max_age_days: float = 30.0

    def prune_vector_index(self, vector_index: Any) -> int:
        """Remove stale entries from *vector_index*.

        Parameters
        ----------
        vector_index:
            A :class:`~manifold.vectorfs.VectorIndex` instance.

        Returns
        -------
        int
            Number of entries removed.
        """
        cutoff = time.time() - self.max_age_days * 86_400.0
        ids_to_remove: list[str] = []

        for entry in vector_index.entries():
            accessed_at = entry.metadata.get("accessed_at")
            if accessed_at is not None:
                try:
                    if float(accessed_at) < cutoff:
                        ids_to_remove.append(entry.vector_id)
                except (ValueError, TypeError):
                    pass

        for vid in ids_to_remove:
            vector_index.remove(vid)

        return len(ids_to_remove)

    def prune_reputation_hub(self, hub: Any, *, decay_threshold: float = 0.05) -> int:
        """Remove near-zero-reliability entries from *hub*.

        Parameters
        ----------
        hub:
            A :class:`~manifold.hub.ReputationHub` instance.
        decay_threshold:
            Tools whose computed reliability falls below this value are pruned.
            Default: ``0.05``.

        Returns
        -------
        int
            Number of tool entries removed.
        """
        tool_names = list(hub.ledger.tool_names()) if hasattr(hub, "ledger") else []
        removed = 0
        for tool_name in tool_names:
            try:
                score = hub.ledger.global_rate(tool_name)
                if score < decay_threshold:
                    hub.ledger.remove_tool(tool_name)
                    removed += 1
            except (AttributeError, KeyError, ValueError):
                pass
        return removed


# ---------------------------------------------------------------------------
# GCRunReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GCRunReport:
    """Immutable summary of a complete GC cycle.

    Attributes
    ----------
    timestamp:
        POSIX timestamp when the GC run completed.
    compaction_results:
        One :class:`CompactionResult` per compacted file.
    vectors_pruned:
        Number of stale vector entries removed from the ``VectorIndex``.
    reputations_pruned:
        Number of decayed reputation entries removed from the ``ReputationHub``.
    total_bytes_saved:
        Sum of ``bytes_saved`` across all compaction results.
    duration_seconds:
        Wall-clock duration of the GC run.
    """

    timestamp: float
    compaction_results: tuple[CompactionResult, ...]
    vectors_pruned: int
    reputations_pruned: int
    total_bytes_saved: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "timestamp": self.timestamp,
            "compaction_results": [r.to_dict() for r in self.compaction_results],
            "vectors_pruned": self.vectors_pruned,
            "reputations_pruned": self.reputations_pruned,
            "total_bytes_saved": self.total_bytes_saved,
            "duration_seconds": round(self.duration_seconds, 4),
            "files_compacted": len(self.compaction_results),
        }


# ---------------------------------------------------------------------------
# ManifoldGC
# ---------------------------------------------------------------------------


@dataclass
class ManifoldGC:
    """Orchestrates log compaction and state pruning for MANIFOLD.

    Parameters
    ----------
    data_dir:
        Directory containing the ``.jsonl`` WAL files.  Defaults to the
        current working directory.
    keep_last_n:
        Forwarded to :class:`LogCompactor`.  Default: ``10_000``.
    ttl_seconds:
        Forwarded to :class:`LogCompactor`.  Default: ``0.0`` (disabled).
    max_age_days:
        Forwarded to :class:`StatePruner`.  Default: ``30``.
    interval_seconds:
        How often the background daemon runs.  Default: ``86_400`` (24 h).

    Example
    -------
    ::

        gc = ManifoldGC(data_dir="/var/manifold", keep_last_n=5000,
                        ttl_seconds=86400 * 7)
        report = gc.run()
        print(f"GC freed {report.total_bytes_saved} bytes")
    """

    data_dir: str | os.PathLike[str] = field(default_factory=os.getcwd)
    keep_last_n: int = 10_000
    ttl_seconds: float = 0.0
    max_age_days: float = 30.0
    interval_seconds: float = 86_400.0

    _compactor: LogCompactor = field(init=False, repr=False)
    _pruner: StatePruner = field(init=False, repr=False)
    _last_report: GCRunReport | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._compactor = LogCompactor(
            keep_last_n=self.keep_last_n,
            ttl_seconds=self.ttl_seconds,
        )
        self._pruner = StatePruner(max_age_days=self.max_age_days)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def run(
        self,
        vector_index: Any = None,
        reputation_hub: Any = None,
    ) -> GCRunReport:
        """Execute one GC cycle synchronously.

        Parameters
        ----------
        vector_index:
            Optional :class:`~manifold.vectorfs.VectorIndex` to prune.
        reputation_hub:
            Optional :class:`~manifold.hub.ReputationHub` to prune.

        Returns
        -------
        GCRunReport
            Summary of the completed cycle.
        """
        t_start = time.monotonic()

        # Compact all JSONL files in data_dir
        data_path = Path(self.data_dir)
        compact_results: list[CompactionResult] = []
        if data_path.exists() and data_path.is_dir():
            compact_results = self._compactor.compact_directory(data_path)

        # Prune in-memory state
        vectors_pruned = 0
        if vector_index is not None:
            try:
                vectors_pruned = self._pruner.prune_vector_index(vector_index)
            except Exception:  # noqa: BLE001
                pass

        reputations_pruned = 0
        if reputation_hub is not None:
            try:
                reputations_pruned = self._pruner.prune_reputation_hub(reputation_hub)
            except Exception:  # noqa: BLE001
                pass

        total_bytes_saved = sum(r.bytes_saved for r in compact_results)
        duration = time.monotonic() - t_start

        report = GCRunReport(
            timestamp=time.time(),
            compaction_results=tuple(compact_results),
            vectors_pruned=vectors_pruned,
            reputations_pruned=reputations_pruned,
            total_bytes_saved=total_bytes_saved,
            duration_seconds=duration,
        )
        with self._lock:
            self._last_report = report
        return report

    @property
    def last_report(self) -> GCRunReport | None:
        """Return the most recent :class:`GCRunReport`, or ``None``."""
        with self._lock:
            return self._last_report

    # ------------------------------------------------------------------
    # Background daemon
    # ------------------------------------------------------------------

    def start_daemon(
        self,
        vector_index: Any = None,
        reputation_hub: Any = None,
        *,
        run_immediately: bool = False,
    ) -> None:
        """Start the background GC daemon thread.

        The daemon runs :meth:`run` every :attr:`interval_seconds` seconds.

        Parameters
        ----------
        vector_index:
            Passed to :meth:`run` on each cycle.
        reputation_hub:
            Passed to :meth:`run` on each cycle.
        run_immediately:
            If ``True``, run one GC cycle immediately before entering the
            sleep loop.
        """
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()

        def _loop() -> None:
            if run_immediately:
                try:
                    self.run(vector_index=vector_index, reputation_hub=reputation_hub)
                except Exception:  # noqa: BLE001
                    pass
            while not self._stop_event.wait(timeout=self.interval_seconds):
                try:
                    self.run(vector_index=vector_index, reputation_hub=reputation_hub)
                except Exception:  # noqa: BLE001
                    pass

        self._thread = threading.Thread(target=_loop, daemon=True, name="manifold-gc")
        self._thread.start()

    def stop_daemon(self) -> None:
        """Signal the background daemon to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def is_running(self) -> bool:
        """Return ``True`` if the background daemon thread is alive."""
        return self._thread is not None and self._thread.is_alive()
