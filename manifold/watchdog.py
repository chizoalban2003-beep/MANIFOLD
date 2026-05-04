"""Phase 42: Daemon Watchdog & Self-Healing — 100% Uptime Supervisor.

``ProcessWatchdog`` monitors background threads registered by MANIFOLD
phases (e.g. ``ActiveProber``, ``ClearingEngine``, ``GraphExecutor``).
If a supervised component misses **3 consecutive heartbeats** the Watchdog:

1. Catches any exception and dumps a stack trace to ``vault.py``
   (``crashlogs.jsonl``).
2. Restarts the component via its registered ``restart_fn`` callback.
3. Never kills the main HTTP server.

If the :class:`~manifold.multisig.MultiSigVault` has a pending entry
older than ``deadlock_timeout_seconds`` the Watchdog purges it.

Key classes
-----------
``WatchedComponent``
    Descriptor for a supervised background component.
``HeartbeatMiss``
    Frozen record of a single missed heartbeat event.
``WatchdogReport``
    Summary of Watchdog activity since startup.
``ProcessWatchdog``
    Thread supervisor with heartbeat monitoring and deadlock resolution.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# WatchedComponent
# ---------------------------------------------------------------------------


@dataclass
class WatchedComponent:
    """Descriptor for a background component supervised by :class:`ProcessWatchdog`.

    Parameters
    ----------
    name:
        Human-readable component name (e.g. ``"active_prober"``).
    heartbeat_fn:
        Zero-argument callable that returns ``True`` if the component is
        healthy and ``False`` (or raises) if it is not.
    restart_fn:
        Zero-argument callable that restarts the component.  Called after
        ``max_missed`` consecutive missed heartbeats.
    max_missed:
        Number of consecutive missed heartbeats before a restart is
        triggered.  Default: ``3``.
    """

    name: str
    heartbeat_fn: Callable[[], bool]
    restart_fn: Callable[[], None]
    max_missed: int = 3

    _consecutive_misses: int = field(default=0, init=False, repr=False)
    _last_heartbeat: float = field(default_factory=time.time, init=False, repr=False)
    _restart_count: int = field(default=0, init=False, repr=False)

    def record_beat(self, healthy: bool) -> None:
        """Update the consecutive-miss counter."""
        if healthy:
            self._consecutive_misses = 0
            self._last_heartbeat = time.time()
        else:
            self._consecutive_misses += 1

    @property
    def needs_restart(self) -> bool:
        """``True`` if consecutive misses have reached ``max_missed``."""
        return self._consecutive_misses >= self.max_missed

    def mark_restarted(self) -> None:
        """Reset the miss counter and record a restart event."""
        self._consecutive_misses = 0
        self._restart_count += 1
        self._last_heartbeat = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "consecutive_misses": self._consecutive_misses,
            "last_heartbeat": self._last_heartbeat,
            "restart_count": self._restart_count,
            "max_missed": self.max_missed,
        }


# ---------------------------------------------------------------------------
# HeartbeatMiss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeartbeatMiss:
    """Frozen record of a single missed heartbeat event.

    Attributes
    ----------
    component_name:
        Name of the component that missed a heartbeat.
    timestamp:
        POSIX timestamp of the miss.
    consecutive_count:
        How many consecutive misses have occurred (including this one).
    stack_trace:
        Stack trace if an exception was raised during heartbeat check
        (empty string otherwise).
    """

    component_name: str
    timestamp: float
    consecutive_count: int
    stack_trace: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "component_name": self.component_name,
            "timestamp": self.timestamp,
            "consecutive_count": self.consecutive_count,
            "stack_trace": self.stack_trace,
        }


# ---------------------------------------------------------------------------
# WatchdogReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchdogReport:
    """Summary of :class:`ProcessWatchdog` activity since startup.

    Attributes
    ----------
    total_components:
        Number of supervised components.
    total_restarts:
        Total component restarts triggered.
    total_missed_heartbeats:
        Total missed-heartbeat events recorded.
    deadlock_purges:
        Number of MultiSig deadlocks resolved by purging.
    is_running:
        Whether the Watchdog background loop is active.
    component_states:
        Per-component state dicts (name, misses, restarts).
    """

    total_components: int
    total_restarts: int
    total_missed_heartbeats: int
    deadlock_purges: int
    is_running: bool
    component_states: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "total_components": self.total_components,
            "total_restarts": self.total_restarts,
            "total_missed_heartbeats": self.total_missed_heartbeats,
            "deadlock_purges": self.deadlock_purges,
            "is_running": self.is_running,
            "component_states": self.component_states,
        }


# ---------------------------------------------------------------------------
# ProcessWatchdog
# ---------------------------------------------------------------------------


@dataclass
class ProcessWatchdog:
    """Thread supervisor with heartbeat monitoring and deadlock resolution.

    Parameters
    ----------
    interval_seconds:
        How often (seconds) the Watchdog checks heartbeats.
        Default: ``10.0``.
    deadlock_timeout_seconds:
        Max age (seconds) for a pending MultiSig entry before it is
        forcefully purged.  Default: ``300.0``.
    crashlog_fn:
        Optional callable ``(record: dict) -> None`` used to persist crash
        logs to ``vault.py``.  If ``None``, crash logs are discarded.

    Example
    -------
    ::

        wd = ProcessWatchdog()
        wd.register(
            WatchedComponent(
                name="active_prober",
                heartbeat_fn=lambda: prober.is_running(),
                restart_fn=prober.start,
            )
        )
        wd.start()
    """

    interval_seconds: float = 10.0
    deadlock_timeout_seconds: float = 300.0
    crashlog_fn: Callable[[dict[str, Any]], None] | None = None

    _components: list[WatchedComponent] = field(
        default_factory=list, init=False, repr=False
    )
    _misses: list[HeartbeatMiss] = field(
        default_factory=list, init=False, repr=False
    )
    _deadlock_purges: int = field(default=0, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _multisig_vault: Any = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, component: WatchedComponent) -> None:
        """Register a component for supervision.

        Parameters
        ----------
        component:
            The component descriptor to supervise.
        """
        with self._lock:
            self._components.append(component)

    def set_multisig_vault(self, vault: Any) -> None:
        """Bind a :class:`~manifold.multisig.MultiSigVault` for deadlock monitoring.

        Parameters
        ----------
        vault:
            The ``MultiSigVault`` instance to watch.
        """
        with self._lock:
            self._multisig_vault = vault

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background watchdog loop (non-blocking).

        Calling ``start()`` when already running is a no-op.
        """
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="manifold-watchdog",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background loop to stop."""
        with self._lock:
            self._running = False

    def is_running(self) -> bool:
        """Return ``True`` if the watchdog loop is active."""
        return self._running

    # ------------------------------------------------------------------
    # On-demand check
    # ------------------------------------------------------------------

    def check_once(self) -> None:
        """Perform one synchronous heartbeat check of all components.

        Exceptions raised by ``heartbeat_fn`` are caught and recorded.
        """
        with self._lock:
            components = list(self._components)

        for comp in components:
            trace = ""
            try:
                healthy = bool(comp.heartbeat_fn())
            except Exception:  # noqa: BLE001
                healthy = False
                trace = traceback.format_exc()

            comp.record_beat(healthy)

            if not healthy:
                miss = HeartbeatMiss(
                    component_name=comp.name,
                    timestamp=time.time(),
                    consecutive_count=comp._consecutive_misses,  # noqa: SLF001
                    stack_trace=trace,
                )
                with self._lock:
                    self._misses.append(miss)
                if self.crashlog_fn is not None:
                    try:
                        self.crashlog_fn(miss.to_dict())
                    except Exception:  # noqa: BLE001
                        pass

            if comp.needs_restart:
                try:
                    comp.restart_fn()
                    comp.mark_restarted()
                except Exception:  # noqa: BLE001
                    pass

        self._resolve_multisig_deadlocks()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> WatchdogReport:
        """Return a :class:`WatchdogReport` snapshot."""
        with self._lock:
            comps = list(self._components)
            total_misses = len(self._misses)
            purges = self._deadlock_purges
            running = self._running

        total_restarts = sum(c._restart_count for c in comps)  # noqa: SLF001
        return WatchdogReport(
            total_components=len(comps),
            total_restarts=total_restarts,
            total_missed_heartbeats=total_misses,
            deadlock_purges=purges,
            is_running=running,
            component_states=[c.to_dict() for c in comps],
        )

    def missed_heartbeats(self) -> list[HeartbeatMiss]:
        """Return a copy of all recorded missed-heartbeat events."""
        with self._lock:
            return list(self._misses)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_multisig_deadlocks(self) -> None:
        """Purge MultiSig entries that exceed ``deadlock_timeout_seconds``."""
        with self._lock:
            vault = self._multisig_vault
        if vault is None:
            return
        try:
            purged = vault.purge_expired(self.deadlock_timeout_seconds)
            if purged > 0:
                with self._lock:
                    self._deadlock_purges += purged
        except Exception:  # noqa: BLE001
            pass

    def _run_loop(self) -> None:
        """Background watchdog loop — runs until ``stop()`` is called."""
        while True:
            with self._lock:
                if not self._running:
                    break
            try:
                self.check_once()
            except Exception:  # noqa: BLE001
                pass  # never crash the watchdog itself
            time.sleep(self.interval_seconds)
