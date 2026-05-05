"""Phase 46: Manifold Shell / TUI — Interactive Admin CLI.

``ManifoldCLI`` is a native, interactive command-line shell (built on
Python's standard-library ``cmd`` module) that lets an operator inspect and
control a running MANIFOLD daemon without touching the public-facing HTTP
API.

Commands
--------
``top``
    Live snapshot of the PID controller's current risk threshold and the
    number of active DAG executions.
``ledger [n]``
    Display the last *n* (default 10) entries from the ``AgentEconomyLedger``.
``veto <tool>``
    Admin override — force a tool's reliability score to ``0.0`` in the
    local reputation hub.
``peers``
    Display the routing table of active Swarm / DHT peers.
``health``
    Show the ProcessWatchdog component health matrix.
``genesis``
    Show the genesis token distribution summary.

Key classes
-----------
``AdminMetrics``
    Lightweight snapshot of daemon metrics that feeds every command.
``ManifoldCLI``
    ``cmd.Cmd`` subclass providing the interactive shell.
"""

from __future__ import annotations

import cmd
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# AdminMetrics
# ---------------------------------------------------------------------------


@dataclass
class AdminMetrics:
    """Lightweight snapshot of daemon metrics used by :class:`ManifoldCLI`.

    Parameters
    ----------
    pid_threshold:
        Current risk-veto threshold from the PID controller.
    dag_count:
        Number of DAG executions recorded.
    ledger_entries:
        Recent :class:`~manifold.b2b.EconomyEntry` records (as dicts).
    swarm_peers:
        Routing table from :class:`~manifold.swarm.SwarmRouter`.
    dht_peers:
        Routing table from :class:`~manifold.sharding.ShardRouter`.
    watchdog_report:
        Dict representation of a :class:`~manifold.watchdog.WatchdogReport`.
    genesis_summary:
        Dict from :meth:`~manifold.genesis.GenesisMint.summary`.
    """

    pid_threshold: float = 0.3
    dag_count: int = 0
    ledger_entries: list[dict[str, Any]] = field(default_factory=list)
    swarm_peers: list[dict[str, Any]] = field(default_factory=list)
    dht_peers: list[dict[str, Any]] = field(default_factory=list)
    watchdog_report: dict[str, Any] = field(default_factory=dict)
    genesis_summary: dict[str, Any] = field(default_factory=dict)
    hub_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ManifoldCLI
# ---------------------------------------------------------------------------


class ManifoldCLI(cmd.Cmd):
    """Interactive admin shell for the MANIFOLD daemon.

    Parameters
    ----------
    metrics_fn:
        Zero-argument callable that returns an :class:`AdminMetrics`
        snapshot on each call.  Useful for dependency-injection in tests.
    veto_fn:
        Callable ``(tool_name: str) -> None`` that sets the tool's
        reliability to ``0.0`` in the reputation hub.
    intro:
        Banner displayed when the shell starts.
    prompt:
        Shell prompt string.

    Example
    -------
    ::

        cli = ManifoldCLI(
            metrics_fn=lambda: AdminMetrics(pid_threshold=0.3),
            veto_fn=lambda tool: hub.update(tool, 0.0),
        )
        cli.cmdloop()   # blocks; user types commands interactively
    """

    intro: str = (
        "\n╔══════════════════════════════════════╗\n"
        "║  MANIFOLD v2.1 — Admin Shell          ║\n"
        "║  Type 'help' for available commands   ║\n"
        "╚══════════════════════════════════════╝\n"
    )
    prompt: str = "manifold> "

    def __init__(
        self,
        metrics_fn: Callable[[], AdminMetrics] | None = None,
        veto_fn: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._metrics_fn: Callable[[], AdminMetrics] = (
            metrics_fn if metrics_fn is not None else lambda: AdminMetrics()
        )
        self._veto_fn: Callable[[str], None] = (
            veto_fn if veto_fn is not None else lambda _tool: None
        )
        self._command_count: int = 0

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def do_top(self, _arg: str) -> None:
        """top — Live snapshot of risk threshold and DAG activity."""
        m = self._metrics_fn()
        self._command_count += 1
        now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        lines = [
            f"\n[{now}]  MANIFOLD System Snapshot",
            "─" * 42,
            f"  PID risk threshold : {m.pid_threshold:.4f}",
            f"  DAG executions     : {m.dag_count}",
            f"  Swarm peers        : {len(m.swarm_peers)}",
            f"  DHT peers          : {len(m.dht_peers)}",
            "",
        ]
        self.stdout.write("\n".join(lines))

    def do_ledger(self, arg: str) -> None:
        """ledger [n] — Show last n AgentEconomyLedger entries (default 10)."""
        try:
            n = int(arg.strip()) if arg.strip() else 10
        except ValueError:
            self.stdout.write("Usage: ledger [n]\n")
            return

        m = self._metrics_fn()
        self._command_count += 1
        entries = m.ledger_entries[-n:]
        if not entries:
            self.stdout.write("  (no ledger entries)\n")
            return

        header = f"\n{'Local Org':<20} {'Remote Org':<20} {'Allowed':<8} {'Net Cost':>10}\n"
        divider = "─" * 62 + "\n"
        self.stdout.write(header)
        self.stdout.write(divider)
        for e in entries:
            local = str(e.get("local_org_id", ""))[:18]
            remote = str(e.get("remote_org_id", ""))[:18]
            allowed = "yes" if e.get("allowed") else "no"
            cost = float(e.get("net_trust_cost", 0.0))
            self.stdout.write(f"  {local:<20} {remote:<20} {allowed:<8} {cost:>10.4f}\n")
        self.stdout.write("\n")

    def do_veto(self, arg: str) -> None:
        """veto <tool> — Force tool reliability to 0.0 (admin override)."""
        tool = arg.strip()
        if not tool:
            self.stdout.write("Usage: veto <tool_name>\n")
            return
        self._command_count += 1
        self._veto_fn(tool)
        self.stdout.write(f"  ⚠  Veto applied: {tool!r} reliability set to 0.0\n")

    def do_peers(self, _arg: str) -> None:
        """peers — Show routing table of active Swarm/DHT nodes."""
        m = self._metrics_fn()
        self._command_count += 1
        all_peers = m.swarm_peers + m.dht_peers
        if not all_peers:
            self.stdout.write("  (no peers registered)\n")
            return

        header = f"\n{'Peer ID':<22} {'Endpoint':<30} {'Routing Value':>14}\n"
        divider = "─" * 70 + "\n"
        self.stdout.write(header)
        self.stdout.write(divider)
        for p in all_peers[:20]:
            org_id = str(p.get("org_id", p.get("raw_id", "?")))[:20]
            endpoint = str(p.get("endpoint", ""))[:28]
            rv = p.get("routing_value", p.get("distance", ""))
            rv_str = f"{rv:.4f}" if isinstance(rv, float) else str(rv)
            self.stdout.write(f"  {org_id:<22} {endpoint:<30} {rv_str:>14}\n")
        self.stdout.write("\n")

    def do_health(self, _arg: str) -> None:
        """health — Show ProcessWatchdog component health matrix."""
        m = self._metrics_fn()
        self._command_count += 1
        wr = m.watchdog_report
        if not wr:
            self.stdout.write("  (no watchdog data available)\n")
            return

        running = "RUNNING" if wr.get("is_running") else "STOPPED"
        self.stdout.write(
            f"\n  Watchdog: {running}"
            f"  |  Restarts: {wr.get('total_restarts', 0)}"
            f"  |  Missed heartbeats: {wr.get('total_missed_heartbeats', 0)}"
            f"  |  Deadlock purges: {wr.get('deadlock_purges', 0)}\n"
        )
        states: list[dict[str, Any]] = wr.get("component_states", [])
        if states:
            header = f"\n  {'Component':<24} {'Misses':>7} {'Restarts':>9} {'Max':>5}\n"
            divider = "  " + "─" * 50 + "\n"
            self.stdout.write(header)
            self.stdout.write(divider)
            for cs in states:
                name = str(cs.get("name", "?"))[:22]
                misses = cs.get("consecutive_misses", 0)
                restarts = cs.get("restart_count", 0)
                max_m = cs.get("max_missed", 3)
                status = "⚠" if misses > 0 else "✓"
                self.stdout.write(
                    f"  {status} {name:<22} {misses:>7} {restarts:>9} {max_m:>5}\n"
                )
        self.stdout.write("\n")

    def do_genesis(self, _arg: str) -> None:
        """genesis — Show genesis token distribution summary."""
        m = self._metrics_fn()
        self._command_count += 1
        gs = m.genesis_summary
        if not gs:
            self.stdout.write("  (no genesis data available)\n")
            return

        self.stdout.write(
            f"\n  Genesis Node   : {gs.get('genesis_node_id', '?')}\n"
            f"  Total Tokens   : {gs.get('total_tokens', 0):.1f}\n"
            f"  Gamma (γ)      : {gs.get('gamma', 0):.4f}\n"
            f"  Mint Events    : {gs.get('mint_events', 0)}\n\n"
        )

    def do_exit(self, _arg: str) -> bool:
        """exit — Quit the shell."""
        self.stdout.write("Goodbye.\n")
        return True

    def do_quit(self, _arg: str) -> bool:
        """quit — Quit the shell."""
        return self.do_exit(_arg)

    def do_EOF(self, _arg: str) -> bool:  # noqa: N802
        """Handle Ctrl-D / EOF."""
        self.stdout.write("\n")
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def command_count(self) -> int:
        """Total number of commands executed in this session."""
        return self._command_count
