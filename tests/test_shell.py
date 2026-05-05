"""Tests for Phase 46: Manifold Shell / TUI (manifold/shell.py)."""

from __future__ import annotations

import io

from manifold.shell import AdminMetrics, ManifoldCLI


# ---------------------------------------------------------------------------
# AdminMetrics
# ---------------------------------------------------------------------------


class TestAdminMetrics:
    def test_defaults(self) -> None:
        m = AdminMetrics()
        assert m.pid_threshold == 0.3
        assert m.dag_count == 0
        assert m.ledger_entries == []
        assert m.swarm_peers == []
        assert m.dht_peers == []
        assert m.watchdog_report == {}
        assert m.genesis_summary == {}
        assert m.hub_scores == {}

    def test_custom_values(self) -> None:
        m = AdminMetrics(pid_threshold=0.55, dag_count=7)
        assert m.pid_threshold == 0.55
        assert m.dag_count == 7


# ---------------------------------------------------------------------------
# ManifoldCLI helpers
# ---------------------------------------------------------------------------


def _make_cli(
    metrics: AdminMetrics | None = None,
    veto_log: list[str] | None = None,
) -> tuple[ManifoldCLI, io.StringIO]:
    """Create a CLI with a captured stdout and injected metrics."""
    buf = io.StringIO()
    if metrics is None:
        metrics = AdminMetrics()
    veto_log = veto_log if veto_log is not None else []

    def _veto(tool: str) -> None:
        veto_log.append(tool)

    cli = ManifoldCLI(
        metrics_fn=lambda: metrics,
        veto_fn=_veto,
        stdout=buf,
    )
    return cli, buf


# ---------------------------------------------------------------------------
# ManifoldCLI — basics
# ---------------------------------------------------------------------------


class TestManifoldCLIBasics:
    def test_instantiation(self) -> None:
        cli, _ = _make_cli()
        assert isinstance(cli, ManifoldCLI)

    def test_default_prompt(self) -> None:
        cli, _ = _make_cli()
        assert "manifold" in cli.prompt

    def test_command_count_starts_zero(self) -> None:
        cli, _ = _make_cli()
        assert cli.command_count == 0

    def test_no_metrics_fn_uses_defaults(self) -> None:
        buf = io.StringIO()
        cli = ManifoldCLI(stdout=buf)
        assert cli.command_count == 0


# ---------------------------------------------------------------------------
# ManifoldCLI — do_top
# ---------------------------------------------------------------------------


class TestManifoldCLITop:
    def test_top_outputs_threshold(self) -> None:
        m = AdminMetrics(pid_threshold=0.42)
        cli, buf = _make_cli(metrics=m)
        cli.do_top("")
        output = buf.getvalue()
        assert "0.4200" in output

    def test_top_outputs_dag_count(self) -> None:
        m = AdminMetrics(dag_count=17)
        cli, buf = _make_cli(metrics=m)
        cli.do_top("")
        assert "17" in buf.getvalue()

    def test_top_increments_command_count(self) -> None:
        cli, _ = _make_cli()
        cli.do_top("")
        assert cli.command_count == 1

    def test_top_shows_peer_counts(self) -> None:
        m = AdminMetrics(
            swarm_peers=[{"org_id": "p1", "endpoint": "http://p1", "routing_value": 0.5}],
            dht_peers=[{"raw_id": "p2", "endpoint": "http://p2", "distance": 100}],
        )
        cli, buf = _make_cli(metrics=m)
        cli.do_top("")
        output = buf.getvalue()
        assert "1" in output  # swarm peers


# ---------------------------------------------------------------------------
# ManifoldCLI — do_ledger
# ---------------------------------------------------------------------------


class TestManifoldCLILedger:
    def test_ledger_empty(self) -> None:
        cli, buf = _make_cli()
        cli.do_ledger("")
        assert "no ledger entries" in buf.getvalue()

    def test_ledger_shows_entries(self) -> None:
        m = AdminMetrics(
            ledger_entries=[
                {
                    "local_org_id": "org-a",
                    "remote_org_id": "org-b",
                    "allowed": True,
                    "net_trust_cost": 0.12,
                }
            ]
        )
        cli, buf = _make_cli(metrics=m)
        cli.do_ledger("")
        output = buf.getvalue()
        assert "org-a" in output
        assert "org-b" in output

    def test_ledger_respects_n_param(self) -> None:
        entries = [
            {"local_org_id": f"o{i}", "remote_org_id": "x", "allowed": True, "net_trust_cost": 0.0}
            for i in range(20)
        ]
        m = AdminMetrics(ledger_entries=entries)
        cli, buf = _make_cli(metrics=m)
        cli.do_ledger("5")
        output = buf.getvalue()
        # Only last 5 — o15..o19
        assert "o15" in output or "o19" in output

    def test_ledger_invalid_n(self) -> None:
        cli, buf = _make_cli()
        cli.do_ledger("not-a-number")
        assert "Usage" in buf.getvalue()

    def test_ledger_increments_count(self) -> None:
        cli, _ = _make_cli()
        cli.do_ledger("")
        assert cli.command_count == 1


# ---------------------------------------------------------------------------
# ManifoldCLI — do_veto
# ---------------------------------------------------------------------------


class TestManifoldCLIVeto:
    def test_veto_calls_fn(self) -> None:
        log: list[str] = []
        cli, _ = _make_cli(veto_log=log)
        cli.do_veto("my-tool")
        assert log == ["my-tool"]

    def test_veto_empty_arg(self) -> None:
        log: list[str] = []
        cli, buf = _make_cli(veto_log=log)
        cli.do_veto("")
        assert "Usage" in buf.getvalue()
        assert log == []

    def test_veto_output_message(self) -> None:
        cli, buf = _make_cli()
        cli.do_veto("bad-tool")
        assert "bad-tool" in buf.getvalue()

    def test_veto_increments_count(self) -> None:
        cli, _ = _make_cli()
        cli.do_veto("tool-x")
        assert cli.command_count == 1

    def test_veto_whitespace_arg(self) -> None:
        log: list[str] = []
        cli, buf = _make_cli(veto_log=log)
        cli.do_veto("  ")
        assert "Usage" in buf.getvalue()
        assert log == []


# ---------------------------------------------------------------------------
# ManifoldCLI — do_peers
# ---------------------------------------------------------------------------


class TestManifoldCLIPeers:
    def test_peers_empty(self) -> None:
        cli, buf = _make_cli()
        cli.do_peers("")
        assert "no peers" in buf.getvalue()

    def test_peers_shows_swarm(self) -> None:
        m = AdminMetrics(
            swarm_peers=[
                {"org_id": "peer-x", "endpoint": "http://x:8080", "routing_value": 0.7}
            ]
        )
        cli, buf = _make_cli(metrics=m)
        cli.do_peers("")
        assert "peer-x" in buf.getvalue()

    def test_peers_shows_dht(self) -> None:
        m = AdminMetrics(
            dht_peers=[
                {"raw_id": "dht-peer", "endpoint": "http://dht:8080", "distance": 42}
            ]
        )
        cli, buf = _make_cli(metrics=m)
        cli.do_peers("")
        assert "dht-peer" in buf.getvalue()

    def test_peers_increments_count(self) -> None:
        cli, _ = _make_cli()
        cli.do_peers("")
        assert cli.command_count == 1

    def test_peers_combined(self) -> None:
        m = AdminMetrics(
            swarm_peers=[{"org_id": "sw1", "endpoint": "http://sw1:8080", "routing_value": 0.5}],
            dht_peers=[{"raw_id": "dht1", "endpoint": "http://dht1:8080", "distance": 10}],
        )
        cli, buf = _make_cli(metrics=m)
        cli.do_peers("")
        output = buf.getvalue()
        assert "sw1" in output
        assert "dht1" in output


# ---------------------------------------------------------------------------
# ManifoldCLI — do_health
# ---------------------------------------------------------------------------


class TestManifoldCLIHealth:
    def test_health_empty_report(self) -> None:
        cli, buf = _make_cli()
        cli.do_health("")
        assert "no watchdog data" in buf.getvalue()

    def test_health_shows_status(self) -> None:
        m = AdminMetrics(
            watchdog_report={
                "is_running": True,
                "total_restarts": 0,
                "total_missed_heartbeats": 2,
                "deadlock_purges": 0,
                "component_states": [
                    {"name": "active_prober", "consecutive_misses": 0, "restart_count": 0, "max_missed": 3}
                ],
            }
        )
        cli, buf = _make_cli(metrics=m)
        cli.do_health("")
        output = buf.getvalue()
        assert "RUNNING" in output
        assert "active_prober" in output

    def test_health_stopped_status(self) -> None:
        m = AdminMetrics(
            watchdog_report={
                "is_running": False,
                "total_restarts": 1,
                "total_missed_heartbeats": 3,
                "deadlock_purges": 0,
                "component_states": [],
            }
        )
        cli, buf = _make_cli(metrics=m)
        cli.do_health("")
        assert "STOPPED" in buf.getvalue()

    def test_health_increments_count(self) -> None:
        cli, _ = _make_cli()
        cli.do_health("")
        assert cli.command_count == 1


# ---------------------------------------------------------------------------
# ManifoldCLI — do_genesis
# ---------------------------------------------------------------------------


class TestManifoldCLIGenesis:
    def test_genesis_empty(self) -> None:
        cli, buf = _make_cli()
        cli.do_genesis("")
        assert "no genesis data" in buf.getvalue()

    def test_genesis_shows_summary(self) -> None:
        m = AdminMetrics(
            genesis_summary={
                "genesis_node_id": "genesis-0",
                "total_tokens": 1000.0,
                "gamma": 1.0,
                "mint_events": 3,
            }
        )
        cli, buf = _make_cli(metrics=m)
        cli.do_genesis("")
        output = buf.getvalue()
        assert "genesis-0" in output
        assert "1000" in output
        assert "3" in output

    def test_genesis_increments_count(self) -> None:
        cli, _ = _make_cli()
        cli.do_genesis("")
        assert cli.command_count == 1


# ---------------------------------------------------------------------------
# ManifoldCLI — do_exit / do_quit
# ---------------------------------------------------------------------------


class TestManifoldCLIExit:
    def test_exit_returns_true(self) -> None:
        cli, _ = _make_cli()
        assert cli.do_exit("") is True

    def test_quit_returns_true(self) -> None:
        cli, _ = _make_cli()
        assert cli.do_quit("") is True

    def test_eof_returns_true(self) -> None:
        cli, _ = _make_cli()
        assert cli.do_EOF("") is True

    def test_exit_prints_goodbye(self) -> None:
        cli, buf = _make_cli()
        cli.do_exit("")
        assert "Goodbye" in buf.getvalue()


# ---------------------------------------------------------------------------
# ManifoldCLI — help system
# ---------------------------------------------------------------------------


class TestManifoldCLIHelp:
    def test_has_top_help(self) -> None:
        cli, _ = _make_cli()
        assert cli.do_top.__doc__ is not None
        assert "top" in cli.do_top.__doc__.lower() or "snapshot" in cli.do_top.__doc__.lower()

    def test_has_ledger_help(self) -> None:
        cli, _ = _make_cli()
        assert cli.do_ledger.__doc__ is not None

    def test_has_veto_help(self) -> None:
        cli, _ = _make_cli()
        assert cli.do_veto.__doc__ is not None

    def test_has_peers_help(self) -> None:
        cli, _ = _make_cli()
        assert cli.do_peers.__doc__ is not None

    def test_has_health_help(self) -> None:
        cli, _ = _make_cli()
        assert cli.do_health.__doc__ is not None

    def test_has_genesis_help(self) -> None:
        cli, _ = _make_cli()
        assert cli.do_genesis.__doc__ is not None
