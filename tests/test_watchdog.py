"""Tests for Phase 42: Daemon Watchdog & Self-Healing (manifold/watchdog.py)."""

from __future__ import annotations

import time

import pytest

from manifold.watchdog import (
    HeartbeatMiss,
    ProcessWatchdog,
    WatchdogReport,
    WatchedComponent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _always_healthy() -> bool:
    return True


def _always_unhealthy() -> bool:
    return False


def _raises() -> bool:
    raise RuntimeError("boom")


def _noop() -> None:
    pass


# ---------------------------------------------------------------------------
# WatchedComponent
# ---------------------------------------------------------------------------


class TestWatchedComponent:
    def test_initial_state(self) -> None:
        comp = WatchedComponent(name="test", heartbeat_fn=_always_healthy, restart_fn=_noop)
        assert comp._consecutive_misses == 0
        assert comp._restart_count == 0
        assert not comp.needs_restart

    def test_record_healthy_resets_misses(self) -> None:
        comp = WatchedComponent(name="test", heartbeat_fn=_always_healthy, restart_fn=_noop)
        comp._consecutive_misses = 2
        comp.record_beat(True)
        assert comp._consecutive_misses == 0

    def test_record_unhealthy_increments(self) -> None:
        comp = WatchedComponent(name="test", heartbeat_fn=_always_healthy, restart_fn=_noop)
        comp.record_beat(False)
        comp.record_beat(False)
        assert comp._consecutive_misses == 2

    def test_needs_restart_at_max_missed(self) -> None:
        comp = WatchedComponent(name="test", heartbeat_fn=_always_healthy, restart_fn=_noop, max_missed=3)
        for _ in range(3):
            comp.record_beat(False)
        assert comp.needs_restart

    def test_needs_restart_false_before_max(self) -> None:
        comp = WatchedComponent(name="test", heartbeat_fn=_always_healthy, restart_fn=_noop, max_missed=3)
        comp.record_beat(False)
        comp.record_beat(False)
        assert not comp.needs_restart

    def test_mark_restarted(self) -> None:
        comp = WatchedComponent(name="test", heartbeat_fn=_always_healthy, restart_fn=_noop)
        comp._consecutive_misses = 5
        comp.mark_restarted()
        assert comp._consecutive_misses == 0
        assert comp._restart_count == 1

    def test_to_dict_keys(self) -> None:
        comp = WatchedComponent(name="mycomp", heartbeat_fn=_always_healthy, restart_fn=_noop)
        d = comp.to_dict()
        assert d["name"] == "mycomp"
        assert "consecutive_misses" in d
        assert "last_heartbeat" in d
        assert "restart_count" in d


# ---------------------------------------------------------------------------
# HeartbeatMiss
# ---------------------------------------------------------------------------


class TestHeartbeatMiss:
    def test_to_dict(self) -> None:
        miss = HeartbeatMiss(
            component_name="prober",
            timestamp=1234567890.0,
            consecutive_count=2,
            stack_trace="",
        )
        d = miss.to_dict()
        assert d["component_name"] == "prober"
        assert d["timestamp"] == 1234567890.0
        assert d["consecutive_count"] == 2
        assert d["stack_trace"] == ""

    def test_frozen(self) -> None:
        miss = HeartbeatMiss(component_name="x", timestamp=0.0, consecutive_count=1, stack_trace="")
        with pytest.raises((AttributeError, TypeError)):
            miss.component_name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WatchdogReport
# ---------------------------------------------------------------------------


class TestWatchdogReport:
    def test_to_dict(self) -> None:
        report = WatchdogReport(
            total_components=2,
            total_restarts=1,
            total_missed_heartbeats=3,
            deadlock_purges=0,
            is_running=True,
            component_states=[],
        )
        d = report.to_dict()
        assert d["total_components"] == 2
        assert d["total_restarts"] == 1
        assert d["total_missed_heartbeats"] == 3
        assert d["deadlock_purges"] == 0
        assert d["is_running"] is True


# ---------------------------------------------------------------------------
# ProcessWatchdog
# ---------------------------------------------------------------------------


class TestProcessWatchdog:
    def test_empty_report(self) -> None:
        wd = ProcessWatchdog()
        r = wd.report()
        assert r.total_components == 0
        assert r.total_restarts == 0
        assert r.total_missed_heartbeats == 0
        assert not r.is_running

    def test_register_component(self) -> None:
        wd = ProcessWatchdog()
        comp = WatchedComponent(name="test", heartbeat_fn=_always_healthy, restart_fn=_noop)
        wd.register(comp)
        assert wd.report().total_components == 1

    def test_healthy_check_no_misses(self) -> None:
        wd = ProcessWatchdog()
        wd.register(WatchedComponent(name="ok", heartbeat_fn=_always_healthy, restart_fn=_noop))
        wd.check_once()
        assert wd.report().total_missed_heartbeats == 0

    def test_unhealthy_check_records_miss(self) -> None:
        wd = ProcessWatchdog()
        wd.register(WatchedComponent(name="bad", heartbeat_fn=_always_unhealthy, restart_fn=_noop))
        wd.check_once()
        assert wd.report().total_missed_heartbeats == 1

    def test_exception_in_heartbeat_recorded_as_miss(self) -> None:
        wd = ProcessWatchdog()
        wd.register(WatchedComponent(name="boom", heartbeat_fn=_raises, restart_fn=_noop))
        wd.check_once()
        misses = wd.missed_heartbeats()
        assert len(misses) == 1
        assert "RuntimeError" in misses[0].stack_trace

    def test_restart_triggered_after_max_missed(self) -> None:
        restart_calls: list[int] = []

        def restart_fn() -> None:
            restart_calls.append(1)

        wd = ProcessWatchdog()
        wd.register(WatchedComponent(name="flaky", heartbeat_fn=_always_unhealthy, restart_fn=restart_fn, max_missed=3))
        for _ in range(3):
            wd.check_once()
        assert len(restart_calls) >= 1

    def test_crashlog_fn_called_on_miss(self) -> None:
        logs: list[dict] = []

        wd = ProcessWatchdog(crashlog_fn=logs.append)
        wd.register(WatchedComponent(name="bad", heartbeat_fn=_always_unhealthy, restart_fn=_noop))
        wd.check_once()
        assert len(logs) == 1
        assert logs[0]["component_name"] == "bad"

    def test_start_stop(self) -> None:
        wd = ProcessWatchdog(interval_seconds=0.05)
        wd.start()
        assert wd.is_running()
        wd.stop()
        time.sleep(0.15)
        assert not wd.is_running()

    def test_start_idempotent(self) -> None:
        wd = ProcessWatchdog(interval_seconds=60.0)
        wd.start()
        wd.start()  # should not raise
        assert wd.is_running()
        wd.stop()

    def test_set_multisig_vault_no_crash(self) -> None:
        wd = ProcessWatchdog()

        class FakeVault:
            def purge_expired(self, timeout: float) -> int:
                return 0

        wd.set_multisig_vault(FakeVault())
        wd.check_once()  # should not raise

    def test_deadlock_purge_counted(self) -> None:
        purged = [0]

        class FakeVault:
            def purge_expired(self, timeout: float) -> int:
                purged[0] = 2
                return 2

        wd = ProcessWatchdog()
        wd.set_multisig_vault(FakeVault())
        wd.check_once()
        assert wd.report().deadlock_purges == 2

    def test_missed_heartbeats_returns_copy(self) -> None:
        wd = ProcessWatchdog()
        wd.register(WatchedComponent(name="x", heartbeat_fn=_always_unhealthy, restart_fn=_noop))
        wd.check_once()
        m1 = wd.missed_heartbeats()
        m1.clear()
        assert len(wd.missed_heartbeats()) == 1

    def test_multiple_components_tracked(self) -> None:
        wd = ProcessWatchdog()
        wd.register(WatchedComponent(name="a", heartbeat_fn=_always_healthy, restart_fn=_noop))
        wd.register(WatchedComponent(name="b", heartbeat_fn=_always_healthy, restart_fn=_noop))
        wd.register(WatchedComponent(name="c", heartbeat_fn=_always_unhealthy, restart_fn=_noop))
        wd.check_once()
        assert wd.report().total_components == 3
        assert wd.report().total_missed_heartbeats == 1

    def test_component_states_in_report(self) -> None:
        wd = ProcessWatchdog()
        wd.register(WatchedComponent(name="probe", heartbeat_fn=_always_healthy, restart_fn=_noop))
        r = wd.report()
        assert len(r.component_states) == 1
        assert r.component_states[0]["name"] == "probe"
