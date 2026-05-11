"""manifold/monitor.py — Proactive agent health monitoring.

Background loop that monitors registered agents, marks stale ones,
detects unhealthy agents, and logs anomalous events.
Runs as a daemon thread — starts with the server.
"""
from __future__ import annotations

import threading
import time

from manifold.agent_registry import AgentRegistry


class AgentMonitor:
    """
    Proactively monitors registered agents.
    Marks stale agents, detects unhealthy agents, logs anomalies.
    Runs as a daemon thread — starts with the server.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        check_interval: int = 30,
        stale_timeout: int = 120,
        health_threshold: float = 0.7,
    ) -> None:
        self._registry = registry
        self._check_interval = check_interval
        self._stale_timeout = stale_timeout
        self._health_threshold = health_threshold
        self._running = False
        self._thread: threading.Thread | None = None
        self._events: list[dict] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._check_interval)
            self._run_checks()

    def _run_checks(self) -> None:
        # 1. Mark stale agents
        stale = self._registry.mark_stale_agents()
        for agent_id in stale:
            self._log_event("stale", agent_id, "Agent missed heartbeat")

        # 2. Check agent health
        for agent in self._registry.active_agents():
            if agent.health_score() < self._health_threshold:
                self._log_event(
                    "unhealthy",
                    agent.agent_id,
                    f"Health score {agent.health_score():.2f} below threshold",
                )

    def _log_event(self, event_type: str, agent_id: str, message: str) -> None:
        with self._lock:
            self._events.append(
                {
                    "type": event_type,
                    "agent_id": agent_id,
                    "message": message,
                    "timestamp": time.time(),
                }
            )
            # Keep last 500 events only
            if len(self._events) > 500:
                self._events = self._events[-500:]

    def recent_events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return self._events[-limit:]

    def status(self) -> dict:
        return {
            "running": self._running,
            "check_interval": self._check_interval,
            "health_threshold": self._health_threshold,
            "event_count": len(self._events),
            "registry_summary": self._registry.summary(),
        }
