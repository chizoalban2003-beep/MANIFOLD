"""manifold/agent_registry.py — Track all registered AI agents.

Agents announce themselves at startup and send heartbeats.
MANIFOLD knows who is running, what they do, and their health.
Thread-safe. In-memory (no external dependencies).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Literal

AgentStatus = Literal["active", "idle", "paused", "stale", "unregistered"]


@dataclass
class AgentRecord:
    agent_id: str
    display_name: str
    capabilities: list[str]  # e.g. ["code", "search", "billing", "email"]
    org_id: str
    endpoint_url: str  # where MANIFOLD can send instructions
    domain: str = "general"
    status: AgentStatus = "active"
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    task_count: int = 0
    error_count: int = 0
    notes: str = ""
    _command_queue: list = field(default_factory=list)

    def is_stale(self, timeout_seconds: int = 120) -> bool:
        return time.time() - self.last_heartbeat > timeout_seconds

    def health_score(self) -> float:
        if self.task_count == 0:
            return 1.0
        return max(0.0, 1.0 - self.error_count / self.task_count)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "capabilities": self.capabilities,
            "org_id": self.org_id,
            "domain": self.domain,
            "status": self.status,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "task_count": self.task_count,
            "error_count": self.error_count,
            "health_score": round(self.health_score(), 4),
            "is_stale": self.is_stale(),
        }


class AgentRegistry:
    """
    Tracks all registered AI agents across the MANIFOLD ecosystem.
    Thread-safe. In-memory (persisted optionally via snapshot()).
    """

    def __init__(self, stale_timeout: int = 120) -> None:
        self._agents: dict[str, AgentRecord] = {}
        self._lock = threading.Lock()
        self.stale_timeout = stale_timeout

    def register(
        self,
        agent_id: str,
        display_name: str,
        capabilities: list[str],
        org_id: str,
        endpoint_url: str = "",
        domain: str = "general",
        notes: str = "",
    ) -> AgentRecord:
        with self._lock:
            record = AgentRecord(
                agent_id=agent_id,
                display_name=display_name,
                capabilities=capabilities,
                org_id=org_id,
                endpoint_url=endpoint_url,
                domain=domain,
                notes=notes,
            )
            self._agents[agent_id] = record
            return record

    def heartbeat(self, agent_id: str, status: AgentStatus = "active") -> bool:
        with self._lock:
            rec = self._agents.get(agent_id)
            if rec is None:
                return False
            rec.last_heartbeat = time.time()
            rec.status = status
            return True

    def record_task(self, agent_id: str, success: bool) -> None:
        with self._lock:
            rec = self._agents.get(agent_id)
            if rec:
                rec.task_count += 1
                if not success:
                    rec.error_count += 1

    def pause(self, agent_id: str) -> bool:
        with self._lock:
            rec = self._agents.get(agent_id)
            if rec:
                rec.status = "paused"
                return True
            return False

    def resume(self, agent_id: str) -> bool:
        with self._lock:
            rec = self._agents.get(agent_id)
            if rec:
                rec.status = "active"
                return True
            return False

    def get(self, agent_id: str) -> AgentRecord | None:
        return self._agents.get(agent_id)

    def all_agents(self) -> list[AgentRecord]:
        with self._lock:
            return list(self._agents.values())

    def active_agents(self) -> list[AgentRecord]:
        with self._lock:
            return [
                a
                for a in self._agents.values()
                if a.status == "active" and not a.is_stale(self.stale_timeout)
            ]

    def agents_with_capability(self, capability: str) -> list[AgentRecord]:
        return [a for a in self.active_agents() if capability in a.capabilities]

    def mark_stale_agents(self) -> list[str]:
        stale = []
        with self._lock:
            for rec in self._agents.values():
                if rec.status == "active" and rec.is_stale(self.stale_timeout):
                    rec.status = "stale"
                    stale.append(rec.agent_id)
        return stale

    def unregister(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                return True
            return False

    def summary(self) -> dict:
        agents = self.all_agents()
        return {
            "total": len(agents),
            "active": sum(1 for a in agents if a.status == "active"),
            "paused": sum(1 for a in agents if a.status == "paused"),
            "stale": sum(1 for a in agents if a.status == "stale"),
            "avg_health": round(
                sum(a.health_score() for a in agents) / max(len(agents), 1), 4
            ),
        }

    def queue_command(
        self,
        agent_id: str,
        command: str,
        payload: dict | None = None,
    ) -> str | None:
        """Queue a command for an agent. Returns command_id or None if not found.

        command: 'pause' | 'resume' | 'redirect' | 'update_policy' | 'message'
        payload: command-specific data dict.
        """
        import uuid
        with self._lock:
            rec = self._agents.get(agent_id)
            if rec is None:
                return None
            cmd = {
                "id": str(uuid.uuid4())[:8],
                "command": command,
                "payload": payload or {},
                "queued_at": time.time(),
            }
            rec._command_queue.append(cmd)
            return cmd["id"]

    def poll_commands(
        self,
        agent_id: str,
        consume: bool = True,
    ) -> list[dict]:
        """Return pending commands for an agent.

        If consume=True, clears the queue after returning.
        Returns empty list if agent not found.
        """
        with self._lock:
            rec = self._agents.get(agent_id)
            if rec is None:
                return []
            cmds = list(rec._command_queue)
            if consume:
                rec._command_queue.clear()
            return cmds
