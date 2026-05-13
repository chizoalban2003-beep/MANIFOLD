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


# ---------------------------------------------------------------------------
# EXP7 — Episode dataclass for per-agent episodic memory
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A record of one completed task episode for an agent."""
    task_description: str
    domain: str
    duration_seconds: float
    success: bool
    crna_at_start: dict   # {c, r, n, a}
    crna_at_end: dict     # {c, r, n, a}
    risk_encountered: float   # max R seen during the task
    timestamp: float = field(default_factory=time.time)


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
    episode_history: list = field(default_factory=list)  # list[Episode]
    max_episodes: int = 100

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

    # ------------------------------------------------------------------
    # EXP7 — Episodic memory
    # ------------------------------------------------------------------

    def record_episode(self, agent_id: str, episode: "Episode") -> bool:
        """Append *episode* to the agent's episode history (capped at max_episodes).

        Returns ``True`` on success, ``False`` if the agent does not exist.
        """
        with self._lock:
            rec = self._agents.get(agent_id)
            if rec is None:
                return False
            rec.episode_history.append(episode)
            if len(rec.episode_history) > rec.max_episodes:
                rec.episode_history = rec.episode_history[-rec.max_episodes:]
        return True

    def agent_risk_estimate(self, agent_id: str, domain: str) -> float:
        """Return the mean risk_encountered for the agent's episodes in *domain*.

        Returns 0.5 if the agent has no episodes for that domain.
        """
        rec = self._agents.get(agent_id)
        if rec is None:
            return 0.5
        domain_eps = [e for e in rec.episode_history if e.domain == domain]
        if not domain_eps:
            return 0.5
        return sum(e.risk_encountered for e in domain_eps) / len(domain_eps)

    def domain_risk_estimate(self, agent_id: str, domain: str) -> float:
        """Return a recency-weighted risk estimate for *agent_id* in *domain*.

        Recent episodes (last 10) get 2× weight to reflect skill improvement
        or environmental drift.  Returns 0.5 (neutral) if no history exists.
        """
        rec = self._agents.get(agent_id)
        if rec is None:
            return 0.5
        domain_eps = [e for e in rec.episode_history if e.domain == domain]
        if not domain_eps:
            return 0.5
        old_eps = domain_eps[:-10] if len(domain_eps) > 10 else []
        recent_eps = domain_eps[-10:]
        total_weight = len(old_eps) + 2 * len(recent_eps)
        if total_weight == 0:
            return 0.5
        weighted_sum = (
            sum(e.risk_encountered for e in old_eps)
            + 2.0 * sum(e.risk_encountered for e in recent_eps)
        )
        return weighted_sum / total_weight

    def agent_task_score(self, agent_id: str, domain: str) -> float:
        """Episodic-aware trust score for task assignment.

        score = health_score * (1 − domain_risk_estimate)
        """
        rec = self._agents.get(agent_id)
        if rec is None:
            return 0.0
        ats = rec.health_score()
        return ats * (1.0 - self.domain_risk_estimate(agent_id, domain))

    def best_agent_for_domain(
        self,
        domain: str,
        required_capabilities: list | None = None,
    ) -> str | None:
        """Return the agent_id with the highest ``agent_task_score`` for *domain*.

        Filters by *required_capabilities* when provided.  Falls back to
        highest ATS if no agent has domain history.  Returns ``None`` when
        no active agents are available.
        """
        active = self.active_agents()
        if not active:
            return None

        if required_capabilities:
            capable = [
                a for a in active
                if all(c in a.capabilities for c in required_capabilities)
            ]
            if capable:
                active = capable

        best_id: str | None = None
        best_score = -1.0
        for agent in active:
            score = self.agent_task_score(agent.agent_id, domain)
            if score > best_score:
                best_score = score
                best_id = agent.agent_id
        return best_id

    def best_agent_for_task(
        self,
        task_domain: str,
        cell_crna: dict | None = None,
    ) -> str | None:
        """Return the agent_id with the highest episodic-risk-adjusted ATS score.

        Delegates to ``best_agent_for_domain`` for backwards compatibility.
        Returns None if no active agents are registered.
        """
        return self.best_agent_for_domain(task_domain)


# ---------------------------------------------------------------------------
# EXP7 benchmark
# ---------------------------------------------------------------------------

def compare_assignment_quality() -> dict:
    """Compare episodic vs random agent assignment quality.

    Registers 3 agents with different episode histories and measures
    which assignment strategy selects lower-risk agents for finance tasks.

    Returns
    -------
    dict with keys: episodic_win_rate, risk_improvement.
    """
    import random as _random

    registry = AgentRegistry(stale_timeout=9999)

    # Register three agents
    for aid, name, caps in [
        ("agent-finance", "Finance Expert", ["finance", "billing"]),
        ("agent-general", "Generalist", ["general", "finance", "search"]),
        ("agent-new", "Newcomer", ["finance"]),
    ]:
        registry.register(aid, name, caps, "org1")

    # Give agent-finance a strong low-risk finance history
    crna_safe = {"c": 0.4, "r": 0.3, "n": 0.2, "a": 0.7}
    for _ in range(10):
        registry.record_episode(
            "agent-finance",
            Episode(
                task_description="Quarterly audit review",
                domain="finance",
                duration_seconds=45.0,
                success=True,
                crna_at_start=crna_safe,
                crna_at_end=crna_safe,
                risk_encountered=0.2,
            ),
        )

    # Give agent-general a medium-risk finance history
    crna_med = {"c": 0.5, "r": 0.5, "n": 0.5, "a": 0.5}
    for _ in range(10):
        registry.record_episode(
            "agent-general",
            Episode(
                task_description="Generic finance task",
                domain="finance",
                duration_seconds=60.0,
                success=True,
                crna_at_start=crna_med,
                crna_at_end=crna_med,
                risk_encountered=0.5,
            ),
        )

    # agent-new has no history (defaults to 0.5)

    rng = _random.Random(42)
    all_agents = ["agent-finance", "agent-general", "agent-new"]

    episodic_wins = 0
    episodic_risk_total = 0.0
    random_risk_total = 0.0
    n_trials = 20

    for _ in range(n_trials):
        best = registry.best_agent_for_task("finance")
        random_agent = rng.choice(all_agents)

        ep_risk = registry.agent_risk_estimate(best or "", "finance")
        rnd_risk = registry.agent_risk_estimate(random_agent, "finance")

        episodic_risk_total += ep_risk
        random_risk_total += rnd_risk

        if ep_risk <= rnd_risk:
            episodic_wins += 1

    return {
        "episodic_win_rate": round(episodic_wins / n_trials, 4),
        "risk_improvement": round((random_risk_total - episodic_risk_total) / n_trials, 6),
    }
