"""manifold/task_router.py — Decompose any problem into governed sub-tasks.

Receives complex problems, decomposes them into governed sub-tasks,
routes each to the best available agent, and returns an execution plan.
This is Mode 3 — arbitrary task intake.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

from manifold.agent_registry import AgentRecord, AgentRegistry
from manifold.brain import BrainTask, ManifoldBrain
from manifold.encoder_v2 import encode_prompt
from manifold.workspace import GlobalWorkspace


@dataclass
class SubTask:
    index: int
    prompt: str
    domain: str
    stakes: float
    risk_score: float
    action: str  # brain's governance decision
    assigned_to: str | None  # agent_id or None if unassigned
    status: str = "pending"  # pending|assigned|blocked|complete
    reason: str = ""


@dataclass
class TaskPlan:
    task_id: str
    original_task: str
    sub_tasks: list[SubTask]
    created_at: float = field(default_factory=time.time)
    executable: bool = False  # True if all sub-tasks are assigned
    blocked_count: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "original_task": self.original_task[:120],
            "sub_tasks": [
                {
                    "index": s.index,
                    "prompt": s.prompt[:80],
                    "domain": s.domain,
                    "stakes": s.stakes,
                    "risk_score": round(s.risk_score, 4),
                    "action": s.action,
                    "assigned_to": s.assigned_to,
                    "status": s.status,
                    "reason": s.reason,
                }
                for s in self.sub_tasks
            ],
            "executable": self.executable,
            "blocked_count": self.blocked_count,
            "total_sub_tasks": len(self.sub_tasks),
        }


class TaskRouter:
    """
    Receives complex problems. Decomposes. Governs. Routes.

    Three steps:
    1. Decompose: split into concrete sub-tasks by sentence/clause
    2. Govern:    run each sub-task through ManifoldBrain
    3. Route:     assign each permitted sub-task to the best agent

    If no agents are registered, returns a plan with no assignments
    (operators can then execute manually or wire agents later).
    """

    def __init__(
        self,
        brain: ManifoldBrain | None = None,
        registry: AgentRegistry | None = None,
    ) -> None:
        self._brain = brain or ManifoldBrain()
        self._registry = registry or AgentRegistry()
        self._workspace = GlobalWorkspace()
        self._plans: dict[str, TaskPlan] = {}

    def _decompose(self, task: str) -> list[str]:
        """
        Split a complex task string into concrete sub-tasks.
        Uses sentence boundaries and common connectors.
        Returns list of non-empty sub-task strings.
        """
        parts = re.split(
            r"(?:[,;]?\s+(?:and|then|also|next|after that|finally)\s+)|"
            r"(?:\.\s+)|(?:\n+)",
            task,
            flags=re.IGNORECASE,
        )
        cleaned = [p.strip().rstrip(".,;") for p in parts if len(p.strip()) > 8]
        return cleaned if cleaned else [task]

    def _best_agent(
        self, domain: str, capabilities_needed: list[str]
    ) -> AgentRecord | None:
        """Find the best available agent for a sub-task."""
        candidates = self._registry.active_agents()
        if not candidates:
            return None

        def score(a: AgentRecord) -> float:
            domain_match = 1.0 if a.domain == domain else 0.3
            cap_overlap = len(set(a.capabilities) & set(capabilities_needed)) / max(
                len(capabilities_needed), 1
            )
            return domain_match * 0.5 + cap_overlap * 0.3 + a.health_score() * 0.2

        return max(candidates, key=score)

    def route(self, task: str, stakes_hint: float = 0.5) -> TaskPlan:
        """
        Main entry point. Receive a complex task, return a TaskPlan.
        """
        task_id = hashlib.sha256(f"{task}{time.time()}".encode()).hexdigest()[:12]
        sub_texts = self._decompose(task)
        sub_tasks: list[SubTask] = []

        for i, sub_text in enumerate(sub_texts):
            domain, _ = self._workspace.route(sub_text)
            encoded = encode_prompt(sub_text, force_keyword=True)
            stakes = max(stakes_hint * 0.8, encoded.risk)
            brain_task = BrainTask(
                prompt=sub_text,
                domain=domain,
                stakes=stakes,
                uncertainty=stakes,  # higher stakes → higher epistemic uncertainty
            )
            decision = self._brain.decide(brain_task)
            caps_needed = [domain, "general"]
            agent = None
            status = "blocked"
            reason = ""

            if decision.action in ("refuse", "stop"):
                status = "blocked"
                reason = f"Governance refused: risk={decision.risk_score:.3f}"
            elif decision.action == "escalate":
                status = "blocked"
                reason = "Escalation required — human review needed"
            else:
                agent = self._best_agent(domain, caps_needed)
                if agent:
                    status = "assigned"
                    reason = f"Assigned to {agent.agent_id}"
                else:
                    status = "pending"
                    reason = "No agent available — register an agent or execute manually"

            sub_tasks.append(
                SubTask(
                    index=i,
                    prompt=sub_text,
                    domain=domain,
                    stakes=round(stakes, 4),
                    risk_score=round(decision.risk_score, 4),
                    action=decision.action,
                    assigned_to=agent.agent_id if agent else None,
                    status=status,
                    reason=reason,
                )
            )

        blocked = sum(1 for s in sub_tasks if s.status == "blocked")
        executable = all(s.status in ("assigned", "pending") for s in sub_tasks)
        plan = TaskPlan(
            task_id=task_id,
            original_task=task,
            sub_tasks=sub_tasks,
            executable=executable,
            blocked_count=blocked,
        )
        self._plans[task_id] = plan
        return plan

    def get_plan(self, task_id: str) -> TaskPlan | None:
        return self._plans.get(task_id)

    def all_plans(self) -> list[TaskPlan]:
        return list(self._plans.values())
