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
    depends_on: list = field(default_factory=list)  # list of sub-task index strings
    execution_mode: str = "sequential"  # "sequential" or "parallel"
    parallel_group: int = 0  # tasks with same group ID run together


# ---------------------------------------------------------------------------
# EXP6 — DependencyGraph for temporal task ordering
# ---------------------------------------------------------------------------

class DependencyGraph:
    """Directed graph for capturing 'must-complete-before' relationships.

    Nodes are string task identifiers.
    """

    def __init__(self) -> None:
        self._edges: dict[str, set] = {}  # from_id -> set of to_ids

    def add_edge(self, from_id: str, to_id: str) -> None:
        """*from_id* must complete before *to_id* can start."""
        self._edges.setdefault(from_id, set()).add(to_id)

    def _all_nodes(self) -> set:
        nodes: set = set(self._edges.keys())
        for tos in self._edges.values():
            nodes.update(tos)
        return nodes

    def topological_sort(self, task_ids: list) -> list | None:
        """Return topologically sorted order, or None if a cycle is detected.

        Uses Kahn's algorithm (BFS on in-degrees).
        """
        nodes = set(task_ids) | self._all_nodes()
        # Build in-degree count
        in_degree: dict = {n: 0 for n in nodes}
        for frm, tos in self._edges.items():
            for to in tos:
                if to in in_degree:
                    in_degree[to] += 1

        queue = sorted(n for n, deg in in_degree.items() if deg == 0)
        sorted_list: list = []

        while queue:
            # Pop first node (use a list for deterministic ordering)
            node = queue.pop(0)
            sorted_list.append(node)
            for to in sorted(self._edges.get(node, set())):
                in_degree[to] -= 1
                if in_degree[to] == 0:
                    queue.append(to)

        if len(sorted_list) != len(nodes):
            return None  # cycle detected
        return sorted_list

    def parallel_groups(self) -> list:
        """Return groups of tasks that can run in parallel.

        Each group is a list of task IDs at the same topological depth.
        Groups are ordered: all tasks in group k must complete before group k+1 starts.
        """
        all_nodes = self._all_nodes()
        if not all_nodes:
            return []

        # BFS level assignment
        predecessors: dict = {n: set() for n in all_nodes}
        for frm, tos in self._edges.items():
            for to in tos:
                if to in predecessors:
                    predecessors[to].add(frm)

        levels: dict = {}
        remaining = set(all_nodes)

        while remaining:
            progress = False
            for n in sorted(remaining):  # sorted for determinism
                preds = predecessors[n]
                if all(p in levels for p in preds):
                    levels[n] = max((levels[p] + 1 for p in preds), default=0)
                    remaining.remove(n)
                    progress = True
            if not progress:
                # Cycle — put all remaining in their own groups
                for n in sorted(remaining):
                    levels[n] = max(levels.values(), default=0) + 1
                break

        if not levels:
            return []

        max_level = max(levels.values())
        return [
            sorted(n for n, lv in levels.items() if lv == lvl)
            for lvl in range(max_level + 1)
        ]

    def to_dict(self) -> dict:
        """Return {node: [dependency_ids]} adjacency dict."""
        result: dict = {}
        for frm, tos in self._edges.items():
            result.setdefault(frm, []).extend(sorted(tos))
        return result


@dataclass
class TaskPlan:
    task_id: str
    original_task: str
    sub_tasks: list[SubTask]
    created_at: float = field(default_factory=time.time)
    executable: bool = False  # True if all sub-tasks are assigned
    blocked_count: int = 0
    dependency_graph: dict = field(default_factory=dict)   # {id: [depends_on_ids]}
    parallel_groups: list = field(default_factory=list)    # list of task-id groups
    has_ordering: bool = False

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
                    "depends_on": s.depends_on,
                    "execution_mode": s.execution_mode,
                    "parallel_group": s.parallel_group,
                }
                for s in self.sub_tasks
            ],
            "executable": self.executable,
            "blocked_count": self.blocked_count,
            "total_sub_tasks": len(self.sub_tasks),
            "dependency_graph": self.dependency_graph,
            "parallel_groups": self.parallel_groups,
            "has_ordering": self.has_ordering,
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

        .. deprecated::
            Replaced by :meth:`_decompose_two_level`.  Kept for backward
            compatibility with callers that expect a flat list.
        """
        slots = self._decompose_two_level(task)
        flat = []
        for slot in slots:
            flat.extend(slot)
        return flat if flat else [task]

    # EXP6 — two-level decomposition: sequential slots → parallel sub-tasks
    _SEQUENTIAL_PATTERN = re.compile(
        r"(?:[,;]?\s+(?:then|after(?:\s+that)?|next|once|when\s+\S+\s+is\s+(?:complete|done|finished)|"
        r"following|subsequently|finally|first|second|third|lastly)\s+)|"
        r"(?:\.\s+)|(?:\n+)",
        re.IGNORECASE,
    )

    _PARALLEL_PATTERN = re.compile(
        r"\s+(?:and|while|simultaneously|in\s+parallel|also|at\s+the\s+same\s+time|"
        r"concurrently|meanwhile)\s+",
        re.IGNORECASE,
    )

    def _decompose_two_level(self, task: str) -> list[list[str]]:
        """Return a list of sequential slots, each slot a list of parallel sub-tasks.

        Step 1: split on SEQUENTIAL_WORDS → sequential slots.
        Step 2: for each slot, split on PARALLEL_WORDS → parallel sub-tasks.
        """
        # Step 1: sequential split
        seq_parts = self._SEQUENTIAL_PATTERN.split(task)
        slots: list[list[str]] = []
        for seq_part in seq_parts:
            seq_part = seq_part.strip()
            if not seq_part:
                continue
            # Step 2: parallel split within this sequential slot
            par_parts = self._PARALLEL_PATTERN.split(seq_part)
            cleaned = [p.strip().rstrip(".,;") for p in par_parts if len(p.strip()) > 4]
            if cleaned:
                slots.append(cleaned)
        return slots if slots else [[task]]

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
        slots = self._decompose_two_level(task)
        sub_tasks: list[SubTask] = []
        index = 0

        for group_id, slot in enumerate(slots):
            for sub_text in slot:
                domain, _ = self._workspace.route(sub_text)
                encoded = encode_prompt(sub_text, force_keyword=True)
                stakes = max(stakes_hint * 0.8, encoded.risk)
                brain_task = BrainTask(
                    prompt=sub_text,
                    domain=domain,
                    stakes=stakes,
                    uncertainty=stakes,
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
                    reason = "Escalation required -- human review needed"
                else:
                    agent = self._best_agent(domain, caps_needed)
                    if agent:
                        status = "assigned"
                        reason = f"Assigned to {agent.agent_id}"
                    else:
                        status = "pending"
                        reason = "No agent available -- register an agent or execute manually"

                sub_tasks.append(
                    SubTask(
                        index=index,
                        prompt=sub_text,
                        domain=domain,
                        stakes=round(stakes, 4),
                        risk_score=round(decision.risk_score, 4),
                        action=decision.action,
                        assigned_to=agent.agent_id if agent else None,
                        status=status,
                        reason=reason,
                        execution_mode="parallel" if len(slot) > 1 else "sequential",
                        parallel_group=group_id,
                    )
                )
                index += 1

        # EXP6: Analyse temporal ordering from the original task text
        dep_graph, par_groups, has_ordering = self._analyse_dependencies(
            task, sub_tasks, slots
        )

        blocked = sum(1 for s in sub_tasks if s.status == "blocked")
        executable = all(s.status in ("assigned", "pending") for s in sub_tasks)
        plan = TaskPlan(
            task_id=task_id,
            original_task=task,
            sub_tasks=sub_tasks,
            executable=executable,
            blocked_count=blocked,
            dependency_graph=dep_graph,
            parallel_groups=par_groups,
            has_ordering=has_ordering,
        )
        self._plans[task_id] = plan
        return plan

    # ------------------------------------------------------------------
    # EXP6 helpers
    # ------------------------------------------------------------------

    _ORDERING_KEYWORDS = re.compile(
        r"\b(then|after|once|when\s+\w+\s+is\s+(complete|done|finished)|"
        r"using\s+the\s+result|after\s+that|following)\b",
        re.IGNORECASE,
    )

    _PARALLEL_KEYWORDS = re.compile(
        r"\b(and|while|simultaneously|in\s+parallel|also|at\s+the\s+same\s+time|"
        r"concurrently|meanwhile)\b",
        re.IGNORECASE,
    )

    def _analyse_dependencies(
        self,
        original_task: str,
        sub_tasks: list,
        slots: list | None = None,
    ) -> tuple:
        """Detect sequential and parallel ordering from the task structure.

        Parameters
        ----------
        original_task:
            The original task string (used for keyword detection).
        sub_tasks:
            Flat list of SubTask objects.
        slots:
            Two-level structure from ``_decompose_two_level``.  When provided,
            dependencies are wired precisely per-slot rather than by keywords.

        Returns
        -------
        (dependency_graph_dict, parallel_groups_list, has_ordering_bool)
        """
        graph = DependencyGraph()
        n = len(sub_tasks)

        has_sequential = bool(self._ORDERING_KEYWORDS.search(original_task))
        has_parallel = bool(self._PARALLEL_KEYWORDS.search(original_task))
        has_ordering = has_sequential or has_parallel

        if slots is not None and len(slots) > 1:
            # Wire sequential dependencies between slots
            # All sub-tasks in slot[k] must complete before any in slot[k+1]
            prev_indices: list[int] = []
            cursor = 0
            for slot in slots:
                slot_indices = list(range(cursor, cursor + len(slot)))
                for next_idx in slot_indices:
                    for prev_idx in prev_indices:
                        from_id = str(sub_tasks[prev_idx].index)
                        to_id = str(sub_tasks[next_idx].index)
                        graph.add_edge(from_id, to_id)
                        if from_id not in sub_tasks[next_idx].depends_on:
                            sub_tasks[next_idx].depends_on.append(from_id)
                prev_indices = slot_indices
                cursor += len(slot)
        elif has_sequential and n > 1 and slots is None:
            # Fallback: assume fully sequential ordering
            for i in range(n - 1):
                from_id = str(sub_tasks[i].index)
                to_id = str(sub_tasks[i + 1].index)
                graph.add_edge(from_id, to_id)
                sub_tasks[i + 1].depends_on = [from_id]

        dep_dict = graph.to_dict()

        # Build parallel_groups: each slot's task IDs form one parallel group
        if slots is not None:
            par_groups: list = []
            cursor = 0
            for slot in slots:
                group = [str(sub_tasks[cursor + j].index) for j in range(len(slot))]
                par_groups.append(group)
                cursor += len(slot)
        else:
            par_groups = graph.parallel_groups() if has_ordering else []

        return dep_dict, par_groups, has_ordering

    def get_plan(self, task_id: str) -> TaskPlan | None:
        return self._plans.get(task_id)

    def all_plans(self) -> list[TaskPlan]:
        return list(self._plans.values())


# ---------------------------------------------------------------------------
# EXP6 benchmark
# ---------------------------------------------------------------------------

def run_task_ordering_benchmark() -> dict:
    """Compare ordered vs flat task plan for a multi-step task.

    Returns
    -------
    dict with ordered/flat plan metadata for comparison.
    """
    router = TaskRouter()

    # Task with explicit "then" ordering
    task = "fetch data, then analyse it, then write the report"
    ordered_plan = router.route(task)

    # Flat version (no ordering keywords)
    flat_task = "fetch data and analyse it and write the report"
    flat_plan = router.route(flat_task)

    return {
        "ordered_plan_task_id": ordered_plan.task_id,
        "has_ordering": ordered_plan.has_ordering,
        "dependency_count": sum(len(st.depends_on) for st in ordered_plan.sub_tasks),
        "parallel_groups": ordered_plan.parallel_groups,
        "flat_has_ordering": flat_plan.has_ordering,
        "ordered_sub_task_count": len(ordered_plan.sub_tasks),
        "flat_sub_task_count": len(flat_plan.sub_tasks),
    }
