"""Phase 38: DAG Orchestration — Multi-Step Workflow Execution.

``TaskGraph`` models a Directed Acyclic Graph (DAG) of
:class:`~manifold.brain.BrainTask` objects.  The ``GraphExecutor`` resolves
execution order via Kahn's topological-sort algorithm (standard-library only:
:mod:`collections.deque` + :mod:`collections.defaultdict`) and processes each
node through the ``ActiveInterceptor`` and, optionally, the ``SwarmRouter``.

Key classes
-----------
``DAGNode``
    A single step in the task graph with optional upstream dependencies.
``DAGNodeStatus``
    Enum-like literals describing a node's lifecycle state.
``DAGNodeResult``
    Outcome of executing one node.
``TaskGraph``
    Container that holds nodes and edges; validates acyclicity.
``GraphExecutor``
    Topological-sort engine that processes a ``TaskGraph``.
``DAGExecutionReport``
    Final report summarising the full graph run.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Literal

from .brain import BrainTask


# ---------------------------------------------------------------------------
# DAGNodeStatus
# ---------------------------------------------------------------------------

DAGNodeStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


# ---------------------------------------------------------------------------
# DAGNode
# ---------------------------------------------------------------------------


@dataclass
class DAGNode:
    """A single step in a task graph.

    Parameters
    ----------
    node_id:
        Unique identifier within the graph (e.g. ``"step-1"``).
    task:
        The :class:`~manifold.brain.BrainTask` to execute at this node.
    depends_on:
        List of *node_id* strings that must complete successfully before this
        node may start.  Empty list = root node.

    Notes
    -----
    Nodes are mutable; the executor updates ``status`` and ``result`` in-place
    as the graph runs.
    """

    node_id: str
    task: BrainTask
    depends_on: list[str] = field(default_factory=list)
    status: DAGNodeStatus = "pending"
    result: "DAGNodeResult | None" = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# DAGNodeResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DAGNodeResult:
    """Outcome of executing one :class:`DAGNode`.

    Attributes
    ----------
    node_id:
        The node that was executed.
    success:
        ``True`` if the node completed without error.
    action:
        Brain decision action string (e.g. ``"use_tool"``).
    risk_score:
        Risk score produced by the brain/interceptor.
    latency_seconds:
        Wall-clock time spent on this node.
    error:
        Human-readable error description, or ``""`` on success.
    swarm_delegated:
        ``True`` if the node was delegated to a swarm peer.
    peer_org_id:
        Org ID of the swarm peer that handled this node (empty if local).
    """

    node_id: str
    success: bool
    action: str
    risk_score: float
    latency_seconds: float
    error: str = ""
    swarm_delegated: bool = False
    peer_org_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "success": self.success,
            "action": self.action,
            "risk_score": round(self.risk_score, 4),
            "latency_seconds": round(self.latency_seconds, 4),
            "error": self.error,
            "swarm_delegated": self.swarm_delegated,
            "peer_org_id": self.peer_org_id,
        }


# ---------------------------------------------------------------------------
# TaskGraph
# ---------------------------------------------------------------------------


class CyclicGraphError(ValueError):
    """Raised when a ``TaskGraph`` contains a cycle."""


@dataclass
class TaskGraph:
    """Directed Acyclic Graph of :class:`DAGNode` objects.

    Parameters
    ----------
    graph_id:
        Human-readable identifier for this graph run.

    Example
    -------
    ::

        graph = TaskGraph(graph_id="pipeline-1")
        graph.add_node(DAGNode("a", task_a))
        graph.add_node(DAGNode("b", task_b, depends_on=["a"]))
        graph.add_node(DAGNode("c", task_c, depends_on=["a"]))
        graph.add_node(DAGNode("d", task_d, depends_on=["b", "c"]))
    """

    graph_id: str = "default"
    _nodes: dict[str, DAGNode] = field(default_factory=dict, init=False, repr=False)

    def add_node(self, node: DAGNode) -> None:
        """Add *node* to the graph.

        Parameters
        ----------
        node:
            The node to add.

        Raises
        ------
        ValueError
            If a node with the same ``node_id`` already exists.
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node_id={node.node_id!r}")
        self._nodes[node.node_id] = node

    def nodes(self) -> list[DAGNode]:
        """Return all nodes in insertion order."""
        return list(self._nodes.values())

    def node_count(self) -> int:
        """Return the total number of nodes."""
        return len(self._nodes)

    def get(self, node_id: str) -> DAGNode | None:
        """Return the node with *node_id*, or ``None``."""
        return self._nodes.get(node_id)

    def topological_order(self) -> list[DAGNode]:
        """Return nodes in a valid topological (Kahn's algorithm) order.

        Raises
        ------
        CyclicGraphError
            If the graph contains a cycle.
        KeyError
            If a ``depends_on`` references an unknown node.
        """
        # Build in-degree map and adjacency list
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        children: dict[str, list[str]] = defaultdict(list)

        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise KeyError(
                        f"Node {node.node_id!r} depends on unknown node {dep!r}"
                    )
                in_degree[node.node_id] += 1
                children[dep].append(node.node_id)

        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        order: list[DAGNode] = []

        while queue:
            nid = queue.popleft()
            order.append(self._nodes[nid])
            for child_id in children[nid]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(order) != len(self._nodes):
            raise CyclicGraphError(
                f"TaskGraph {self.graph_id!r} contains a cycle — "
                f"only {len(order)}/{len(self._nodes)} nodes resolved."
            )
        return order


# ---------------------------------------------------------------------------
# DAGExecutionReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DAGExecutionReport:
    """Final report from a :class:`GraphExecutor` run.

    Attributes
    ----------
    graph_id:
        Identifier of the graph that was executed.
    total_nodes:
        Number of nodes in the graph.
    succeeded:
        Number of nodes that completed successfully.
    failed:
        Number of nodes that failed.
    skipped:
        Number of nodes skipped because an upstream dependency failed.
    total_latency_seconds:
        Total wall-clock time for the entire execution.
    node_results:
        Ordered list of per-node results.
    """

    graph_id: str
    total_nodes: int
    succeeded: int
    failed: int
    skipped: int
    total_latency_seconds: float
    node_results: tuple[DAGNodeResult, ...]

    @property
    def all_succeeded(self) -> bool:
        """``True`` if every node succeeded (no failures or skips)."""
        return self.failed == 0 and self.skipped == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "total_nodes": self.total_nodes,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_latency_seconds": round(self.total_latency_seconds, 4),
            "all_succeeded": self.all_succeeded,
            "node_results": [r.to_dict() for r in self.node_results],
        }


# ---------------------------------------------------------------------------
# GraphExecutor
# ---------------------------------------------------------------------------


@dataclass
class GraphExecutor:
    """Topological-sort engine that processes a :class:`TaskGraph`.

    Each node is evaluated through the
    :class:`~manifold.interceptor.ActiveInterceptor` (required) and
    optionally routed to a swarm peer via
    :class:`~manifold.swarm.SwarmRouter`.

    Parameters
    ----------
    interceptor:
        The ``ActiveInterceptor`` used for pre-flight risk checks.  If
        ``None``, the executor runs in *headless* mode — it evaluates the
        task directly via the brain without routing through the interceptor.
    swarm_router:
        Optional ``SwarmRouter``.  When provided and a node's task has
        ``stakes > swarm_threshold``, the executor will attempt delegation.
    swarm_threshold:
        Minimum task stakes required for swarm delegation consideration.
        Default: ``0.8``.
    fail_fast:
        When ``True``, abort the graph on the first node failure.
        Default: ``False``.

    Example
    -------
    ::

        executor = GraphExecutor(interceptor=interceptor, swarm_router=router)
        report = executor.execute(graph)
        if report.all_succeeded:
            print("Pipeline complete!")
    """

    interceptor: Any = None  # ActiveInterceptor | None
    swarm_router: Any = None  # SwarmRouter | None
    swarm_threshold: float = 0.8
    fail_fast: bool = False

    def execute(self, graph: TaskGraph) -> DAGExecutionReport:
        """Execute all nodes in *graph* respecting topological order.

        Nodes whose dependencies failed are automatically skipped.

        Parameters
        ----------
        graph:
            The task graph to run.

        Returns
        -------
        DAGExecutionReport
        """
        start = time.monotonic()
        order = graph.topological_order()
        failed_nodes: set[str] = set()
        results: list[DAGNodeResult] = []
        succeeded = failed = skipped = 0

        for node in order:
            # Skip if any upstream dependency failed
            if any(dep in failed_nodes for dep in node.depends_on):
                node.status = "skipped"
                results.append(
                    DAGNodeResult(
                        node_id=node.node_id,
                        success=False,
                        action="skipped",
                        risk_score=0.0,
                        latency_seconds=0.0,
                        error="upstream dependency failed",
                    )
                )
                skipped += 1
                continue

            node.status = "running"
            node_result = self._execute_node(node)
            node.result = node_result
            results.append(node_result)

            if node_result.success:
                node.status = "succeeded"
                succeeded += 1
            else:
                node.status = "failed"
                failed += 1
                failed_nodes.add(node.node_id)
                if self.fail_fast:
                    # Mark remaining nodes as skipped
                    processed_ids = {r.node_id for r in results}
                    for remaining in order:
                        if remaining.node_id not in processed_ids:
                            remaining.status = "skipped"
                            results.append(
                                DAGNodeResult(
                                    node_id=remaining.node_id,
                                    success=False,
                                    action="skipped",
                                    risk_score=0.0,
                                    latency_seconds=0.0,
                                    error="fail_fast triggered",
                                )
                            )
                            skipped += 1
                    break

        total_latency = time.monotonic() - start
        return DAGExecutionReport(
            graph_id=graph.graph_id,
            total_nodes=graph.node_count(),
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            total_latency_seconds=total_latency,
            node_results=tuple(results),
        )

    def _execute_node(self, node: DAGNode) -> DAGNodeResult:
        """Execute a single node, routing through interceptor/swarm."""
        t0 = time.monotonic()
        task = node.task

        # Try swarm delegation first when stakes exceed threshold
        if (
            self.swarm_router is not None
            and task.stakes >= self.swarm_threshold
        ):
            swarm_result = self.swarm_router.route(task)
            if swarm_result.delegated:
                latency = time.monotonic() - t0
                return DAGNodeResult(
                    node_id=node.node_id,
                    success=True,
                    action="delegate",
                    risk_score=task.stakes,
                    latency_seconds=latency,
                    swarm_delegated=True,
                    peer_org_id=swarm_result.peer.org_id
                    if swarm_result.peer
                    else "",
                )

        # Run through interceptor if available
        if self.interceptor is not None:
            try:
                intercept_result = self.interceptor.intercept(
                    task, requested_tool="dag_executor"
                )
                latency = time.monotonic() - t0
                return DAGNodeResult(
                    node_id=node.node_id,
                    success=intercept_result.permitted,
                    action=intercept_result.manifold_decision.action,
                    risk_score=intercept_result.risk_score,
                    latency_seconds=latency,
                    error="" if intercept_result.permitted else intercept_result.reason,
                )
            except KeyError:
                # Tool not found — fall through to direct brain evaluation
                pass

        # Headless mode: evaluate via brain directly
        try:
            from .brain import ManifoldBrain  # noqa: PLC0415

            brain: ManifoldBrain | None = None
            if self.interceptor is not None:
                brain = self.interceptor.brain
            if brain is None:
                latency = time.monotonic() - t0
                return DAGNodeResult(
                    node_id=node.node_id,
                    success=True,
                    action="execute",
                    risk_score=task.stakes,
                    latency_seconds=latency,
                )
            decision = brain.decide(task)
            latency = time.monotonic() - t0
            success = decision.action not in {"refuse", "escalate"}
            return DAGNodeResult(
                node_id=node.node_id,
                success=success,
                action=decision.action,
                risk_score=decision.risk_score,
                latency_seconds=latency,
                error="" if success else f"brain action={decision.action!r}",
            )
        except Exception as exc:  # noqa: BLE001
            latency = time.monotonic() - t0
            return DAGNodeResult(
                node_id=node.node_id,
                success=False,
                action="error",
                risk_score=0.0,
                latency_seconds=latency,
                error=str(exc),
            )
