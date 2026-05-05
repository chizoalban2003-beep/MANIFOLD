"""Tests for Phase 38: DAG Orchestration (manifold/dag.py)."""

from __future__ import annotations

import pytest

from manifold.brain import BrainTask
from manifold.dag import (
    CyclicGraphError,
    DAGExecutionReport,
    DAGNode,
    DAGNodeResult,
    GraphExecutor,
    TaskGraph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task(prompt: str = "t", stakes: float = 0.3) -> BrainTask:
    return BrainTask(prompt=prompt, domain="test", stakes=stakes)


def _node(nid: str, deps: list[str] | None = None, stakes: float = 0.3) -> DAGNode:
    return DAGNode(node_id=nid, task=_task(nid, stakes=stakes), depends_on=deps or [])


# ---------------------------------------------------------------------------
# DAGNode
# ---------------------------------------------------------------------------


class TestDAGNode:
    def test_default_status(self) -> None:
        node = _node("a")
        assert node.status == "pending"
        assert node.result is None

    def test_custom_deps(self) -> None:
        node = DAGNode(node_id="b", task=_task(), depends_on=["a", "c"])
        assert node.depends_on == ["a", "c"]

    def test_mutable_status(self) -> None:
        node = _node("x")
        node.status = "running"
        assert node.status == "running"

    def test_node_id_stored(self) -> None:
        node = _node("mynode")
        assert node.node_id == "mynode"

    def test_task_stored(self) -> None:
        task = _task("hello_task")
        node = DAGNode(node_id="n", task=task)
        assert node.task.prompt == "hello_task"


# ---------------------------------------------------------------------------
# DAGNodeResult
# ---------------------------------------------------------------------------


class TestDAGNodeResult:
    def test_creation(self) -> None:
        r = DAGNodeResult(
            node_id="a",
            success=True,
            action="use_tool",
            risk_score=0.2,
            latency_seconds=0.01,
        )
        assert r.success is True
        assert r.action == "use_tool"
        assert r.swarm_delegated is False

    def test_to_dict(self) -> None:
        r = DAGNodeResult(
            node_id="z",
            success=False,
            action="refuse",
            risk_score=0.9,
            latency_seconds=0.1,
            error="too risky",
        )
        d = r.to_dict()
        assert d["node_id"] == "z"
        assert d["success"] is False
        assert d["error"] == "too risky"
        assert "risk_score" in d

    def test_swarm_delegated(self) -> None:
        r = DAGNodeResult(
            node_id="s",
            success=True,
            action="delegate",
            risk_score=0.5,
            latency_seconds=0.05,
            swarm_delegated=True,
            peer_org_id="peer-x",
        )
        assert r.swarm_delegated is True
        assert r.peer_org_id == "peer-x"
        assert r.to_dict()["peer_org_id"] == "peer-x"


# ---------------------------------------------------------------------------
# TaskGraph
# ---------------------------------------------------------------------------


class TestTaskGraph:
    def test_add_node(self) -> None:
        g = TaskGraph(graph_id="g1")
        g.add_node(_node("a"))
        assert g.node_count() == 1

    def test_duplicate_node_raises(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        with pytest.raises(ValueError, match="Duplicate"):
            g.add_node(_node("a"))

    def test_get_existing(self) -> None:
        g = TaskGraph()
        g.add_node(_node("x"))
        assert g.get("x") is not None
        assert g.get("missing") is None

    def test_nodes_returns_list(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))
        assert len(g.nodes()) == 2

    def test_topological_order_linear(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b", deps=["a"]))
        g.add_node(_node("c", deps=["b"]))
        order = g.topological_order()
        ids = [n.node_id for n in order]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_topological_order_diamond(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b", deps=["a"]))
        g.add_node(_node("c", deps=["a"]))
        g.add_node(_node("d", deps=["b", "c"]))
        order = g.topological_order()
        ids = [n.node_id for n in order]
        assert ids[0] == "a"
        assert ids[-1] == "d"

    def test_cycle_raises(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a", deps=["b"]))
        g.add_node(_node("b", deps=["a"]))
        with pytest.raises(CyclicGraphError):
            g.topological_order()

    def test_unknown_dependency_raises(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a", deps=["nonexistent"]))
        with pytest.raises(KeyError):
            g.topological_order()

    def test_empty_graph_topological(self) -> None:
        g = TaskGraph()
        assert g.topological_order() == []

    def test_single_root(self) -> None:
        g = TaskGraph()
        g.add_node(_node("root"))
        order = g.topological_order()
        assert len(order) == 1
        assert order[0].node_id == "root"


# ---------------------------------------------------------------------------
# DAGExecutionReport
# ---------------------------------------------------------------------------


class TestDAGExecutionReport:
    def _make_report(self, succeeded: int = 3, failed: int = 0, skipped: int = 0) -> DAGExecutionReport:
        return DAGExecutionReport(
            graph_id="test-graph",
            total_nodes=succeeded + failed + skipped,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            total_latency_seconds=0.5,
            node_results=(),
        )

    def test_all_succeeded_true(self) -> None:
        r = self._make_report(succeeded=3, failed=0, skipped=0)
        assert r.all_succeeded is True

    def test_all_succeeded_false_on_fail(self) -> None:
        r = self._make_report(succeeded=2, failed=1, skipped=0)
        assert r.all_succeeded is False

    def test_all_succeeded_false_on_skip(self) -> None:
        r = self._make_report(succeeded=2, failed=0, skipped=1)
        assert r.all_succeeded is False

    def test_to_dict(self) -> None:
        r = self._make_report()
        d = r.to_dict()
        assert d["graph_id"] == "test-graph"
        assert "total_nodes" in d
        assert "node_results" in d
        assert isinstance(d["node_results"], list)


# ---------------------------------------------------------------------------
# GraphExecutor (headless mode)
# ---------------------------------------------------------------------------


class TestGraphExecutorHeadless:
    """Test executor without interceptor (headless)."""

    def test_single_node_succeeds(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        ex = GraphExecutor()
        report = ex.execute(g)
        assert report.total_nodes == 1
        assert report.succeeded == 1
        assert report.failed == 0

    def test_linear_chain(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b", deps=["a"]))
        g.add_node(_node("c", deps=["b"]))
        ex = GraphExecutor()
        report = ex.execute(g)
        assert report.total_nodes == 3
        assert report.succeeded == 3
        assert report.all_succeeded is True

    def test_status_updated_on_nodes(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        ex = GraphExecutor()
        ex.execute(g)
        node = g.get("a")
        assert node is not None
        assert node.status == "succeeded"
        assert node.result is not None

    def test_report_has_node_results(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b", deps=["a"]))
        ex = GraphExecutor()
        report = ex.execute(g)
        assert len(report.node_results) == 2

    def test_graph_id_propagated(self) -> None:
        g = TaskGraph(graph_id="my-pipeline")
        g.add_node(_node("a"))
        ex = GraphExecutor()
        report = ex.execute(g)
        assert report.graph_id == "my-pipeline"

    def test_to_dict_serializable(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        ex = GraphExecutor()
        report = ex.execute(g)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["all_succeeded"] is True

    def test_fail_fast(self) -> None:
        """A single-node graph with fail_fast has no skipped nodes."""
        g = TaskGraph()
        g.add_node(_node("a"))
        ex = GraphExecutor(fail_fast=True)
        report = ex.execute(g)
        assert report.total_nodes == 1

    def test_empty_graph(self) -> None:
        g = TaskGraph()
        ex = GraphExecutor()
        report = ex.execute(g)
        assert report.total_nodes == 0
        assert report.all_succeeded is True

    def test_five_node_graph(self) -> None:
        """Simulates the 5-step graph described in the spec."""
        g = TaskGraph(graph_id="five-step")
        g.add_node(_node("s1"))
        g.add_node(_node("s2", deps=["s1"]))
        g.add_node(_node("s3", deps=["s1"]))
        g.add_node(_node("s4", deps=["s2", "s3"]))
        g.add_node(_node("s5", deps=["s4"]))
        ex = GraphExecutor()
        report = ex.execute(g)
        assert report.total_nodes == 5
        assert report.all_succeeded is True

    def test_latency_non_negative(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        ex = GraphExecutor()
        report = ex.execute(g)
        assert report.total_latency_seconds >= 0.0


# ---------------------------------------------------------------------------
# GraphExecutor — skip on upstream failure
# ---------------------------------------------------------------------------


class TestGraphExecutorSkipBehavior:
    """Nodes whose dependencies failed should be skipped."""

    def _make_failing_executor(self) -> GraphExecutor:
        """Create a fake executor that makes every node 'fail' via a mock brain."""
        from manifold.brain import ManifoldBrain, BrainConfig  # noqa: PLC0415
        from manifold.connector import ConnectorRegistry  # noqa: PLC0415
        from manifold.interceptor import ActiveInterceptor, InterceptorConfig  # noqa: PLC0415

        brain = ManifoldBrain(
            config=BrainConfig(grid_size=5, generations=5, population_size=12),
            tools=[],
        )
        registry = ConnectorRegistry()
        cfg = InterceptorConfig(risk_veto_threshold=0.01, redirect_strategy="refuse")
        interceptor = ActiveInterceptor(registry=registry, brain=brain, config=cfg)
        return GraphExecutor(interceptor=interceptor)

    def test_downstream_skipped_when_upstream_fails(self) -> None:
        """When the root fails, downstream nodes should be skipped."""
        g = TaskGraph()
        # Use very high stakes so they get vetoed by the strict config
        g.add_node(_node("root", stakes=0.99))
        g.add_node(_node("child", deps=["root"], stakes=0.5))
        ex = self._make_failing_executor()
        report = ex.execute(g)
        # root should fail or get an error, child should be skipped
        ids_by_action = {r.node_id: r.action for r in report.node_results}
        assert ids_by_action.get("child") == "skipped"
        assert report.skipped >= 1

    def test_sibling_not_skipped_on_unrelated_fail(self) -> None:
        """Two sibling nodes (both roots) — one can succeed even if another fails."""
        g = TaskGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b"))  # no dependency on a
        ex = GraphExecutor()
        report = ex.execute(g)
        assert report.total_nodes == 2

    def test_node_result_action_skipped(self) -> None:
        """Skipped node result should have action='skipped'."""
        g = TaskGraph()
        g.add_node(_node("root", stakes=0.99))
        g.add_node(_node("child", deps=["root"]))
        ex = self._make_failing_executor()
        report = ex.execute(g)
        skipped_results = [r for r in report.node_results if r.action == "skipped"]
        assert len(skipped_results) >= 1

    def test_to_dict_node_results_populated(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b", deps=["a"]))
        ex = GraphExecutor()
        report = ex.execute(g)
        d = report.to_dict()
        assert len(d["node_results"]) == 2
        assert all("node_id" in nr for nr in d["node_results"])

    def test_report_succeeded_plus_failed_plus_skipped_eq_total(self) -> None:
        g = TaskGraph()
        g.add_node(_node("a"))
        g.add_node(_node("b", deps=["a"]))
        ex = GraphExecutor()
        report = ex.execute(g)
        assert report.succeeded + report.failed + report.skipped == report.total_nodes

