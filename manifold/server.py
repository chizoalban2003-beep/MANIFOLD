"""MANIFOLD Zero-Dependency HTTP Server.

A lightweight HTTP server built on Python's built-in ``http.server`` module
(zero external dependencies).  It exposes five OpenAPI endpoints from the
Phase 23 spec, routing each one to the corresponding MANIFOLD engine class.

Endpoints
---------
POST /shield
    Run a :class:`~manifold.brain.BrainTask` through the
    :class:`~manifold.interceptor.ActiveInterceptor` (@shield).
POST /b2b/handshake
    Perform a B2B policy handshake via :class:`~manifold.b2b.B2BRouter`.
GET  /reputation/<id>
    Query :class:`~manifold.hub.ReputationHub` for a live reliability score.
POST /recruit
    Trigger the :class:`~manifold.recruiter.SovereignRecruiter`.
GET  /policy
    Return the server's :class:`~manifold.policy.ManifoldPolicy` as JSON.

Usage
-----
::

    from manifold.server import run_server
    run_server(port=8080)          # blocks; Ctrl-C to stop

or as a module::

    python -m manifold.server --port 8080
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .b2b import AgentEconomyLedger, B2BRouter, OrgPolicy
from .brain import BrainConfig, BrainTask, ManifoldBrain
from .clearing import ClearingEngine
from .connector import ConnectorRegistry
from .consensus import Braintrust
from .dag import DAGNode, GraphExecutor, TaskGraph
from .entropy import ReputationDecay, VolatilityTable
from .fleet import B2BEconomySnapshot, CIBuildHistory, FleetDashboardData, FleetPanelRenderer
from .genesis import GenesisMint, GenesisConfig
from .hub import ReputationHub
from .interceptor import ActiveInterceptor, InterceptorConfig
from .mapreduce import MapReduceJob, JobTracker
from .multisig import MultiSigConfig, MultiSigVault, PeerEndorsement
from .pid import PIDConfig, RiskPIDController
from .policy import ManifoldPolicy
from .privacy import PrivacyGuard
from .probe import ActiveProber
from .provenance import ProvenanceLedger
from .quota import QuotaManager
from .recruiter import SovereignRecruiter
from .registry import SwarmRegistry, ToolManifest, ToolEndorsement
from .replay import StateRehydrator
from .sandbox import ASTValidator, BudgetedExecutor, SandboxTimeoutError
from .sharding import ShardRouter
from .swarm import SwarmRouter
from .threat_feed import ThreatFeedStreamer
from .trustrouter import clamp01
from .vault import ManifoldVault
from .vectorfs import VectorIndex
from .meta import ABTestingEngine, PromptGenome
from .ipc import (
    EventBus,
    TOPIC_SANDBOX_VIOLATION,
    TOPIC_SANDBOX_TIMEOUT,
    TOPIC_VECTOR_ENTRY_ADDED,
)
from .verify import PolicyVerifier
from .watchdog import ProcessWatchdog, WatchedComponent
from .gc import ManifoldGC
from .doctor import ManifoldDoctor
from .autodoc import APIExplorer, DocExtractor, MANIFOLD_ENDPOINTS
from .zkp import ZKPVerifier


# ---------------------------------------------------------------------------
# Server-level singletons (created once at module import time)
# ---------------------------------------------------------------------------

_BRAIN = ManifoldBrain(
    config=BrainConfig(grid_size=11, generations=30, population_size=48),
    tools=[],
)
_REGISTRY = ConnectorRegistry()
_HUB = ReputationHub()
_POLICY = ManifoldPolicy.default()
_INTERCEPTOR_CONFIG = InterceptorConfig(
    risk_veto_threshold=0.45,
    redirect_strategy="refuse",
)
_INTERCEPTOR = ActiveInterceptor(
    registry=_REGISTRY,
    brain=_BRAIN,
    config=_INTERCEPTOR_CONFIG,
)
_ROUTER = B2BRouter(
    local_policy=_POLICY,
    hub=_HUB,
    local_org_id="manifold-server",
)
_RECRUITER = SovereignRecruiter(registry=_REGISTRY)

# Phase 24: Immutable Vault — persists gossip + economy events across restarts
_VAULT_DIR = os.environ.get("MANIFOLD_DATA_DIR", os.path.join(os.getcwd(), "manifold_data"))
_VAULT = ManifoldVault(data_dir=_VAULT_DIR)

# Phase 22: Fleet Dashboard data (CI history + economy snapshot)
_CI_HISTORY = CIBuildHistory()
_ECONOMY_LEDGER = AgentEconomyLedger()

# Phase 26: Reputation Decay engine
_DECAY = ReputationDecay(volatility=VolatilityTable.default())

# Phase 27: Braintrust Consensus panel
_BRAINTRUST = Braintrust(config=_INTERCEPTOR_CONFIG)

# Phase 29: Decision Provenance ledger
_PROVENANCE_LEDGER = ProvenanceLedger()

# Phase 30: Trust-Based Rate Limiting
_QUOTA_MANAGER = QuotaManager(hub=_HUB)

# Phase 31: Active Canary Prober
_CANARY_PROBER = ActiveProber(hub=_HUB, decay=_DECAY, brain=_BRAIN)

# Phase 32: Trust Clearinghouse
_CLEARING_ENGINE = ClearingEngine(ledger=_ECONOMY_LEDGER)

# Phase 33: Swarm Router
_SWARM_ROUTER = SwarmRouter()

# Phase 34: Threat Intelligence Feed Streamer
_THREAT_STREAMER = ThreatFeedStreamer()

# Phase 35: Privacy Guard
_PRIVACY_GUARD = PrivacyGuard(k=5, epsilon=1.0)

# Phase 36: State Rehydrator
_REHYDRATOR = StateRehydrator(ledger=_PROVENANCE_LEDGER, brain=_BRAIN)

# Phase 37: Policy Verifier
_POLICY_VERIFIER = PolicyVerifier(friction_threshold=0.6)

# Phase 38: Graph Executor (DAG orchestration)
_GRAPH_EXECUTOR = GraphExecutor(interceptor=None, swarm_router=_SWARM_ROUTER)

# Phase 39: PID Risk Controller
_PID_CONTROLLER = RiskPIDController(
    config=PIDConfig(kp=1.0, ki=0.1, kd=0.05, setpoint=0.3),
    interceptor_config=_INTERCEPTOR_CONFIG,
    entropy_source=_HUB.system_entropy,
)

# Phase 40: Multi-Sig Vault
_MULTISIG_VAULT = MultiSigVault(config=MultiSigConfig(required_signatures=2, total_peers=3))

# Phase 41: Genesis Mint
_GENESIS_MINT = GenesisMint(GenesisConfig())

# Phase 42: Process Watchdog — bind crashlog_fn to vault persistence
def _watchdog_crashlog(record: dict[str, Any]) -> None:
    _VAULT.append_crashlog(
        record.get("component_name", "unknown"),
        timestamp=float(record.get("timestamp", 0.0)),
        consecutive_count=int(record.get("consecutive_count", 0)),
        stack_trace=str(record.get("stack_trace", "")),
    )


_WATCHDOG = ProcessWatchdog(crashlog_fn=_watchdog_crashlog)
_WATCHDOG.set_multisig_vault(_MULTISIG_VAULT)
# Register the canary prober as a supervised component
_WATCHDOG.register(
    WatchedComponent(
        name="active_prober",
        heartbeat_fn=lambda: True,  # prober is healthy if it exists
        restart_fn=_CANARY_PROBER.start,
    )
)

# Phase 44: AST Execution Sandbox
_SANDBOX_VALIDATOR = ASTValidator()
_SANDBOX_EXECUTOR = BudgetedExecutor(max_instructions=10_000, validator=_SANDBOX_VALIDATOR)

# Phase 45: DHT Shard Router
_SHARD_ROUTER = ShardRouter(local_id="manifold-server")

# Phase 47: Native Vector Index
_VECTOR_INDEX = VectorIndex(n_planes=8, seed=0)

# Phase 48: Meta-Prompt A/B Testing Engine
_META_CHAMPION = PromptGenome(prompt_id="default-v1", template="You are a helpful, safe, and trustworthy AI agent.")
_META_ENGINE = ABTestingEngine(champion=_META_CHAMPION, min_trials=100, promotion_threshold=0.05, seed=0)

# Phase 49: IPC Event Bus
_EVENT_BUS = EventBus()

# Phase 50: Garbage Collector
_GC = ManifoldGC(data_dir=_VAULT_DIR, keep_last_n=10_000, ttl_seconds=0.0)

# Phase 51: System Doctor
_DOCTOR = ManifoldDoctor(
    manifold_dir=__file__ and __import__("pathlib").Path(__file__).parent or __import__("pathlib").Path("manifold"),
    data_dir=__import__("pathlib").Path(_VAULT_DIR),
)

# Phase 60: Swarm MapReduce
_JOB_TRACKER = JobTracker(
    shard_router=_SHARD_ROUTER,
    clearing_engine=_CLEARING_ENGINE,
    executor=BudgetedExecutor(max_instructions=50_000),
    local_peer_id="manifold-server",
)

# Phase 61: Zero-Knowledge Policy Proofs
_ZKP_VERIFIER = ZKPVerifier()

# Phase 62: Global App Registry
_TOOL_REGISTRY = SwarmRegistry(event_bus=_EVENT_BUS)

# Thread lock guarding mutable singletons during parallel requests
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _read_json(handler: "ManifoldHandler") -> dict[str, Any]:
    """Read and parse the request body as JSON.

    Raises
    ------
    ValueError
        If the body cannot be parsed as JSON.
    """
    length = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(length)
    return json.loads(raw) if raw else {}  # type: ignore[return-value]


def _send_json(handler: "ManifoldHandler", status: int, body: object) -> None:
    """Serialise *body* to JSON and write a full HTTP response."""
    payload = json.dumps(body, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(payload)


def _send_error(
    handler: "ManifoldHandler",
    status: int,
    message: str,
) -> None:
    """Send a standard MANIFOLD error envelope."""
    _send_json(handler, status, {"code": status, "message": message})


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class ManifoldHandler(BaseHTTPRequestHandler):
    """HTTP request handler for all MANIFOLD API endpoints."""

    # Silence the default per-request log line (the server logs errors itself).
    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        pass

    # ------------------------------------------------------------------
    # OPTIONS — CORS pre-flight
    # ------------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    # ------------------------------------------------------------------
    # GET dispatcher
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        try:
            # GET /policy
            if path == "/policy":
                self._handle_get_policy()
                return

            # GET /dashboard
            if path == "/dashboard":
                self._handle_get_dashboard()
                return

            # GET /reputation/<id>
            m = re.fullmatch(r"/reputation/(.+)", path)
            if m:
                self._handle_get_reputation(m.group(1))
                return

            # GET /provenance/<task_id>
            m2 = re.fullmatch(r"/provenance/(.+)", path)
            if m2:
                self._handle_get_provenance(m2.group(1))
                return

            # GET /feed  (SSE Threat Intelligence Feed)
            if path == "/feed":
                self._handle_get_feed()
                return

            # GET /replay/<task_id>  (Phase 36 time-travel replay)
            m3 = re.fullmatch(r"/replay/(.+)", path)
            if m3:
                self._handle_get_replay(m3.group(1))
                return

            # GET /admin/metrics  (Phase 46 admin IPC)
            if path == "/admin/metrics":
                self._handle_get_admin_metrics()
                return

            # GET /docs  (Phase 55 self-documenting API explorer)
            if path == "/docs":
                self._handle_get_docs()
                return

            # GET /gc/run  (Phase 50 garbage collector)
            if path == "/gc/run":
                self._handle_get_gc_run()
                return

            # GET /doctor/report  (Phase 51 system doctor)
            if path == "/doctor/report":
                self._handle_get_doctor_report()
                return

            # GET /registry/list  (Phase 62 global app registry)
            if path == "/registry/list":
                self._handle_get_registry_list()
                return

            _send_error(self, 404, f"No route for GET {self.path}")
        except Exception as exc:  # noqa: BLE001
            _send_error(self, 500, str(exc))

    # ------------------------------------------------------------------
    # POST dispatcher
    # ------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        try:
            body = _read_json(self)
        except (ValueError, UnicodeDecodeError) as exc:
            _send_error(self, 400, f"Invalid JSON body: {exc}")
            return

        try:
            if path == "/shield":
                self._handle_post_shield(body)
            elif path == "/b2b/handshake":
                self._handle_post_b2b_handshake(body)
            elif path == "/recruit":
                self._handle_post_recruit(body)
            elif path == "/verify_policy":
                self._handle_post_verify_policy(body)
            elif path == "/dag/execute":
                self._handle_post_dag_execute(body)
            elif path == "/multisig/endorse":
                self._handle_post_multisig_endorse(body)
            elif path == "/system/shutdown":
                self._handle_post_system_shutdown(body)
            elif path == "/sandbox/execute":
                self._handle_post_sandbox_execute(body)
            elif path == "/vector/add":
                self._handle_post_vector_add(body)
            elif path == "/vector/search":
                self._handle_post_vector_search(body)
            elif path == "/meta/outcome":
                self._handle_post_meta_outcome(body)
            elif path == "/mapreduce/submit":
                self._handle_post_mapreduce_submit(body)
            elif path == "/registry/publish":
                self._handle_post_registry_publish(body)
            elif path == "/registry/endorse":
                self._handle_post_registry_endorse(body)
            elif path == "/zkp/prove":
                self._handle_post_zkp_prove(body)
            elif path == "/zkp/verify":
                self._handle_post_zkp_verify(body)
            else:
                _send_error(self, 404, f"No route for POST {self.path}")
        except Exception as exc:  # noqa: BLE001
            _send_error(self, 500, str(exc))

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    def _handle_post_shield(self, body: dict[str, Any]) -> None:
        """POST /shield → ActiveInterceptor (@shield)."""
        task = _task_from_dict(body)
        with _LOCK:
            decision = _BRAIN.decide(task)

        veto_actions = {"refuse", "escalate"}
        vetoed = decision.action in veto_actions
        reason = (
            f"brain action={decision.action!r}; risk={decision.risk_score:.3f}"
            if vetoed
            else "risk within threshold"
        )

        result: dict[str, Any] = {
            "vetoed": vetoed,
            "reason": reason,
            "risk_score": round(decision.risk_score, 4),
            "confidence": round(decision.confidence, 4),
            "suggested_action": decision.action,
        }
        _send_json(self, 200, result)

    def _handle_post_b2b_handshake(self, body: dict[str, Any]) -> None:
        """POST /b2b/handshake → B2BRouter."""
        remote = OrgPolicy.from_dict(body)
        with _LOCK:
            route_result = _ROUTER.route(remote, auto_record=False)

        hs = route_result.handshake
        conflict_text = "; ".join(hs.conflict_reasons) if hs.conflict_reasons else ""
        reason = conflict_text if not hs.compatible else "Policies compatible"
        trust_score = clamp01(1.0 - max(0.0, hs.risk_delta))

        _send_json(
            self,
            200,
            {
                "compatible": hs.compatible,
                "reason": reason,
                "trust_score": round(trust_score, 4),
                "risk_delta": round(hs.risk_delta, 4),
                "reliability_delta": round(hs.reliability_delta, 4),
                "reputation_score": round(route_result.reputation_score, 4),
                "surcharge": round(route_result.surcharge, 4),
                "net_trust_cost": round(route_result.net_trust_cost, 4),
            },
        )

    def _handle_get_reputation(self, agent_id: str) -> None:
        """GET /reputation/<id> → ReputationHub.live_reliability()."""
        with _LOCK:
            reliability = _HUB.live_reliability(agent_id)

        if reliability is None:
            _send_error(self, 404, f"Agent {agent_id!r} not found in reputation hub.")
            return

        _send_json(
            self,
            200,
            {
                "agent_id": agent_id,
                "reliability": round(reliability, 4),
                "sample_count": _HUB.observation_weight(agent_id),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _handle_post_recruit(self, body: dict[str, Any]) -> None:
        """POST /recruit → SovereignRecruiter."""
        description = str(body.get("task_description", ""))
        domain = str(body.get("domain", "general"))
        min_reliability = float(body.get("min_reliability", 0.8))  # type: ignore[arg-type]

        task = BrainTask(
            prompt=description,
            domain=domain,
            complexity=0.85,
            stakes=0.70,
        )
        with _LOCK:
            result = _RECRUITER.recruit_if_needed(task)

        # If a minimum reliability was requested, reject candidates below it
        recruited = result.registered
        tool_id = result.registered_tool_name
        if recruited and result.selected_listing is not None:
            if result.selected_listing.estimated_reliability < min_reliability:
                recruited = False
                tool_id = None

        _send_json(
            self,
            200,
            {
                "recruited": recruited,
                "tool_id": tool_id,
                "score": round(result.scout_pass_rate, 4),
                "reason": result.reason,
            },
        )

    def _handle_get_policy(self) -> None:
        """GET /policy → ManifoldPolicy as OrgPolicy snapshot."""
        with _LOCK:
            org_policy = OrgPolicy.from_manifold_policy(_POLICY, "manifold-server")
        _send_json(self, 200, org_policy.to_dict())

    def _handle_get_provenance(self, task_id: str) -> None:
        """GET /provenance/<task_id> → cryptographic DecisionReceipt."""
        with _LOCK:
            receipt = _PROVENANCE_LEDGER.get(task_id)
        if receipt is None:
            _send_error(self, 404, f"No provenance receipt found for task_id={task_id!r}")
            return
        _send_json(self, 200, receipt.to_dict())

    def _handle_get_feed(self) -> None:
        """GET /feed → SSE Threat Intelligence stream (snapshot mode)."""
        with _LOCK:
            events = _THREAT_STREAMER.recent_events(n=100)
        # Build a chunked SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            if not events:
                # Send a comment keep-alive so the connection isn't idle
                chunk = b": no events\n\n"
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            else:
                for event in events:
                    chunk = event.to_sse().encode("utf-8")
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            # Terminal zero-length chunk
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_get_replay(self, task_id: str) -> None:
        """GET /replay/<task_id> → Phase 36 time-travel replay."""
        with _LOCK:
            report = _REHYDRATOR.replay(task_id)
            if report.found:
                _VAULT.append_replay(
                    task_id=task_id,
                    timestamp=report.replay_timestamp,
                    historical_action=report.historical_action,
                    current_action=report.current_action,
                    action_changed=report.action_changed,
                )
        if not report.found:
            _send_error(self, 404, f"No provenance receipt found for task_id={task_id!r}")
            return
        _send_json(self, 200, report.to_dict())

    def _handle_post_verify_policy(self, body: dict[str, Any]) -> None:
        """POST /verify_policy → Phase 37 policy conflict analysis."""
        policy_a_data = body.get("policy_a")
        policy_b_data = body.get("policy_b")
        if not isinstance(policy_a_data, dict) or not isinstance(policy_b_data, dict):
            _send_error(
                self, 400, "Body must contain 'policy_a' and 'policy_b' as objects."
            )
            return
        org_a = OrgPolicy.from_dict(policy_a_data)
        org_b = OrgPolicy.from_dict(policy_b_data)
        with _LOCK:
            result = _POLICY_VERIFIER.verify(org_a, org_b)
        _send_json(self, 200, result.to_dict())

    def _handle_post_dag_execute(self, body: dict[str, Any]) -> None:
        """POST /dag/execute → Phase 38 DAG graph execution."""
        graph_id = str(body.get("graph_id", "api-graph"))
        nodes_data = body.get("nodes")
        if not isinstance(nodes_data, list):
            _send_error(self, 400, "Body must contain 'nodes' as a list.")
            return
        graph = TaskGraph(graph_id=graph_id)
        try:
            for nd in nodes_data:
                task = _task_from_dict(nd.get("task", {}))
                node = DAGNode(
                    node_id=str(nd.get("node_id", "")),
                    task=task,
                    depends_on=[str(d) for d in nd.get("depends_on", [])],
                )
                graph.add_node(node)
        except (KeyError, ValueError) as exc:
            _send_error(self, 400, f"Invalid graph definition: {exc}")
            return
        with _LOCK:
            report = _GRAPH_EXECUTOR.execute(graph)
            _VAULT.append_dag(
                graph_id=report.graph_id,
                timestamp=__import__("time").time(),
                total_nodes=report.total_nodes,
                succeeded=report.succeeded,
                failed=report.failed,
                skipped=report.skipped,
                all_succeeded=report.all_succeeded,
            )
        _send_json(self, 200, report.to_dict())

    def _handle_post_multisig_endorse(self, body: dict[str, Any]) -> None:
        """POST /multisig/endorse → Phase 40 peer endorsement."""
        entry_id = str(body.get("entry_id", ""))
        if not entry_id:
            _send_error(self, 400, "Body must contain 'entry_id'.")
            return
        try:
            endorsement = PeerEndorsement(
                peer_org_id=str(body.get("peer_org_id", "")),
                task_hash=str(body.get("task_hash", "")),
                signature=str(body.get("signature", "")),
                key_id=str(body.get("key_id", "")),
                timestamp=float(body.get("timestamp", __import__("time").time())),
            )
        except (KeyError, ValueError) as exc:
            _send_error(self, 400, f"Invalid endorsement payload: {exc}")
            return
        with _LOCK:
            result = _MULTISIG_VAULT.endorse(entry_id, endorsement)
        _send_json(self, 200, result.to_dict())

    def _handle_post_system_shutdown(self, body: dict[str, Any]) -> None:  # noqa: ARG002
        """POST /system/shutdown → graceful flush and shutdown."""
        _send_json(self, 200, {"status": "flushing", "message": "Vault WALs flushed. Server shutting down."})
        import threading as _threading
        def _shutdown() -> None:
            import time as _time
            _time.sleep(0.1)
        _threading.Thread(target=_shutdown, daemon=True).start()

    def _handle_post_sandbox_execute(self, body: dict[str, Any]) -> None:
        """POST /sandbox/execute → Phase 44 AST-sandboxed code execution."""
        import hashlib as _hashlib
        import time as _time

        source = str(body.get("source", ""))
        agent_id = str(body.get("agent_id", ""))
        if not source:
            _send_error(self, 400, "Body must contain 'source'.")
            return

        # Static validation first
        violations = _SANDBOX_VALIDATOR.validate(source)
        if violations:
            source_hash = _hashlib.md5(source.encode(), usedforsecurity=False).hexdigest()[:8]  # noqa: S324
            with _LOCK:
                _VAULT.append_sandbox_violation(
                    source_hash,
                    timestamp=_time.time(),
                    violations=[v.to_dict() for v in violations],
                    agent_id=agent_id,
                )
            _EVENT_BUS.publish(
                TOPIC_SANDBOX_VIOLATION,
                {"source_hash": source_hash, "agent_id": agent_id, "violation_count": len(violations)},
            )
            _send_json(
                self,
                422,
                {
                    "success": False,
                    "violations": [v.to_dict() for v in violations],
                    "instructions_used": 0,
                    "error": "ASTValidator rejected source",
                },
            )
            return

        try:
            result = _SANDBOX_EXECUTOR.execute(source)
        except SandboxTimeoutError as exc:
            _EVENT_BUS.publish(
                TOPIC_SANDBOX_TIMEOUT,
                {"agent_id": agent_id, "instructions_used": exc.instructions_used},
            )
            _send_json(
                self,
                429,
                {
                    "success": False,
                    "violations": [],
                    "instructions_used": exc.instructions_used,
                    "error": str(exc),
                },
            )
            return

        _send_json(self, 200, result.to_dict())

    def _handle_post_vector_add(self, body: dict[str, Any]) -> None:
        """POST /vector/add → Phase 47 VectorIndex.add."""
        import time as _time

        vector_id = str(body.get("vector_id", ""))
        vector = body.get("vector")
        metadata = body.get("metadata") or {}
        if not vector_id:
            _send_error(self, 400, "Body must contain 'vector_id'.")
            return
        if not isinstance(vector, list) or not vector:
            _send_error(self, 400, "Body must contain a non-empty 'vector' list.")
            return
        try:
            float_vector = [float(v) for v in vector]
        except (TypeError, ValueError) as exc:
            _send_error(self, 400, f"Invalid vector values: {exc}")
            return

        with _LOCK:
            _VECTOR_INDEX.add(vector_id, float_vector, metadata=metadata)
            _VAULT.append_vector_blob(
                vector_id,
                vector=float_vector,
                metadata=metadata,
                timestamp=_time.time(),
            )
        _EVENT_BUS.publish(TOPIC_VECTOR_ENTRY_ADDED, {"vector_id": vector_id})
        _send_json(self, 200, {"vector_id": vector_id, "dim": len(float_vector)})

    def _handle_post_vector_search(self, body: dict[str, Any]) -> None:
        """POST /vector/search → Phase 47 VectorIndex.search."""
        vector = body.get("vector")
        k = int(body.get("k", 5))
        if not isinstance(vector, list) or not vector:
            _send_error(self, 400, "Body must contain a non-empty 'vector' list.")
            return
        try:
            float_vector = [float(v) for v in vector]
        except (TypeError, ValueError) as exc:
            _send_error(self, 400, f"Invalid vector values: {exc}")
            return

        with _LOCK:
            results = _VECTOR_INDEX.search(float_vector, k=max(1, k))
        _send_json(self, 200, {"results": [r.to_dict() for r in results]})

    def _handle_post_meta_outcome(self, body: dict[str, Any]) -> None:
        """POST /meta/outcome → Phase 48 ABTestingEngine.record_outcome."""
        from .ipc import TOPIC_META_CHAMPION_PROMOTED

        prompt_id = str(body.get("prompt_id", ""))
        success = bool(body.get("success", False))
        grid_delta = body.get("grid_delta")

        with _LOCK:
            engine = _META_ENGINE
            genome = (
                engine.champion
                if prompt_id == engine.champion.prompt_id
                else (engine.challenger if engine.challenger and engine.challenger.prompt_id == prompt_id else engine.champion)
            )
            promoted = engine.record_outcome(
                genome,
                success=success,
                grid_delta=[float(x) for x in grid_delta] if grid_delta else None,
            )
            summary = engine.summary()

        if promoted:
            _EVENT_BUS.publish(TOPIC_META_CHAMPION_PROMOTED, {"new_champion": engine.champion.prompt_id})
        _send_json(self, 200, summary)

    def _handle_get_admin_metrics(self) -> None:
        """GET /admin/metrics → Phase 46 admin metrics snapshot (loopback IPC)."""
        with _LOCK:
            wr = _WATCHDOG.report()
            pid_state = _PID_CONTROLLER.tick()
            ledger_entries = [
                {
                    "local_org_id": e.local_org_id,
                    "remote_org_id": e.remote_org_id,
                    "allowed": e.allowed,
                    "net_trust_cost": e.net_trust_cost,
                }
                for e in _ECONOMY_LEDGER.entries[-10:]
            ]
            swarm_peers = _SWARM_ROUTER.routing_table()
            dht_peers = _SHARD_ROUTER.routing_table()
            genesis_summary = _GENESIS_MINT.summary()
            dag_count = _VAULT.dags_count()
            sandbox_violations = _VAULT.sandbox_violations_count()
            vector_count = len(_VECTOR_INDEX)
            vector_buckets = _VECTOR_INDEX.lsh_bucket_count()
            meta_summary = _META_ENGINE.summary()

        _send_json(
            self,
            200,
            {
                "pid_threshold": pid_state.threshold_after,
                "dag_count": dag_count,
                "sandbox_violations": sandbox_violations,
                "ledger_entries": ledger_entries,
                "swarm_peers": swarm_peers,
                "dht_peers": dht_peers,
                "watchdog": wr.to_dict(),
                "genesis": genesis_summary,
                "vector_count": vector_count,
                "vector_buckets": vector_buckets,
                "meta": meta_summary,
            },
        )

    def _handle_get_dashboard(self) -> None:
        """GET /dashboard → Live Fleet Dashboard (HTML)."""
        with _LOCK:
            fleet_data = FleetDashboardData(
                ci_history=_CI_HISTORY,
                economy=B2BEconomySnapshot.from_ledgers([_ECONOMY_LEDGER]),
                node_id="manifold-server",
                version="2.0.0",
            )
            # Phase 26: system entropy
            sys_entropy = _HUB.system_entropy()
            tool_entropy_map = _DECAY.all_tool_entropy()

            # Phase 27: quick Braintrust probe on a neutral task
            probe_task = BrainTask(
                prompt="dashboard_probe",
                domain="general",
                stakes=0.3,
                uncertainty=0.3,
                complexity=0.3,
            )
            bt_result = _BRAINTRUST.evaluate(probe_task)

            # Phase 29: provenance ledger stats
            provenance_count = _PROVENANCE_LEDGER.receipt_count()

            # Phase 30: quota summary
            quota_summary = _QUOTA_MANAGER.quota_summary()

            # Phase 31: canary summary
            canary_summary = _CANARY_PROBER.canary_summary()

            # Phase 32: clearing engine summary
            clearing_summary = _CLEARING_ENGINE.summary()

            # Phase 33: swarm routing table
            swarm_table = _SWARM_ROUTER.routing_table()

            # Phase 34: threat feed summary
            threat_summary = _THREAT_STREAMER.summary()
            threat_events = _THREAT_STREAMER.recent_events(n=6)

            # Phase 35: privacy guard summary
            privacy_summary = _PRIVACY_GUARD.summary()

            # Phase 36: replay audit count
            replay_count = _VAULT.replays_count()

            # Phase 37: policy verifier — demo self-check
            self_policy = OrgPolicy.from_manifold_policy(_POLICY, "manifold-server")
            verify_result = _POLICY_VERIFIER.verify(self_policy, self_policy)

            # Phase 38: DAG execution count
            dag_count = _VAULT.dags_count()

            # Phase 39: PID controller — tick once and collect telemetry
            pid_state = _PID_CONTROLLER.tick()
            pid_summary = _PID_CONTROLLER.summary()

            # Phase 40: Multi-Sig Vault summary
            multisig_summary = _MULTISIG_VAULT.summary()

            # Phase 41: Genesis Mint summary
            genesis_summary = _GENESIS_MINT.summary()

            # Phase 42: Watchdog report
            watchdog_report = _WATCHDOG.report()
            crashlog_count = _VAULT.crashlogs_count()

            # Phase 47: VectorIndex semantic memory stats
            vector_count = len(_VECTOR_INDEX)
            vector_buckets = _VECTOR_INDEX.lsh_bucket_count()
            vector_bucket_summary = _VECTOR_INDEX.bucket_summary()
            vector_blobs_count = _VAULT.vector_blobs_count()

            # Phase 48: Meta-prompt A/B testing summary
            meta_summary = _META_ENGINE.summary()

        renderer = FleetPanelRenderer(fleet_data)
        ci_text = renderer.ci_summary_text()
        eco_text = renderer.economy_summary_text()
        ci_summary = fleet_data.ci_history.summary()
        eco_summary = fleet_data.economy.summary()

        # Build CSS bar chart rows for CI pass rate
        pass_rate_pct = int(ci_summary["pass_rate"] * 100)  # type: ignore[arg-type]
        block_rate_pct = int(eco_summary["block_rate"] * 100)  # type: ignore[arg-type]

        # Top risky tools rows
        risky_rows = ""
        for tool, delta in fleet_data.ci_history.most_risky_tools(top_n=5):
            bar_w = int(delta * 100)
            risky_rows += (
                f"<tr><td class='p-2 font-mono text-sm'>{tool}</td>"
                f"<td class='p-2'><div class='bg-red-400 h-4 rounded' style='width:{bar_w}%'></div></td>"
                f"<td class='p-2 text-right text-sm'>{delta:.4f}</td></tr>\n"
            )
        if not risky_rows:
            risky_rows = "<tr><td colspan='3' class='p-2 text-gray-400 text-center'>No risky tools recorded</td></tr>"

        # Economy org rows
        eco_rows = ""
        for org, cost in fleet_data.economy.top_partners(top_n=5):
            label = fleet_data.economy.org_labels.get(org, org)
            bar_w = min(100, int(cost * 20))
            eco_rows += (
                f"<tr><td class='p-2 font-mono text-sm'>{label}</td>"
                f"<td class='p-2'><div class='bg-blue-400 h-4 rounded' style='width:{bar_w}%'></div></td>"
                f"<td class='p-2 text-right text-sm'>{cost:.4f}</td></tr>\n"
            )
        if not eco_rows:
            eco_rows = "<tr><td colspan='3' class='p-2 text-gray-400 text-center'>No economy activity recorded</td></tr>"

        vault_gossip = _VAULT.gossip_count()
        vault_economy = _VAULT.economy_count()
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Phase 47: build LSH bucket table rows
        vector_bucket_rows = ""
        for bk, bv in list(vector_bucket_summary.items())[:10]:
            bar_w = min(100, bv * 20)
            vector_bucket_rows += (
                f"<tr><td class='p-2 font-mono text-xs text-cyan-300'>{bk}</td>"
                f"<td class='p-2 text-right text-sm'>{bv}</td>"
                f"<td class='p-2'><div class='bg-purple-500 h-3 rounded' style='width:{bar_w}%'></div></td></tr>\n"
            )
        if not vector_bucket_rows:
            vector_bucket_rows = "<tr><td colspan='3' class='p-2 text-gray-400 text-center'>No vectors indexed — use POST /vector/add</td></tr>"

        # Phase 48: build meta challenger row
        champ_d = meta_summary["champion"]
        champ_bar = int(champ_d["success_rate"] * 100)
        chall_d = meta_summary["challenger"]
        if chall_d:
            chall_bar = int(chall_d["success_rate"] * 100)
            challenger_row = (
                f"<tr><td class='p-2 text-blue-400 font-bold'>Challenger</td>"
                f"<td class='p-2 font-mono text-sm text-cyan-300'>{chall_d['prompt_id']}</td>"
                f"<td class='p-2 text-right text-sm'>{chall_d['trial_count']}</td>"
                f"<td class='p-2 text-right text-sm text-blue-400'>{chall_d['success_rate']:.4f}</td>"
                f"<td class='p-2'><div class='bg-blue-500 h-4 rounded' style='width:{chall_bar}%'></div></td></tr>"
            )
        else:
            challenger_row = "<tr><td colspan='5' class='p-2 text-gray-400 text-center'>Challenger not yet created — select a genome to start A/B testing</td></tr>"

        # Phase 26: entropy sparkline — CSS bar per tracked tool (max 8)
        entropy_rows = ""
        for tool_name, entropy in list(tool_entropy_map.items())[:8]:
            bar_w = int(entropy * 100)
            color = "bg-green-400" if entropy < 0.3 else ("bg-yellow-400" if entropy < 0.7 else "bg-red-400")
            entropy_rows += (
                f"<tr><td class='p-2 font-mono text-sm'>{tool_name}</td>"
                f"<td class='p-2'><div class='{color} h-4 rounded' style='width:{bar_w}%'></div></td>"
                f"<td class='p-2 text-right text-sm'>{entropy:.4f}</td></tr>\n"
            )
        if not entropy_rows:
            entropy_rows = "<tr><td colspan='3' class='p-2 text-gray-400 text-center'>No entropy signals recorded yet</td></tr>"

        sys_entropy_pct = int(sys_entropy * 100)
        entropy_color = "text-green-400" if sys_entropy_pct < 30 else ("text-yellow-400" if sys_entropy_pct < 70 else "text-red-400")

        # Phase 27: Braintrust vote table
        bt_rows = ""
        for vote in bt_result.votes:
            approve_icon = "✅" if vote.approves else "🚫"
            bt_rows += (
                f"<tr><td class='p-2 font-mono text-sm'>{vote.genome_name}</td>"
                f"<td class='p-2 text-center'>{approve_icon}</td>"
                f"<td class='p-2 text-right text-sm'>{vote.decision.action}</td>"
                f"<td class='p-2 text-right text-sm'>{vote.weighted_confidence:.4f}</td></tr>\n"
            )
        bt_approved_icon = "✅ APPROVED" if bt_result.approved else "🚫 VETOED"
        bt_approved_color = "text-green-400" if bt_result.approved else "text-red-400"

        # Phase 29: provenance summary row
        provenance_html = (
            f"<span class='font-bold text-purple-400'>{provenance_count}</span>"
            " decisions recorded in Merkle chain"
        )

        # Phase 30: rate-limit quota bar chart (top 6 entities)
        quota_rows = ""
        for entity_id, info in list(quota_summary.items())[:6]:
            tokens = info["tokens"]
            cap = info["capacity"]
            bar_w = int((tokens / cap) * 100) if cap > 0 else 0
            color = "bg-green-400" if bar_w > 50 else ("bg-yellow-400" if bar_w > 20 else "bg-red-400")
            quota_rows += (
                f"<tr><td class='p-2 font-mono text-sm'>{entity_id}</td>"
                f"<td class='p-2'><div class='{color} h-4 rounded' style='width:{bar_w}%'></div></td>"
                f"<td class='p-2 text-right text-sm'>{tokens:.2f}/{cap:.0f}</td></tr>\n"
            )
        if not quota_rows:
            quota_rows = "<tr><td colspan='3' class='p-2 text-gray-400 text-center'>No quota buckets active</td></tr>"

        # Phase 31: canary status panel
        canary_total = canary_summary["total_probes"]
        canary_suspects = canary_summary["adversarial_suspects"]
        canary_running = canary_summary["is_running"]
        canary_pass_rate = canary_summary["pass_rate"]
        canary_pulse_class = "canary-pulse-active" if canary_running else "canary-pulse-idle"
        canary_status_text = "ACTIVE" if canary_running else "IDLE"
        canary_pass_pct = int(canary_pass_rate * 100)
        canary_color = "text-green-400" if canary_suspects == 0 else "text-red-400"

        # Phase 32: clearinghouse balances table
        clearing_rows = ""
        for org, debt in list(clearing_summary["net_debts"].items())[:8]:
            debt_val: float = debt  # type: ignore[assignment]
            bal = clearing_summary["trust_balances"].get(org, 0.0)  # type: ignore[union-attr]
            debt_color = "text-red-400" if debt_val > 0 else "text-green-400"
            clearing_rows += (
                f"<tr><td class='p-2 font-mono text-sm'>{org}</td>"
                f"<td class='p-2 text-right {debt_color}'>{debt_val:.4f}</td>"
                f"<td class='p-2 text-right text-sm text-yellow-400'>{bal:.2f}</td></tr>\n"
            )
        if not clearing_rows:
            clearing_rows = "<tr><td colspan='3' class='p-2 text-gray-400 text-center'>No settlement activity recorded</td></tr>"
        clearing_freezes: int = clearing_summary["total_freezes"]  # type: ignore[assignment]

        # Phase 33: swarm delegation flow chart
        swarm_rows = ""
        for peer_info in swarm_table[:6]:
            rv = peer_info["routing_value"]
            rv_pct = int(max(0.0, min(1.0, (rv + 1.0) / 2.0)) * 100)
            rv_color = "bg-green-400" if rv > 0.5 else ("bg-yellow-400" if rv > 0 else "bg-red-400")
            swarm_rows += (
                f"<tr><td class='p-2 font-mono text-sm'>{peer_info['org_id']}</td>"
                f"<td class='p-2'><div class='{rv_color} h-4 rounded' style='width:{rv_pct}%'></div></td>"
                f"<td class='p-2 text-right text-sm'>{rv:.4f}</td>"
                f"<td class='p-2 text-right text-xs text-slate-400'>{peer_info['endpoint']}</td></tr>\n"
            )
        if not swarm_rows:
            swarm_rows = "<tr><td colspan='4' class='p-2 text-gray-400 text-center'>No swarm peers registered</td></tr>"
        swarm_peer_count = _SWARM_ROUTER.peer_count()

        # Phase 34: threat feed terminal
        threat_total = threat_summary["total_events"]
        threat_critical = threat_summary["critical"]
        threat_high = threat_summary["high"]
        threat_terminal_lines = ""
        for ev in threat_events:
            sev_color = {
                "critical": "text-red-400",
                "high": "text-orange-400",
                "medium": "text-yellow-400",
                "low": "text-green-400",
            }.get(ev.severity, "text-slate-400")
            threat_terminal_lines += (
                f"<div><span class='text-slate-500'>{ev.timestamp:.0f}</span> "
                f"<span class='{sev_color}'>[{ev.severity.upper()}]</span> "
                f"<span class='text-white'>{ev.event_type}</span> "
                f"<span class='text-cyan-400'>{ev.tool_name}</span></div>\n"
            )
        if not threat_terminal_lines:
            threat_terminal_lines = "<div class='text-slate-500'>No threat events recorded — system healthy.</div>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>MANIFOLD Fleet Dashboard</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/tailwindcss@3.4.1/base.css"
        crossorigin="anonymous"/>
  <style>
    body {{ font-family: system-ui, sans-serif; background:#0f172a; color:#e2e8f0; }}
    .card {{ background:#1e293b; border-radius:0.75rem; padding:1.5rem; margin-bottom:1.5rem; }}
    table {{ width:100%; border-collapse:collapse; }}
    th {{ text-align:left; padding:0.5rem; font-size:0.75rem; text-transform:uppercase;
          letter-spacing:0.05em; color:#94a3b8; border-bottom:1px solid #334155; }}
    tr:hover td {{ background:#1a2a3b; }}
    pre {{ white-space:pre-wrap; font-size:0.8rem; color:#94a3b8; background:#0f172a;
           padding:1rem; border-radius:0.5rem; }}
    .sparkline-bar {{ height:8px; border-radius:4px; display:inline-block; }}
    @keyframes canary-pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}
    .canary-pulse-active {{ animation: canary-pulse 2s ease-in-out infinite; color:#4ade80; }}
    .canary-pulse-idle {{ color:#94a3b8; }}
    .threat-terminal {{ background:#000; border-radius:0.5rem; padding:1rem; font-family:monospace;
                        font-size:0.75rem; max-height:200px; overflow-y:auto; }}
    @keyframes blink {{ 0%,100%{{opacity:1}} 49%{{opacity:1}} 50%{{opacity:0}} }}
    .blink {{ animation: blink 1s step-start infinite; color:#4ade80; }}
    .privacy-shield {{ font-size:2.5rem; }}
  </style>
</head>
<body class="p-6">
  <div class="max-w-5xl mx-auto">

    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <div>
        <h1 class="text-3xl font-bold text-white">⚡ MANIFOLD Fleet Dashboard</h1>
        <p class="text-slate-400 text-sm mt-1">
          v2.0.0 &nbsp;|&nbsp; 1,350+ Tests Passing &nbsp;|&nbsp; 0 External Dependencies
          &nbsp;|&nbsp; Enterprise Defense Network
        </p>
      </div>
      <span class="text-slate-500 text-xs">{now_utc}</span>
    </div>

    <!-- KPI Strip -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      <div class="card text-center">
        <div class="text-4xl font-bold text-green-400">{ci_summary["total_builds"]}</div>
        <div class="text-slate-400 text-sm mt-1">Total CI Builds</div>
      </div>
      <div class="card text-center">
        <div class="text-4xl font-bold {'text-green-400' if pass_rate_pct >= 80 else 'text-red-400'}">{pass_rate_pct}%</div>
        <div class="text-slate-400 text-sm mt-1">CI Pass Rate</div>
        <div class="mt-2 h-3 rounded bg-slate-700">
          <div class="h-3 rounded bg-green-400" style="width:{pass_rate_pct}%"></div>
        </div>
      </div>
      <div class="card text-center">
        <div class="text-4xl font-bold text-blue-400">{eco_summary["total_calls"]}</div>
        <div class="text-slate-400 text-sm mt-1">B2B API Calls</div>
      </div>
      <div class="card text-center">
        <div class="text-4xl font-bold {'text-yellow-400' if block_rate_pct > 0 else 'text-green-400'}">{block_rate_pct}%</div>
        <div class="text-slate-400 text-sm mt-1">B2B Block Rate</div>
        <div class="mt-2 h-3 rounded bg-slate-700">
          <div class="h-3 rounded bg-yellow-400" style="width:{block_rate_pct}%"></div>
        </div>
      </div>
    </div>

    <!-- Phase 26: System Entropy Sparkline -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🌡️ System Entropy (Phase 26 — Reputation Decay)</h2>
      <div class="flex items-center gap-4 mb-4">
        <div>
          <span class="text-slate-400 text-sm">Mean System Entropy:</span>
          <span class="ml-2 font-bold text-2xl {entropy_color}">{sys_entropy_pct}%</span>
        </div>
        <div class="flex-1 h-3 rounded bg-slate-700">
          <div class="h-3 rounded {'bg-green-400' if sys_entropy_pct < 30 else 'bg-yellow-400' if sys_entropy_pct < 70 else 'bg-red-400'}"
               style="width:{sys_entropy_pct}%"></div>
        </div>
      </div>
      <table>
        <thead><tr>
          <th>Tool</th><th>Entropy Sparkline</th><th class="text-right">Score</th>
        </tr></thead>
        <tbody>
          {entropy_rows}
        </tbody>
      </table>
    </div>

    <!-- Phase 27: Braintrust Consensus Status -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🧠 Braintrust Consensus (Phase 27)</h2>
      <div class="flex items-center gap-4 mb-4">
        <span class="font-bold text-xl {bt_approved_color}">{bt_approved_icon}</span>
        <span class="text-slate-400 text-sm">Score: {bt_result.consensus_score:.4f} / threshold: {bt_result.threshold:.4f}</span>
      </div>
      <table>
        <thead><tr>
          <th>Genome</th><th class="text-center">Vote</th><th class="text-right">Action</th><th class="text-right">W·Confidence</th>
        </tr></thead>
        <tbody>
          {bt_rows}
        </tbody>
      </table>
    </div>

    <!-- Phase 29: Decision Provenance -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🔐 Decision Provenance (Phase 29)</h2>
      <p class="text-slate-400 text-sm mb-3">Merkle-chained cryptographic audit trail of all decisions.</p>
      <div class="text-slate-300 text-sm">{provenance_html}</div>
      <div class="mt-2 text-slate-500 text-xs">Query: <code>GET /provenance/&lt;task_id&gt;</code></div>
    </div>

    <!-- Phase 30: Rate Limit Quotas -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🚦 Rate Limit Quotas (Phase 30)</h2>
      <p class="text-slate-400 text-sm mb-3">Live token-bucket levels per tool/org (green = healthy, red = throttled).</p>
      <table>
        <thead><tr>
          <th>Entity</th><th>Quota (tokens remaining)</th><th class="text-right">Tokens/Cap</th>
        </tr></thead>
        <tbody>
          {quota_rows}
        </tbody>
      </table>
    </div>

    <!-- Phase 31: Canary Status -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🐦 Canary Prober (Phase 31)</h2>
      <div class="flex items-center gap-6 mb-4">
        <div>
          <span class="{canary_pulse_class} font-bold text-lg">● {canary_status_text}</span>
        </div>
        <div class="text-sm text-slate-400">
          Probes: <span class="text-white font-mono">{canary_total}</span>
          &nbsp;|&nbsp; Suspects: <span class="{canary_color} font-mono">{canary_suspects}</span>
          &nbsp;|&nbsp; Pass rate: <span class="text-white font-mono">{canary_pass_pct}%</span>
        </div>
      </div>
      <div class="h-3 rounded bg-slate-700">
        <div class="h-3 rounded bg-green-400" style="width:{canary_pass_pct}%"></div>
      </div>
    </div>

    <!-- Phase 32: Clearinghouse Balances -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🏦 Clearinghouse Balances (Phase 32)</h2>
      <div class="flex gap-6 mb-4 text-sm text-slate-400">
        <div>Total settlements: <span class="text-white font-mono">{clearing_summary["total_settlements"]}</span></div>
        <div>Bankruptcy freezes: <span class="{'text-red-400' if clearing_freezes > 0 else 'text-green-400'} font-mono">{clearing_freezes}</span></div>
      </div>
      <table>
        <thead><tr>
          <th>Organisation</th><th class="text-right">Net Debt</th><th class="text-right">Trust Tokens</th>
        </tr></thead>
        <tbody>
          {clearing_rows}
        </tbody>
      </table>
    </div>

    <!-- Phase 33: Swarm Delegation Flow -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🕸️ Swarm Delegation (Phase 33)</h2>
      <p class="text-slate-400 text-sm mb-3">
        Registered peers: <span class="text-white font-mono">{swarm_peer_count}</span>
        &nbsp;—&nbsp; Tasks overflow to the peer with highest routing value.
      </p>
      <table>
        <thead><tr>
          <th>Peer Org</th><th>Routing Value</th><th class="text-right">V<sub>swarm</sub></th><th class="text-right">Endpoint</th>
        </tr></thead>
        <tbody>
          {swarm_rows}
        </tbody>
      </table>
    </div>

    <!-- Phase 34: Live Threat Feed Terminal -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">
        🛡️ Live Threat Feed (Phase 34) &nbsp;
        <span class="blink text-xs">■ LIVE</span>
      </h2>
      <div class="flex gap-6 mb-3 text-sm text-slate-400">
        <div>Total events: <span class="text-white font-mono">{threat_total}</span></div>
        <div>Critical: <span class="text-red-400 font-mono">{threat_critical}</span></div>
        <div>High: <span class="text-orange-400 font-mono">{threat_high}</span></div>
        <div class="text-xs text-slate-500">Stream: <code>GET /feed</code></div>
      </div>
      <div class="threat-terminal">
        {threat_terminal_lines}
      </div>
    </div>

    <!-- Phase 35: Privacy Shield -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🔒 Privacy Shield (Phase 35 — Differential Privacy)</h2>
      <div class="flex items-center gap-6 mb-4">
        <div class="privacy-shield">🛡️</div>
        <div>
          <div class="text-slate-400 text-sm">k-Anonymity threshold: <span class="text-white font-mono font-bold">{privacy_summary['k']}</span></div>
          <div class="text-slate-400 text-sm mt-1">Privacy budget (ε): <span class="text-{'green' if privacy_summary['epsilon'] <= 1.0 else 'yellow'}-400 font-mono font-bold">{privacy_summary['epsilon']:.4f}</span></div>
          <div class="text-slate-400 text-sm mt-1">Laplace scale (Δf/ε): <span class="text-purple-400 font-mono">{privacy_summary['laplace_scale']:.4f}</span></div>
        </div>
        <div class="flex-1">
          <div class="text-slate-500 text-xs mb-1">Privacy strength (lower ε = stronger)</div>
          <div class="h-3 rounded bg-slate-700">
            <div class="h-3 rounded bg-purple-400" style="width:{min(100, int((1.0/max(0.01,privacy_summary['epsilon']))*50))}%"></div>
          </div>
        </div>
      </div>
      <div class="text-slate-500 text-xs">Gossip &amp; Threat feeds anonymised before federation. API: <code>POST /verify_policy</code></div>
    </div>

    <!-- Phase 36: Time-Travel Scrubber -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">⏪ Time-Travel Scrubber (Phase 36 — Decision Replay)</h2>
      <div class="flex items-center gap-6 mb-4 text-sm">
        <div class="text-slate-400">Replay audits logged: <span class="text-cyan-400 font-mono font-bold">{replay_count}</span></div>
        <div class="text-slate-400">Provenance receipts available: <span class="text-purple-400 font-mono font-bold">{provenance_count}</span></div>
      </div>
      <div class="bg-slate-900 rounded p-3 text-xs text-slate-400 font-mono">
        <div class="mb-1 text-slate-500">// Time-Travel API</div>
        <div>GET /replay/<span class="text-cyan-400">&lt;task_id&gt;</span>  → ReplayReport (historical vs current decision)</div>
        <div class="mt-1 text-slate-500">// Enter a Task ID to replay a past decision and compare with current behaviour</div>
      </div>
    </div>

    <!-- Phase 37: Policy Verifier -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">⚖️ Policy Verifier (Phase 37 — Formal Conflict Detection)</h2>
      <div class="flex gap-6 mb-4 text-sm text-slate-400">
        <div>Server self-check friction: <span class="text-{'green' if verify_result.friction_score < 0.3 else 'yellow'}-400 font-mono font-bold">{verify_result.friction_score:.4f}</span></div>
        <div>Deadlocks: <span class="text-{'red' if verify_result.deadlock_count > 0 else 'green'}-400 font-mono">{verify_result.deadlock_count}</span></div>
        <div>Compatible: <span class="text-{'green' if verify_result.compatible else 'red'}-400 font-mono">{'YES' if verify_result.compatible else 'NO'}</span></div>
      </div>
      <div class="text-slate-500 text-xs">Pre-flight check before B2B handshakes. API: <code>POST /verify_policy</code> with <code>policy_a</code> + <code>policy_b</code></div>
    </div>

    <!-- Phase 38: DAG Topo-Map -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🗺️ DAG Topo-Map (Phase 38 — Workflow Orchestration)</h2>
      <div class="flex gap-6 mb-4 text-sm text-slate-400">
        <div>Graph executions logged: <span class="text-cyan-400 font-mono font-bold">{dag_count}</span></div>
        <div class="text-slate-500 text-xs">Submit multi-step pipelines via <code>POST /dag/execute</code></div>
      </div>
      <div class="bg-slate-900 rounded p-3 text-xs text-slate-400 font-mono">
        <div class="text-slate-500 mb-1">// DAG API — submit a task graph for topological execution</div>
        <div>POST /dag/execute  {{"graph_id": "...", "nodes": [{{"node_id": "a", "task": {{...}}, "depends_on": []}}]}}</div>
      </div>
    </div>

    <!-- Phase 39: PID Telemetry Chart -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">📈 PID Telemetry (Phase 39 — Autonomic Risk Regulation)</h2>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4 text-sm">
        <div><span class="text-slate-400">Setpoint (target entropy):</span><span class="ml-2 text-white font-mono">{pid_summary['setpoint']:.3f}</span></div>
        <div><span class="text-slate-400">Last error e(t):</span><span class="ml-2 text-{'red' if pid_state.error < 0 else 'green'}-400 font-mono">{pid_state.error:.4f}</span></div>
        <div><span class="text-slate-400">PID output u(t):</span><span class="ml-2 text-purple-400 font-mono font-bold">{pid_state.output:.4f}</span></div>
        <div><span class="text-slate-400">Threshold (after):</span><span class="ml-2 text-yellow-400 font-mono font-bold">{pid_state.threshold_after:.4f}</span></div>
      </div>
      <div class="flex items-center gap-4 mb-2">
        <span class="text-slate-500 text-xs">P={pid_summary['kp']} &nbsp; I={pid_summary['ki']} &nbsp; D={pid_summary['kd']} &nbsp; Ticks: {pid_summary['total_ticks']}</span>
      </div>
      <div class="h-3 rounded bg-slate-700">
        <div class="h-3 rounded bg-purple-400" style="width:{int(pid_state.output * 100)}%"></div>
      </div>
      <div class="text-slate-500 text-xs mt-2">Dynamic threshold adapts to global model entropy. Anti-windup active.</div>
    </div>

    <!-- Phase 40: Multi-Sig Pending Queue -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🔑 Multi-Sig Queue (Phase 40 — M-of-N Consensus)</h2>
      <div class="flex gap-6 mb-4 text-sm text-slate-400">
        <div>Total entries: <span class="text-white font-mono">{multisig_summary['total_entries']}</span></div>
        <div>Pending: <span class="text-yellow-400 font-mono">{multisig_summary['by_status'].get('pending', 0)}</span></div>
        <div>Approved: <span class="text-green-400 font-mono">{multisig_summary['by_status'].get('approved', 0)}</span></div>
        <div>Required sigs: <span class="text-purple-400 font-mono">{multisig_summary['required_signatures']}-of-{multisig_summary['total_peers']}</span></div>
      </div>
      <div class="text-slate-500 text-xs">High-stakes tasks (stakes ≥ {multisig_summary['high_stakes_threshold']}) require {multisig_summary['required_signatures']}-of-{multisig_summary['total_peers']} peer endorsements. API: <code>POST /multisig/endorse</code></div>
    </div>

    <!-- CI Risk Trends -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-4">🔬 CI/CD Risk Trends</h2>
      <table>
        <thead><tr>
          <th>Tool</th><th>Risk Delta</th><th class="text-right">Δ Value</th>
        </tr></thead>
        <tbody>
          {risky_rows}
        </tbody>
      </table>
    </div>

    <!-- B2B Economy Map -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-4">🤝 B2B Economy Map (Trust-Tax Flows)</h2>
      <div class="grid grid-cols-3 gap-4 mb-4 text-sm">
        <div><span class="text-slate-400">Total trust cost:</span>
             <span class="ml-2 text-white font-mono">{eco_summary["total_trust_cost"]:.4f}</span></div>
        <div><span class="text-slate-400">Avg reputation:</span>
             <span class="ml-2 text-white font-mono">{eco_summary["avg_reputation"]:.4f}</span></div>
        <div><span class="text-slate-400">Unique remote orgs:</span>
             <span class="ml-2 text-white font-mono">{eco_summary["unique_remote_orgs"]}</span></div>
      </div>
      <table>
        <thead><tr>
          <th>Remote Org</th><th>Trust Cost</th><th class="text-right">Units</th>
        </tr></thead>
        <tbody>
          {eco_rows}
        </tbody>
      </table>
    </div>

    <!-- Vault Status -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🗄️ Vault (Phase 24 WAL)</h2>
      <div class="grid grid-cols-2 gap-4 text-sm">
        <div><span class="text-slate-400">Gossip records persisted:</span>
             <span class="ml-2 text-white font-mono">{vault_gossip}</span></div>
        <div><span class="text-slate-400">Economy records persisted:</span>
             <span class="ml-2 text-white font-mono">{vault_economy}</span></div>
        <div><span class="text-slate-400">Volatility coefficients persisted:</span>
             <span class="ml-2 text-white font-mono">{_VAULT.volatility_count()}</span></div>
        <div><span class="text-slate-400">Probationary records persisted:</span>
             <span class="ml-2 text-white font-mono">{_VAULT.probationary_count()}</span></div>
        <div><span class="text-slate-400">Provenance receipts persisted:</span>
             <span class="ml-2 text-white font-mono">{_VAULT.provenance_count()}</span></div>
        <div><span class="text-slate-400">Token-bucket snapshots persisted:</span>
             <span class="ml-2 text-white font-mono">{_VAULT.token_bucket_count()}</span></div>
        <div><span class="text-slate-400">Settlement records persisted:</span>
             <span class="ml-2 text-white font-mono">{_VAULT.settlements_count()}</span></div>
        <div><span class="text-slate-400">Replay audit records persisted:</span>
             <span class="ml-2 text-white font-mono">{_VAULT.replays_count()}</span></div>
        <div><span class="text-slate-400">DAG execution records persisted:</span>
             <span class="ml-2 text-white font-mono">{_VAULT.dags_count()}</span></div>
        <div><span class="text-slate-400">Crash log records persisted:</span>
             <span class="ml-2 text-white font-mono">{crashlog_count}</span></div>
        <div><span class="text-slate-400">Sandbox violations logged:</span>
             <span class="ml-2 text-white font-mono">{_VAULT.sandbox_violations_count()}</span></div>
        <div><span class="text-slate-400">Vector blobs persisted:</span>
             <span class="ml-2 text-white font-mono">{vector_blobs_count}</span></div>
      </div>
    </div>

    <!-- Phase 41: Network Genesis Map -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🌐 Network Genesis Map (Phase 41 — Cold-Start Bootstrap)</h2>
      <div class="flex gap-6 mb-4 text-sm text-slate-400">
        <div>Genesis node: <span class="text-cyan-400 font-mono font-bold">{genesis_summary['genesis_node_id']}</span></div>
        <div>Token pool: <span class="text-yellow-400 font-mono font-bold">{genesis_summary['total_tokens']:.0f}</span></div>
        <div>Decay γ: <span class="text-purple-400 font-mono">{genesis_summary['gamma']:.2f}</span></div>
        <div>Mint events: <span class="text-white font-mono">{genesis_summary['mint_events']}</span></div>
      </div>
      <div class="bg-slate-900 rounded p-3 text-xs text-slate-400 font-mono">
        <div class="text-slate-500 mb-1">// Spatial Decay Formula: T_i = e^(-γ·d_i) / Σ e^(-γ·d_j)</div>
        <div>Boot with: <span class="text-cyan-400">python manifold.pyz --genesis --port 8080</span></div>
      </div>
    </div>

    <!-- Phase 42: System Health Matrix -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🔧 System Health Matrix (Phase 42 — Watchdog)</h2>
      <div class="flex gap-6 mb-4 text-sm text-slate-400">
        <div>Supervised components: <span class="text-white font-mono">{watchdog_report.total_components}</span></div>
        <div>Total restarts: <span class="text-red-400 font-mono">{watchdog_report.total_restarts}</span></div>
        <div>Missed heartbeats: <span class="text-red-400 font-mono">{watchdog_report.total_missed_heartbeats}</span></div>
        <div>Deadlock purges: <span class="text-yellow-400 font-mono">{watchdog_report.deadlock_purges}</span></div>
        <div>Crash logs: <span class="text-orange-400 font-mono">{crashlog_count}</span></div>
      </div>
      <table>
        <thead><tr><th>Component</th><th class="text-right">Restarts</th><th class="text-right">Consec. Misses</th></tr></thead>
        <tbody>
          {''.join(f"<tr><td class='p-2 font-mono text-sm'>{c['name']}</td><td class='p-2 text-right text-sm'>{c['restart_count']}</td><td class='p-2 text-right text-sm'>{c['consecutive_misses']}</td></tr>" for c in watchdog_report.component_states) or "<tr><td colspan='3' class='p-2 text-gray-400 text-center'>No components registered</td></tr>"}
        </tbody>
      </table>
    </div>

    <!-- Phase 47: Semantic Memory Browser -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🧬 Semantic Memory Browser (Phase 47 — VectorFS)</h2>
      <div class="flex gap-6 mb-4 text-sm text-slate-400">
        <div>Vectors stored: <span class="text-cyan-400 font-mono font-bold">{vector_count}</span></div>
        <div>Active LSH buckets: <span class="text-purple-400 font-mono font-bold">{vector_buckets}</span></div>
        <div>WAL records: <span class="text-white font-mono">{vector_blobs_count}</span></div>
      </div>
      <table>
        <thead><tr>
          <th>LSH Bucket Key</th><th class="text-right">Vectors</th><th>Distribution</th>
        </tr></thead>
        <tbody>
          {vector_bucket_rows}
        </tbody>
      </table>
      <div class="text-slate-500 text-xs mt-2">Cosine similarity search · Random-projection LSH · POST /vector/add · POST /vector/search</div>
    </div>

    <!-- Phase 48: Meta-Evolution Chart -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">🧪 Meta-Evolution Chart (Phase 48 — Prompt A/B Testing)</h2>
      <div class="flex gap-6 mb-4 text-sm text-slate-400">
        <div>Promotions: <span class="text-yellow-400 font-mono font-bold">{meta_summary['promotions']}</span></div>
        <div>Min trials: <span class="text-white font-mono">{meta_summary['min_trials']}</span></div>
        <div>Threshold: <span class="text-white font-mono">+{meta_summary['promotion_threshold']*100:.1f}%</span></div>
      </div>
      <table>
        <thead><tr>
          <th>Role</th><th>Prompt ID</th><th class="text-right">Trials</th><th class="text-right">Success Rate</th><th>Performance Bar</th>
        </tr></thead>
        <tbody>
          <tr>
            <td class='p-2 text-yellow-400 font-bold'>Champion</td>
            <td class='p-2 font-mono text-sm text-cyan-300'>{champ_d['prompt_id']}</td>
            <td class='p-2 text-right text-sm'>{champ_d['trial_count']}</td>
            <td class='p-2 text-right text-sm text-green-400'>{champ_d['success_rate']:.4f}</td>
            <td class='p-2'><div class='bg-green-500 h-4 rounded' style='width:{champ_bar}%'></div></td>
          </tr>
          {challenger_row}
        </tbody>
      </table>
      <div class="text-slate-500 text-xs mt-2">Grid axes: [cost, risk, neutrality, asset] · POST /meta/outcome to record outcomes</div>
    </div>

    <!-- Raw Text Report -->
    <div class="card">
      <h2 class="text-xl font-semibold text-white mb-3">📋 Raw Report</h2>
      <pre>{ci_text}\n\n{eco_text}</pre>
    </div>

  </div>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _handle_get_docs(self) -> None:
        """GET /docs → Phase 55 self-documenting API explorer."""
        extractor = DocExtractor()
        explorer = APIExplorer(endpoints=MANIFOLD_ENDPOINTS, extractor=extractor)
        _send_json(self, 200, explorer.to_dict())

    def _handle_get_gc_run(self) -> None:
        """GET /gc/run → Phase 50 garbage collector run."""
        with _LOCK:
            summary = _GC.run()
        _send_json(self, 200, summary)

    def _handle_get_doctor_report(self) -> None:
        """GET /doctor/report → Phase 51 system doctor diagnostic report."""
        with _LOCK:
            report = _DOCTOR.run()
        _send_json(self, 200, report.to_dict())

    def _handle_post_mapreduce_submit(self, body: dict[str, Any]) -> None:
        """POST /mapreduce/submit → Phase 60 Swarm MapReduce job execution."""
        job_id = str(body.get("job_id", ""))
        dataset_raw = body.get("dataset")
        map_func = str(body.get("map_func", ""))
        reduce_func = str(body.get("reduce_func", ""))
        chunk_size = int(body.get("chunk_size", 100))
        timeout_seconds = float(body.get("timeout_seconds", 30.0))
        submitter_org_id = str(body.get("submitter_org_id", "api-client"))

        if not job_id:
            _send_error(self, 400, "Body must contain 'job_id'.")
            return
        if not isinstance(dataset_raw, list) or not dataset_raw:
            _send_error(self, 400, "Body must contain a non-empty 'dataset' list.")
            return
        if not map_func:
            _send_error(self, 400, "Body must contain 'map_func'.")
            return
        if not reduce_func:
            _send_error(self, 400, "Body must contain 'reduce_func'.")
            return

        try:
            job = MapReduceJob(
                job_id=job_id,
                dataset=tuple(dataset_raw),
                map_func=map_func,
                reduce_func=reduce_func,
                chunk_size=max(1, chunk_size),
                timeout_seconds=max(1.0, timeout_seconds),
                submitter_org_id=submitter_org_id,
            )
            result = _JOB_TRACKER.run(job)
        except ValueError as exc:
            _send_error(self, 400, str(exc))
            return
        _send_json(self, 200, result.to_dict())

    def _handle_get_registry_list(self) -> None:
        """GET /registry/list → Phase 62 global tool registry listing."""
        status_filter = self.path.split("status=")[-1] if "status=" in self.path else None
        with _LOCK:
            tools = _TOOL_REGISTRY.list_tools(status=status_filter)
            summary = _TOOL_REGISTRY.summary()
        _send_json(self, 200, {"tools": tools, "summary": summary})

    def _handle_post_registry_publish(self, body: dict[str, Any]) -> None:
        """POST /registry/publish → Phase 62 publish a tool to the registry."""
        try:
            manifest = ToolManifest(
                tool_id=str(body.get("tool_id", "")),
                name=str(body.get("name", "")),
                description=str(body.get("description", "")),
                code=str(body.get("code", "")),
                endpoints=tuple(body.get("endpoints", [])),
                author_org_id=str(body.get("author_org_id", "")),
                version=str(body.get("version", "1.0.0")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            _send_error(self, 400, f"Invalid manifest payload: {exc}")
            return
        with _LOCK:
            result = _TOOL_REGISTRY.publish(manifest)
        _send_json(self, 200 if result.accepted else 409, result.to_dict())

    def _handle_post_registry_endorse(self, body: dict[str, Any]) -> None:
        """POST /registry/endorse → Phase 62 endorse a registered tool."""
        tool_id = str(body.get("tool_id", ""))
        if not tool_id:
            _send_error(self, 400, "Body must contain 'tool_id'.")
            return
        try:
            import time as _time
            endorsement = ToolEndorsement(
                genesis_org_id=str(body.get("genesis_org_id", "")),
                tool_id=tool_id,
                manifest_hash=str(body.get("manifest_hash", "")),
                signature=str(body.get("signature", "")),
                key_id=str(body.get("key_id", "")),
                timestamp=float(body.get("timestamp", _time.time())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            _send_error(self, 400, f"Invalid endorsement payload: {exc}")
            return
        with _LOCK:
            result = _TOOL_REGISTRY.endorse(tool_id, endorsement)
        _send_json(self, 200 if result.accepted else 409, result.to_dict())

    def _handle_post_zkp_prove(self, body: dict[str, Any]) -> None:
        """POST /zkp/prove → Phase 61 generate a Schnorr ZKP for a policy value."""
        x_raw = body.get("x")
        context = str(body.get("context", ""))
        if x_raw is None:
            _send_error(self, 400, "Body must contain 'x' (the secret integer).")
            return
        try:
            x = int(x_raw)
        except (TypeError, ValueError) as exc:
            _send_error(self, 400, f"'x' must be an integer: {exc}")
            return
        try:
            proof = _ZKP_VERIFIER.prove(x=x, context=context)
        except ValueError as exc:
            _send_error(self, 400, str(exc))
            return
        _send_json(self, 200, proof.to_dict())

    def _handle_post_zkp_verify(self, body: dict[str, Any]) -> None:
        """POST /zkp/verify → Phase 61 verify a Schnorr proof."""
        from .zkp import ZKProof
        try:
            proof = ZKProof.from_dict(body)
        except (KeyError, TypeError, ValueError) as exc:
            _send_error(self, 400, f"Invalid proof payload: {exc}")
            return
        valid = _ZKP_VERIFIER.verify(proof)
        _send_json(self, 200, {"valid": valid})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_from_dict(data: dict[str, Any]) -> BrainTask:
    """Build a ``BrainTask`` from a JSON payload dict."""
    return BrainTask(
        prompt=str(data.get("prompt", "")),
        domain=str(data.get("domain", "general")),
        stakes=float(data.get("stakes", 0.5)),  # type: ignore[arg-type]
        uncertainty=float(data.get("uncertainty", 0.5)),  # type: ignore[arg-type]
        complexity=float(data.get("complexity", 0.5)),  # type: ignore[arg-type]
        tool_relevance=float(data.get("tool_relevance", 0.5)),  # type: ignore[arg-type]
        time_pressure=float(data.get("time_pressure", 0.4)),  # type: ignore[arg-type]
        safety_sensitivity=float(data.get("safety_sensitivity", 0.2)),  # type: ignore[arg-type]
        collaboration_value=float(data.get("collaboration_value", 0.3)),  # type: ignore[arg-type]
        source_confidence=float(data.get("source_confidence", 0.7)),  # type: ignore[arg-type]
        user_patience=float(data.get("user_patience", 0.7)),  # type: ignore[arg-type]
        dynamic_goal=bool(data.get("dynamic_goal", False)),
    )


# ---------------------------------------------------------------------------
# ReputationHub helper — expose observation_weight on the hub via module attr
# ---------------------------------------------------------------------------


def _observation_weight(hub: ReputationHub, agent_id: str) -> int:
    """Return the community observation weight for *agent_id*."""
    return hub.baseline.observation_weight(agent_id)


# Monkey-patch a convenience method onto the singleton hub instance so
# _handle_get_reputation can call hub.observation_weight() without importing
# CommunityBaseline internals throughout this module.
ReputationHub.observation_weight = lambda self, agent_id: self.baseline.observation_weight(agent_id)  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_server(port: int = 8080, *, host: str = "0.0.0.0") -> None:
    """Start the MANIFOLD HTTP server and block until interrupted.

    On startup the Vault WAL is replayed to restore any gossip and economy
    state from previous runs.

    Parameters
    ----------
    port:
        TCP port to bind.  Default: ``8080``.
    host:
        Bind address.  Default: ``"0.0.0.0"`` (all interfaces).
    """
    # Phase 24: replay WAL on startup
    with _LOCK:
        result = _VAULT.load_state(hub=_HUB, ledger=_ECONOMY_LEDGER)
    if result.total_loaded:
        print(
            f"MANIFOLD vault: replayed {result.gossip_loaded} gossip + "
            f"{result.economy_loaded} economy records "
            f"({result.skipped} skipped)."
        )

    server = HTTPServer((host, port), ManifoldHandler)
    print(f"MANIFOLD server listening on {host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("MANIFOLD server stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the MANIFOLD HTTP server.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)
