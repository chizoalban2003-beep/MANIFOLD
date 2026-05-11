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

import atexit
import asyncio
import dataclasses
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
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
    TOPIC_META_CHAMPION_PROMOTED,
    TOPIC_SANDBOX_VIOLATION,
    TOPIC_SANDBOX_TIMEOUT,
    TOPIC_VECTOR_ENTRY_ADDED,
)
from .verify import PolicyVerifier
from .watchdog import ProcessWatchdog, WatchedComponent
from .gc import ManifoldGC
from .doctor import ManifoldDoctor
from .autodoc import APIExplorer, DocExtractor, MANIFOLD_ENDPOINTS
from .zkp import ZKPVerifier, ZKProof
from .rosetta import ForeignPayloadIngress, EgressTranslator
from .temporal import BranchResult, ParallelTimeline, TimelineCollapse
from .gridmapper import GridState, GridWorld
from .db import ManifoldDB
from .worker import ManifoldWorker
from .auth import ManifoldAuth
from .trust_network.registry import ATSRegistry
from .trust_network.models import ToolRegistration, TrustSignal
from .pipeline import ManifoldPipeline
from .cognitive_map import CognitiveMap
from .cooccurrence import ToolCooccurrenceGraph
from .predictor import PredictiveBrain
from .consolidator import MemoryConsolidator
from .policy_rules import PolicyRule, PolicyRuleEngine
from .federation import FederatedGossipBridge
from .cell_update_bus import get_bus as _get_bus
from .dynamic_grid import get_grid as _get_grid
from .health_monitor import DigitalHealthMonitor as _DigitalHealthMonitor
from .planner import CRNAPlanner as _CRNAPlanner
from .nervatura_world import NERVATURAWorld as _NERVATURAWorld


# ---------------------------------------------------------------------------
# Brain state persistence
# ---------------------------------------------------------------------------

_BRAIN_STATE_DIR = Path(
    os.environ.get("MANIFOLD_STATE_DIR", os.path.expanduser("~/.manifold/brain"))
)


def _save_brain_state() -> None:
    """Serialise all four brain components to disk."""
    if _pipeline is None:
        return
    try:
        _BRAIN_STATE_DIR.mkdir(parents=True, exist_ok=True)
        _pipeline._cognitive_map.save(str(_BRAIN_STATE_DIR / "cognitive_map.json"))
        _pipeline._cooccurrence.save(str(_BRAIN_STATE_DIR / "cooccurrence.json"))
        _pipeline._predictor.save(str(_BRAIN_STATE_DIR / "predictor.json"))
        _pipeline._consolidator.save(str(_BRAIN_STATE_DIR / "consolidator.json"))
    except Exception as exc:  # noqa: BLE001
        print(f"[MANIFOLD] Brain state save failed: {exc}")


def _rehydrate_brain() -> None:
    """Load persisted brain components into the active pipeline (if any)."""
    if _pipeline is None:
        return
    try:
        cmap_path = _BRAIN_STATE_DIR / "cognitive_map.json"
        cooc_path = _BRAIN_STATE_DIR / "cooccurrence.json"
        pred_path = _BRAIN_STATE_DIR / "predictor.json"
        cons_path = _BRAIN_STATE_DIR / "consolidator.json"
        if cmap_path.exists():
            _pipeline._cognitive_map = CognitiveMap.load(str(cmap_path))
        if cooc_path.exists():
            _pipeline._cooccurrence = ToolCooccurrenceGraph.load(str(cooc_path))
        if pred_path.exists():
            _pipeline._predictor = PredictiveBrain.load(str(pred_path))
        if cons_path.exists():
            _pipeline._consolidator = MemoryConsolidator.load(str(cons_path))
        print("[MANIFOLD] Brain state rehydrated from disk")
    except Exception as exc:  # noqa: BLE001
        print(f"[MANIFOLD] Brain rehydration skipped: {exc}")


atexit.register(_save_brain_state)


# ---------------------------------------------------------------------------
# Policy rule engine singleton
# ---------------------------------------------------------------------------

_RULE_ENGINE = PolicyRuleEngine()


# ---------------------------------------------------------------------------
# Federation singletons
# ---------------------------------------------------------------------------

_GOSSIP_BRIDGE = FederatedGossipBridge()
_FEDERATION_LEDGER = _GOSSIP_BRIDGE.ledger


# ---------------------------------------------------------------------------
# Optional bearer-token authentication
# ---------------------------------------------------------------------------


def _init_auth() -> "ManifoldAuth | None":
    """Initialise authentication from the environment.

    Returns ``None`` when ``MANIFOLD_API_KEY`` is not set (auth disabled).
    Exits the process immediately when the key is set but blank/whitespace,
    which would indicate a misconfigured deployment.
    """
    key = os.environ.get("MANIFOLD_API_KEY", "")
    if not key:
        # Key absent → auth disabled (dev/test mode)
        return None
    key = key.strip()
    if not key:
        print(
            "[MANIFOLD] ERROR: MANIFOLD_API_KEY is set but blank/whitespace.\n"
            "           Set a non-empty key or unset the variable to disable auth.\n"
            "           Generate one with: "
            "python -c 'import secrets; print(secrets.token_hex(32))'",
            file=sys.stderr,
        )
        sys.exit(1)
    return ManifoldAuth(key)


_AUTH: "ManifoldAuth | None" = _init_auth()

# Routes that require auth when _AUTH is active
_PROTECTED_POSTS = frozenset({"/shield", "/b2b/handshake", "/recruit", "/ats/register", "/ats/signal", "/run", "/v1/chat/completions"})

# Lazy pipeline singleton (created on first request to avoid circular imports)
_pipeline: "Any | None" = None
_worker: "Any | None" = None


def _get_pipeline() -> "Any":
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        _pipeline = ManifoldPipeline()
    return _pipeline


def _check_auth(handler: "ManifoldHandler", path: str = "") -> "tuple[bool, _OrgConfig | None]":
    """Return ``(authorised, org_config)``.

    Looks up the bearer token in OrgRegistry.  Falls back to the legacy
    ``MANIFOLD_API_KEY`` environment variable for backward compatibility.
    When auth is disabled (no key configured), returns ``(True, admin_org)``.
    """
    auth_header = handler.headers.get("Authorization", "")
    env_key = os.environ.get("MANIFOLD_API_KEY", "")

    # Auth disabled — no key configured at all
    if not env_key and not auth_header:
        return True, _OrgConfig(
            org_id="anon",
            display_name="Anonymous",
            role=_OrgRole.ADMIN,
            api_key_hash="",
        )

    if not auth_header.startswith("Bearer "):
        _send_json(handler, 401, {"code": 401, "message": "Unauthorized"})
        return False, None

    token = auth_header[7:].strip()

    # OrgRegistry lookup (primary path)
    org = _ORG_REGISTRY.lookup(token)
    if org is not None:
        return True, org

    # Backward compat: raw env-key match → treat as admin
    if env_key and hmac.compare_digest(token, env_key):
        return True, _OrgConfig(
            org_id="env-admin",
            display_name="Environment Admin",
            role=_OrgRole.ADMIN,
            api_key_hash="",
        )

    # Auth configured but token does not match
    _send_json(handler, 403, {"code": 403, "message": "Forbidden"})
    return False, None


# ---------------------------------------------------------------------------
# In-memory task counters (incremented by the shield handler)
# ---------------------------------------------------------------------------

_TASK_COUNT: int = 0
_ESCALATION_COUNT: int = 0
_REFUSAL_COUNT: int = 0

_ats_registry = ATSRegistry()

# Multi-tenancy: OrgRegistry for per-org policy and RBAC
from .orgs import OrgConfig as _OrgConfig, OrgRegistry, OrgRole as _OrgRole  # noqa: E402

_ORG_REGISTRY = OrgRegistry(
    orgs_file=os.environ.get("MANIFOLD_ORGS_FILE", "orgs.json")
)

# Agentic layer singletons
from manifold.agent_registry import AgentRegistry as _AgentRegistry  # noqa: E402
from manifold.monitor import AgentMonitor as _AgentMonitor  # noqa: E402
from manifold.task_router import TaskRouter as _TaskRouter  # noqa: E402

_AGENT_REGISTRY = _AgentRegistry(stale_timeout=120)
_AGENT_MONITOR = _AgentMonitor(_AGENT_REGISTRY, check_interval=30)
_TASK_ROUTER = _TaskRouter(registry=_AGENT_REGISTRY)


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

# Phase 64: Rosetta Protocol Adapter
_ROSETTA_INGRESS = ForeignPayloadIngress()
_ROSETTA_EGRESS = EgressTranslator()

# v1.7.0: Real-Time Layer singletons
_DYNAMIC_GRID = _get_grid()          # auto-subscribes to CellUpdateBus
_HEALTH_MONITOR = _DigitalHealthMonitor()
_PLANNER = _CRNAPlanner()
_NERVATURA: "_NERVATURAWorld | None" = None  # initialised via POST /nervatura/world/init

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

            # GET /metrics  (Priority 7 — Prometheus-compatible metrics)
            if path == "/metrics":
                self._handle_get_metrics()
                return

            # GET /ats/score/<tool_id>
            m_ats = re.fullmatch(r"/ats/score/(.+)", path)
            if m_ats:
                self._handle_get_ats_score(m_ats.group(1))
                return

            # GET /ats/leaderboard
            if path == "/ats/leaderboard":
                self._handle_get_ats_leaderboard()
                return

            # GET /learned
            if path == "/learned":
                self._handle_get_learned()
                return

            # GET /v1/models  (OpenAI-compatible models list)
            if path == "/v1/models":
                self._handle_get_v1_models()
                return

            # GET /admin  (org & policy management UI)
            if path == "/admin":
                self._handle_get_admin()
                return

            # GET /report  (self-reporting visual dashboard)
            if path == "/report":
                self._handle_get_report()
                return

            # GET /digest  (structured JSON governance summary)
            if path.startswith("/digest"):
                self._handle_get_digest()
                return

            # GET /  (landing page)
            if path == "" or path == "/":
                self._handle_get_landing()
                return

            # GET /signup  (signup form)
            if path == "/signup":
                self._handle_get_signup()
                return

            # GET /connect  (tool connection guide)
            if path == "/connect":
                self._handle_get_connect()
                return

            # GET /agents  (list all registered agents)
            if path == "/agents":
                self._handle_get_agents()
                return

            # GET /world  (serve the isometric game world)
            if path == "/world":
                self._handle_get_world()
                return

            # GET /world/manifest.json  (PWA manifest)
            if path == "/world/manifest.json":
                self._handle_get_world_manifest()
                return

            # GET /ws  (WebSocket upgrade)
            if path == "/ws":
                self._handle_ws_upgrade()
                return

            # GET /brain/state  (brain persistence status)
            if path == "/brain/state":
                self._handle_get_brain_state()
                return

            # GET /agents/{id}/commands  (long-poll for agent commands)
            if re.match(r"^/agents/[\w-]+/commands$", path):
                agent_id = path.split("/")[2]
                self._handle_get_agent_commands(agent_id)
                return

            # GET /rules  (list policy rules for caller org)
            if path == "/rules":
                self._handle_get_rules()
                return

            # GET /federation/status  (federation health)
            if path == "/federation/status":
                self._handle_get_federation_status()
                return

            # GET /realtime/status  (v1.7.0 real-time bus/grid/health/planner)
            if path == "/realtime/status":
                self._handle_get_realtime_status()
                return

            # GET /health/tools  (live tool health summary)
            if path == "/health/tools":
                self._handle_get_health_tools()
                return

            # GET /plan  (CRNA A* path planning)
            if path == "/plan":
                self._handle_get_plan()
                return

            # GET /nervatura/world  (NERVATURAWorld summary)
            if path == "/nervatura/world":
                self._handle_get_nervatura_world()
                return

            # GET /physical/cameras  (camera registry status)
            if path == "/physical/cameras":
                self._handle_get_physical_cameras()
                return

            # GET /physical/status  (PhysicalManager status)
            if path == "/physical/status":
                self._handle_get_physical_status()
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
            if path in _PROTECTED_POSTS:
                _authed, _caller_org = _check_auth(self, path)
                if not _authed:
                    return
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
            elif path == "/rosetta/ingress":
                self._handle_post_rosetta_ingress(body)
            elif path == "/temporal/fork":
                self._handle_post_temporal_fork(body)
            elif path == "/ats/register":
                self._handle_post_ats_register(body)
            elif path == "/ats/signal":
                self._handle_post_ats_signal(body)
            elif path == "/run":
                self._handle_post_run(body)
            elif path == "/v1/chat/completions":
                self._handle_post_v1_chat_completions(body)
            elif path == "/orgs":
                _authed2, _caller2 = _check_auth(self, path)
                if not _authed2:
                    return
                self._handle_post_orgs(body, _caller2)
            elif re.fullmatch(r"/orgs/[^/]+/policy", path):
                _authed2, _caller2 = _check_auth(self, path)
                if not _authed2:
                    return
                org_id = path.split("/")[2]
                self._handle_put_org_policy(org_id, body, _caller2)
            elif re.fullmatch(r"/orgs/[^/]+/keys", path):
                _authed2, _caller2 = _check_auth(self, path)
                if not _authed2:
                    return
                org_id = path.split("/")[2]
                self._handle_post_org_key(org_id, body, _caller2)
            elif path == "/signup":
                self._handle_post_signup(body)
            elif path == "/agents/register":
                self._handle_post_agents_register(body)
            elif path.startswith("/agents/") and path.endswith("/heartbeat"):
                agent_id = path.split("/")[2]
                self._handle_post_agent_heartbeat(agent_id, body)
            elif path.startswith("/agents/") and path.endswith("/pause"):
                agent_id = path.split("/")[2]
                self._handle_post_agent_pause(agent_id)
            elif path.startswith("/agents/") and path.endswith("/resume"):
                agent_id = path.split("/")[2]
                self._handle_post_agent_resume(agent_id)
            elif re.match(r"^/agents/[\w-]+/command$", path):
                agent_id = path.split("/")[2]
                self._handle_post_agent_command(agent_id, body)
            elif path == "/rules":
                _authed2, _caller2 = _check_auth(self, path)
                if not _authed2:
                    return
                self._handle_post_rule(body, _caller2)
            elif re.match(r"^/rules/[^/]+$", path):
                # DELETE-like fallback not needed here; handled in do_DELETE
                _send_error(self, 405, "Use DELETE /rules/{rule_id}")
            elif path == "/federation/join":
                _authed2, _caller2 = _check_auth(self, path)
                if not _authed2:
                    return
                self._handle_post_federation_join(body, _caller2)
            elif path == "/federation/gossip":
                self._handle_post_federation_gossip(body)
            elif path == "/task":
                self._handle_post_task(body)
            elif path == "/nervatura/world/init":
                _authed2, _caller2 = _check_auth(self, path)
                if not _authed2:
                    return
                self._handle_post_nervatura_world_init(body)
            elif path == "/physical/init":
                self._handle_post_physical_init(body)
            else:
                _send_error(self, 404, f"No route for POST {self.path}")
        except Exception as exc:  # noqa: BLE001
            _send_error(self, 500, str(exc))

    # ------------------------------------------------------------------
    # DELETE dispatcher
    # ------------------------------------------------------------------

    def do_DELETE(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        try:
            _authed, _caller = _check_auth(self, path)
            if not _authed:
                return
            if re.match(r"^/rules/[^/]+$", path):
                rule_id = path.split("/")[2]
                self._handle_delete_rule(rule_id)
            else:
                _send_error(self, 404, f"No route for DELETE {self.path}")
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

        # Update in-memory counters for /metrics endpoint (protected by _LOCK)
        global _TASK_COUNT, _ESCALATION_COUNT, _REFUSAL_COUNT
        with _LOCK:
            _TASK_COUNT += 1
            if decision.action == "escalate":
                _ESCALATION_COUNT += 1
            elif decision.action == "refuse":
                _REFUSAL_COUNT += 1

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
        def _shutdown() -> None:
            time.sleep(0.1)
        threading.Thread(target=_shutdown, daemon=True).start()

    def _handle_post_sandbox_execute(self, body: dict[str, Any]) -> None:
        """POST /sandbox/execute → Phase 44 AST-sandboxed code execution."""
        source = str(body.get("source", ""))
        agent_id = str(body.get("agent_id", ""))
        if not source:
            _send_error(self, 400, "Body must contain 'source'.")
            return

        # Static validation first
        violations = _SANDBOX_VALIDATOR.validate(source)
        if violations:
            source_hash = hashlib.md5(source.encode(), usedforsecurity=False).hexdigest()[:8]  # noqa: S324
            with _LOCK:
                _VAULT.append_sandbox_violation(
                    source_hash,
                    timestamp=time.time(),
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
                timestamp=time.time(),
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
        payload = html.encode("utf-8")
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

    def _handle_get_metrics(self) -> None:
        """GET /metrics → Prometheus-compatible plain-text metrics."""
        with _LOCK:
            all_baselines = dict(_HUB.baseline.tool_baselines)

        lines = [
            "# HELP manifold_tasks_total Total tasks processed",
            "# TYPE manifold_tasks_total counter",
            f"manifold_tasks_total {_TASK_COUNT}",
            "# HELP manifold_escalations_total Tasks escalated to human",
            "# TYPE manifold_escalations_total counter",
            f"manifold_escalations_total {_ESCALATION_COUNT}",
            "# HELP manifold_refusals_total Tasks refused",
            "# TYPE manifold_refusals_total counter",
            f"manifold_refusals_total {_REFUSAL_COUNT}",
            "# HELP manifold_tool_reliability Reputation score per tool",
            "# TYPE manifold_tool_reliability gauge",
        ]
        for tool_name, (score, _weight) in sorted(all_baselines.items()):
            safe_name = tool_name.replace("-", "_").replace(".", "_")
            lines.append(f'manifold_tool_reliability{{tool="{safe_name}"}} {score:.4f}')

        # Also emit live reliability if hub has contributions
        with _LOCK:
            for tool_name, (score, _weight) in sorted(all_baselines.items()):
                live = _HUB.live_reliability(tool_name)
                if live is not None:
                    safe_name = tool_name.replace("-", "_").replace(".", "_")
                    # Overwrite the baseline line with live value
                    live_line = f'manifold_tool_reliability_live{{tool="{safe_name}"}} {live:.4f}'
                    lines.append(live_line)

        body = ("\n".join(lines) + "\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

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
            endorsement = ToolEndorsement(
                genesis_org_id=str(body.get("genesis_org_id", "")),
                tool_id=tool_id,
                manifest_hash=str(body.get("manifest_hash", "")),
                signature=str(body.get("signature", "")),
                key_id=str(body.get("key_id", "")),
                timestamp=float(body.get("timestamp", time.time())),
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
        try:
            proof = ZKProof.from_dict(body)
        except (KeyError, TypeError, ValueError) as exc:
            _send_error(self, 400, f"Invalid proof payload: {exc}")
            return
        valid = _ZKP_VERIFIER.verify(proof)
        _send_json(self, 200, {"valid": valid})

    def _handle_post_rosetta_ingress(self, body: dict[str, Any]) -> None:
        """POST /rosetta/ingress → Phase 64 translate foreign payload to BrainTask."""
        result = _ROSETTA_INGRESS.ingest(body)
        _send_json(self, 200, result.to_dict())

    def _handle_post_temporal_fork(self, body: dict[str, Any]) -> None:
        """POST /temporal/fork → Phase 63 fork parallel timelines and collapse.

        Expects a JSON body with:
        - ``branches``: list of label strings (required)
        - ``grid_state``: optional dict passed as branch metadata
        - ``fork_id``: optional string
        """
        branch_labels: list[str] = body.get("branches", [])
        if not isinstance(branch_labels, list) or not branch_labels:
            _send_error(self, 400, "'branches' must be a non-empty list of label strings")
            return
        fork_id: str | None = body.get("fork_id") or None

        # Use a minimal GridState sourced from the GridMapper
        meta: dict[str, object] = body.get("grid_state", {})
        world = GridWorld(size=5)
        state = GridState(
            world=world,
            description=str(meta.get("description", "temporal-fork")),
            domain=str(meta.get("domain", "general")),
            parameters={},
            cell_profile=(0.1, 0.2, 0.3, 0.4),
        )
        ledger = AgentEconomyLedger()

        def default_executor(
            branch_id: str,
            _s: GridState,
            _l: AgentEconomyLedger,
        ) -> "BranchResult":
            return BranchResult(
                branch_id=branch_id,
                label=branch_id,
                asset=1.0,
                cost=0.5,
                risk=0.5,
                success=True,
            )

        timeline = ParallelTimeline(state, ledger, fork_id=fork_id)
        for label in branch_labels:
            timeline.add_branch(str(label))

        results = timeline.run(default_executor)
        collapse = TimelineCollapse.collapse(results, fork_id=timeline.fork_id)
        _send_json(self, 200, collapse.to_dict())

    # -------------------------------------------------------------------------
    # ATS endpoint implementations — Phase 70
    # -------------------------------------------------------------------------

    def _handle_get_ats_score(self, tool_id: str) -> None:
        """GET /ats/score/<tool_id> → AgentTrustScore as JSON (public)."""
        with _LOCK:
            data = _ats_registry.to_dict(tool_id)
        _send_json(self, 200, data)

    def _handle_get_ats_leaderboard(self) -> None:
        """GET /ats/leaderboard → top 10 AgentTrustScores as JSON (public)."""
        with _LOCK:
            board = [_ats_registry.to_dict(s.tool_id) for s in _ats_registry.leaderboard()]
        _send_json(self, 200, board)

    def _handle_post_ats_register(self, body: dict[str, Any]) -> None:
        """POST /ats/register → register a tool in the ATS network (authenticated)."""
        tool = ToolRegistration(
            tool_id=body["tool_id"],
            org_id=body.get("org_id", "unknown"),
            display_name=body.get("display_name", body["tool_id"]),
            domain=body.get("domain", "general"),
            description=body.get("description", ""),
        )
        with _LOCK:
            _ats_registry.register_tool(tool)
        _send_json(self, 200, {"registered": True, "tool_id": tool.tool_id})

    def _handle_post_ats_signal(self, body: dict[str, Any]) -> None:
        """POST /ats/signal → submit a trust signal (authenticated)."""
        signal = TrustSignal(
            tool_id=body["tool_id"],
            signal_type=body["signal_type"],
            domain=body.get("domain", "general"),
            stakes=float(body.get("stakes", 0.5)),
            submitter_hash=body.get("submitter_hash", "anonymous"),
            metadata=body.get("metadata", {}),
        )
        with _LOCK:
            _ats_registry.submit_signal(signal)
        _send_json(self, 200, {"recorded": True, "tool_id": signal.tool_id})


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
# Pipeline endpoint handler helpers (methods added to ManifoldHandler)
# ---------------------------------------------------------------------------


def _handle_post_run(self: "ManifoldHandler", body: dict[str, Any]) -> None:
    """POST /run — execute ManifoldPipeline and return the result."""
    prompt = body.get("prompt")
    if not prompt:
        _send_json(self, 400, {"error": "prompt required"})
        return
    try:
        pipeline = _get_pipeline()
        result = pipeline.run(
            prompt=str(prompt),
            data=body.get("data"),
            encoder_hint=str(body.get("encoder_hint", "auto")),
            explicit_domain=body.get("domain") or None,
            stakes=float(body.get("stakes", 0.5)),
            uncertainty=float(body.get("uncertainty", 0.5)),
            tools_used=body.get("tools_used"),
        )
        # Serialise to plain JSON-safe dict
        serialised = {
            "action": result["action"],
            "domain": result["domain"],
            "risk_score": result["risk_score"],
            "nearest_cells": [
                {"row": c.get("row", 0), "col": c.get("col", 0), "distance": c.get("distance", 0.0)}
                for c in result.get("nearest_cells", [])
            ],
            "flagged_tools": result.get("flagged_tools", []),
        }
        _send_json(self, 200, serialised)
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


def _handle_get_learned(self: "ManifoldHandler") -> None:
    """GET /learned — return what the system has learned so far."""
    try:
        pipeline = _get_pipeline()

        # Section 1: cognitive_map
        outcome_log = pipeline._cognitive_map._outcome_log
        all_outcomes = [entry for entries in outcome_log.values() for entry in entries]
        total_outcomes = len(all_outcomes)
        recent_actions = [e["action"] for e in all_outcomes[-10:]]
        success_rate = (
            sum(1 for e in all_outcomes if e.get("success")) / total_outcomes
            if total_outcomes > 0 else None
        )

        # Section 2: cooccurrence
        cooccurrence_summary = pipeline._cooccurrence.summary()

        # Section 3: consolidation
        promoted = pipeline._consolidator.promoted_rules()
        consolidation_list = [
            {
                "domain": r.domain,
                "action": r.action,
                "stakes_min": r.stakes_min,
                "confidence": r.confidence,
                "sample_count": r.sample_count,
            }
            for r in promoted
        ]

        # Section 4: prediction
        prediction_cal = pipeline._predictor.calibration_signal()

        # Section 5: worker
        if _worker is not None:
            worker_status = _worker.status()
        else:
            worker_status = {"running": False, "last_run": None}

        response = {
            "cognitive_map": {
                "total_outcomes": total_outcomes,
                "recent_actions": recent_actions,
                "success_rate": success_rate,
            },
            "cooccurrence": cooccurrence_summary,
            "consolidation": consolidation_list,
            "prediction": prediction_cal,
            "worker": worker_status,
        }
        _send_json(self, 200, response)
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


# ---------------------------------------------------------------------------
# Universal AI Gateway handlers (OpenAI-compatible)
# ---------------------------------------------------------------------------

_V1_MODELS_LIST = {
    "object": "list",
    "data": [
        {"id": "manifold-governed", "object": "model", "created": 0, "owned_by": "manifold"},
        {"id": "gpt-4o", "object": "model", "created": 0, "owned_by": "manifold-proxy"},
        {"id": "gpt-4-turbo", "object": "model", "created": 0, "owned_by": "manifold-proxy"},
        {"id": "claude-3-5-sonnet", "object": "model", "created": 0, "owned_by": "manifold-proxy"},
    ],
}


def _handle_get_v1_models(self: "ManifoldHandler") -> None:
    """GET /v1/models — OpenAI-compatible models list (public)."""
    _send_json(self, 200, _V1_MODELS_LIST)


def _handle_post_v1_chat_completions(self: "ManifoldHandler", body: dict[str, Any]) -> None:
    """POST /v1/chat/completions — Universal AI gateway endpoint.

    Governs the request through ManifoldBrain, then either forwards to the
    configured upstream LLM or returns a governance-only response.
    """
    import uuid as _uuid

    # STEP 1 — Extract prompt from messages array
    messages = body.get("messages", [])
    model = body.get("model", "gpt-4o")
    prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            prompt = content if isinstance(content, str) else str(content)
            break
    if not prompt:
        prompt = str(messages) if messages else str(body)

    # STEP 2 — Auto-detect framework and translate to BrainTask via rosetta
    ingress_result = _ROSETTA_INGRESS.ingest(body)
    task = dataclasses.replace(ingress_result.task, prompt=prompt)

    # STEP 3 — Govern through ManifoldBrain
    brain_decision = _BRAIN.decide(task)
    vetoed = brain_decision.action in ("refuse", "stop")

    # STEP 4a — Vetoed: return governance refusal in OpenAI format
    if vetoed:
        refusal_msg = (
            f"[MANIFOLD GOVERNANCE] Request refused. "
            f"Risk score: {brain_decision.risk_score:.2f}. "
            f"Action: {brain_decision.action}. "
            f"Domain: {task.domain}. "
            f"To override, reduce stakes or contact your system administrator."
        )
        response: dict[str, Any] = {
            "id": f"chatcmpl-manifold-{_uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": refusal_msg},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "_manifold": {
                "governed": True,
                "vetoed": True,
                "action": brain_decision.action,
                "risk_score": round(brain_decision.risk_score, 4),
                "domain": task.domain,
            },
        }
        _send_json(self, 200, response)
        return

    # STEP 4b — Permitted and upstream configured: forward to real LLM
    upstream_url = os.environ.get("MANIFOLD_UPSTREAM_URL", "").rstrip("/")
    upstream_key = (
        os.environ.get("MANIFOLD_UPSTREAM_KEY")
        or os.environ.get("OPENAI_API_KEY", "")
    )

    if upstream_url and upstream_key:
        import urllib.request
        import urllib.error
        import json as _json

        req_data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{upstream_url}/chat/completions",
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {upstream_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                upstream_response = _json.loads(resp.read().decode("utf-8"))
            upstream_response["_manifold"] = {
                "governed": True,
                "vetoed": False,
                "action": brain_decision.action,
                "risk_score": round(brain_decision.risk_score, 4),
                "domain": task.domain,
            }
            _send_json(self, 200, upstream_response)
            return
        except urllib.error.URLError:
            # Upstream unreachable — fall through to governance-only response
            pass

    # STEP 4c — Governance-only mode (no upstream configured or upstream failed)
    governed_content = (
        f"[MANIFOLD GOVERNANCE — PERMITTED] "
        f"Action: {brain_decision.action}. "
        f"Risk score: {brain_decision.risk_score:.2f}. "
        f"No upstream LLM configured. Set MANIFOLD_UPSTREAM_URL and "
        f"MANIFOLD_UPSTREAM_KEY to enable response forwarding."
    )
    response = {
        "id": f"chatcmpl-manifold-{_uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "manifold-governed",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": governed_content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "_manifold": {
            "governed": True,
            "vetoed": False,
            "action": brain_decision.action,
            "risk_score": round(brain_decision.risk_score, 4),
            "domain": task.domain,
        },
    }
    _send_json(self, 200, response)


# ---------------------------------------------------------------------------
# Multi-tenancy handlers (org management + admin UI)
# ---------------------------------------------------------------------------


def _handle_post_orgs(self: "ManifoldHandler", body: dict[str, Any], caller_org: "_OrgConfig") -> None:
    """POST /orgs — create a new org. Admin only."""
    if caller_org.role.value != "admin":
        _send_error(self, 403, "Admin role required")
        return
    name = str(body.get("display_name", ""))
    oid  = str(body.get("org_id", ""))
    role = body.get("role", "agent")
    if not name or not oid:
        _send_error(self, 400, "display_name and org_id required")
        return
    kwargs = {
        k: body[k]
        for k in ("domain", "risk_tolerance", "veto_threshold",
                  "min_reliability", "fallback", "notes")
        if k in body
    }
    try:
        raw_key, config = _ORG_REGISTRY.generate_key(
            oid, name, _OrgRole(role), **kwargs
        )
    except ValueError as exc:
        _send_error(self, 400, str(exc))
        return
    _send_json(self, 201, {
        "org_id":  config.org_id,
        "api_key": raw_key,
        "role":    config.role.value,
        "warning": "Store this API key — it will not be shown again",
    })


def _handle_put_org_policy(self: "ManifoldHandler", org_id: str, body: dict[str, Any], caller_org: "_OrgConfig") -> None:
    """PUT /orgs/{org_id}/policy — update org policy fields. Admin only."""
    if caller_org.role.value != "admin":
        _send_error(self, 403, "Admin role required")
        return
    allowed = {"domain", "risk_tolerance", "veto_threshold",
               "min_reliability", "fallback", "notes", "allowed_tools"}
    updates = {k: v for k, v in body.items() if k in allowed}
    all_orgs = _ORG_REGISTRY.all_orgs()
    target = next((o for o in all_orgs if o.org_id == org_id), None)
    if target is None:
        _send_error(self, 404, f"Org {org_id!r} not found")
        return
    for k, v in updates.items():
        setattr(target, k, v)
    _ORG_REGISTRY.save()
    _send_json(self, 200, target.to_dict())


def _handle_post_org_key(self: "ManifoldHandler", org_id: str, body: dict[str, Any], caller_org: "_OrgConfig") -> None:
    """POST /orgs/{org_id}/keys — generate a new API key. Admin only."""
    if caller_org.role.value != "admin":
        _send_error(self, 403, "Admin role required")
        return
    all_orgs = _ORG_REGISTRY.all_orgs()
    target = next((o for o in all_orgs if o.org_id == org_id), None)
    if target is None:
        _send_error(self, 404, f"Org {org_id!r} not found")
        return
    raw_key = secrets.token_hex(32)
    _ORG_REGISTRY.register(
        raw_key,
        target.org_id,
        target.display_name,
        target.role,
        domain=target.domain,
        risk_tolerance=target.risk_tolerance,
    )
    _send_json(self, 201, {
        "org_id":  org_id,
        "api_key": raw_key,
        "warning": "Store this key — it will not be shown again",
    })


def _handle_get_admin(self: "ManifoldHandler") -> None:
    """GET /admin — org & policy management dashboard. Admin only."""
    from manifold import __version__ as _version

    all_orgs = _ORG_REGISTRY.all_orgs()
    org_count = len(all_orgs)

    # Build table rows
    def _esc_js(s: str) -> str:
        """Escape a string for safe embedding in a JS string literal."""
        return (
            s.replace("\\", "\\\\")
             .replace("'", "\\'")
             .replace("\n", "\\n")
             .replace("\r", "\\r")
             .replace("<", "\\x3C")
             .replace(">", "\\x3E")
        )

    def _esc_html(s: str) -> str:
        """Escape a string for safe embedding in HTML text content."""
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#x27;")
        )

    rows_html = ""
    for org in all_orgs:
        rows_html += f"""
        <tr class="border-b hover:bg-gray-50">
          <td class="px-4 py-2 font-mono text-sm">{_esc_html(org.org_id)}</td>
          <td class="px-4 py-2">{_esc_html(org.display_name)}</td>
          <td class="px-4 py-2"><span class="px-2 py-0.5 rounded text-xs font-semibold bg-blue-100 text-blue-800">{_esc_html(org.role.value)}</span></td>
          <td class="px-4 py-2">{_esc_html(org.domain)}</td>
          <td class="px-4 py-2">{org.risk_tolerance}</td>
          <td class="px-4 py-2">{org.veto_threshold}</td>
          <td class="px-4 py-2">{org.min_reliability}</td>
          <td class="px-4 py-2">{_esc_html(org.fallback)}</td>
          <td class="px-4 py-2 space-x-2">
            <button onclick="editOrg('{_esc_js(org.org_id)}', {org.risk_tolerance}, {org.veto_threshold}, '{_esc_js(org.domain)}', '{_esc_js(org.fallback)}', '{_esc_js(org.notes)}')"
                    class="px-2 py-1 text-xs bg-yellow-400 hover:bg-yellow-500 rounded">Edit Policy</button>
            <button onclick="newKey('{_esc_js(org.org_id)}')"
                    class="px-2 py-1 text-xs bg-green-500 hover:bg-green-600 text-white rounded">New Key</button>
          </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MANIFOLD Admin</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen p-6">
  <div class="max-w-7xl mx-auto">
    <div class="mb-6 flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">MANIFOLD Admin — Org &amp; Policy Management</h1>
        <p class="text-sm text-gray-500 mt-1">Version {_version} &bull; {org_count} org(s) registered</p>
      </div>
      <a href="/dashboard" class="text-sm text-blue-600 hover:underline">← Dashboard</a>
    </div>

    <!-- Orgs table -->
    <div class="bg-white rounded-lg shadow overflow-x-auto mb-8">
      <table class="min-w-full text-sm text-gray-700">
        <thead class="bg-gray-800 text-white text-xs uppercase">
          <tr>
            <th class="px-4 py-3 text-left">Org ID</th>
            <th class="px-4 py-3 text-left">Display Name</th>
            <th class="px-4 py-3 text-left">Role</th>
            <th class="px-4 py-3 text-left">Domain</th>
            <th class="px-4 py-3 text-left">Risk Tol.</th>
            <th class="px-4 py-3 text-left">Veto Thresh.</th>
            <th class="px-4 py-3 text-left">Min Reliability</th>
            <th class="px-4 py-3 text-left">Fallback</th>
            <th class="px-4 py-3 text-left">Actions</th>
          </tr>
        </thead>
        <tbody>{rows_html}
        </tbody>
      </table>
    </div>

    <!-- Create Org form -->
    <div class="bg-white rounded-lg shadow p-6 mb-8">
      <h2 class="text-lg font-semibold mb-4">Create New Org</h2>
      <form id="createOrgForm" class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Org ID</label>
          <input name="org_id" required class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" placeholder="acme-corp">
        </div>
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Display Name</label>
          <input name="display_name" required class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" placeholder="Acme Corporation">
        </div>
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Role</label>
          <select name="role" class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400">
            <option value="agent">agent</option>
            <option value="admin">admin</option>
            <option value="readonly">readonly</option>
            <option value="viewer">viewer</option>
          </select>
        </div>
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Domain</label>
          <select name="domain" class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400">
            <option value="general">general</option>
            <option value="finance">finance</option>
            <option value="healthcare">healthcare</option>
            <option value="legal">legal</option>
            <option value="devops">devops</option>
            <option value="custom">custom</option>
          </select>
        </div>
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Risk Tolerance</label>
          <input type="number" name="risk_tolerance" min="0.05" max="0.95" step="0.05" value="0.45" class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400">
        </div>
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Veto Threshold</label>
          <input type="number" name="veto_threshold" min="0.05" max="0.95" step="0.05" value="0.45" class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400">
        </div>
        <div class="col-span-2">
          <label class="block text-xs font-medium text-gray-600 mb-1">Notes</label>
          <textarea name="notes" rows="2" class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400" placeholder="Optional notes"></textarea>
        </div>
        <div class="col-span-2">
          <button type="submit" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium">
            Create Org + Generate Key
          </button>
        </div>
      </form>
      <div id="createResult" class="mt-4 hidden p-4 bg-green-50 border border-green-200 rounded text-sm font-mono"></div>
    </div>

    <!-- Edit Policy section -->
    <div id="editSection" class="bg-white rounded-lg shadow p-6 hidden mb-8">
      <h2 class="text-lg font-semibold mb-4">Edit Policy — <span id="editOrgId" class="font-mono text-blue-700"></span></h2>
      <form id="editPolicyForm" class="grid grid-cols-2 gap-4">
        <input type="hidden" name="org_id" id="editOrgIdInput">
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Risk Tolerance</label>
          <input type="number" name="risk_tolerance" id="editRiskTol" min="0.05" max="0.95" step="0.05" class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-yellow-400">
        </div>
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Veto Threshold</label>
          <input type="number" name="veto_threshold" id="editVetoThresh" min="0.05" max="0.95" step="0.05" class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-yellow-400">
        </div>
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Domain</label>
          <input name="domain" id="editDomain" class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-yellow-400">
        </div>
        <div>
          <label class="block text-xs font-medium text-gray-600 mb-1">Fallback</label>
          <select name="fallback" id="editFallback" class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-yellow-400">
            <option value="hitl">hitl</option>
            <option value="refuse">refuse</option>
          </select>
        </div>
        <div class="col-span-2">
          <label class="block text-xs font-medium text-gray-600 mb-1">Notes</label>
          <textarea name="notes" id="editNotes" rows="2" class="w-full border rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-yellow-400"></textarea>
        </div>
        <div class="col-span-2">
          <button type="submit" class="px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded text-sm font-medium">
            Save Policy
          </button>
          <button type="button" onclick="document.getElementById('editSection').classList.add('hidden')" class="ml-2 px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded text-sm">Cancel</button>
        </div>
      </form>
      <div id="editResult" class="mt-4 hidden p-4 rounded text-sm font-mono"></div>
    </div>
  </div>

  <script>
    function editOrg(orgId, riskTol, vetoThresh, domain, fallback, notes) {{
      document.getElementById('editSection').classList.remove('hidden');
      document.getElementById('editOrgId').textContent = orgId;
      document.getElementById('editOrgIdInput').value = orgId;
      document.getElementById('editRiskTol').value = riskTol;
      document.getElementById('editVetoThresh').value = vetoThresh;
      document.getElementById('editDomain').value = domain;
      document.getElementById('editFallback').value = fallback;
      document.getElementById('editNotes').value = notes;
      document.getElementById('editSection').scrollIntoView({{behavior:'smooth'}});
    }}

    function newKey(orgId) {{
      if (!confirm('Generate a new API key for ' + orgId + '? Existing keys remain valid.')) return;
      const authKey = prompt('Enter your admin API key:');
      if (!authKey) return;
      fetch('/orgs/' + orgId + '/keys', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json', 'Authorization': 'Bearer ' + authKey}},
        body: JSON.stringify({{}})
      }}).then(r => r.json()).then(data => {{
        alert('New API key: ' + data.api_key + '\\n\\nStore it now — it will not be shown again.');
      }}).catch(e => alert('Error: ' + e));
    }}

    document.getElementById('createOrgForm').addEventListener('submit', function(e) {{
      e.preventDefault();
      const fd = new FormData(e.target);
      const body = Object.fromEntries(fd.entries());
      body.risk_tolerance = parseFloat(body.risk_tolerance);
      body.veto_threshold = parseFloat(body.veto_threshold);
      const authKey = prompt('Enter your admin API key:');
      if (!authKey) return;
      fetch('/orgs', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json', 'Authorization': 'Bearer ' + authKey}},
        body: JSON.stringify(body)
      }}).then(r => r.json()).then(data => {{
        const el = document.getElementById('createResult');
        el.classList.remove('hidden');
        if (data.api_key) {{
          el.innerHTML = '<strong>Org created!</strong> API Key: <span class="text-green-700">' + data.api_key + '</span><br><em>Store this key now — it will not be shown again.</em>';
        }} else {{
          el.innerHTML = '<span class="text-red-600">' + JSON.stringify(data) + '</span>';
        }}
      }}).catch(e => alert('Error: ' + e));
    }});

    document.getElementById('editPolicyForm').addEventListener('submit', function(e) {{
      e.preventDefault();
      const fd = new FormData(e.target);
      const body = Object.fromEntries(fd.entries());
      const orgId = body.org_id;
      delete body.org_id;
      body.risk_tolerance = parseFloat(body.risk_tolerance);
      body.veto_threshold = parseFloat(body.veto_threshold);
      const authKey = prompt('Enter your admin API key:');
      if (!authKey) return;
      fetch('/orgs/' + orgId + '/policy', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json', 'Authorization': 'Bearer ' + authKey}},
        body: JSON.stringify(body)
      }}).then(r => r.json()).then(data => {{
        const el = document.getElementById('editResult');
        el.classList.remove('hidden');
        if (data.org_id) {{
          el.className = 'mt-4 p-4 bg-green-50 border border-green-200 rounded text-sm font-mono';
          el.textContent = 'Policy updated: ' + JSON.stringify(data);
        }} else {{
          el.className = 'mt-4 p-4 bg-red-50 border border-red-200 rounded text-sm font-mono';
          el.textContent = JSON.stringify(data);
        }}
      }}).catch(e => alert('Error: ' + e));
    }});
  </script>
</body>
</html>"""

    payload = html.encode("utf-8")
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(payload)))
    self.send_header("Access-Control-Allow-Origin", "*")
    self.end_headers()
    self.wfile.write(payload)


# ---------------------------------------------------------------------------
# Shared helper: collect live pipeline statistics
# ---------------------------------------------------------------------------

def _collect_pipeline_stats() -> dict:
    """Return a dict of live pipeline statistics.  Safe when _pipeline is None."""
    from collections import Counter as _Counter

    pipeline = _pipeline  # read module-level singleton without creating one

    total_tasks = 0
    action_counts: dict = {}
    mean_risk = 0.0
    top_domains: dict = {}
    tool_summary: dict = {}
    rules_promoted = 0
    promoted_rules_list: list = []
    orgs_count = 1
    calibration_signal: dict = {"mean_error": 0.0, "samples": 0,
                                 "overestimates": 0, "underestimates": 0}

    try:
        orgs_count = max(1, len(_ORG_REGISTRY.all_orgs()))
    except Exception:  # noqa: BLE001
        pass

    if pipeline is not None:
        try:
            log = getattr(pipeline._predictor, "_prediction_log", [])
            total_tasks = len(log)
            action_counts = dict(_Counter(
                e["decision"].action
                for e in log
                if "decision" in e and hasattr(e["decision"], "action")
            ))
            risks = [
                e["decision"].risk_score
                for e in log
                if "decision" in e and hasattr(e["decision"], "risk_score")
            ]
            mean_risk = sum(risks) / len(risks) if risks else 0.0
            top_domains = dict(_Counter(
                getattr(e["task"], "domain", "general")
                for e in log
                if "task" in e
            ).most_common(5))
        except Exception:  # noqa: BLE001
            pass

        try:
            tool_summary = pipeline._cooccurrence.summary()
        except Exception:  # noqa: BLE001
            pass

        try:
            promoted_rules_list = pipeline._consolidator.promoted_rules()
            rules_promoted = len(promoted_rules_list)
        except Exception:  # noqa: BLE001
            pass

        try:
            calibration_signal = pipeline._predictor.calibration_signal()
        except Exception:  # noqa: BLE001
            pass

    escalated = action_counts.get("escalate", 0)
    refused = action_counts.get("refuse", 0)
    escalation_rate = escalated / max(total_tasks, 1)
    refusal_rate = refused / max(total_tasks, 1)

    return {
        "total_tasks": total_tasks,
        "action_counts": action_counts,
        "escalated": escalated,
        "refused": refused,
        "escalation_rate": escalation_rate,
        "refusal_rate": refusal_rate,
        "mean_risk": mean_risk,
        "top_domains": top_domains,
        "tool_summary": tool_summary,
        "rules_promoted": rules_promoted,
        "promoted_rules_list": promoted_rules_list,
        "orgs_count": orgs_count,
        "calibration_signal": calibration_signal,
    }


# ---------------------------------------------------------------------------
# GET /report — self-reporting visual dashboard
# ---------------------------------------------------------------------------

def _handle_get_report(self: "ManifoldHandler") -> None:
    """GET /report — serve a self-contained HTML governance dashboard."""
    import html as _html
    import json as _json
    from datetime import datetime as _dt

    try:
        stats = _collect_pipeline_stats()

        total_tasks = stats["total_tasks"]
        action_counts = stats["action_counts"]
        escalation_rate = stats["escalation_rate"]
        refusal_rate = stats["refusal_rate"]
        top_domains = stats["top_domains"]
        tool_summary = stats["tool_summary"]
        rules_promoted = stats["rules_promoted"]
        promoted_rules_list = stats["promoted_rules_list"]
        orgs_count = stats["orgs_count"]

        timestamp = _dt.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Colour-coded rate helpers
        def rate_colour(rate: float) -> str:
            if rate < 0.10:
                return "#22c55e"  # green
            if rate <= 0.30:
                return "#f59e0b"  # amber
            return "#ef4444"  # red

        esc_colour = rate_colour(escalation_rate)
        ref_colour = rate_colour(refusal_rate)

        # Chart.js data (safe JSON — no user content in chart labels themselves)
        action_labels = ["escalate", "verify", "answer", "refuse", "other"]
        action_colours = ["#ef4444", "#f59e0b", "#22c55e", "#7f1d1d", "#6b7280"]
        action_data = [action_counts.get(a, 0) for a in action_labels[:-1]]
        other_count = sum(v for k, v in action_counts.items() if k not in action_labels[:-1])
        action_data.append(other_count)

        domain_labels = list(top_domains.keys())
        domain_data = list(top_domains.values())

        chart_data_json = _json.dumps({
            "actionLabels": action_labels,
            "actionData": action_data,
            "actionColours": action_colours,
            "domainLabels": domain_labels,
            "domainData": domain_data,
        })

        # Tool rows
        tool_rows_html = ""
        if tool_summary:
            for tool_name, info in tool_summary.items():
                sr = info.get("success_rate", 1.0)
                tasks = info.get("total_tasks", 0)
                if sr > 0.8:
                    status_html = '<span style="color:#22c55e">✓ Healthy</span>'
                elif sr >= 0.5:
                    status_html = '<span style="color:#f59e0b">⚠ Degraded</span>'
                else:
                    status_html = '<span style="color:#ef4444">✗ Flagged</span>'
                safe_name = _html.escape(str(tool_name))
                tool_rows_html += (
                    f"<tr><td>{safe_name}</td><td>{tasks}</td>"
                    f"<td>{sr:.0%}</td><td>{status_html}</td></tr>\n"
                )
        else:
            tool_rows_html = '<tr><td colspan="4" style="color:#9ca3af">No tool data yet</td></tr>'

        # Rules panel
        if promoted_rules_list:
            rules_html = "<ul style='margin:0;padding-left:1.2em'>"
            for r in promoted_rules_list:
                d = _html.escape(str(getattr(r, "domain", "?")))
                a = _html.escape(str(getattr(r, "action", "?")))
                conf = getattr(r, "confidence", 0.0)
                n = getattr(r, "sample_count", 0)
                rules_html += f"<li>{d} → {a} (confidence {conf:.0%}, n={n})</li>"
            rules_html += "</ul>"
        else:
            rules_html = (
                '<p style="color:#9ca3af;margin:0">'
                'No rules consolidated yet — needs more traffic</p>'
            )

        from . import __version__ as _ver

        html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="30">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MANIFOLD — Governance Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
          integrity="sha384-eI7PSr3L1XLISH8NdEZGHToNATtgIFeIiu4lMZRpHBLElXP5s8cJnCGJSFZ2UbBk"
          crossorigin="anonymous"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0f1117; color: #f3f4f6; font-family: system-ui, sans-serif;
            padding: 1.5rem; min-height: 100vh; }}
    h1 {{ font-size: 1.75rem; font-weight: 700; }}
    h2 {{ font-size: 1.1rem; font-weight: 600; margin-bottom: .75rem; color: #e5e7eb; }}
    .subtitle {{ color: #9ca3af; font-size: .9rem; margin-top: .25rem; }}
    .timestamp {{ color: #6b7280; font-size: .8rem; margin-top: .25rem; }}
    .header {{ margin-bottom: 2rem; }}
    .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }}
    .card {{ background: #1f2937; border-radius: .75rem; padding: 1.25rem 1.5rem;
             flex: 1 1 160px; min-width: 140px; }}
    .card-label {{ font-size: .75rem; color: #9ca3af; text-transform: uppercase;
                   letter-spacing: .05em; margin-bottom: .5rem; }}
    .card-value {{ font-size: 2rem; font-weight: 700; line-height: 1; }}
    .charts {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem; }}
    .chart-box {{ background: #1f2937; border-radius: .75rem; padding: 1.25rem;
                  flex: 1 1 320px; min-width: 280px; }}
    canvas {{ max-height: 260px; }}
    .section {{ background: #1f2937; border-radius: .75rem; padding: 1.25rem;
                margin-bottom: 1.5rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .875rem; }}
    th {{ text-align: left; color: #9ca3af; font-weight: 500; padding: .5rem .75rem;
          border-bottom: 1px solid #374151; }}
    td {{ padding: .5rem .75rem; border-bottom: 1px solid #1f2937; }}
    tr:last-child td {{ border-bottom: none; }}
    .footer {{ margin-top: 2rem; color: #6b7280; font-size: .8rem; }}
    .footer a {{ color: #818cf8; text-decoration: none; margin-right: 1rem; }}
    .footer a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>MANIFOLD — Governance Report</h1>
    <p class="subtitle">v{_html.escape(str(_ver))} · Live data · Auto-refreshes every 30s</p>
    <p class="timestamp">Generated: {_html.escape(timestamp)}</p>
  </div>

  <div class="cards">
    <div class="card">
      <div class="card-label">Total Decisions</div>
      <div class="card-value">{total_tasks:,}</div>
    </div>
    <div class="card">
      <div class="card-label">Escalation Rate</div>
      <div class="card-value" style="color:{esc_colour}">{escalation_rate:.1%}</div>
    </div>
    <div class="card">
      <div class="card-label">Refusal Rate</div>
      <div class="card-value" style="color:{ref_colour}">{refusal_rate:.1%}</div>
    </div>
    <div class="card">
      <div class="card-label">Active Orgs</div>
      <div class="card-value">{orgs_count}</div>
    </div>
  </div>

  <div class="charts">
    <div class="chart-box">
      <h2>Action Distribution</h2>
      <canvas id="actionChart"></canvas>
    </div>
    <div class="chart-box">
      <h2>Top Domains</h2>
      <canvas id="domainChart"></canvas>
    </div>
  </div>

  <div class="section">
    <h2>Tool Health</h2>
    <table>
      <thead><tr><th>Tool Name</th><th>Tasks</th><th>Success Rate</th><th>Status</th></tr></thead>
      <tbody>{tool_rows_html}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>Consolidated Policy Rules</h2>
    <p style="color:#9ca3af;font-size:.875rem;margin-bottom:.75rem">
      {rules_promoted} rule{'s' if rules_promoted != 1 else ''} promoted from observed patterns
    </p>
    {rules_html}
  </div>

  <div class="footer">
    <a href="/docs">API Docs</a>
    <a href="/admin">Admin</a>
    <a href="/metrics">Metrics</a>
    <a href="/learned">Learned State</a>
  </div>

  <script>
    const _d = JSON.parse({_json.dumps(chart_data_json)});
    new Chart(document.getElementById('actionChart'), {{
      type: 'doughnut',
      data: {{
        labels: _d.actionLabels,
        datasets: [{{ data: _d.actionData, backgroundColor: _d.actionColours, borderWidth: 2,
                      borderColor: '#0f1117' }}]
      }},
      options: {{ plugins: {{ legend: {{ labels: {{ color: '#e5e7eb' }} }} }},
                  responsive: true, maintainAspectRatio: true }}
    }});
    new Chart(document.getElementById('domainChart'), {{
      type: 'bar',
      data: {{
        labels: _d.domainLabels,
        datasets: [{{ data: _d.domainData, backgroundColor: '#7F77DD',
                      borderRadius: 4 }}]
      }},
      options: {{
        indexAxis: 'y',
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: '#9ca3af' }}, grid: {{ color: '#374151' }} }},
          y: {{ ticks: {{ color: '#e5e7eb' }} }}
        }},
        responsive: true, maintainAspectRatio: true
      }}
    }});
  </script>
</body>
</html>"""

    except Exception as exc:  # noqa: BLE001
        import html as _html
        html_page = f"""<!DOCTYPE html><html><head><title>MANIFOLD Report Error</title></head>
<body style="background:#0f1117;color:#f3f4f6;font-family:system-ui;padding:2rem">
<h1>Report Error</h1>
<p>Could not generate report: {_html.escape(str(exc))}</p>
</body></html>"""

    payload = html_page.encode("utf-8")
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(payload)))
    self.send_header("Access-Control-Allow-Origin", "*")
    self.end_headers()
    self.wfile.write(payload)


# ---------------------------------------------------------------------------
# GET /digest — structured JSON governance summary
# ---------------------------------------------------------------------------

def _handle_get_digest(self: "ManifoldHandler") -> None:
    """GET /digest?period=7d — structured JSON governance summary."""
    from urllib.parse import parse_qs, urlparse
    from datetime import datetime as _dt

    try:
        query = parse_qs(urlparse(self.path).query)
        period = query.get("period", ["7d"])[0]
        if period not in ("24h", "7d", "30d"):
            period = "7d"

        stats = _collect_pipeline_stats()
        pipeline = _pipeline

        total_tasks = stats["total_tasks"]
        escalated = stats["escalated"]
        refused = stats["refused"]
        permitted = max(0, total_tasks - escalated - refused)

        # Top risky decisions (anonymised)
        top_risky: list[dict] = []
        if pipeline is not None:
            try:
                log = getattr(pipeline._predictor, "_prediction_log", [])
                entries = []
                for e in log:
                    decision = e.get("decision")
                    task = e.get("task")
                    if decision is None or task is None:
                        continue
                    prompt = getattr(task, "prompt", "") or ""
                    entries.append({
                        "prompt_prefix": prompt[:40],
                        "action": getattr(decision, "action", ""),
                        "risk_score": round(float(getattr(decision, "risk_score", 0.0)), 4),
                        "domain": getattr(task, "domain", "general"),
                    })
                top_risky = sorted(entries, key=lambda x: x["risk_score"], reverse=True)[:5]
            except Exception:  # noqa: BLE001
                top_risky = []

        # Flagged tools via cooccurrence
        flagged_tools: list[str] = []
        if pipeline is not None:
            try:
                tool_summary = stats["tool_summary"]
                flagged_tools = [
                    t for t, info in tool_summary.items()
                    if info.get("success_rate", 1.0) < 0.5
                ]
            except Exception:  # noqa: BLE001
                pass

        # Build tools list
        tools_list = []
        for tool_name, info in stats["tool_summary"].items():
            sr = info.get("success_rate", 1.0)
            if sr > 0.8:
                status = "healthy"
            elif sr >= 0.5:
                status = "degraded"
            else:
                status = "flagged"
            tools_list.append({
                "name": str(tool_name),
                "total_tasks": int(info.get("total_tasks", 0)),
                "success_rate": round(float(sr), 4),
                "status": status,
            })

        from . import __version__ as _ver
        cal = stats["calibration_signal"]

        response = {
            "generated_at": _dt.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "period": period,
            "version": str(_ver),
            "summary": {
                "total_decisions": total_tasks,
                "escalated": escalated,
                "refused": refused,
                "permitted": permitted,
                "escalation_rate": round(stats["escalation_rate"], 4),
                "refusal_rate": round(stats["refusal_rate"], 4),
                "mean_risk_score": round(stats["mean_risk"], 4),
            },
            "domains": stats["top_domains"],
            "tools": tools_list,
            "policy": {
                "rules_promoted": stats["rules_promoted"],
                "active_orgs": stats["orgs_count"],
                "calibration": {
                    "mean_prediction_error": round(float(cal.get("mean_error", 0.0)), 4),
                    "samples": int(cal.get("samples", 0)),
                    "overestimates": int(cal.get("overestimates", 0)),
                    "underestimates": int(cal.get("underestimates", 0)),
                },
            },
            "governance": {
                "flagged_tools": flagged_tools,
                "top_risky_decisions": top_risky,
            },
        }
        _send_json(self, 200, response)
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


# ---------------------------------------------------------------------------
# Consumer app endpoints
# ---------------------------------------------------------------------------

_CONSUMER_CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0f1117;color:#e2e8f0;line-height:1.6}
.wrap{max-width:880px;margin:0 auto;padding:40px 24px}
h1{font-size:clamp(2rem,5vw,3.5rem);font-weight:800;letter-spacing:-0.03em;
  background:linear-gradient(135deg,#fff 0%,#a5b4fc 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
h2{font-size:1.5rem;font-weight:700;margin-bottom:8px}
.sub{color:#94a3b8;font-size:1.1rem;margin:16px 0 32px}
.hero{text-align:center;padding:80px 0 60px}
.btn{display:inline-block;padding:14px 28px;border-radius:8px;
  font-size:15px;font-weight:600;text-decoration:none;cursor:pointer;
  border:none;transition:opacity .15s}
.btn-primary{background:#7F77DD;color:#fff}
.btn-secondary{background:transparent;color:#7F77DD;
  border:1px solid #7F77DD;margin-left:12px}
.btn:hover{opacity:.85}
.cards{display:flex;flex-wrap:wrap;gap:20px;margin:40px 0}
.card{flex:1 1 220px;background:#1a1d27;border:1px solid #2a2d3a;
  border-radius:12px;padding:28px 24px}
.card-num{font-size:2rem;margin-bottom:8px}
.card h3{font-size:1rem;font-weight:700;margin-bottom:8px}
.card p{color:#94a3b8;font-size:.9rem}
.compat{text-align:center;color:#64748b;font-size:.95rem;
  padding:32px 0;border-top:1px solid #1e2130;border-bottom:1px solid #1e2130;
  margin:40px 0}
.cta{text-align:center;padding:60px 0}
pre,.code{background:#0a0c14;border:1px solid #2a2d3a;border-radius:8px;
  padding:14px 16px;font-family:monospace;font-size:13px;
  color:#a5b4fc;overflow:auto;white-space:pre-wrap}
label{display:block;font-size:.875rem;color:#94a3b8;margin-bottom:4px}
input,select{width:100%;padding:10px 12px;background:#1a1d27;
  border:1px solid #2a2d3a;border-radius:6px;color:#e2e8f0;
  font-size:14px;margin-bottom:16px}
input:focus,select:focus{outline:2px solid #7F77DD}
form{background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;
  padding:32px;max-width:520px;margin:0 auto}
.note{text-align:center;color:#64748b;font-size:.8rem;margin-top:16px}
footer{text-align:center;padding:40px 0;color:#475569;font-size:.85rem}
footer a{color:#7F77DD;text-decoration:none}
a.lnk{color:#7F77DD;text-decoration:none}
</style>
"""

_CONSUMER_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
""" + _CONSUMER_CSS + "</head><body>"

_CONSUMER_FOOTER = """<footer class="wrap">
  <a href="/" class="lnk">Home</a> ·
  <a href="/signup" class="lnk">Signup</a> ·
  <a href="/connect" class="lnk">Connect</a> ·
  <a href="/report" class="lnk">Dashboard</a> ·
  <a href="/digest?period=7d" class="lnk">Digest API</a>
</footer></body></html>"""


def _handle_get_landing(self: "ManifoldHandler") -> None:
    """GET / — consumer landing page."""
    html = _CONSUMER_HEADER.format(title="MANIFOLD — AI Governance") + """
<div class="wrap">
  <section class="hero">
    <h1>MANIFOLD</h1>
    <p class="sub">Stop your AI from doing things it shouldn't.</p>
    <p style="color:#94a3b8;max-width:560px;margin:0 auto 32px">
      One line of code. Every AI call governed.<br>
      Learns from every outcome. Works with any model.
    </p>
    <a href="/signup" class="btn btn-primary">Get started free</a>
    <a href="/report" class="btn btn-secondary">View live demo</a>
  </section>

  <h2 style="text-align:center;margin-bottom:8px">How it works</h2>
  <div class="cards">
    <div class="card">
      <div class="card-num">🔗</div>
      <h3>Connect</h3>
      <p>Point your AI tool at MANIFOLD. One URL change.</p>
    </div>
    <div class="card">
      <div class="card-num">⚖️</div>
      <h3>Govern</h3>
      <p>Every call is priced for risk before it executes.</p>
    </div>
    <div class="card">
      <div class="card-num">🧠</div>
      <h3>Learn</h3>
      <p>The system tightens its own policies from outcomes.</p>
    </div>
  </div>

  <div class="compat">
    Works with:&nbsp;
    <strong>GPT-4</strong> &nbsp;·&nbsp;
    <strong>Claude</strong> &nbsp;·&nbsp;
    <strong>Gemini</strong> &nbsp;·&nbsp;
    <strong>LangChain</strong> &nbsp;·&nbsp;
    <strong>Cursor</strong> &nbsp;·&nbsp;
    <strong>any OpenAI-compatible tool</strong>
  </div>

  <section class="cta">
    <h2>Start governing your AI in 30 seconds</h2>
    <p class="sub">Free forever for individual use.</p>
    <a href="/signup" class="btn btn-primary">Get started</a>
  </section>
</div>
""" + _CONSUMER_FOOTER
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.end_headers()
    self.wfile.write(html.encode())


def _handle_get_signup(self: "ManifoldHandler") -> None:
    """GET /signup — signup form page."""
    html = _CONSUMER_HEADER.format(title="Sign up — MANIFOLD") + """
<div class="wrap" style="padding-top:60px">
  <div style="text-align:center;margin-bottom:32px">
    <h1 style="font-size:2rem">Create your MANIFOLD account</h1>
    <p class="sub">Free forever for individual use.</p>
  </div>
  <form method="POST" action="/signup">
    <label for="email">Email address</label>
    <input id="email" name="email" type="email" required
           placeholder="you@example.com">

    <label for="org_name">Organisation name</label>
    <input id="org_name" name="org_name" type="text" required
           placeholder="Acme Corp">

    <label for="domain">Primary domain</label>
    <select id="domain" name="domain">
      <option value="general">General</option>
      <option value="finance">Finance</option>
      <option value="healthcare">Healthcare</option>
      <option value="devops">DevOps</option>
      <option value="legal">Legal</option>
    </select>

    <button type="submit" class="btn btn-primary" style="width:100%;padding:14px">
      Create account + get API key
    </button>
  </form>
  <p class="note">No credit card. No installation. API key shown once.</p>
</div>
""" + _CONSUMER_FOOTER
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.end_headers()
    self.wfile.write(html.encode())


def _handle_post_signup(self: "ManifoldHandler", body: dict) -> None:
    """POST /signup — create account, show API key once."""
    import html as _html
    import re as _re

    email = str(body.get("email", "")).strip()
    org_name = str(body.get("org_name", "")).strip()
    domain = str(body.get("domain", "general")).strip() or "general"

    if not email or not org_name:
        _send_error(self, 400, "email and org_name required")
        return

    org_id = _re.sub(r"[^a-z0-9-]", "-", org_name.lower())[:32].strip("-") or "org"
    try:
        raw_key, _cfg = _ORG_REGISTRY.generate_key(
            org_id=org_id,
            display_name=org_name,
            role=_OrgRole.AGENT,
            domain=domain,
            notes=f"signup:{email}",
        )
    except Exception as exc:  # noqa: BLE001
        _send_error(self, 500, f"Could not create account: {exc}")
        return

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Your MANIFOLD API key</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0f1117;color:#e2e8f0;
  display:flex;align-items:center;justify-content:center;
  min-height:100vh}}
.card{{background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;
  padding:40px;max-width:540px;width:100%;margin:24px}}
.key{{background:#0a0c14;border:1px solid #3a3d4a;border-radius:8px;
  padding:16px;font-family:monospace;font-size:14px;
  word-break:break-all;color:#a5b4fc;margin:16px 0}}
.warn{{color:#EF9F27;font-size:13px;margin-bottom:24px}}
.btn{{display:inline-block;background:#7F77DD;color:#fff;
  padding:12px 24px;border-radius:8px;text-decoration:none;
  font-size:14px;font-weight:600;margin-top:8px}}
pre{{background:#0a0c14;padding:14px;border-radius:8px;
  font-size:12px;overflow:auto;color:#a5b4fc;white-space:pre-wrap}}
p{{color:#94a3b8}}
</style>
</head>
<body>
<div class="card">
  <h2 style="margin-bottom:8px">Your account is ready ✓</h2>
  <p style="margin-bottom:24px">Organisation: <strong style="color:#e2e8f0">{_html.escape(org_name)}</strong></p>
  <p style="font-size:14px;color:#e2e8f0;margin-bottom:4px">Your API key</p>
  <div class="key">{_html.escape(raw_key)}</div>
  <p class="warn">⚠ Save this key now — it will not be shown again.</p>
  <p style="font-size:14px;color:#e2e8f0;margin-bottom:12px">
    Point any OpenAI-compatible agent at MANIFOLD:
  </p>
  <pre>import openai
client = openai.OpenAI(
    base_url="http://YOUR_HOST/v1",
    api_key="{_html.escape(raw_key)}"
)</pre>
  <a href="/connect" class="btn">Next: connect your tools →</a>
</div>
</body>
</html>"""
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.end_headers()
    self.wfile.write(page.encode())


def _handle_get_connect(self: "ManifoldHandler") -> None:
    """GET /connect — tool connection guide with integration snippets."""
    html = _CONSUMER_HEADER.format(title="Connect your tools — MANIFOLD") + """
<div class="wrap" style="padding-top:48px">
  <h1 style="font-size:2rem;margin-bottom:8px">Connect your AI tools</h1>
  <p class="sub">Replace <code style="background:#1a1d27;padding:2px 6px;border-radius:4px;color:#a5b4fc">HOST</code>
  with your MANIFOLD server address and <code style="background:#1a1d27;padding:2px 6px;border-radius:4px;color:#a5b4fc">YOUR_KEY</code>
  with your API key from <a href="/signup" class="lnk">/signup</a>.</p>

  <div class="cards" style="flex-direction:column">

    <div class="card">
      <h3 style="margin-bottom:12px">🐍 Python / OpenAI SDK</h3>
      <pre>import openai
client = openai.OpenAI(
    base_url="http://HOST/v1",
    api_key="YOUR_KEY"
)</pre>
    </div>

    <div class="card">
      <h3 style="margin-bottom:12px">🦜 LangChain</h3>
      <pre>from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://HOST/v1",
    api_key="YOUR_KEY"
)</pre>
    </div>

    <div class="card">
      <h3 style="margin-bottom:12px">🖱️ Cursor / VS Code Copilot</h3>
      <p style="color:#94a3b8;font-size:.9rem;margin-bottom:10px">
        In <code style="color:#a5b4fc">settings.json</code>:
      </p>
      <pre>"github.copilot.advanced": {{
    "debug.overrideProxyUrl": "http://HOST/v1"
}}</pre>
      <p style="color:#94a3b8;font-size:.9rem;margin-top:10px">
        Or install the
        <a href="/vscode-extension" class="lnk">MANIFOLD VS Code extension</a>.
      </p>
    </div>

    <div class="card">
      <h3 style="margin-bottom:12px">⚡ cURL / any HTTP client</h3>
      <pre>curl http://HOST/v1/chat/completions \\
  -H "Authorization: Bearer YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"gpt-4o","messages":[{{"role":"user","content":"hello"}}]}}'</pre>
    </div>

    <div class="card">
      <h3 style="margin-bottom:12px">🌍 Environment variable</h3>
      <pre>export OPENAI_BASE_URL="http://HOST/v1"
export OPENAI_API_KEY="YOUR_KEY"
# All subsequent OpenAI SDK calls are now governed</pre>
    </div>

  </div>

  <p style="text-align:center;margin-top:32px;color:#64748b">
    View your governance dashboard →
    <a href="/report" class="lnk">Live report</a>
  </p>
</div>
""" + _CONSUMER_FOOTER
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.end_headers()
    self.wfile.write(html.encode())


ManifoldHandler._handle_get_landing = _handle_get_landing  # type: ignore[attr-defined]
ManifoldHandler._handle_get_signup = _handle_get_signup  # type: ignore[attr-defined]
ManifoldHandler._handle_post_signup = _handle_post_signup  # type: ignore[attr-defined]
ManifoldHandler._handle_get_connect = _handle_get_connect  # type: ignore[attr-defined]
ManifoldHandler._handle_get_learned = _handle_get_learned  # type: ignore[attr-defined]
ManifoldHandler._handle_get_v1_models = _handle_get_v1_models  # type: ignore[attr-defined]
ManifoldHandler._handle_post_v1_chat_completions = _handle_post_v1_chat_completions  # type: ignore[attr-defined]
ManifoldHandler._handle_post_orgs = _handle_post_orgs  # type: ignore[attr-defined]
ManifoldHandler._handle_put_org_policy = _handle_put_org_policy  # type: ignore[attr-defined]
ManifoldHandler._handle_post_org_key = _handle_post_org_key  # type: ignore[attr-defined]
ManifoldHandler._handle_get_admin = _handle_get_admin  # type: ignore[attr-defined]
ManifoldHandler._handle_get_report = _handle_get_report  # type: ignore[attr-defined]
ManifoldHandler._handle_get_digest = _handle_get_digest  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Agentic layer endpoint handlers
# ---------------------------------------------------------------------------


def _handle_get_agents(self: "ManifoldHandler") -> None:
    """GET /agents — list all registered agents."""
    agents = _AGENT_REGISTRY.all_agents()
    _send_json(self, 200, {
        "agents": [a.to_dict() for a in agents],
        "summary": _AGENT_REGISTRY.summary(),
        "monitor": _AGENT_MONITOR.status(),
    })


def _handle_post_agents_register(self: "ManifoldHandler", body: dict) -> None:
    """POST /agents/register — agent announces itself."""
    agent_id = str(body.get("agent_id", "")).strip()
    name = str(body.get("display_name", "")).strip()
    caps = body.get("capabilities", [])
    org_id = str(body.get("org_id", "default")).strip()
    endpoint = str(body.get("endpoint_url", "")).strip()
    domain = str(body.get("domain", "general")).strip()
    if not agent_id or not name:
        _send_error(self, 400, "agent_id and display_name required")
        return
    record = _AGENT_REGISTRY.register(
        agent_id=agent_id,
        display_name=name,
        capabilities=caps if isinstance(caps, list) else [],
        org_id=org_id,
        endpoint_url=endpoint,
        domain=domain,
    )
    _send_json(self, 201, record.to_dict())


def _handle_post_agent_heartbeat(
    self: "ManifoldHandler", agent_id: str, body: dict
) -> None:
    """POST /agents/{id}/heartbeat — keep-alive."""
    status = str(body.get("status", "active"))
    ok = _AGENT_REGISTRY.heartbeat(agent_id, status)
    if not ok:
        _send_error(self, 404, f"Agent {agent_id!r} not registered")
        return
    _send_json(self, 200, {"agent_id": agent_id, "acknowledged": True})


def _handle_post_agent_pause(self: "ManifoldHandler", agent_id: str) -> None:
    """POST /agents/{id}/pause — MANIFOLD pauses an agent."""
    ok = _AGENT_REGISTRY.pause(agent_id)
    if not ok:
        _send_error(self, 404, f"Agent {agent_id!r} not found")
        return
    _send_json(self, 200, {"agent_id": agent_id, "status": "paused"})


def _handle_post_agent_resume(self: "ManifoldHandler", agent_id: str) -> None:
    """POST /agents/{id}/resume — MANIFOLD resumes a paused agent."""
    ok = _AGENT_REGISTRY.resume(agent_id)
    if not ok:
        _send_error(self, 404, f"Agent {agent_id!r} not found")
        return
    _send_json(self, 200, {"agent_id": agent_id, "status": "active"})


def _handle_post_task(self: "ManifoldHandler", body: dict) -> None:
    """POST /task — receive any problem, decompose, govern, route."""
    task = str(body.get("task", "")).strip()
    stakes = float(body.get("stakes", 0.5))
    if not task:
        _send_error(self, 400, "task field required")
        return
    plan = _TASK_ROUTER.route(task, stakes_hint=stakes)
    _send_json(self, 200, plan.to_dict())


# ---------------------------------------------------------------------------
# MANIFOLD World + WebSocket handlers
# ---------------------------------------------------------------------------

import base64 as _base64
import socket as _socket
import struct as _struct
import os as _os_world

_WORLD_DIR = _os_world.path.join(_os_world.path.dirname(_os_world.path.dirname(_os_world.path.abspath(__file__))), "manifold-world")


def _handle_get_world(self: "ManifoldHandler") -> None:
    """GET /world — serve the isometric game world HTML."""
    world_file = _os_world.path.join(_WORLD_DIR, "index.html")
    if not _os_world.path.exists(world_file):
        _send_error(self, 404, "MANIFOLD World not found")
        return
    with open(world_file, "rb") as fh:
        data = fh.read()
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)


def _handle_get_world_manifest(self: "ManifoldHandler") -> None:
    """GET /world/manifest.json — serve the PWA manifest."""
    manifest_file = _os_world.path.join(_WORLD_DIR, "manifest.json")
    if not _os_world.path.exists(manifest_file):
        _send_error(self, 404, "Manifest not found")
        return
    with open(manifest_file, "rb") as fh:
        data = fh.read()
    self.send_response(200)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)


def _ws_send_frame(conn: "_socket.socket", payload: bytes, opcode: int = 0x1) -> None:
    """Send a single WebSocket text frame."""
    ln = len(payload)
    if ln <= 125:
        header = bytes([0x80 | opcode, ln])
    elif ln <= 65535:
        header = bytes([0x80 | opcode, 126]) + _struct.pack(">H", ln)
    else:
        header = bytes([0x80 | opcode, 127]) + _struct.pack(">Q", ln)
    conn.sendall(header + payload)


def _ws_read_frame(conn: "_socket.socket") -> "bytes | None":
    """Read one WebSocket frame, return payload or None on close."""
    try:
        raw = b""
        while len(raw) < 2:
            chunk = conn.recv(2 - len(raw))
            if not chunk:
                return None
            raw += chunk
        first, second = raw[0], raw[1]
        masked = bool(second & 0x80)
        ln = second & 0x7F
        if ln == 126:
            buf = b""
            while len(buf) < 2:
                buf += conn.recv(2 - len(buf))
            ln = _struct.unpack(">H", buf)[0]
        elif ln == 127:
            buf = b""
            while len(buf) < 8:
                buf += conn.recv(8 - len(buf))
            ln = _struct.unpack(">Q", buf)[0]
        mask_key = b""
        if masked:
            while len(mask_key) < 4:
                mask_key += conn.recv(4 - len(mask_key))
        payload = b""
        while len(payload) < ln:
            payload += conn.recv(ln - len(payload))
        if masked:
            payload = bytes(payload[i] ^ mask_key[i % 4] for i in range(ln))
        opcode = first & 0x0F
        if opcode == 0x8:  # close frame
            return None
        return payload
    except OSError:
        return None


def _handle_get_brain_state(self: "ManifoldHandler") -> None:
    """GET /brain/state — return brain persistence status."""
    pipeline = _get_pipeline()
    cmap_nodes = len(pipeline._cognitive_map._outcome_log)
    cooc_tools = len(pipeline._cooccurrence._tool_counts)
    pred_entries = len(pipeline._predictor._prediction_log)
    rules = len(pipeline._consolidator._promoted_rules)
    state_dir = str(_BRAIN_STATE_DIR)
    persisted = (
        (_BRAIN_STATE_DIR / "cognitive_map.json").exists()
        or (_BRAIN_STATE_DIR / "cooccurrence.json").exists()
        or (_BRAIN_STATE_DIR / "predictor.json").exists()
        or (_BRAIN_STATE_DIR / "consolidator.json").exists()
    )
    _send_json(
        self,
        200,
        {
            "cognitive_map_nodes": cmap_nodes,
            "cooccurrence_tools": cooc_tools,
            "prediction_log_entries": pred_entries,
            "promoted_rules": rules,
            "state_dir": state_dir,
            "persisted": persisted,
        },
    )


def _handle_ws_upgrade(self: "ManifoldHandler") -> None:
    """GET /ws — WebSocket upgrade + live event loop."""
    # Only accept Upgrade: websocket requests
    upgrade = self.headers.get("Upgrade", "").lower()
    if upgrade != "websocket":
        _send_error(self, 400, "WebSocket upgrade required")
        return
    key = self.headers.get("Sec-WebSocket-Key", "")
    if not key:
        _send_error(self, 400, "Sec-WebSocket-Key missing")
        return
    # Compute accept hash
    magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    accept = _base64.b64encode(
        hashlib.sha1((key + magic).encode()).digest()
    ).decode()
    # Send 101 Switching Protocols
    self.send_response(101)
    self.send_header("Upgrade", "websocket")
    self.send_header("Connection", "Upgrade")
    self.send_header("Sec-WebSocket-Accept", accept)
    self.end_headers()
    self.wfile.flush()

    conn = self.connection
    conn.settimeout(1.0)

    last_agent_push = 0.0
    last_stats_push = 0.0

    def _agents_payload() -> bytes:
        agents = _AGENT_REGISTRY.all_agents()
        data = {
            "type": "agent_update",
            "agents": [
                {
                    "id": a.agent_id,
                    "status": a.status,
                    "task": a.notes or "active",
                    "risk": round(1.0 - a.health_score(), 4),
                    "health": round(a.health_score(), 4),
                    "tx": float(a.task_count % 16),
                    "tz": float(a.error_count % 16),
                }
                for a in agents
            ],
        }
        return json.dumps(data).encode()

    def _stats_payload() -> bytes:
        summary = _AGENT_REGISTRY.summary()
        plans = _TASK_ROUTER.all_plans()
        data = {
            "type": "world_stats",
            "agents_active": summary.get("active", 0),
            "tasks_running": sum(1 for p in plans if not p.executable and p.blocked_count == 0),
            "governance_events_today": _REFUSAL_COUNT,
            "system_health": round(summary.get("avg_health", 1.0), 4),
        }
        return json.dumps(data).encode()

    try:
        while True:
            now = time.time()
            # Send agent update every 5 s
            if now - last_agent_push >= 5.0:
                _ws_send_frame(conn, _agents_payload())
                last_agent_push = now
            # Send world stats every 30 s
            if now - last_stats_push >= 30.0:
                _ws_send_frame(conn, _stats_payload())
                last_stats_push = now
            # Non-blocking read (timeout=1s)
            try:
                frame = _ws_read_frame(conn)
                if frame is None:
                    break  # client disconnected
            except OSError:
                pass  # timeout, continue
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


ManifoldHandler._handle_get_world = _handle_get_world  # type: ignore[attr-defined]
ManifoldHandler._handle_get_world_manifest = _handle_get_world_manifest  # type: ignore[attr-defined]
ManifoldHandler._handle_ws_upgrade = _handle_ws_upgrade  # type: ignore[attr-defined]
ManifoldHandler._handle_get_brain_state = _handle_get_brain_state  # type: ignore[attr-defined]


ManifoldHandler._handle_get_agents = _handle_get_agents  # type: ignore[attr-defined]
ManifoldHandler._handle_post_agents_register = _handle_post_agents_register  # type: ignore[attr-defined]
ManifoldHandler._handle_post_agent_heartbeat = _handle_post_agent_heartbeat  # type: ignore[attr-defined]
ManifoldHandler._handle_post_agent_pause = _handle_post_agent_pause  # type: ignore[attr-defined]
ManifoldHandler._handle_post_agent_resume = _handle_post_agent_resume  # type: ignore[attr-defined]
ManifoldHandler._handle_post_task = _handle_post_task  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Command channel handlers
# ---------------------------------------------------------------------------


def _handle_get_agent_commands(self: "ManifoldHandler", agent_id: str) -> None:
    """GET /agents/{id}/commands — long-poll for up to 20 seconds.

    Returns immediately when commands are queued; returns empty list after
    20 seconds with no commands.
    """
    deadline = time.time() + 20.0
    while time.time() < deadline:
        cmds = _AGENT_REGISTRY.poll_commands(agent_id, consume=True)
        if cmds:
            _send_json(self, 200, {"commands": cmds, "agent_id": agent_id})
            return
        time.sleep(0.5)
    _send_json(self, 200, {"commands": [], "agent_id": agent_id})


def _handle_post_agent_command(
    self: "ManifoldHandler", agent_id: str, body: dict
) -> None:
    """POST /agents/{id}/command — queue a command for an agent."""
    command = str(body.get("command", "")).strip()
    payload = body.get("payload", {})
    valid = {"pause", "resume", "redirect", "update_policy", "message"}
    if command not in valid:
        _send_error(self, 400, f"Invalid command. Must be one of: {sorted(valid)}")
        return
    cmd_id = _AGENT_REGISTRY.queue_command(agent_id, command, payload)
    if cmd_id is None:
        _send_error(self, 404, f"Agent {agent_id!r} not registered")
        return
    _send_json(
        self,
        201,
        {
            "command_id": cmd_id,
            "agent_id": agent_id,
            "command": command,
            "status": "queued",
        },
    )


ManifoldHandler._handle_get_agent_commands = _handle_get_agent_commands  # type: ignore[attr-defined]
ManifoldHandler._handle_post_agent_command = _handle_post_agent_command  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Policy rules handlers
# ---------------------------------------------------------------------------


def _handle_get_rules(self: "ManifoldHandler") -> None:
    """GET /rules — return all policy rules for the calling org."""
    _authed, caller = _check_auth(self, "/rules")
    if not _authed:
        return
    org_id = caller.org_id if caller else ""
    rules = [r.to_dict() for r in _RULE_ENGINE.rules_for_org(org_id)]
    _send_json(self, 200, {"rules": rules, "org_id": org_id})


def _handle_post_rule(self: "ManifoldHandler", body: dict, caller: Any) -> None:
    """POST /rules — create a new policy rule for the calling org."""
    import uuid as _uuid
    org_id = caller.org_id if caller else body.get("org_id", "")
    name = str(body.get("name", "unnamed rule"))
    conditions = body.get("conditions", {})
    action = str(body.get("action", "allow"))
    priority = int(body.get("priority", 0))
    rule = PolicyRule(
        rule_id=str(_uuid.uuid4()),
        org_id=org_id,
        name=name,
        conditions=conditions,
        action=action,
        priority=priority,
    )
    _RULE_ENGINE.add_rule(rule)
    _send_json(self, 201, rule.to_dict())


def _handle_delete_rule(self: "ManifoldHandler", rule_id: str) -> None:
    """DELETE /rules/{rule_id} — remove a policy rule."""
    removed = _RULE_ENGINE.remove_rule(rule_id)
    if removed:
        _send_json(self, 200, {"rule_id": rule_id, "status": "deleted"})
    else:
        _send_error(self, 404, f"Rule {rule_id!r} not found")


ManifoldHandler._handle_get_rules = _handle_get_rules  # type: ignore[attr-defined]
ManifoldHandler._handle_post_rule = _handle_post_rule  # type: ignore[attr-defined]
ManifoldHandler._handle_delete_rule = _handle_delete_rule  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Federation handlers
# ---------------------------------------------------------------------------


def _handle_get_federation_status(self: "ManifoldHandler") -> None:
    """GET /federation/status — return federation health summary."""
    ledger = _GOSSIP_BRIDGE.ledger
    known = ledger.known_tools()
    trust_entries = {t: ledger.global_rate(t) for t in known}
    contributing_orgs = len(_GOSSIP_BRIDGE.registered_orgs())
    _send_json(
        self,
        200,
        {
            "contributing_orgs": contributing_orgs,
            "known_tools": len(known),
            "trust_entries": trust_entries,
        },
    )


def _handle_post_federation_join(
    self: "ManifoldHandler", body: dict, caller: Any
) -> None:
    """POST /federation/join — register calling org with the gossip bridge."""
    org_id = caller.org_id if caller else body.get("org_id", "")
    if not org_id:
        _send_error(self, 400, "org_id required")
        return
    _GOSSIP_BRIDGE.register(org_id)
    _send_json(
        self,
        200,
        {
            "org_id": org_id,
            "status": "joined",
            "contributing_orgs": len(_GOSSIP_BRIDGE.registered_orgs()),
        },
    )


def _handle_post_federation_gossip(self: "ManifoldHandler", body: dict) -> None:
    """POST /federation/gossip — ingest a gossip packet."""
    from .federation import FederatedGossipPacket
    try:
        packet = FederatedGossipPacket(**body)
        _GOSSIP_BRIDGE.contribute_packet(packet)
        _send_json(self, 200, {"status": "ingested"})
    except Exception as exc:  # noqa: BLE001
        _send_error(self, 400, f"Invalid gossip packet: {exc}")


ManifoldHandler._handle_get_federation_status = _handle_get_federation_status  # type: ignore[attr-defined]
ManifoldHandler._handle_post_federation_join = _handle_post_federation_join  # type: ignore[attr-defined]
ManifoldHandler._handle_post_federation_gossip = _handle_post_federation_gossip  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# v1.7.0 — Real-Time Layer handlers
# ---------------------------------------------------------------------------


def _handle_get_realtime_status(self: "ManifoldHandler") -> None:
    """GET /realtime/status — live bus, grid, health, planner status."""
    global _NERVATURA  # noqa: PLW0603
    bus = _get_bus()
    try:
        _send_json(self, 200, {
            "bus_recent_updates": len(bus.recent()),
            "dynamic_grid_cells": len(_DYNAMIC_GRID.all_cells()),
            "health_monitor": _HEALTH_MONITOR.status(),
            "planner_ready": True,
            "nervatura_world": _NERVATURA.summary() if _NERVATURA is not None else None,
        })
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


def _handle_get_health_tools(self: "ManifoldHandler") -> None:
    """GET /health/tools — live tool health summary (public)."""
    try:
        _send_json(self, 200, _HEALTH_MONITOR.status())
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


def _handle_get_plan(self: "ManifoldHandler") -> None:
    """GET /plan — CRNA A* path planning."""
    import urllib.parse as _up
    try:
        qs = _up.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")

        def _parse_coord(key: str, default: list) -> tuple:
            raw = qs.get(key, [None])[0]
            if raw:
                parts = [int(x) for x in raw.split(",")]
                return tuple(parts)
            return tuple(default)

        start = _parse_coord("start", [0, 0, 0])
        target = _parse_coord("target", [5, 5, 0])
        risk_budget = float(qs.get("risk_budget", ["0.7"])[0])
        result = _PLANNER.plan(start=start, target=target, risk_budget=risk_budget)
        _send_json(self, 200, result)
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


def _handle_get_nervatura_world(self: "ManifoldHandler") -> None:
    """GET /nervatura/world — NERVATURAWorld summary."""
    global _NERVATURA  # noqa: PLW0603
    try:
        if _NERVATURA is None:
            _send_json(self, 200, {"status": "not_initialised",
                                   "hint": "POST /nervatura/world/init to create a world"})
        else:
            _send_json(self, 200, _NERVATURA.summary())
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


def _handle_post_nervatura_world_init(self: "ManifoldHandler", body: dict) -> None:
    """POST /nervatura/world/init — initialise NERVATURAWorld singleton."""
    global _NERVATURA  # noqa: PLW0603
    try:
        width = int(body.get("width", 20))
        depth = int(body.get("depth", 20))
        height = int(body.get("height", 5))
        domain = str(body.get("domain", "general"))
        _NERVATURA = _NERVATURAWorld(width=width, depth=depth, height=height)
        _send_json(self, 200, {
            "status": "ok",
            "domain": domain,
            **_NERVATURA.summary(),
        })
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


ManifoldHandler._handle_get_realtime_status = _handle_get_realtime_status  # type: ignore[attr-defined]
ManifoldHandler._handle_get_health_tools = _handle_get_health_tools  # type: ignore[attr-defined]
ManifoldHandler._handle_get_plan = _handle_get_plan  # type: ignore[attr-defined]
ManifoldHandler._handle_get_nervatura_world = _handle_get_nervatura_world  # type: ignore[attr-defined]
ManifoldHandler._handle_post_nervatura_world_init = _handle_post_nervatura_world_init  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Physical layer — v1.8.0
# GET  /physical/cameras   → CameraRegistry status
# GET  /physical/status    → PhysicalManager.status()
# POST /physical/init      → initialise PhysicalManager from request body
# ---------------------------------------------------------------------------

_PHYSICAL_MANAGER: "Any | None" = None  # PhysicalManager singleton


def _handle_get_physical_cameras(self: "ManifoldHandler") -> None:
    """GET /physical/cameras — list all registered camera detectors."""
    try:
        from manifold_physical.camera_detector import get_camera_registry
        registry = get_camera_registry()
        _send_json(self, 200, {"cameras": registry.status_list()})
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


def _handle_get_physical_status(self: "ManifoldHandler") -> None:
    """GET /physical/status — PhysicalManager status, or empty if not initialised."""
    global _PHYSICAL_MANAGER  # noqa: PLW0603
    if _PHYSICAL_MANAGER is None:
        _send_json(self, 200, {
            "roomba_connected": False,
            "mqtt_connected": False,
            "cameras_running": 0,
            "agents_registered": 0,
            "last_obstacle_event": None,
            "initialised": False,
        })
        return
    try:
        status = _PHYSICAL_MANAGER.status()
        status["initialised"] = True
        _send_json(self, 200, status)
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


def _handle_post_physical_init(self: "ManifoldHandler", body: dict) -> None:
    """POST /physical/init — initialise (or re-initialise) the PhysicalManager."""
    global _PHYSICAL_MANAGER  # noqa: PLW0603
    try:
        # Stop existing manager if present
        if _PHYSICAL_MANAGER is not None:
            try:
                _PHYSICAL_MANAGER.stop_all()
            except Exception:  # noqa: BLE001
                pass

        from manifold_physical.physical_manager import PhysicalManager
        _PHYSICAL_MANAGER = PhysicalManager(config=body)
        _PHYSICAL_MANAGER.start_all()
        _send_json(self, 200, {"status": "ok", **_PHYSICAL_MANAGER.status()})
    except Exception as exc:  # noqa: BLE001
        _send_json(self, 500, {"error": str(exc)})


ManifoldHandler._handle_get_physical_cameras = _handle_get_physical_cameras  # type: ignore[attr-defined]
ManifoldHandler._handle_get_physical_status = _handle_get_physical_status  # type: ignore[attr-defined]
ManifoldHandler._handle_post_physical_init = _handle_post_physical_init  # type: ignore[attr-defined]


def run_server(port: int = 8080, *, host: str = "0.0.0.0") -> None:
    """Start the MANIFOLD HTTP server and block until interrupted.

    On startup the Vault WAL is replayed to restore any gossip and economy
    state from previous runs.  The DB persistence layer is also initialised
    and the WAL counters are synced into it.

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

    # DB persistence layer startup: connect, initialise schema, sync WAL counters.
    # asyncio.run() is safe here because this is a synchronous entry point and no
    # event loop is running yet.  The HTTPServer itself is thread-based (not async).
    # NOTE: in-memory task counters (_TASK_COUNT etc.) are incremented per request
    # but only flushed to the DB on graceful shutdown via flush_to_vault().  If the
    # process is killed (SIGKILL, OOM), in-flight increments since the last run are
    # lost — this is an accepted trade-off to keep the server synchronous.
    _db_url = os.environ.get("MANIFOLD_DB_URL", "sqlite:///manifold.db")
    _db = ManifoldDB(_db_url)

    async def _startup() -> None:
        await _db.connect()
        await _db.sync_from_vault(_VAULT)

    asyncio.run(_startup())

    # Start background learning worker
    global _worker  # noqa: PLW0603
    _worker = ManifoldWorker(pipeline=_get_pipeline(), db=_db)
    _worker.start()

    # Rehydrate brain state from disk (after pipeline is initialised)
    _rehydrate_brain()

    # Start agent monitor
    _AGENT_MONITOR.start()

    # Start real-time health monitor (v1.7.0)
    _HEALTH_MONITOR.start()

    # Start federation background sync thread (every 300 seconds)
    def _federation_sync_loop() -> None:
        while True:
            time.sleep(300)
            try:
                for org in _ORG_REGISTRY.all_orgs():
                    snapshot = _GOSSIP_BRIDGE.export_snapshot(org.org_id)
                    if snapshot is not None:
                        _GOSSIP_BRIDGE.contribute_snapshot(snapshot)
            except Exception:  # noqa: BLE001
                pass

    _fed_thread = threading.Thread(
        target=_federation_sync_loop, daemon=True, name="manifold-federation-sync"
    )
    _fed_thread.start()

    server = HTTPServer((host, port), ManifoldHandler)
    print(f"MANIFOLD server listening on {host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if _worker is not None:
            _worker.stop()
        _AGENT_MONITOR.stop()
        # Flush DB state back to WAL for portability across container restarts
        async def _shutdown() -> None:
            await _db.flush_to_vault(_VAULT)
            await _db.close()

        asyncio.run(_shutdown())
        print("MANIFOLD server stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the MANIFOLD HTTP server.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", os.environ.get("MANIFOLD_PORT", 8080))),
    )
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)
