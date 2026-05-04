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
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .b2b import B2BRouter, OrgPolicy
from .brain import BrainConfig, BrainTask, ManifoldBrain
from .connector import ConnectorRegistry
from .hub import ReputationHub
from .interceptor import ActiveInterceptor, InterceptorConfig
from .policy import ManifoldPolicy
from .recruiter import SovereignRecruiter
from .trustrouter import clamp01


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

            # GET /reputation/<id>
            m = re.fullmatch(r"/reputation/(.+)", path)
            if m:
                self._handle_get_reputation(m.group(1))
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

    Parameters
    ----------
    port:
        TCP port to bind.  Default: ``8080``.
    host:
        Bind address.  Default: ``"0.0.0.0"`` (all interfaces).
    """
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
