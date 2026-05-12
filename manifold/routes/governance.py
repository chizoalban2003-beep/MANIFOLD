"""manifold/routes/governance.py — Governance pipeline endpoint handlers.

Handlers for:
  POST /run
  POST /shield       (delegated from ManifoldHandler class method)
  POST /task
  POST /task/cooperative
  GET  /rules
  POST /rules
  DELETE /rules/{id}
  GET  /plan
  GET  /brain/state
  POST /llm/chat
  GET  /llm/history
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manifold.server import ManifoldHandler


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

def _srv():
    import manifold.server as _s  # noqa: PLC0415
    return _s


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_post_run(self: "ManifoldHandler", body: dict[str, Any]) -> None:
    """POST /run — execute ManifoldPipeline and return the result."""
    s = _srv()
    prompt = body.get("prompt")
    if not prompt:
        s._send_json(self, 400, {"error": "prompt required"})
        return
    try:
        pipeline = s._get_pipeline()
        result = pipeline.run(
            prompt=str(prompt),
            data=body.get("data"),
            encoder_hint=str(body.get("encoder_hint", "auto")),
            explicit_domain=body.get("domain") or None,
            stakes=float(body.get("stakes", 0.5)),
            uncertainty=float(body.get("uncertainty", 0.5)),
            tools_used=body.get("tools_used"),
        )
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
        s._send_json(self, 200, serialised)
    except Exception as exc:  # noqa: BLE001
        s._send_json(self, 500, {"error": str(exc)})


def handle_post_task(self: "ManifoldHandler", body: dict) -> None:
    """POST /task — receive any problem, decompose, govern, route."""
    s = _srv()
    task = str(body.get("task", "")).strip()
    stakes = float(body.get("stakes", 0.5))
    if not task:
        s._send_error(self, 400, "task field required")
        return
    plan = s._TASK_ROUTER.route(task, stakes_hint=stakes)
    s._send_json(self, 200, plan.to_dict())


def handle_get_brain_state(self: "ManifoldHandler") -> None:
    """GET /brain/state — return brain persistence status."""
    s = _srv()
    pipeline = s._get_pipeline()
    cmap_nodes = len(pipeline._cognitive_map._outcome_log)
    cooc_tools = len(pipeline._cooccurrence._tool_counts)
    pred_entries = len(pipeline._predictor._prediction_log)
    rules = len(pipeline._consolidator._promoted_rules)
    state_dir = str(s._BRAIN_STATE_DIR)
    persisted = (
        (s._BRAIN_STATE_DIR / "cognitive_map.json").exists()
        or (s._BRAIN_STATE_DIR / "cooccurrence.json").exists()
        or (s._BRAIN_STATE_DIR / "predictor.json").exists()
        or (s._BRAIN_STATE_DIR / "consolidator.json").exists()
    )
    s._send_json(
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


def handle_get_rules(self: "ManifoldHandler") -> None:
    """GET /rules — return all policy rules for the calling org."""
    s = _srv()
    _authed, caller = s._check_auth(self, "/rules")
    if not _authed:
        return
    org_id = caller.org_id if caller else ""
    rules = [r.to_dict() for r in s._RULE_ENGINE.rules_for_org(org_id)]
    s._send_json(self, 200, {"rules": rules, "org_id": org_id})


def handle_post_rule(self: "ManifoldHandler", body: dict, caller: Any) -> None:
    """POST /rules — create a new policy rule for the calling org."""
    import uuid as _uuid  # noqa: PLC0415
    s = _srv()
    from manifold.policy_rules import PolicyRule  # noqa: PLC0415
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
    s._RULE_ENGINE.add_rule(rule)
    s._send_json(self, 201, rule.to_dict())


def handle_delete_rule(self: "ManifoldHandler", rule_id: str) -> None:
    """DELETE /rules/{rule_id} — remove a policy rule."""
    s = _srv()
    removed = s._RULE_ENGINE.remove_rule(rule_id)
    if removed:
        s._send_json(self, 200, {"rule_id": rule_id, "status": "deleted"})
    else:
        s._send_error(self, 404, f"Rule {rule_id!r} not found")


def handle_get_plan(self: "ManifoldHandler") -> None:
    """GET /plan — CRNA A* path planning."""
    import urllib.parse as _up  # noqa: PLC0415
    s = _srv()
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
        result = s._PLANNER.plan(start=start, target=target, risk_budget=risk_budget)
        s._send_json(self, 200, result)
    except Exception as exc:  # noqa: BLE001
        s._send_json(self, 500, {"error": str(exc)})


def handle_get_llm_history(self: "ManifoldHandler") -> None:
    """GET /llm/history — last 20 LLM chat exchanges."""
    from manifold.llm_interface import get_llm_history  # noqa: PLC0415
    s = _srv()
    s._send_json(self, 200, {"history": get_llm_history()})


def handle_post_llm_chat(self: "ManifoldHandler", body: dict) -> None:
    """POST /llm/chat — send a natural language message to MANIFOLD governance AI."""
    s = _srv()
    message = body.get("message") or body.get("user_message") or ""
    if not message:
        s._send_error(self, 400, "Body must include a 'message' field.")
        return
    org_id = body.get("org_id", "default")
    model_endpoint = body.get("model_endpoint", "http://localhost:8080/v1/chat/completions")
    api_key = body.get("api_key", "")
    model = body.get("model", "gpt-4o")

    from manifold.llm_interface import ManifoldLLM  # noqa: PLC0415
    llm = ManifoldLLM(
        org_id=org_id,
        model_endpoint=model_endpoint,
        api_key=api_key,
        model=model,
    )
    response = llm.chat(message)
    llm.apply_response(response)
    s._send_json(self, 200, {
        "reply": response.plain_text,
        "action_type": response.action_type,
        "action_payload": response.action_payload,
        "applied": response.applied,
        "apply_error": response.apply_error,
    })
