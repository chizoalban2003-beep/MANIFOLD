"""Phase 23: The Polyglot Protocol — OpenAPI/REST Spec Generator.

MANIFOLD is currently a Python library.  For it to become a *protocol* (like
HTTP or TCP), it must be expressible in a language-agnostic format that any
client — TypeScript, Go, Java, Rust — can consume.

Phase 23 delivers the **OpenAPI 3.0 specification generator**.  It produces a
standard ``dict`` (serialisable to JSON or YAML) that describes every
MANIFOLD operation as a REST endpoint.  External agents can use this spec to:

* Generate client SDKs automatically (``openapi-generator``,
  ``swagger-codegen``, or ``oapi-codegen`` for Go).
* Validate requests/responses against the schema.
* Import the spec into API gateways (AWS API Gateway, Kong, etc.).

Key ideas
---------
* **Zero external dependencies** — the spec is built from plain Python dicts;
  no ``openapi-core`` or ``pydantic`` required.
* **Shield endpoint** — describes the ``@shield`` decorator as a ``POST``
  endpoint, including the JSON schema for ``BrainTask``, ``InterceptResult``,
  and ``InterceptorVeto``.
* **Policy handshake endpoint** — describes the B2B ``PolicyHandshake`` as a
  ``POST`` endpoint, including ``OrgPolicy`` and ``HandshakeResult`` schemas.
* **Reputation endpoint** — describes the ``ReputationHub.live_reliability``
  query as a ``GET`` endpoint.
* **Recruitment endpoint** — describes ``SovereignRecruiter.maybe_recruit`` as
  a ``POST`` endpoint.

Key classes / functions
-----------------------
``ManifoldOpenAPISpec``
    Builds the full OpenAPI 3.0 spec dict.
``generate_openapi_spec``
    Convenience function returning the spec dict.
``spec_to_json``
    Serialise the spec to a JSON string.
``spec_to_yaml``
    Serialise the spec to a YAML string (zero-dependency hand-written emitter).
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Schema building helpers
# ---------------------------------------------------------------------------


def _string(description: str = "", example: str = "") -> dict[str, Any]:
    s: dict[str, Any] = {"type": "string"}
    if description:
        s["description"] = description
    if example:
        s["example"] = example
    return s


def _number(description: str = "", minimum: float | None = None, maximum: float | None = None) -> dict[str, Any]:
    s: dict[str, Any] = {"type": "number", "format": "float"}
    if description:
        s["description"] = description
    if minimum is not None:
        s["minimum"] = minimum
    if maximum is not None:
        s["maximum"] = maximum
    return s


def _boolean(description: str = "") -> dict[str, Any]:
    s: dict[str, Any] = {"type": "boolean"}
    if description:
        s["description"] = description
    return s


def _array(items: dict[str, Any], description: str = "") -> dict[str, Any]:
    s: dict[str, Any] = {"type": "array", "items": items}
    if description:
        s["description"] = description
    return s


def _object(
    properties: dict[str, Any],
    required: list[str] | None = None,
    description: str = "",
) -> dict[str, Any]:
    s: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        s["required"] = required
    if description:
        s["description"] = description
    return s


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{name}"}


# ---------------------------------------------------------------------------
# Component schemas
# ---------------------------------------------------------------------------


def _build_schemas() -> dict[str, Any]:
    """Return all OpenAPI component schemas for MANIFOLD types."""

    schemas: dict[str, Any] = {}

    # --- BrainTask ---
    schemas["BrainTask"] = _object(
        description=(
            "A task submitted to the ManifoldBrain for pricing and routing."
        ),
        properties={
            "task_id": _string("Unique task identifier.", example="task-001"),
            "domain": _string(
                "Domain context (e.g. 'finance', 'medical', 'general').",
                example="finance",
            ),
            "complexity": _number("Task complexity score [0, 1].", minimum=0.0, maximum=1.0),
            "risk": _number("Estimated risk score [0, 1].", minimum=0.0, maximum=1.0),
            "cost_budget": _number("Maximum acceptable cost (trust units)."),
            "description": _string("Free-text description of the task."),
        },
        required=["task_id", "domain"],
    )

    # --- InterceptResult ---
    schemas["InterceptResult"] = _object(
        description="Result of an ActiveInterceptor evaluation.",
        properties={
            "task_id": _string("Task that was evaluated."),
            "vetoed": _boolean("True if the task was vetoed."),
            "veto_reason": _string("Human-readable veto reason (empty if not vetoed)."),
            "risk_score": _number("Computed risk score for the task.", minimum=0.0, maximum=1.0),
            "threshold": _number("Veto threshold used.", minimum=0.0, maximum=1.0),
        },
        required=["task_id", "vetoed", "risk_score"],
    )

    # --- InterceptorVeto ---
    schemas["InterceptorVeto"] = _object(
        description="Detailed veto record from the ActiveInterceptor.",
        properties={
            "task_id": _string("Vetoed task identifier."),
            "reason": _string("Veto reason."),
            "risk_score": _number("Risk score that triggered the veto."),
            "threshold": _number("Threshold that was exceeded."),
            "timestamp": _string("ISO-8601 timestamp of the veto.", example="2026-01-01T12:00:00Z"),
        },
        required=["task_id", "reason", "risk_score"],
    )

    # --- OrgPolicy ---
    schemas["OrgPolicy"] = _object(
        description="Lightweight snapshot of a remote organisation's published policy.",
        properties={
            "org_id": _string("Unique organisation identifier.", example="org-b"),
            "min_reliability": _number(
                "Minimum tool reliability this org guarantees [0, 1].",
                minimum=0.0,
                maximum=1.0,
            ),
            "max_risk": _number(
                "Maximum risk score any call from this org may produce [0, 1].",
                minimum=0.0,
                maximum=1.0,
            ),
            "domain": _string("Primary operational domain.", example="finance"),
            "version": _string("Policy version string.", example="1.0.0"),
            "notes": _string("Free-text notes."),
        },
        required=["org_id"],
    )

    # --- HandshakeResult ---
    schemas["HandshakeResult"] = _object(
        description="Outcome of a PolicyHandshake between two organisations.",
        properties={
            "compatible": _boolean("True if the policies are compatible."),
            "local_org_id": _string("Calling organisation ID."),
            "remote_org_id": _string("Target organisation ID."),
            "local_domain": _string("Domain the local policy is configured for."),
            "remote_domain": _string("Domain the remote org declared."),
            "conflict_reasons": _array(
                _string(),
                description="Human-readable reasons for incompatibility.",
            ),
            "risk_delta": _number("remote.max_risk − local.risk_tolerance."),
            "reliability_delta": _number("remote.min_reliability − local.min_reliability."),
        },
        required=["compatible", "local_org_id", "remote_org_id"],
    )

    # --- ReputationScore ---
    schemas["ReputationScore"] = _object(
        description="Live reliability score for a tool or org.",
        properties={
            "agent_id": _string("Tool or org identifier."),
            "reliability": _number("Live reliability score [0, 1].", minimum=0.0, maximum=1.0),
            "sample_count": {"type": "integer", "description": "Number of samples in the window."},
        },
        required=["agent_id", "reliability"],
    )

    # --- RecruitmentRequest ---
    schemas["RecruitmentRequest"] = _object(
        description="Request body for the SovereignRecruiter endpoint.",
        properties={
            "domain": _string("Target domain for recruitment.", example="medical"),
            "complexity": _number("Task complexity that triggered recruitment.", minimum=0.0, maximum=1.0),
        },
        required=["domain"],
    )

    # --- RecruitmentResult ---
    schemas["RecruitmentResult"] = _object(
        description="Outcome of a SovereignRecruiter recruitment cycle.",
        properties={
            "recruited": _boolean("True if a new tool was hired."),
            "tool_name": _string("Name of the recruited tool (empty if none)."),
            "domain": _string("Domain the recruitment targeted."),
            "probationary_reliability": _number(
                "Reliability score assigned during probation [0, 1].",
                minimum=0.0,
                maximum=1.0,
            ),
            "reason": _string("Human-readable reason for the outcome."),
        },
        required=["recruited", "domain"],
    )

    # --- Error ---
    schemas["Error"] = _object(
        description="Standard error response.",
        properties={
            "code": {"type": "integer", "description": "HTTP status code."},
            "message": _string("Error message."),
            "detail": _string("Additional detail (optional)."),
        },
        required=["code", "message"],
    )

    return schemas


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------


def _json_response(schema_ref: str, description: str = "Success") -> dict[str, Any]:
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": _ref(schema_ref),
            }
        },
    }


def _error_response(code: int, description: str) -> dict[str, Any]:
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": _ref("Error"),
                "example": {"code": code, "message": description},
            }
        },
    }


def _build_paths() -> dict[str, Any]:
    """Return all OpenAPI path definitions."""

    paths: dict[str, Any] = {}

    # --- POST /shield ---
    paths["/shield"] = {
        "post": {
            "summary": "Evaluate a BrainTask through the ActiveInterceptor (@shield).",
            "operationId": "evaluateShield",
            "tags": ["Governance"],
            "description": (
                "Submits a `BrainTask` to the `ActiveInterceptor`.  "
                "Returns an `InterceptResult` indicating whether the task was "
                "vetoed.  Vetoed tasks must not be executed by the caller."
            ),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": _ref("BrainTask"),
                    }
                },
            },
            "responses": {
                "200": _json_response("InterceptResult", "Intercept evaluation complete."),
                "400": _error_response(400, "Invalid request body."),
                "422": _error_response(422, "Task vetoed — see InterceptResult.vetoed."),
            },
        }
    }

    # --- POST /b2b/handshake ---
    paths["/b2b/handshake"] = {
        "post": {
            "summary": "Perform a B2B policy handshake with a remote organisation.",
            "operationId": "b2bHandshake",
            "tags": ["B2B Routing"],
            "description": (
                "Compares the local `ManifoldPolicy` with a remote `OrgPolicy`.  "
                "Returns a `HandshakeResult` indicating compatibility.  "
                "Only proceed with the cross-org API call if `compatible` is `true`."
            ),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": _ref("OrgPolicy"),
                        "example": {
                            "org_id": "org-b",
                            "min_reliability": 0.85,
                            "max_risk": 0.25,
                            "domain": "finance",
                            "version": "1.0.0",
                        },
                    }
                },
            },
            "responses": {
                "200": _json_response("HandshakeResult", "Handshake evaluation complete."),
                "400": _error_response(400, "Invalid OrgPolicy payload."),
                "403": _error_response(403, "Handshake failed — policies incompatible."),
            },
        }
    }

    # --- GET /reputation/{agent_id} ---
    paths["/reputation/{agent_id}"] = {
        "get": {
            "summary": "Query the live reliability score for a tool or org.",
            "operationId": "getReputation",
            "tags": ["Reputation"],
            "description": (
                "Returns the live reliability score for the given `agent_id` from "
                "the `ReputationHub`.  Scores are computed over a sliding window of "
                "recent outcomes."
            ),
            "parameters": [
                {
                    "name": "agent_id",
                    "in": "path",
                    "required": True,
                    "schema": _string(example="wolfram-alpha"),
                    "description": "Tool or organisation identifier.",
                }
            ],
            "responses": {
                "200": _json_response("ReputationScore", "Reputation score returned."),
                "404": _error_response(404, "Agent not found in the ReputationHub."),
            },
        }
    }

    # --- POST /recruit ---
    paths["/recruit"] = {
        "post": {
            "summary": "Trigger the SovereignRecruiter to find and hire a new tool.",
            "operationId": "triggerRecruitment",
            "tags": ["Recruitment"],
            "description": (
                "Triggers the `SovereignRecruiter` for the given domain.  "
                "If complexity > 0.8 and domain reliability < 0.6, the recruiter "
                "searches the marketplace, runs scout probes, and registers the "
                "best candidate on a 25 %% probationary reliability discount."
            ),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": _ref("RecruitmentRequest"),
                    }
                },
            },
            "responses": {
                "200": _json_response("RecruitmentResult", "Recruitment cycle complete."),
                "400": _error_response(400, "Invalid recruitment request."),
            },
        }
    }

    # --- GET /policy ---
    paths["/policy"] = {
        "get": {
            "summary": "Export the local ManifoldPolicy as an OrgPolicy snapshot.",
            "operationId": "exportPolicy",
            "tags": ["Policy"],
            "description": (
                "Returns a serialised `OrgPolicy` snapshot of the local "
                "`ManifoldPolicy`.  Remote organisations call this endpoint "
                "to retrieve the policy before initiating a handshake."
            ),
            "responses": {
                "200": _json_response("OrgPolicy", "Policy snapshot returned."),
            },
        }
    }

    return paths


# ---------------------------------------------------------------------------
# ManifoldOpenAPISpec
# ---------------------------------------------------------------------------


class ManifoldOpenAPISpec:
    """Builds the full OpenAPI 3.0 specification for MANIFOLD.

    The generated spec describes every MANIFOLD operation as a REST endpoint,
    enabling non-Python clients to generate SDKs and participate in the
    protocol.

    Parameters
    ----------
    title:
        API title in the spec.
    version:
        API version string.
    server_url:
        Base URL of the MANIFOLD API server.
    description:
        Extended API description.

    Example
    -------
    ::

        spec = ManifoldOpenAPISpec()
        d = spec.to_dict()
        print(spec_to_json(d))
    """

    def __init__(
        self,
        title: str = "MANIFOLD Trust API",
        version: str = "1.1.0",
        server_url: str = "http://localhost:8080",
        description: str = (
            "The Trust Operating System for AI agents. "
            "Prices risk, delegates hierarchically, detects adversarial gaming, "
            "recruits its own tools, and governs via policy-as-code."
        ),
    ) -> None:
        self.title = title
        self.version = version
        self.server_url = server_url
        self.description = description

    def to_dict(self) -> dict[str, Any]:
        """Build and return the full OpenAPI 3.0 spec as a plain dict.

        Returns
        -------
        dict
            The complete OpenAPI specification.
        """
        return {
            "openapi": "3.0.3",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description,
                "license": {"name": "MIT"},
                "contact": {
                    "name": "MANIFOLD Contributors",
                    "url": "https://github.com/chizoalban2003-beep/MANIFOLD",
                },
            },
            "servers": [
                {
                    "url": self.server_url,
                    "description": "MANIFOLD API server",
                }
            ],
            "tags": [
                {"name": "Governance", "description": "Task interception and risk evaluation."},
                {"name": "B2B Routing", "description": "Cross-org policy handshakes."},
                {"name": "Reputation", "description": "Agent and org reputation queries."},
                {"name": "Recruitment", "description": "Sovereign Recruiter operations."},
                {"name": "Policy", "description": "Policy export and management."},
            ],
            "paths": _build_paths(),
            "components": {
                "schemas": _build_schemas(),
                "securitySchemes": {
                    "ManifoldHMAC": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-Manifold-Signature",
                        "description": (
                            "HMAC-SHA256 signature of the request body, hex-encoded. "
                            "Used for zero-trust policy verification (Phase 21)."
                        ),
                    }
                },
            },
        }

    def endpoint_ids(self) -> list[str]:
        """Return list of all operationId values in the spec."""
        spec = self.to_dict()
        ids: list[str] = []
        for path_item in spec["paths"].values():
            for method_obj in path_item.values():
                if isinstance(method_obj, dict) and "operationId" in method_obj:
                    ids.append(method_obj["operationId"])
        return sorted(ids)

    def schema_names(self) -> list[str]:
        """Return list of all component schema names."""
        return sorted(self.to_dict()["components"]["schemas"].keys())


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def generate_openapi_spec(
    title: str = "MANIFOLD Trust API",
    version: str = "1.1.0",
    server_url: str = "http://localhost:8080",
) -> dict[str, Any]:
    """Generate the MANIFOLD OpenAPI 3.0 spec.

    Parameters
    ----------
    title:
        API title.
    version:
        API version.
    server_url:
        Base URL of the server.

    Returns
    -------
    dict
        OpenAPI 3.0 specification dict.
    """
    return ManifoldOpenAPISpec(title=title, version=version, server_url=server_url).to_dict()


def spec_to_json(spec: dict[str, Any], indent: int = 2) -> str:
    """Serialise an OpenAPI spec dict to a JSON string.

    Parameters
    ----------
    spec:
        The spec dict returned by ``generate_openapi_spec``.
    indent:
        JSON indentation level.

    Returns
    -------
    str
        JSON string.
    """
    return json.dumps(spec, indent=indent, ensure_ascii=False)


def spec_to_yaml(spec: dict[str, Any]) -> str:
    """Serialise an OpenAPI spec dict to a YAML string.

    Uses a hand-written emitter (zero external dependencies).

    Parameters
    ----------
    spec:
        The spec dict.

    Returns
    -------
    str
        YAML string.
    """
    return _dict_to_yaml(spec, indent=0)


# ---------------------------------------------------------------------------
# Hand-written YAML emitter (zero-dependency)
# ---------------------------------------------------------------------------


def _yaml_scalar(value: Any) -> str:
    """Render a scalar value as a YAML scalar string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    s = str(value)
    # Quote strings that contain special YAML characters or look like
    # other types to avoid ambiguity.
    needs_quote = (
        not s
        or s[0] in "{[\"'|>&*!%@`"
        or s in ("true", "false", "null", "yes", "no", "on", "off")
        or ":" in s
        or "#" in s
        or s[0] == "-"
        or "\n" in s
    )
    if needs_quote:
        escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return s


def _dict_to_yaml(obj: Any, indent: int) -> str:
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines: list[str] = []
        for k, v in obj.items():
            key_str = _yaml_scalar(k)
            rendered = _dict_to_yaml(v, indent + 1)
            if isinstance(v, dict) and v:
                lines.append(f"{pad}{key_str}:")
                lines.append(rendered)
            elif isinstance(v, list) and v:
                lines.append(f"{pad}{key_str}:")
                lines.append(rendered)
            else:
                # inline scalar or empty collection
                lines.append(f"{pad}{key_str}: {rendered}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        lines = []
        for item in obj:
            rendered = _dict_to_yaml(item, indent + 1)
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}-")
                lines.append(rendered)
            else:
                lines.append(f"{pad}- {rendered}")
        return "\n".join(lines)
    else:
        return _yaml_scalar(obj)
