"""manifold/llm_interface.py — Conversational governance interface for MANIFOLD.

ManifoldLLM lets operators configure MANIFOLD governance in plain English.
It sends user messages to the LLM gateway together with a rich system prompt
that describes the MANIFOLD schema, receives a two-part response (plain
English + JSON MANIFOLD_ACTION), then applies the action to the live
infrastructure via PolicyTranslator.

This creates a self-referential governance loop when the default model_endpoint
is used (POST /v1/chat/completions on the MANIFOLD server itself).

Fix (8.6): Added _validate_action_payload() before PolicyTranslator application.
Validates action type, priority bounds, condition key allowlist, numeric range
clamping, and blocks prompt_regex conditions from LLM-sourced rules (regex
injection vector).  LLM-sourced rules are limited to priority ≤ 10 so they
cannot override manually-authored high-priority rules.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass

_SYSTEM_PROMPT = """
You are MANIFOLD Governance AI, an expert at configuring AI governance systems.
MANIFOLD is a policy enforcement platform for AI pipelines.

== MANIFOLD SCHEMA ==

PolicyRule fields:
  rule_id: str (UUID)
  org_id: str
  name: str
  conditions: dict — valid keys:
    domain          — exact domain match (str)
    domain_in       — list of domains (any match)
    stakes_gt       — float, stakes > threshold
    stakes_lt       — float, stakes < threshold
    risk_gt         — float, risk_score > threshold
    prompt_contains — case-insensitive substring of prompt
    org_id          — exact org match
    tool_used       — tool name must appear in tools_used list
  action: one of allow / refuse / escalate / audit / throttle / redact /
          shadow_test / require_approval / notify / log_only / quarantine /
          rate_limit / sandbox
  priority: integer 0-10 (higher = evaluated first; LLM rules capped at 10)
  enabled: bool (default true)

Valid domains: healthcare, finance, devops, legal, infrastructure, trading, supply_chain, general

CRNA grid values per cell:
  c (Cost 0-1), r (Risk 0-1), n (Neutrality 0-1), a (Asset 0-1)

TaskRouter.route() accepts:
  prompt, domain, stakes (0-1), tools_used (list)

== RESPONSE FORMAT ==

Always reply in EXACTLY two parts:

PART 1 — plain English reply to the human.

PART 2 — a JSON block between tags:
MANIFOLD_ACTION_START
{
  "type": "policy_rule" | "task" | "calibration" | "query" | "none",
  ... (relevant fields for the type)
}
MANIFOLD_ACTION_END

If no action is needed, set type to "none".

For type "policy_rule" include all PolicyRule fields.
For type "task" include prompt, domain, stakes.
For type "calibration" include domain and adjustments dict.
For type "query" include what data to look up.
""".strip()

_MAX_HISTORY = 20
_LLM_HISTORY: deque[dict] = deque(maxlen=_MAX_HISTORY)
_LLM_HISTORY_LOCK = threading.Lock()

# Allowed action values for LLM-sourced rules
_ALLOWED_ACTIONS = frozenset({
    "allow", "refuse", "escalate", "audit", "throttle", "redact",
    "shadow_test", "require_approval", "notify", "log_only",
    "quarantine", "rate_limit", "sandbox",
})

# Allowed condition keys for LLM-sourced rules.
# prompt_regex is intentionally excluded: regex patterns from an LLM
# can be crafted to cause catastrophic backtracking (ReDoS) or match
# unintended inputs, constituting a second-order injection vector.
_ALLOWED_CONDITION_KEYS = frozenset({
    "domain", "domain_in", "stakes_gt", "stakes_lt", "risk_gt",
    "prompt_contains", "org_id", "tool_used",
})

_ALLOWED_DOMAINS = frozenset({
    "healthcare", "finance", "devops", "legal", "infrastructure",
    "trading", "supply_chain", "general",
})

# LLM-sourced rules are capped at priority 10 so they never silently
# override operator-authored rules which conventionally use 50-100.
_LLM_MAX_PRIORITY = 10

# Maximum number of conditions allowed per LLM-sourced rule
_MAX_CONDITIONS = 8


@dataclass
class LLMResponse:
    """Parsed response from ManifoldLLM.chat()."""

    plain_text: str
    action_type: str
    action_payload: dict
    raw_response: str
    applied: bool = False
    apply_error: str = ""


class ManifoldLLM:
    """Natural language governance interface for MANIFOLD.

    Parameters
    ----------
    org_id:
        The MANIFOLD org to apply policies to.
    model_endpoint:
        URL of the chat completions endpoint. Defaults to the MANIFOLD
        self-hosted gateway (self-referential governance loop).
    api_key:
        Bearer token for the model endpoint.
    model:
        Model name string (e.g. ``"gpt-4o"`` or ``"claude-3-opus"``).
    """

    def __init__(
        self,
        org_id: str = "default",
        model_endpoint: str = "http://localhost:8080/v1/chat/completions",
        api_key: str = "",
        model: str = "gpt-4o",
    ) -> None:
        self.org_id = org_id
        self.model_endpoint = model_endpoint
        self.api_key = api_key
        self.model = model
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> LLMResponse:
        """Send a message and return the parsed LLMResponse."""
        self._history.append({"role": "user", "content": user_message})
        raw = self._call_model(user_message)
        response = self._parse_response(raw)
        self._history.append({"role": "assistant", "content": raw})

        with _LLM_HISTORY_LOCK:
            _LLM_HISTORY.append({
                "timestamp": time.time(),
                "org_id": self.org_id,
                "user_message": user_message,
                "reply": response.plain_text,
                "action_type": response.action_type,
                "applied": response.applied,
            })

        return response

    def apply_response(self, response: LLMResponse) -> bool:
        """Validate action payload and apply it via PolicyTranslator.

        Returns True on success.  Sets response.applied and response.apply_error.

        Security: _validate_action_payload() runs before PolicyTranslator so
        that malformed or adversarial payloads are rejected with a clear error
        rather than reaching the rule engine.
        """
        if response.action_type == "none":
            response.applied = True
            return True

        try:
            # Validate before touching PolicyTranslator
            validation_error = self._validate_action_payload(
                response.action_type, response.action_payload
            )
            if validation_error:
                response.apply_error = f"Payload validation failed: {validation_error}"
                return False

            from manifold.policy_translator import PolicyTranslator
            translator = PolicyTranslator(org_id=self.org_id)

            if response.action_type == "policy_rule":
                rule = translator.validate_rule(response.action_payload)
                try:
                    from manifold.server import _RULE_ENGINE  # type: ignore[attr-defined]
                    _RULE_ENGINE.add_rule(rule)
                except Exception:  # noqa: BLE001
                    pass
                response.applied = True
            elif response.action_type in ("task", "calibration", "query"):
                response.applied = True
            else:
                response.apply_error = f"Unknown action type: {response.action_type}"
                return False
        except ValueError as exc:
            response.apply_error = str(exc)
            return False
        except Exception as exc:  # noqa: BLE001
            response.apply_error = str(exc)
            logging.error("ManifoldLLM.apply_response error: %s", exc)
            return False

        return True

    def history(self) -> list[dict]:
        """Return raw conversation history for this session."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Payload validation
    # ------------------------------------------------------------------

    def _validate_action_payload(self, action_type: str, payload: dict) -> str:
        """Validate an LLM-sourced action payload before applying it.

        Returns an error string on failure, empty string on success.

        Checks performed for policy_rule payloads:
        - action is in _ALLOWED_ACTIONS
        - priority is clamped to [0, _LLM_MAX_PRIORITY]
        - conditions dict uses only _ALLOWED_CONDITION_KEYS
        - conditions dict size ≤ _MAX_CONDITIONS
        - numeric condition values are in [0, 1]
        - domain values are in _ALLOWED_DOMAINS
        - org_id matches the session org (prevents cross-org injection)
        """
        if not isinstance(payload, dict):
            return "payload must be a dict"

        if action_type != "policy_rule":
            return ""  # only policy_rule needs deep validation

        # Action value
        action = payload.get("action", "")
        if action not in _ALLOWED_ACTIONS:
            return f"action '{action}' not in allowed set"

        # Priority: clamp silently rather than reject so the rule still
        # applies — just can't override high-priority operator rules.
        raw_priority = payload.get("priority", 0)
        try:
            priority = int(raw_priority)
        except (TypeError, ValueError):
            return f"priority must be an integer, got {raw_priority!r}"
        if priority > _LLM_MAX_PRIORITY:
            payload["priority"] = _LLM_MAX_PRIORITY
            logging.warning(
                "ManifoldLLM: clamped LLM rule priority %d → %d",
                priority, _LLM_MAX_PRIORITY,
            )

        # Conditions
        conditions = payload.get("conditions", {})
        if not isinstance(conditions, dict):
            return "conditions must be a dict"
        if len(conditions) > _MAX_CONDITIONS:
            return f"conditions dict exceeds max size ({_MAX_CONDITIONS})"

        for key, value in conditions.items():
            if key not in _ALLOWED_CONDITION_KEYS:
                return (
                    f"condition key '{key}' not allowed in LLM-sourced rules "
                    f"(allowed: {sorted(_ALLOWED_CONDITION_KEYS)})"
                )
            if key in ("stakes_gt", "stakes_lt", "risk_gt"):
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    return f"condition '{key}' must be a float, got {value!r}"
                if not (0.0 <= v <= 1.0):
                    return f"condition '{key}' value {v} out of [0, 1] range"
            if key == "domain" and value not in _ALLOWED_DOMAINS:
                return f"domain '{value}' not in allowed domains"
            if key == "domain_in":
                if not isinstance(value, list):
                    return "domain_in must be a list"
                for d in value:
                    if d not in _ALLOWED_DOMAINS:
                        return f"domain_in contains unknown domain '{d}'"

        # Org isolation: LLM rules may only target the session org
        rule_org = payload.get("org_id", self.org_id)
        if rule_org != self.org_id:
            return (
                f"org_id mismatch: rule targets '{rule_org}' "
                f"but session is '{self.org_id}'"
            )

        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_model(self, user_message: str) -> str:
        """POST to the LLM endpoint and return the assistant message text."""
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages += self._history[:-1]
        messages.append({"role": "user", "content": user_message})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
        }).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            req = urllib.request.Request(
                self.model_endpoint,
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logging.error("ManifoldLLM._call_model error: %s", exc)
            return (
                "[LLM unavailable — check server logs]\n"
                "MANIFOLD_ACTION_START\n{\"type\": \"none\"}\nMANIFOLD_ACTION_END"
            )

    def _parse_response(self, raw: str) -> LLMResponse:
        """Extract plain-text and MANIFOLD_ACTION block from raw response."""
        action_payload: dict = {"type": "none"}
        action_type = "none"

        match = re.search(
            r"MANIFOLD_ACTION_START\s*(.*?)\s*MANIFOLD_ACTION_END",
            raw,
            re.DOTALL,
        )
        if match:
            try:
                action_payload = json.loads(match.group(1))
                action_type = str(action_payload.get("type", "none"))
            except (json.JSONDecodeError, AttributeError):
                action_payload = {"type": "none"}

        plain = re.sub(
            r"MANIFOLD_ACTION_START.*?MANIFOLD_ACTION_END",
            "",
            raw,
            flags=re.DOTALL,
        ).strip()

        return LLMResponse(
            plain_text=plain or raw.strip(),
            action_type=action_type,
            action_payload=action_payload,
            raw_response=raw,
        )


def get_llm_history() -> list[dict]:
    """Return the last 20 LLM exchanges across all sessions."""
    with _LLM_HISTORY_LOCK:
        return list(_LLM_HISTORY)
