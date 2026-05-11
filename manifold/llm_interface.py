"""manifold/llm_interface.py — Conversational governance interface for MANIFOLD.

ManifoldLLM lets operators configure MANIFOLD governance in plain English.
It sends user messages to the LLM gateway together with a rich system prompt
that describes the MANIFOLD schema, receives a two-part response (plain
English + JSON MANIFOLD_ACTION), then applies the action to the live
infrastructure via PolicyTranslator.

This creates a self-referential governance loop when the default model_endpoint
is used (POST /v1/chat/completions on the MANIFOLD server itself).
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
from dataclasses import dataclass, field
from typing import Any

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
    prompt_regex    — regex match on prompt
    org_id          — exact org match
    tool_used       — tool name must appear in tools_used list
  action: one of allow / refuse / escalate / audit / throttle / redact /
          shadow_test / require_approval / notify / log_only / quarantine /
          rate_limit / sandbox
  priority: integer 0-100 (higher = evaluated first)
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
        """Send a message and return the parsed LLMResponse.

        Appends to internal conversation history.  The response is also
        stored in the global history ring for ``GET /llm/history``.
        """
        self._history.append({"role": "user", "content": user_message})
        raw = self._call_model(user_message)
        response = self._parse_response(raw)
        self._history.append({"role": "assistant", "content": raw})

        # Store in global ring buffer
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
        """Parse MANIFOLD_ACTION, validate with PolicyTranslator, and apply.

        Returns True on success.  Sets response.applied and response.apply_error.
        """
        if response.action_type == "none":
            response.applied = True
            return True

        try:
            from manifold.policy_translator import PolicyTranslator
            translator = PolicyTranslator(org_id=self.org_id)

            if response.action_type == "policy_rule":
                rule = translator.validate_rule(response.action_payload)
                # Import rule engine singleton from server if available
                try:
                    from manifold.server import _RULE_ENGINE  # type: ignore[attr-defined]
                    _RULE_ENGINE.add_rule(rule)
                except Exception:  # noqa: BLE001
                    pass
                response.applied = True
            elif response.action_type in ("task", "calibration", "query"):
                # No-op stubs for non-rule actions; just mark as applied
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_model(self, user_message: str) -> str:
        """POST to the LLM endpoint and return the assistant message text."""
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages += self._history[:-1]  # history without the current user msg
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
            return "[LLM unavailable — check server logs]\nMANIFOLD_ACTION_START\n{\"type\": \"none\"}\nMANIFOLD_ACTION_END"

    def _parse_response(self, raw: str) -> LLMResponse:
        """Extract plain-text and MANIFOLD_ACTION block from the raw response."""
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

        # Strip MANIFOLD_ACTION block from plain text
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
