"""Phase 64: Rosetta Protocol Adapter — Universal Framework Interoperability.

MANIFOLD must communicate with non-MANIFOLD agents (AutoGPT, LangChain,
OpenAI Swarm, raw JSON APIs).  This module provides two complementary
translation surfaces:

* :class:`ForeignPayloadIngress` — detects the schema of an incoming JSON
  payload and translates it into a native :class:`~manifold.brain.BrainTask`.
* :class:`EgressTranslator` — takes a native
  :class:`~manifold.provenance.DecisionReceipt` and formats it into the
  shape expected by the target framework, allowing MANIFOLD to act as a
  silent governance proxy for legacy AI apps.

Supported frameworks
--------------------
The ingress layer recognises four payload shapes:

``MANIFOLD``
    Native :class:`~manifold.brain.BrainTask` JSON (pass-through after
    validation).

``LANGCHAIN``
    LangChain ``AgentAction`` / ``run`` payloads:
    ``{"input": ..., "tool": ..., "tool_input": ...}``

``AUTOGPT``
    AutoGPT command payloads:
    ``{"command": {"name": ..., "args": {"task": ...}}}``

``OPENAI_SWARM``
    OpenAI Swarm message payloads:
    ``{"messages": [...], "model": ..., "agent": ...}``

``GENERIC``
    Fallback: any JSON object with at least a ``"prompt"`` or ``"text"``
    or ``"input"`` or ``"content"`` key.

Architecture
------------
::

    foreign_json
        │
    ForeignPayloadIngress.detect_schema()  → FrameworkSchema
        │
    ForeignPayloadIngress.ingest()         → BrainTask
        │
    MANIFOLD decision pipeline …
        │
    EgressTranslator.translate()           → dict (framework-native shape)

Zero external dependencies — only stdlib ``json`` and dataclasses.

Key classes
-----------
``FrameworkSchema``
    Enum-like frozen dataclass describing the detected framework.
``IngressResult``
    Immutable result of a single ingress translation attempt.
``EgressResult``
    Immutable result of a single egress translation attempt.
``ForeignPayloadIngress``
    Translates foreign JSON payloads into native ``BrainTask`` objects.
``EgressTranslator``
    Translates native ``DecisionReceipt`` objects into framework-native JSON.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .brain import BrainTask
from .provenance import DecisionReceipt


# ---------------------------------------------------------------------------
# FrameworkSchema
# ---------------------------------------------------------------------------

_KNOWN_FRAMEWORKS = frozenset(
    {"MANIFOLD", "LANGCHAIN", "AUTOGPT", "OPENAI_SWARM", "GENERIC"}
)


@dataclass(frozen=True)
class FrameworkSchema:
    """Identifies the detected source framework of a foreign payload.

    Attributes
    ----------
    name:
        One of ``"MANIFOLD"``, ``"LANGCHAIN"``, ``"AUTOGPT"``,
        ``"OPENAI_SWARM"``, or ``"GENERIC"``.
    confidence:
        Detection confidence in ``[0.0, 1.0]``.  Values above 0.8 mean the
        schema was positively identified; below 0.5 means generic fallback.
    """

    name: str
    confidence: float

    def __post_init__(self) -> None:
        if self.name not in _KNOWN_FRAMEWORKS:
            raise ValueError(
                f"Unknown framework '{self.name}'. "
                f"Expected one of {sorted(_KNOWN_FRAMEWORKS)}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {"name": self.name, "confidence": self.confidence}


# ---------------------------------------------------------------------------
# IngressResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngressResult:
    """Immutable result of a :meth:`ForeignPayloadIngress.ingest` call.

    Attributes
    ----------
    task:
        The translated :class:`~manifold.brain.BrainTask`.
    schema:
        The :class:`FrameworkSchema` that was detected.
    raw_payload:
        The original incoming payload dict (unmodified).
    translated_at:
        POSIX timestamp of the translation.
    warnings:
        Tuple of non-fatal warning strings emitted during translation.
    """

    task: BrainTask
    schema: FrameworkSchema
    raw_payload: dict[str, Any]
    translated_at: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "schema": self.schema.to_dict(),
            "translated_at": self.translated_at,
            "warnings": list(self.warnings),
            "task": {
                "prompt": self.task.prompt,
                "domain": self.task.domain,
                "uncertainty": self.task.uncertainty,
                "complexity": self.task.complexity,
                "stakes": self.task.stakes,
                "source_confidence": self.task.source_confidence,
                "tool_relevance": self.task.tool_relevance,
                "time_pressure": self.task.time_pressure,
                "safety_sensitivity": self.task.safety_sensitivity,
            },
        }


# ---------------------------------------------------------------------------
# EgressResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EgressResult:
    """Immutable result of a :meth:`EgressTranslator.translate` call.

    Attributes
    ----------
    payload:
        Framework-native dict ready for JSON serialisation.
    schema:
        The :class:`FrameworkSchema` the receipt was translated *into*.
    translated_at:
        POSIX timestamp of the translation.
    """

    payload: dict[str, Any]
    schema: FrameworkSchema
    translated_at: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "schema": self.schema.to_dict(),
            "translated_at": self.translated_at,
            "payload": self.payload,
        }


# ---------------------------------------------------------------------------
# ForeignPayloadIngress
# ---------------------------------------------------------------------------


class ForeignPayloadIngress:
    """Detects and translates foreign JSON payloads into native BrainTask.

    Detection is performed heuristically using key-presence checks; no
    external schema registry is required.

    Example
    -------
    ::

        ingress = ForeignPayloadIngress()
        result = ingress.ingest({"input": "refund customer", "tool": "billing_api"})
        print(result.schema.name)  # "LANGCHAIN"
        print(result.task.prompt)  # "refund customer"
    """

    # Default values used when the foreign payload lacks a field
    _DEFAULTS: dict[str, Any] = {
        "domain": "general",
        "uncertainty": 0.5,
        "complexity": 0.5,
        "stakes": 0.5,
        "source_confidence": 0.7,
        "tool_relevance": 0.5,
        "time_pressure": 0.4,
        "safety_sensitivity": 0.2,
        "collaboration_value": 0.3,
        "user_patience": 0.7,
        "dynamic_goal": False,
    }

    # ------------------------------------------------------------------ #
    # Schema detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_schema(payload: dict[str, Any]) -> FrameworkSchema:
        """Detect which framework produced *payload*.

        Parameters
        ----------
        payload:
            The raw incoming JSON dict.

        Returns
        -------
        FrameworkSchema
            The best-matching framework and detection confidence.
        """
        # --- Native MANIFOLD ---
        if "prompt" in payload and "domain" in payload and "uncertainty" in payload:
            return FrameworkSchema(name="MANIFOLD", confidence=0.95)

        # --- LangChain: AgentAction / chain run ---
        if "tool" in payload and "tool_input" in payload:
            return FrameworkSchema(name="LANGCHAIN", confidence=0.95)
        if "input" in payload and "tool" in payload:
            return FrameworkSchema(name="LANGCHAIN", confidence=0.85)

        # --- AutoGPT command format ---
        if "command" in payload and isinstance(payload.get("command"), dict):
            cmd = payload["command"]
            if "name" in cmd or "args" in cmd:
                return FrameworkSchema(name="AUTOGPT", confidence=0.9)

        # --- OpenAI Swarm / Assistants API ---
        if "messages" in payload and isinstance(payload.get("messages"), list):
            return FrameworkSchema(name="OPENAI_SWARM", confidence=0.9)
        if "model" in payload and "agent" in payload:
            return FrameworkSchema(name="OPENAI_SWARM", confidence=0.8)

        # --- Generic fallback ---
        generic_keys = {"prompt", "text", "input", "content", "query", "task"}
        if generic_keys & set(payload.keys()):
            return FrameworkSchema(name="GENERIC", confidence=0.5)

        return FrameworkSchema(name="GENERIC", confidence=0.3)

    # ------------------------------------------------------------------ #
    # Per-framework translators
    # ------------------------------------------------------------------ #

    def _translate_manifold(
        self, payload: dict[str, Any]
    ) -> tuple[BrainTask, list[str]]:
        """Translate a native MANIFOLD payload (pass-through + validation)."""
        warnings: list[str] = []
        prompt = str(payload.get("prompt", ""))
        if not prompt:
            warnings.append("MANIFOLD payload has empty 'prompt' field")
        task = BrainTask(
            prompt=prompt,
            domain=str(payload.get("domain", self._DEFAULTS["domain"])),
            uncertainty=float(payload.get("uncertainty", self._DEFAULTS["uncertainty"])),
            complexity=float(payload.get("complexity", self._DEFAULTS["complexity"])),
            stakes=float(payload.get("stakes", self._DEFAULTS["stakes"])),
            source_confidence=float(
                payload.get("source_confidence", self._DEFAULTS["source_confidence"])
            ),
            tool_relevance=float(
                payload.get("tool_relevance", self._DEFAULTS["tool_relevance"])
            ),
            time_pressure=float(
                payload.get("time_pressure", self._DEFAULTS["time_pressure"])
            ),
            safety_sensitivity=float(
                payload.get("safety_sensitivity", self._DEFAULTS["safety_sensitivity"])
            ),
            collaboration_value=float(
                payload.get("collaboration_value", self._DEFAULTS["collaboration_value"])
            ),
            user_patience=float(
                payload.get("user_patience", self._DEFAULTS["user_patience"])
            ),
            dynamic_goal=bool(payload.get("dynamic_goal", self._DEFAULTS["dynamic_goal"])),
        )
        return task, warnings

    def _translate_langchain(
        self, payload: dict[str, Any]
    ) -> tuple[BrainTask, list[str]]:
        """Translate a LangChain AgentAction / run payload."""
        warnings: list[str] = []
        # Prefer tool_input as prompt text; fall back to input or output
        tool_input = payload.get("tool_input", "")
        if isinstance(tool_input, dict):
            prompt = str(tool_input.get("query", tool_input.get("input", str(tool_input))))
        else:
            prompt = str(tool_input) if tool_input else str(payload.get("input", ""))

        if not prompt:
            warnings.append("LangChain payload has no resolvable prompt text")

        tool_name = str(payload.get("tool", ""))
        task = BrainTask(
            prompt=prompt,
            domain=str(payload.get("domain", self._DEFAULTS["domain"])),
            uncertainty=float(payload.get("uncertainty", self._DEFAULTS["uncertainty"])),
            complexity=float(payload.get("complexity", self._DEFAULTS["complexity"])),
            stakes=float(payload.get("stakes", self._DEFAULTS["stakes"])),
            source_confidence=float(
                payload.get("source_confidence", self._DEFAULTS["source_confidence"])
            ),
            tool_relevance=1.0 if tool_name else float(self._DEFAULTS["tool_relevance"]),
            time_pressure=float(
                payload.get("time_pressure", self._DEFAULTS["time_pressure"])
            ),
            safety_sensitivity=float(
                payload.get("safety_sensitivity", self._DEFAULTS["safety_sensitivity"])
            ),
            collaboration_value=float(
                payload.get(
                    "collaboration_value", self._DEFAULTS["collaboration_value"]
                )
            ),
            user_patience=float(
                payload.get("user_patience", self._DEFAULTS["user_patience"])
            ),
            dynamic_goal=bool(payload.get("dynamic_goal", False)),
        )
        return task, warnings

    def _translate_autogpt(
        self, payload: dict[str, Any]
    ) -> tuple[BrainTask, list[str]]:
        """Translate an AutoGPT command payload."""
        warnings: list[str] = []
        cmd: dict[str, Any] = payload.get("command", {})
        cmd_name = str(cmd.get("name", ""))
        args: dict[str, Any] = cmd.get("args", {})
        # Derive prompt from task arg, falling back to command name
        prompt = str(args.get("task", args.get("query", args.get("input", cmd_name))))
        if not prompt:
            warnings.append("AutoGPT payload has no resolvable task text")

        # AutoGPT commands are inherently tool-usage
        task = BrainTask(
            prompt=prompt,
            domain=str(args.get("domain", payload.get("domain", self._DEFAULTS["domain"]))),
            uncertainty=float(payload.get("uncertainty", self._DEFAULTS["uncertainty"])),
            complexity=float(payload.get("complexity", self._DEFAULTS["complexity"])),
            stakes=float(payload.get("stakes", self._DEFAULTS["stakes"])),
            source_confidence=float(
                payload.get("source_confidence", self._DEFAULTS["source_confidence"])
            ),
            tool_relevance=0.9,  # AutoGPT is always tool-using
            time_pressure=float(
                payload.get("time_pressure", self._DEFAULTS["time_pressure"])
            ),
            safety_sensitivity=float(
                payload.get("safety_sensitivity", self._DEFAULTS["safety_sensitivity"])
            ),
            collaboration_value=float(
                payload.get(
                    "collaboration_value", self._DEFAULTS["collaboration_value"]
                )
            ),
            user_patience=float(
                payload.get("user_patience", self._DEFAULTS["user_patience"])
            ),
            dynamic_goal=True,  # AutoGPT is always goal-directed
        )
        return task, warnings

    def _translate_openai_swarm(
        self, payload: dict[str, Any]
    ) -> tuple[BrainTask, list[str]]:
        """Translate an OpenAI Swarm / Assistants message payload."""
        warnings: list[str] = []
        messages: list[dict[str, Any]] = payload.get("messages", [])
        # Extract the last user message as the prompt
        user_messages = [
            m for m in messages if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user_messages:
            content = user_messages[-1].get("content", "")
            prompt = str(content) if content else ""
        else:
            prompt = ""
            if messages:
                warnings.append(
                    "OpenAI Swarm payload has no user-role message; "
                    "prompt set to empty string"
                )

        if not prompt:
            warnings.append("OpenAI Swarm payload has no resolvable prompt text")

        task = BrainTask(
            prompt=prompt,
            domain=str(payload.get("domain", self._DEFAULTS["domain"])),
            uncertainty=float(payload.get("uncertainty", self._DEFAULTS["uncertainty"])),
            complexity=float(payload.get("complexity", self._DEFAULTS["complexity"])),
            stakes=float(payload.get("stakes", self._DEFAULTS["stakes"])),
            source_confidence=float(
                payload.get("source_confidence", self._DEFAULTS["source_confidence"])
            ),
            tool_relevance=float(
                payload.get("tool_relevance", self._DEFAULTS["tool_relevance"])
            ),
            time_pressure=float(
                payload.get("time_pressure", self._DEFAULTS["time_pressure"])
            ),
            safety_sensitivity=float(
                payload.get("safety_sensitivity", self._DEFAULTS["safety_sensitivity"])
            ),
            collaboration_value=float(
                payload.get(
                    "collaboration_value", self._DEFAULTS["collaboration_value"]
                )
            ),
            user_patience=float(
                payload.get("user_patience", self._DEFAULTS["user_patience"])
            ),
            dynamic_goal=bool(payload.get("dynamic_goal", False)),
        )
        return task, warnings

    def _translate_generic(
        self, payload: dict[str, Any]
    ) -> tuple[BrainTask, list[str]]:
        """Translate a generic JSON payload using best-effort heuristics."""
        warnings: list[str] = []
        # Try common text fields in priority order
        for key in ("prompt", "text", "input", "content", "query", "task"):
            val = payload.get(key)
            if val is not None:
                prompt = str(val)
                break
        else:
            prompt = ""
            warnings.append(
                "Generic payload has no known text field "
                "(prompt/text/input/content/query/task); prompt set to empty"
            )

        task = BrainTask(
            prompt=prompt,
            domain=str(payload.get("domain", self._DEFAULTS["domain"])),
            uncertainty=float(payload.get("uncertainty", self._DEFAULTS["uncertainty"])),
            complexity=float(payload.get("complexity", self._DEFAULTS["complexity"])),
            stakes=float(payload.get("stakes", self._DEFAULTS["stakes"])),
            source_confidence=float(
                payload.get("source_confidence", self._DEFAULTS["source_confidence"])
            ),
            tool_relevance=float(
                payload.get("tool_relevance", self._DEFAULTS["tool_relevance"])
            ),
            time_pressure=float(
                payload.get("time_pressure", self._DEFAULTS["time_pressure"])
            ),
            safety_sensitivity=float(
                payload.get("safety_sensitivity", self._DEFAULTS["safety_sensitivity"])
            ),
            collaboration_value=float(
                payload.get(
                    "collaboration_value", self._DEFAULTS["collaboration_value"]
                )
            ),
            user_patience=float(
                payload.get("user_patience", self._DEFAULTS["user_patience"])
            ),
            dynamic_goal=bool(payload.get("dynamic_goal", False)),
        )
        return task, warnings

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest(self, payload: dict[str, Any]) -> IngressResult:
        """Translate *payload* into a native :class:`~manifold.brain.BrainTask`.

        Parameters
        ----------
        payload:
            Raw JSON object from the foreign framework.

        Returns
        -------
        IngressResult
            The translated task, detected schema, and any warnings.
        """
        schema = self.detect_schema(payload)
        translators = {
            "MANIFOLD": self._translate_manifold,
            "LANGCHAIN": self._translate_langchain,
            "AUTOGPT": self._translate_autogpt,
            "OPENAI_SWARM": self._translate_openai_swarm,
            "GENERIC": self._translate_generic,
        }
        translator = translators.get(schema.name, self._translate_generic)
        task, warnings = translator(payload)
        return IngressResult(
            task=task,
            schema=schema,
            raw_payload=payload,
            translated_at=time.time(),
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# EgressTranslator
# ---------------------------------------------------------------------------


class EgressTranslator:
    """Translates native DecisionReceipt into framework-native JSON.

    MANIFOLD acts as a *silent governance proxy*: downstream frameworks
    receive a response in their native format, unaware that MANIFOLD
    intercepted and governed the decision.

    Example
    -------
    ::

        translator = EgressTranslator()
        result = translator.translate(receipt, target_framework="LANGCHAIN")
        # send result.payload to the LangChain caller
    """

    # ------------------------------------------------------------------ #
    # Per-framework egress formatters
    # ------------------------------------------------------------------ #

    @staticmethod
    def _egress_manifold(receipt: DecisionReceipt) -> dict[str, Any]:
        """Return the native MANIFOLD receipt dict."""
        return receipt.to_dict()

    @staticmethod
    def _egress_langchain(receipt: DecisionReceipt) -> dict[str, Any]:
        """Format receipt as a LangChain tool observation dict."""
        return {
            "tool": "manifold_governance",
            "tool_input": receipt.grid_state_summary,
            "observation": receipt.final_decision,
            "return_values": {
                "output": receipt.final_decision,
                "task_id": receipt.task_id,
                "timestamp": receipt.timestamp,
            },
        }

    @staticmethod
    def _egress_autogpt(receipt: DecisionReceipt) -> dict[str, Any]:
        """Format receipt as an AutoGPT command result dict."""
        return {
            "command": {
                "name": receipt.final_decision,
                "result": "success",
            },
            "task_id": receipt.task_id,
            "timestamp": receipt.timestamp,
            "governance": {
                "policy_hash": receipt.policy_hash,
                "grid_state_summary": receipt.grid_state_summary,
            },
        }

    @staticmethod
    def _egress_openai_swarm(receipt: DecisionReceipt) -> dict[str, Any]:
        """Format receipt as an OpenAI Swarm assistant message dict."""
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": receipt.final_decision,
                    "metadata": {
                        "task_id": receipt.task_id,
                        "timestamp": receipt.timestamp,
                        "policy_hash": receipt.policy_hash,
                    },
                }
            ],
            "agent": "manifold_governance",
        }

    @staticmethod
    def _egress_generic(receipt: DecisionReceipt) -> dict[str, Any]:
        """Format receipt as a minimal generic JSON response."""
        return {
            "decision": receipt.final_decision,
            "task_id": receipt.task_id,
            "timestamp": receipt.timestamp,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def translate(
        self,
        receipt: DecisionReceipt,
        target_framework: str,
    ) -> EgressResult:
        """Translate *receipt* into *target_framework* native JSON.

        Parameters
        ----------
        receipt:
            The native MANIFOLD :class:`~manifold.provenance.DecisionReceipt`.
        target_framework:
            One of ``"MANIFOLD"``, ``"LANGCHAIN"``, ``"AUTOGPT"``,
            ``"OPENAI_SWARM"``, or ``"GENERIC"``.

        Returns
        -------
        EgressResult

        Raises
        ------
        ValueError
            If *target_framework* is not a recognised name.
        """
        if target_framework not in _KNOWN_FRAMEWORKS:
            raise ValueError(
                f"Unknown target framework '{target_framework}'. "
                f"Expected one of {sorted(_KNOWN_FRAMEWORKS)}."
            )
        formatters = {
            "MANIFOLD": self._egress_manifold,
            "LANGCHAIN": self._egress_langchain,
            "AUTOGPT": self._egress_autogpt,
            "OPENAI_SWARM": self._egress_openai_swarm,
            "GENERIC": self._egress_generic,
        }
        formatter = formatters[target_framework]
        payload = formatter(receipt)
        schema = FrameworkSchema(name=target_framework, confidence=1.0)
        return EgressResult(
            payload=payload,
            schema=schema,
            translated_at=time.time(),
        )
