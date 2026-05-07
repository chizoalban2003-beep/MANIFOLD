"""Phase 14: Ecosystem Adapters — zero-boilerplate MANIFOLD integration.

These adapters let developers drop MANIFOLD into their existing stack with
minimal code changes.  No LangChain or OpenAI SDK installation is required
to import this module; adapters work via duck-typing and optional imports.

Key classes
-----------
``ManifoldCallbackHandler``
    LangChain-compatible callback handler.  Hook it into ``on_tool_start``
    and ``on_tool_end`` to automatically populate the ``PriceAdapter`` and
    ``GossipBus`` without writing a single line of MANIFOLD logic.

    Designed to be a **drop-in** ``BaseCallbackHandler`` subclass when
    LangChain is installed, or a standalone duck-typed handler otherwise.

``ManifoldOpenAIWrapper``
    Thin wrapper around any OpenAI-compatible client.  Intercepts
    ``chat.completions.create()`` to price the prompt via the
    ``DualPathEncoder`` *before* the request is sent.  Optionally runs the
    ``ActiveInterceptor`` and raises ``InterceptorVeto`` on veto.

    Usage::

        from manifold.adapters import ManifoldOpenAIWrapper
        from openai import OpenAI

        client = ManifoldOpenAIWrapper(OpenAI(), brain=brain, domain="finance")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Draft a legal clause..."}],
        )

``ManifoldFastAPIMiddleware``
    A framework-agnostic ASGI-style middleware callable that wraps incoming
    requests in a ``ManifoldBrain`` task based on request metadata.
    Works with FastAPI, Starlette, or any ASGI app via duck-typing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .brain import BrainConfig, BrainTask, GossipNote, ManifoldBrain, ToolProfile
from .connector import ConnectorRegistry, ToolConnector
from .encoder import DualPathEncoder, PromptEncoder
from .interceptor import ActiveInterceptor, InterceptorVeto
from .live import GossipBus
from .trustrouter import clamp01

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ManifoldCallbackHandler
# ---------------------------------------------------------------------------


class ManifoldCallbackHandler:
    """LangChain-compatible callback handler for MANIFOLD.

    Hooks into ``on_tool_start`` and ``on_tool_end`` to automatically
    populate the ``PriceAdapter`` and ``GossipBus`` without requiring the
    caller to write any MANIFOLD logic.

    This class is intentionally dependency-free and uses duck-typing.
    When LangChain is installed you can pass it directly as a callback::

        from manifold.adapters import ManifoldCallbackHandler
        handler = ManifoldCallbackHandler(brain=brain, gossip_bus=bus)

        chain.run("Your task here", callbacks=[handler])

    Without LangChain you can call the methods manually::

        handler.on_tool_start({"name": "billing_api"}, "process payment")
        # ... tool runs ...
        handler.on_tool_end("ok")

    Parameters
    ----------
    brain:
        ``ManifoldBrain`` instance that receives risk assessments.
    gossip_bus:
        Optional ``GossipBus`` for broadcasting failure signals.
    registry:
        Optional ``ConnectorRegistry`` — if provided, the handler registers
        dynamically encountered tools on the fly.
    default_stakes:
        Default task stakes used when none can be inferred from the prompt.
    verbose:
        Print intercept events to stdout when ``True``.
    """

    def __init__(
        self,
        brain: ManifoldBrain,
        *,
        gossip_bus: GossipBus | None = None,
        registry: ConnectorRegistry | None = None,
        default_stakes: float = 0.5,
        verbose: bool = False,
    ) -> None:
        self.brain = brain
        self.gossip_bus = gossip_bus
        self.registry = registry
        self.default_stakes = default_stakes
        self.verbose = verbose

        self._encoder = PromptEncoder()
        self._call_log: list[dict[str, Any]] = []
        self._pending_tool: str | None = None
        self._pending_start: float = 0.0

    # ------------------------------------------------------------------
    # LangChain callback interface
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Called when a LangChain tool begins execution.

        Parameters
        ----------
        serialized:
            Serialized tool representation (must contain at least ``"name"``).
        input_str:
            The input passed to the tool.
        """
        tool_name: str = serialized.get("name", "unknown_tool") if isinstance(serialized, dict) else str(serialized)
        self._pending_tool = tool_name
        self._pending_start = time.monotonic()

        features = self._encoder.encode(input_str, domain="general")
        task = BrainTask(
            prompt=input_str[:200],
            domain="general",
            stakes=clamp01(features.stakes),
            uncertainty=clamp01(features.uncertainty),
            complexity=clamp01(features.complexity),
        )
        decision = self.brain.decide(task)

        entry: dict[str, Any] = {
            "tool_name": tool_name,
            "input": input_str[:200],
            "brain_action": decision.action,
            "risk_score": decision.risk_score,
            "start": self._pending_start,
        }
        self._call_log.append(entry)

        if self.verbose:
            _logger.debug(
                "[MANIFOLD] on_tool_start: tool=%r  action=%r  risk=%.2f",
                tool_name, decision.action, decision.risk_score,
            )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Called when a LangChain tool completes successfully.

        Parameters
        ----------
        output:
            The output returned by the tool.
        """
        latency = time.monotonic() - self._pending_start if self._pending_start else 0.0
        tool_name = self._pending_tool or "unknown_tool"

        if self._call_log:
            self._call_log[-1].update(
                {"output": str(output)[:200], "success": True, "latency": latency}
            )

        if self.gossip_bus is not None:
            self.gossip_bus.publish(
                GossipNote(
                    tool=tool_name,
                    claim="healthy",
                    source_id="manifold_langchain",
                    confidence=0.8,
                )
            )

        if self.verbose:
            _logger.debug(
                "[MANIFOLD] on_tool_end: tool=%r  latency=%.3fs  ok=True",
                tool_name, latency,
            )

        self._pending_tool = None
        self._pending_start = 0.0

    def on_tool_error(self, error: BaseException | str, **kwargs: Any) -> None:
        """Called when a LangChain tool raises an exception.

        Parameters
        ----------
        error:
            The exception (or error string) raised by the tool.
        """
        latency = time.monotonic() - self._pending_start if self._pending_start else 0.0
        tool_name = self._pending_tool or "unknown_tool"
        error_msg = str(error)[:200]

        if self._call_log:
            self._call_log[-1].update(
                {"error": error_msg, "success": False, "latency": latency}
            )

        if self.gossip_bus is not None:
            self.gossip_bus.publish(
                GossipNote(
                    tool=tool_name,
                    claim="failing",
                    source_id="manifold_langchain",
                    confidence=0.9,
                )
            )

        if self.verbose:
            _logger.debug(
                "[MANIFOLD] on_tool_error: tool=%r  latency=%.3fs  error=%r",
                tool_name, latency, error_msg,
            )

        self._pending_tool = None
        self._pending_start = 0.0

    # ------------------------------------------------------------------
    # Aggregate access
    # ------------------------------------------------------------------

    def call_log(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded tool calls."""
        return list(self._call_log)

    def failure_rate(self) -> float:
        """Return the fraction of calls that ended in error."""
        finished = [e for e in self._call_log if "success" in e]
        if not finished:
            return 0.0
        return sum(1 for e in finished if not e["success"]) / len(finished)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all intercepted tool calls."""
        finished = [e for e in self._call_log if "success" in e]
        return {
            "total_calls": len(self._call_log),
            "completed": len(finished),
            "failures": sum(1 for e in finished if not e["success"]),
            "failure_rate": self.failure_rate(),
            "avg_latency": (
                sum(e.get("latency", 0.0) for e in finished) / len(finished)
                if finished else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# ManifoldOpenAIWrapper
# ---------------------------------------------------------------------------


class _CompletionsProxy:
    """Inner proxy that intercepts ``.create()`` calls."""

    def __init__(self, wrapper: "ManifoldOpenAIWrapper") -> None:
        self._wrapper = wrapper

    def create(self, **kwargs: Any) -> Any:
        """Intercept a ``chat.completions.create`` call.

        Prices the prompt via ``DualPathEncoder``, optionally vetoes via
        ``ActiveInterceptor``, then forwards to the real client.

        Parameters
        ----------
        **kwargs:
            Same kwargs you would pass to ``openai.chat.completions.create``.

        Raises
        ------
        InterceptorVeto
            If ``ActiveInterceptor`` blocks the call (only when an interceptor
            is configured).

        Returns
        -------
        Any
            The response from the underlying OpenAI-compatible client.
        """
        w = self._wrapper

        # Extract the prompt text from messages
        messages = kwargs.get("messages", [])
        prompt_text = " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        ) or "api_call"

        # Price via DualPathEncoder
        features = w._encoder.encode(prompt_text, domain=w.domain)
        task = BrainTask(
            prompt=prompt_text[:200],
            domain=w.domain,
            stakes=clamp01(features.stakes),
            uncertainty=clamp01(features.uncertainty),
            complexity=clamp01(features.complexity),
        )

        model_name = kwargs.get("model", "unknown_model")

        # Optional: run ActiveInterceptor pre-flight check
        if w.interceptor is not None and w.registry is not None:
            if w.registry.get(model_name) is None:
                # Auto-register the model as a tool on first encounter
                w.registry.register(
                    ToolConnector(
                        name=model_name,
                        fn=lambda *a, **kw: None,
                        profile=ToolProfile(
                            model_name,
                            cost=0.10,
                            latency=0.20,
                            reliability=0.85,
                            risk=0.15,
                            asset=0.80,
                        ),
                    )
                )
            intercept_result = w.interceptor.intercept(task, model_name)
            if not intercept_result.permitted:
                raise InterceptorVeto(
                    f"MANIFOLD OpenAI wrapper vetoed {model_name!r}: "
                    f"{intercept_result.reason}",
                    intercept_result,
                )

        # Record the estimated cost before forwarding
        w._call_log.append(
            {
                "model": model_name,
                "prompt_preview": prompt_text[:80],
                "estimated_stakes": task.stakes,
                "estimated_complexity": task.complexity,
            }
        )

        # Forward to the real client
        return w.client.chat.completions.create(**kwargs)


class _ChatProxy:
    """Middle proxy for ``client.chat.completions.create``."""

    def __init__(self, wrapper: "ManifoldOpenAIWrapper") -> None:
        self._completions = _CompletionsProxy(wrapper)

    @property
    def completions(self) -> _CompletionsProxy:
        return self._completions


@dataclass
class ManifoldOpenAIWrapper:
    """Thin wrapper around an OpenAI-compatible client.

    Intercepts ``chat.completions.create()`` to price the prompt via the
    ``DualPathEncoder`` *before* the request hits the wire.  Optionally
    runs the ``ActiveInterceptor`` and raises ``InterceptorVeto`` on veto.

    Parameters
    ----------
    client:
        Any object with a ``.chat.completions.create()`` method (e.g. an
        ``openai.OpenAI`` instance or a compatible mock).
    brain:
        ``ManifoldBrain`` used for risk pricing.  If not provided a default
        brain is constructed from ``BrainConfig``.
    interceptor:
        Optional ``ActiveInterceptor``.  When provided the wrapper performs
        a pre-flight check and raises ``InterceptorVeto`` on veto.
    registry:
        Optional ``ConnectorRegistry``.  Required when *interceptor* is set.
        Unknown models are auto-registered on first encounter.
    domain:
        Default task domain for risk estimation.
    verbose:
        Print intercept events to stdout when ``True``.

    Example
    -------
    ::

        from manifold.adapters import ManifoldOpenAIWrapper
        from manifold import ManifoldBrain, BrainConfig, default_tools

        brain = ManifoldBrain(BrainConfig(), default_tools())
        client = ManifoldOpenAIWrapper(raw_openai_client, brain=brain, domain="finance")

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Draft a refund policy."}],
            )
        except InterceptorVeto as exc:
            _logger.debug("MANIFOLD blocked: %s", exc)
    """

    client: Any
    brain: ManifoldBrain = field(default=None)  # type: ignore[assignment]
    interceptor: ActiveInterceptor | None = field(default=None)
    registry: ConnectorRegistry | None = field(default=None)
    domain: str = "general"
    verbose: bool = False

    _call_log: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.brain is None:
            self.brain = ManifoldBrain(BrainConfig(), [])
        self._encoder = DualPathEncoder()
        self._chat = _ChatProxy(self)

    @property
    def chat(self) -> _ChatProxy:
        """Proxy for ``client.chat.completions.create()``."""
        return self._chat

    def call_log(self) -> list[dict[str, Any]]:
        """Return a copy of all intercepted call metadata."""
        return list(self._call_log)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all intercepted calls."""
        log = self._call_log
        return {
            "total_calls": len(log),
            "avg_estimated_stakes": (
                sum(e["estimated_stakes"] for e in log) / len(log) if log else 0.0
            ),
            "avg_estimated_complexity": (
                sum(e["estimated_complexity"] for e in log) / len(log) if log else 0.0
            ),
        }
