"""manifold/plugin.py — Universal plugin interface for MANIFOLD.

Any hardware device or digital AI agent implements ManifoldPlugin to instantly
become a governed MANIFOLD citizen.  Zero external dependencies.

Usage::

    from manifold.plugin import ManifoldPlugin, PluginRegistry

    class MyCamera(ManifoldPlugin):
        def manifest(self) -> dict:
            return {
                "agent_id": "cam-01",
                "display_name": "Front Door Camera",
                "version": "1.0.0",
                "capabilities": ["detect", "stream"],
                "domain": "security",
                "layer": "physical/device",
                "crna_profile": {"c": 0.1, "r": 0.2, "n": 0.5, "a": 0.8},
                "input_schema": ["start", "stop", "snapshot"],
                "output_schema": ["detection", "frame"],
            }

        def on_command(self, command: str, payload: dict) -> bool:
            ...

        def get_state(self) -> dict:
            return {"position": None, "health": 1.0, "status": "active", "battery": None}

    registry = PluginRegistry()
    registry.register(MyCamera())
    registry.start_all()
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable

from manifold.agent_registry import AgentRegistry

# Permitted layer values
VALID_LAYERS = frozenset({
    "physical/floor",
    "physical/aerial",
    "physical/device",
    "digital/llm",
    "digital/pipeline",
    "digital/rpa",
    "digital/sensor",
})


class ManifoldPlugin(ABC):
    """Abstract base class for all MANIFOLD plugins.

    Subclass and implement all three abstract methods to register any
    physical or digital device with MANIFOLD.
    """

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every plugin
    # ------------------------------------------------------------------

    @abstractmethod
    def manifest(self) -> dict:
        """Return the plugin manifest.

        Required keys
        -------------
        agent_id : str
            Unique identifier for this agent.
        display_name : str
            Human-readable name.
        version : str
            Semantic version string.
        capabilities : list[str]
            What this agent can do (e.g. ``["vacuum", "map"]``).
        domain : str
            Business or physical domain (e.g. ``"home"``).
        layer : str
            One of: ``physical/floor``, ``physical/aerial``,
            ``physical/device``, ``digital/llm``,
            ``digital/pipeline``, ``digital/rpa``, ``digital/sensor``.
        crna_profile : dict
            Default CRNA values for this device type.
            Keys: ``c``, ``r``, ``n``, ``a`` (all floats 0–1).
        input_schema : list[str]
            Command names this plugin accepts.
        output_schema : list[str]
            Event names this plugin emits.
        """

    @abstractmethod
    def on_command(self, command: str, payload: dict) -> bool:
        """Receive and execute a governance command from MANIFOLD.

        Parameters
        ----------
        command:
            Command name (must appear in ``manifest()["input_schema"]``).
        payload:
            Arbitrary command-specific data.

        Returns
        -------
        bool
            ``True`` if the command was handled, ``False`` otherwise.
        """

    @abstractmethod
    def get_state(self) -> dict:
        """Return the current state of the plugin.

        Required keys in the returned dict
        ------------------------------------
        position : dict | None
            Current 3-D position, e.g. ``{"x": 1, "y": 2, "z": 0}``.
        health : float
            Health score 0–1.
        status : str
            One of ``active``, ``paused``, ``offline``, ``error``.
        battery : float | None
            Battery percentage 0–100, or ``None`` if not applicable.
        """

    # ------------------------------------------------------------------
    # Optional hooks — override as needed
    # ------------------------------------------------------------------

    def on_connect(self) -> bool:
        """Called after the plugin is successfully registered with MANIFOLD.

        Returns True on success (default no-op returns True).
        """
        return True

    def on_disconnect(self) -> None:
        """Called on graceful shutdown (default no-op)."""

    def stream_events(self, callback: Callable[[dict], None]) -> None:
        """For sensor plugins that push events continuously.

        Parameters
        ----------
        callback:
            Function to call with each event dict.

        Default implementation is a no-op; override for streaming sensors.
        """


class PluginRegistry:
    """Registry that bridges ManifoldPlugin instances with AgentRegistry.

    Parameters
    ----------
    agent_registry:
        An existing :class:`~manifold.agent_registry.AgentRegistry` to
        delegate to.  A new one is created if not provided.
    """

    def __init__(self, agent_registry: AgentRegistry | None = None) -> None:
        self._agent_registry = agent_registry or AgentRegistry()
        self._plugins: list[ManifoldPlugin] = []

    # ------------------------------------------------------------------

    def register(self, plugin: ManifoldPlugin) -> str:
        """Register a plugin with MANIFOLD.

        Calls ``plugin.manifest()``, then delegates to
        :meth:`~manifold.agent_registry.AgentRegistry.register`.

        Returns
        -------
        str
            The ``agent_id`` from the plugin's manifest.
        """
        m = plugin.manifest()
        agent_id: str = m["agent_id"]
        self._agent_registry.register(
            agent_id=agent_id,
            display_name=m.get("display_name", agent_id),
            capabilities=m.get("capabilities", []),
            org_id=m.get("org_id", "default"),
            endpoint_url=m.get("endpoint_url", ""),
            domain=m.get("domain", "general"),
            notes=m.get("version", ""),
        )
        self._plugins.append(plugin)
        return agent_id

    def start_all(self) -> None:
        """Call ``on_connect()`` for every registered plugin.

        Also starts ``stream_events()`` for each plugin in a background
        daemon thread.
        """
        for plugin in self._plugins:
            plugin.on_connect()
            t = threading.Thread(
                target=plugin.stream_events,
                args=(lambda evt: None,),
                daemon=True,
                name=f"plugin-stream-{plugin.manifest()['agent_id']}",
            )
            t.start()

    def stop_all(self) -> None:
        """Call ``on_disconnect()`` for every registered plugin."""
        for plugin in self._plugins:
            plugin.on_disconnect()
