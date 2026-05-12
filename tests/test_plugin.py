"""Tests for manifold/plugin.py — ManifoldPlugin universal interface."""

import pytest

from manifold.plugin import ManifoldPlugin, PluginRegistry
from manifold.plugins.roomba_plugin import RoombaPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConcretePlugin(ManifoldPlugin):
    """Minimal concrete plugin for testing."""

    def manifest(self) -> dict:
        return {
            "agent_id": "test-agent",
            "display_name": "Test Agent",
            "version": "0.1.0",
            "capabilities": ["test"],
            "domain": "testing",
            "layer": "digital/pipeline",
            "crna_profile": {"c": 0.1, "r": 0.1, "n": 0.1, "a": 0.1},
            "input_schema": ["run"],
            "output_schema": ["result"],
        }

    def on_command(self, command: str, payload: dict) -> bool:
        return command == "run"

    def get_state(self) -> dict:
        return {
            "position": None,
            "health": 1.0,
            "status": "active",
            "battery": None,
        }


# ---------------------------------------------------------------------------
# Test 1: ManifoldPlugin cannot be instantiated directly (abstract)
# ---------------------------------------------------------------------------

def test_manifold_plugin_is_abstract():
    """ManifoldPlugin is an ABC and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ManifoldPlugin()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Test 2: Concrete plugin instantiates and manifest() returns required keys
# ---------------------------------------------------------------------------

def test_concrete_plugin_instantiates():
    """A concrete plugin implementing all methods should instantiate."""
    plugin = _ConcretePlugin()
    m = plugin.manifest()
    required_keys = {
        "agent_id", "display_name", "version", "capabilities",
        "domain", "layer", "crna_profile", "input_schema", "output_schema",
    }
    assert required_keys <= set(m), f"Missing keys: {required_keys - set(m)}"
    state = plugin.get_state()
    assert set(state) >= {"position", "health", "status", "battery"}


# ---------------------------------------------------------------------------
# Test 3: PluginRegistry.register() returns an agent_id string
# ---------------------------------------------------------------------------

def test_plugin_registry_register_returns_agent_id():
    """PluginRegistry.register() returns the agent_id from the manifest."""
    plugin = _ConcretePlugin()
    registry = PluginRegistry()
    agent_id = registry.register(plugin)
    assert isinstance(agent_id, str)
    assert agent_id == "test-agent"


# ---------------------------------------------------------------------------
# Test 4: on_command() is called when MANIFOLD queues a command
# ---------------------------------------------------------------------------

def test_on_command_dispatched_correctly():
    """on_command() should return True for handled commands and False otherwise."""
    plugin = _ConcretePlugin()
    assert plugin.on_command("run", {}) is True
    assert plugin.on_command("unknown", {}) is False


# ---------------------------------------------------------------------------
# Test 5: get_state() returns a dict with the required keys
# ---------------------------------------------------------------------------

def test_get_state_returns_required_keys():
    """get_state() must return a dict containing position, health, status, battery."""
    plugin = RoombaPlugin(robot_id="roomba-test")
    state = plugin.get_state()
    assert "position" in state
    assert "health" in state
    assert "status" in state
    assert "battery" in state
    assert 0.0 <= state["health"] <= 1.0
    assert state["status"] in ("active", "paused", "offline", "error", "idle")
