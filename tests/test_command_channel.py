"""Tests for bidirectional agent command channel."""
from __future__ import annotations

import json
import time
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from manifold.agent_registry import AgentRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> AgentRegistry:
    reg = AgentRegistry(stale_timeout=120)
    reg.register(
        agent_id="test-agent",
        display_name="Test Agent",
        capabilities=["billing"],
        org_id="org1",
        endpoint_url="",
    )
    return reg


# ---------------------------------------------------------------------------
# AgentRegistry unit tests
# ---------------------------------------------------------------------------


def test_queue_command_returns_string_id() -> None:
    """queue_command() returns a non-empty string command id."""
    reg = _make_registry()
    cmd_id = reg.queue_command("test-agent", "pause")
    assert isinstance(cmd_id, str)
    assert len(cmd_id) > 0


def test_poll_commands_returns_queued_command_and_clears_queue() -> None:
    """poll_commands() returns the queued command and clears the queue (consume=True)."""
    reg = _make_registry()
    reg.queue_command("test-agent", "pause", {"reason": "maintenance"})
    cmds = reg.poll_commands("test-agent", consume=True)
    assert len(cmds) == 1
    assert cmds[0]["command"] == "pause"
    assert cmds[0]["payload"] == {"reason": "maintenance"}
    # Queue should be empty now
    assert reg.poll_commands("test-agent", consume=False) == []


def test_poll_commands_consume_false_does_not_clear_queue() -> None:
    """poll_commands(consume=False) returns command without clearing."""
    reg = _make_registry()
    reg.queue_command("test-agent", "resume")
    cmds1 = reg.poll_commands("test-agent", consume=False)
    cmds2 = reg.poll_commands("test-agent", consume=False)
    assert len(cmds1) == 1
    assert len(cmds2) == 1
    assert cmds1[0]["command"] == "resume"


def test_queue_command_unknown_agent_returns_none() -> None:
    """queue_command() for an unknown agent_id returns None."""
    reg = _make_registry()
    result = reg.queue_command("nonexistent-agent", "pause")
    assert result is None


def test_multiple_commands_queue_in_fifo_order() -> None:
    """Multiple commands queue and are returned in FIFO order."""
    reg = _make_registry()
    reg.queue_command("test-agent", "pause")
    reg.queue_command("test-agent", "message", {"text": "hello"})
    reg.queue_command("test-agent", "resume")
    cmds = reg.poll_commands("test-agent", consume=True)
    assert len(cmds) == 3
    assert cmds[0]["command"] == "pause"
    assert cmds[1]["command"] == "message"
    assert cmds[2]["command"] == "resume"


# ---------------------------------------------------------------------------
# Server endpoint tests
# ---------------------------------------------------------------------------


import manifold.server as server_mod


def _call_server_handler(func_name: str, *args, **kwargs) -> dict:
    """Call a server handler function on a mock handler, capture JSON response."""
    handler = MagicMock()
    captured = {}

    def fake_send_json(h, status, data):
        captured["status"] = status
        captured["data"] = data

    def fake_send_error(h, status, msg):
        captured["status"] = status
        captured["data"] = {"error": msg}

    with (
        patch.object(server_mod, "_send_json", side_effect=fake_send_json),
        patch.object(server_mod, "_send_error", side_effect=fake_send_error),
    ):
        func = getattr(server_mod, func_name)
        func(handler, *args, **kwargs)

    return captured


def _register_agent_in_server(agent_id: str = "srv-agent") -> None:
    """Register an agent directly in the server-level _AGENT_REGISTRY."""
    server_mod._AGENT_REGISTRY.register(
        agent_id=agent_id,
        display_name="Srv Agent",
        capabilities=["code"],
        org_id="org1",
        endpoint_url="",
    )


def test_post_agent_command_returns_201_with_command_id() -> None:
    """POST /agents/{id}/command returns 201 with command_id."""
    _register_agent_in_server("cmd-agent-1")
    result = _call_server_handler(
        "_handle_post_agent_command",
        "cmd-agent-1",
        {"command": "pause", "payload": {}},
    )
    assert result["status"] == 201
    assert "command_id" in result["data"]
    assert result["data"]["status"] == "queued"


def test_post_agent_command_invalid_command_returns_400() -> None:
    """POST /agents/{id}/command with invalid command returns 400."""
    _register_agent_in_server("cmd-agent-2")
    result = _call_server_handler(
        "_handle_post_agent_command",
        "cmd-agent-2",
        {"command": "self-destruct", "payload": {}},
    )
    assert result["status"] == 400


def test_get_agent_commands_returns_queued_commands_immediately() -> None:
    """GET /agents/{id}/commands returns queued commands without waiting."""
    _register_agent_in_server("cmd-agent-3")
    # Pre-queue a command directly in the registry
    server_mod._AGENT_REGISTRY.queue_command("cmd-agent-3", "message", {"text": "hi"})

    # Override time so the long-poll exits immediately after finding commands
    result = _call_server_handler("_handle_get_agent_commands", "cmd-agent-3")
    assert result["status"] == 200
    assert len(result["data"]["commands"]) >= 1
    assert result["data"]["commands"][0]["command"] == "message"
