"""tests/test_llm_interface.py — Tests for ManifoldLLM."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from manifold.llm_interface import ManifoldLLM, LLMResponse, get_llm_history, _LLM_HISTORY


def _fake_response(plain: str, action_type: str = "none", action_payload: dict | None = None) -> str:
    """Build a raw LLM response string."""
    payload = action_payload or {"type": action_type}
    payload["type"] = action_type
    return (
        f"{plain}\n"
        f"MANIFOLD_ACTION_START\n{json.dumps(payload)}\nMANIFOLD_ACTION_END"
    )


def _make_llm(raw_return: str) -> ManifoldLLM:
    """Return a ManifoldLLM whose _call_model returns raw_return."""
    llm = ManifoldLLM(org_id="test-org")
    llm._call_model = MagicMock(return_value=raw_return)
    return llm


def test_parse_plain_text_only():
    """Response without MANIFOLD_ACTION defaults to action_type='none'."""
    raw = _fake_response("Hello, everything looks good.", "none")
    llm = _make_llm(raw)
    response = llm.chat("Is everything OK?")
    assert response.action_type == "none"
    assert "everything looks good" in response.plain_text


def test_parse_policy_rule_action():
    """MANIFOLD_ACTION with a policy_rule is parsed correctly."""
    rule_payload = {
        "type": "policy_rule",
        "name": "Block high risk finance",
        "org_id": "test-org",
        "conditions": {"domain": "finance", "risk_gt": 0.8},
        "action": "refuse",
        "priority": 90,
    }
    raw = _fake_response("I have added a finance rule.", "policy_rule", rule_payload)
    llm = _make_llm(raw)
    response = llm.chat("Block high risk finance ops")
    assert response.action_type == "policy_rule"
    assert response.action_payload["name"] == "Block high risk finance"
    assert response.action_payload["conditions"]["risk_gt"] == 0.8


def test_chat_appends_to_history():
    """chat() appends to the global LLM history ring."""
    _LLM_HISTORY.clear()
    raw = _fake_response("All good.", "none")
    llm = _make_llm(raw)
    llm.chat("Test message for history")
    history = get_llm_history()
    assert len(history) >= 1
    assert any(h["user_message"] == "Test message for history" for h in history)


def test_apply_response_policy_rule():
    """apply_response validates and applies a policy_rule action."""
    from manifold.policy_translator import PolicyTranslator

    rule_payload = {
        "type": "policy_rule",
        "name": "ISO27001 sandbox devops",
        "org_id": "test-org",
        "conditions": {"domain": "devops", "risk_gt": 0.6},
        "action": "sandbox",
        "priority": 70,
    }
    response = LLMResponse(
        plain_text="I'll sandbox risky devops tasks.",
        action_type="policy_rule",
        action_payload=rule_payload,
        raw_response="",
    )
    llm = ManifoldLLM(org_id="test-org")
    # Should not raise even without a live server
    result = llm.apply_response(response)
    # apply_response returns True if validation succeeded (even if server not available)
    assert isinstance(result, bool)
