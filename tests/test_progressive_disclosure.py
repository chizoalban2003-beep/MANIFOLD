"""tests/test_progressive_disclosure.py — 6 tests for GovernanceReporter.generate_escalation_message."""
from __future__ import annotations

import json

import pytest

from manifold.governance_reporter import GovernanceReporter


ESCALATION_BASE = {
    "action": "execute_trade",
    "domain": "finance",
    "risk_score": 0.82,
    "agent_id": "agent-finance-001",
    "agent_name": "Finance Bot",
    "vault_id": "vault-abc-123",
    "crna_values": {"c": 0.4, "r": 0.7, "n": 0.5, "a": 0.6},
    "policy_rule_id": "rule-42",
}

HEALTHCARE_ESC = {
    "action": "dispense_meds",
    "domain": "healthcare",
    "risk_score": 0.65,
    "agent_id": "pharma-robot-1",
    "agent_name": "Pharmacy Robot",
    "medication": "Amoxicillin",
    "dose": "500mg",
    "patient_room": "Room 7B",
    "standing_order": "SO-2024-001",
}

FINANCE_ESC = {
    "action": "execute_trade",
    "domain": "finance",
    "risk_score": 0.55,
    "agent_id": "trader-bot-1",
    "agent_name": "Trader Bot",
    "instrument": "AAPL",
    "notional_value": "$1,000,000",
    "desk": "Equities",
    "regulatory_flag": True,
}


reporter = GovernanceReporter()


# 1. developer message contains "risk_score" and is valid JSON-parseable fragment
def test_developer_message_contains_risk_score():
    msg = reporter.generate_escalation_message(ESCALATION_BASE, "developer")
    assert "risk_score" in msg
    # Should be valid JSON
    parsed = json.loads(msg)
    assert "risk_score" in parsed
    assert parsed["risk_score"] == pytest.approx(0.82, abs=0.01)


# 2. executive message is < 100 words and contains no raw floats (like 0.82)
def test_executive_message_is_short_and_no_raw_floats():
    msg = reporter.generate_escalation_message(ESCALATION_BASE, "executive")
    words = msg.split()
    assert len(words) < 100, f"Too many words: {len(words)}"
    # Should not contain the raw float 0.82 — only risk level words
    assert "0.82" not in msg


# 3. doctor message contains clinical vocabulary for healthcare domain
def test_doctor_message_clinical_vocabulary():
    msg = reporter.generate_escalation_message(HEALTHCARE_ESC, "doctor")
    # Should contain medication or dispense language
    assert any(kw in msg.lower() for kw in ("medication", "dispense", "clinical", "patient", "dose"))


# 4. trader message contains "notional" or "instrument" for finance domain
def test_trader_message_contains_financial_vocabulary():
    msg = reporter.generate_escalation_message(FINANCE_ESC, "trader")
    assert any(kw in msg.lower() for kw in ("notional", "instrument", "trade", "percentile"))


# 5. non_technical message is < 30 words
def test_non_technical_message_is_short():
    msg = reporter.generate_escalation_message(ESCALATION_BASE, "non_technical")
    words = msg.split()
    assert len(words) < 30, f"Too many words: {len(words)}"


# 6. All user types return non-empty string for the same escalation
def test_all_user_types_return_non_empty():
    user_types = ["developer", "executive", "doctor", "lawyer", "trader", "non_technical"]
    for ut in user_types:
        msg = reporter.generate_escalation_message(ESCALATION_BASE, ut)
        assert isinstance(msg, str) and len(msg) > 0, f"Empty message for user_type={ut}"
