"""Tests for Phase 64: Rosetta Protocol Adapter (manifold/rosetta.py)."""

from __future__ import annotations

import time

import pytest

from manifold.brain import BrainTask
from manifold.provenance import DecisionReceipt
from manifold.rosetta import (
    EgressResult,
    EgressTranslator,
    ForeignPayloadIngress,
    FrameworkSchema,
    IngressResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_receipt(
    final_decision: str = "use_tool",
    task_id: str = "tid-001",
) -> DecisionReceipt:
    return DecisionReceipt(
        timestamp=1700000000.0,
        task_id=task_id,
        grid_state_summary={"action": final_decision, "risk_score": 0.2, "confidence": 0.9},
        braintrust_votes=(),
        policy_hash="abc123",
        final_decision=final_decision,
        previous_hash="genesis",
    )


# ---------------------------------------------------------------------------
# FrameworkSchema
# ---------------------------------------------------------------------------


class TestFrameworkSchema:
    def test_valid_name(self) -> None:
        s = FrameworkSchema(name="LANGCHAIN", confidence=0.9)
        assert s.name == "LANGCHAIN"

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown framework"):
            FrameworkSchema(name="PYTORCH", confidence=0.5)

    def test_frozen(self) -> None:
        s = FrameworkSchema(name="MANIFOLD", confidence=1.0)
        with pytest.raises((AttributeError, TypeError)):
            s.name = "GENERIC"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        s = FrameworkSchema(name="AUTOGPT", confidence=0.8)
        d = s.to_dict()
        assert d["name"] == "AUTOGPT"
        assert d["confidence"] == 0.8

    def test_all_known_frameworks_are_valid(self) -> None:
        for name in ("MANIFOLD", "LANGCHAIN", "AUTOGPT", "OPENAI_SWARM", "GENERIC"):
            s = FrameworkSchema(name=name, confidence=0.5)
            assert s.name == name


# ---------------------------------------------------------------------------
# ForeignPayloadIngress — schema detection
# ---------------------------------------------------------------------------


class TestSchemaDetection:
    def setup_method(self) -> None:
        self.ingress = ForeignPayloadIngress()

    def test_detect_manifold_native(self) -> None:
        payload = {"prompt": "test", "domain": "finance", "uncertainty": 0.5}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "MANIFOLD"
        assert schema.confidence >= 0.9

    def test_detect_langchain_tool_and_input(self) -> None:
        payload = {"tool": "calculator", "tool_input": "2+2"}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "LANGCHAIN"

    def test_detect_langchain_input_and_tool(self) -> None:
        payload = {"input": "refund request", "tool": "billing_api"}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "LANGCHAIN"

    def test_detect_autogpt_command(self) -> None:
        payload = {"command": {"name": "web_search", "args": {"task": "find flights"}}}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "AUTOGPT"

    def test_detect_autogpt_minimal(self) -> None:
        payload = {"command": {"name": "do_task"}}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "AUTOGPT"

    def test_detect_openai_swarm_messages(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4"}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "OPENAI_SWARM"

    def test_detect_openai_swarm_model_agent(self) -> None:
        payload = {"model": "gpt-4o", "agent": "assistant"}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "OPENAI_SWARM"

    def test_detect_generic_prompt_key(self) -> None:
        payload = {"prompt": "hello there"}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "GENERIC"

    def test_detect_generic_text_key(self) -> None:
        payload = {"text": "some text"}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "GENERIC"

    def test_detect_generic_input_only(self) -> None:
        payload = {"input": "some question"}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "GENERIC"

    def test_detect_generic_unknown_payload(self) -> None:
        payload = {"foo": "bar", "baz": 42}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "GENERIC"
        assert schema.confidence <= 0.4

    def test_manifold_takes_priority_over_generic(self) -> None:
        payload = {"prompt": "x", "domain": "test", "uncertainty": 0.5}
        schema = self.ingress.detect_schema(payload)
        assert schema.name == "MANIFOLD"


# ---------------------------------------------------------------------------
# ForeignPayloadIngress — ingest
# ---------------------------------------------------------------------------


class TestIngress:
    def setup_method(self) -> None:
        self.ingress = ForeignPayloadIngress()

    def test_ingest_manifold_payload(self) -> None:
        payload = {
            "prompt": "check user balance",
            "domain": "finance",
            "uncertainty": 0.3,
            "complexity": 0.6,
            "stakes": 0.7,
        }
        result = self.ingress.ingest(payload)
        assert isinstance(result, IngressResult)
        assert result.schema.name == "MANIFOLD"
        assert result.task.prompt == "check user balance"
        assert result.task.domain == "finance"
        assert abs(result.task.uncertainty - 0.3) < 1e-9
        assert abs(result.task.stakes - 0.7) < 1e-9

    def test_ingest_langchain_tool_input(self) -> None:
        payload = {"tool": "search", "tool_input": "best coffee shops"}
        result = self.ingress.ingest(payload)
        assert result.schema.name == "LANGCHAIN"
        assert result.task.prompt == "best coffee shops"
        assert result.task.tool_relevance == 1.0

    def test_ingest_langchain_tool_input_dict(self) -> None:
        payload = {"tool": "calculator", "tool_input": {"query": "2+2"}}
        result = self.ingress.ingest(payload)
        assert result.schema.name == "LANGCHAIN"
        assert "2+2" in result.task.prompt

    def test_ingest_langchain_no_tool_input_uses_input(self) -> None:
        payload = {"input": "process invoice", "tool": "billing"}
        result = self.ingress.ingest(payload)
        assert result.schema.name == "LANGCHAIN"
        assert result.task.prompt == "process invoice"

    def test_ingest_autogpt_task_arg(self) -> None:
        payload = {
            "command": {
                "name": "web_search",
                "args": {"task": "find best Python frameworks"},
            }
        }
        result = self.ingress.ingest(payload)
        assert result.schema.name == "AUTOGPT"
        assert "Python frameworks" in result.task.prompt
        assert result.task.dynamic_goal is True
        assert result.task.tool_relevance == 0.9

    def test_ingest_autogpt_query_arg(self) -> None:
        payload = {"command": {"name": "search", "args": {"query": "AI news"}}}
        result = self.ingress.ingest(payload)
        assert result.schema.name == "AUTOGPT"
        assert result.task.prompt == "AI news"

    def test_ingest_autogpt_no_args_uses_command_name(self) -> None:
        payload = {"command": {"name": "do_something"}}
        result = self.ingress.ingest(payload)
        assert result.schema.name == "AUTOGPT"
        assert result.task.prompt == "do_something"

    def test_ingest_openai_swarm_user_message(self) -> None:
        payload = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Refund my subscription"},
            ],
            "model": "gpt-4",
        }
        result = self.ingress.ingest(payload)
        assert result.schema.name == "OPENAI_SWARM"
        assert result.task.prompt == "Refund my subscription"

    def test_ingest_openai_swarm_last_user_message(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ],
        }
        result = self.ingress.ingest(payload)
        assert result.task.prompt == "second"

    def test_ingest_openai_swarm_no_user_message_warns(self) -> None:
        payload = {
            "messages": [{"role": "system", "content": "be helpful"}],
        }
        result = self.ingress.ingest(payload)
        assert result.schema.name == "OPENAI_SWARM"
        assert any("no user-role" in w for w in result.warnings)

    def test_ingest_generic_prompt(self) -> None:
        payload = {"prompt": "help me debug this"}
        result = self.ingress.ingest(payload)
        assert result.task.prompt == "help me debug this"

    def test_ingest_generic_text(self) -> None:
        payload = {"text": "write a report"}
        result = self.ingress.ingest(payload)
        assert result.task.prompt == "write a report"

    def test_ingest_generic_content(self) -> None:
        payload = {"content": "explain quantum computing"}
        result = self.ingress.ingest(payload)
        assert result.task.prompt == "explain quantum computing"

    def test_ingest_generic_empty_warns(self) -> None:
        payload = {"unknown_key": 42}
        result = self.ingress.ingest(payload)
        assert result.task.prompt == ""
        assert len(result.warnings) > 0

    def test_ingest_preserves_raw_payload(self) -> None:
        payload = {"prompt": "x", "domain": "test", "uncertainty": 0.5}
        result = self.ingress.ingest(payload)
        assert result.raw_payload is payload

    def test_ingest_translated_at_is_recent(self) -> None:
        before = time.time()
        result = self.ingress.ingest({"prompt": "hello", "domain": "general", "uncertainty": 0.5})
        after = time.time()
        assert before <= result.translated_at <= after

    def test_ingest_result_to_dict(self) -> None:
        result = self.ingress.ingest(
            {"prompt": "test", "domain": "general", "uncertainty": 0.5}
        )
        d = result.to_dict()
        for key in ("schema", "translated_at", "warnings", "task"):
            assert key in d

    def test_ingest_defaults_are_valid_brain_task(self) -> None:
        """Even minimal payloads must produce a valid BrainTask."""
        result = self.ingress.ingest({"prompt": "hi"})
        task = result.task
        assert isinstance(task, BrainTask)
        assert 0.0 <= task.uncertainty <= 1.0
        assert 0.0 <= task.complexity <= 1.0
        assert 0.0 <= task.stakes <= 1.0

    def test_ingest_manifold_empty_prompt_warns(self) -> None:
        result = self.ingress.ingest({"prompt": "", "domain": "general", "uncertainty": 0.5})
        assert any("empty" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# EgressTranslator
# ---------------------------------------------------------------------------


class TestEgressTranslator:
    def setup_method(self) -> None:
        self.translator = EgressTranslator()
        self.receipt = _make_receipt()

    def test_translate_manifold(self) -> None:
        result = self.translator.translate(self.receipt, "MANIFOLD")
        assert isinstance(result, EgressResult)
        assert result.schema.name == "MANIFOLD"
        assert result.payload["final_decision"] == "use_tool"

    def test_translate_langchain(self) -> None:
        result = self.translator.translate(self.receipt, "LANGCHAIN")
        assert result.schema.name == "LANGCHAIN"
        payload = result.payload
        assert "tool" in payload
        assert payload["tool"] == "manifold_governance"
        assert "observation" in payload
        assert payload["observation"] == "use_tool"
        assert "return_values" in payload
        assert payload["return_values"]["output"] == "use_tool"

    def test_translate_autogpt(self) -> None:
        result = self.translator.translate(self.receipt, "AUTOGPT")
        assert result.schema.name == "AUTOGPT"
        payload = result.payload
        assert "command" in payload
        assert payload["command"]["name"] == "use_tool"
        assert "task_id" in payload

    def test_translate_openai_swarm(self) -> None:
        result = self.translator.translate(self.receipt, "OPENAI_SWARM")
        assert result.schema.name == "OPENAI_SWARM"
        payload = result.payload
        assert "messages" in payload
        assert payload["messages"][0]["role"] == "assistant"
        assert payload["messages"][0]["content"] == "use_tool"

    def test_translate_generic(self) -> None:
        result = self.translator.translate(self.receipt, "GENERIC")
        assert result.schema.name == "GENERIC"
        payload = result.payload
        assert payload["decision"] == "use_tool"
        assert "task_id" in payload

    def test_translate_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown target framework"):
            self.translator.translate(self.receipt, "PYTORCH")

    def test_translate_all_frameworks_produce_dict(self) -> None:
        for fw in ("MANIFOLD", "LANGCHAIN", "AUTOGPT", "OPENAI_SWARM", "GENERIC"):
            result = self.translator.translate(self.receipt, fw)
            assert isinstance(result.payload, dict)
            assert len(result.payload) > 0

    def test_translated_at_is_recent(self) -> None:
        before = time.time()
        result = self.translator.translate(self.receipt, "GENERIC")
        after = time.time()
        assert before <= result.translated_at <= after

    def test_egress_result_to_dict(self) -> None:
        result = self.translator.translate(self.receipt, "LANGCHAIN")
        d = result.to_dict()
        for key in ("schema", "translated_at", "payload"):
            assert key in d

    def test_confidence_is_1_for_egress(self) -> None:
        result = self.translator.translate(self.receipt, "MANIFOLD")
        assert result.schema.confidence == 1.0

    def test_langchain_task_id_in_return_values(self) -> None:
        receipt = _make_receipt(task_id="my-task-id")
        result = self.translator.translate(receipt, "LANGCHAIN")
        assert result.payload["return_values"]["task_id"] == "my-task-id"

    def test_autogpt_timestamp(self) -> None:
        result = self.translator.translate(self.receipt, "AUTOGPT")
        assert result.payload["timestamp"] == 1700000000.0

    def test_openai_swarm_policy_hash_in_metadata(self) -> None:
        result = self.translator.translate(self.receipt, "OPENAI_SWARM")
        meta = result.payload["messages"][0]["metadata"]
        assert meta["policy_hash"] == "abc123"

    def test_different_decisions_produce_different_outputs(self) -> None:
        r1 = _make_receipt("escalate")
        r2 = _make_receipt("refuse")
        out1 = self.translator.translate(r1, "GENERIC")
        out2 = self.translator.translate(r2, "GENERIC")
        assert out1.payload["decision"] == "escalate"
        assert out2.payload["decision"] == "refuse"


# ---------------------------------------------------------------------------
# Integration: ingress → decision → egress round-trip
# ---------------------------------------------------------------------------


class TestRosettaRoundTrip:
    def test_langchain_to_manifold_to_langchain(self) -> None:
        ingress = ForeignPayloadIngress()
        translator = EgressTranslator()

        # Simulate incoming LangChain payload
        lc_payload = {
            "tool": "customer_db",
            "tool_input": "lookup invoice INV-9001",
        }
        ingress_result = ingress.ingest(lc_payload)
        assert ingress_result.schema.name == "LANGCHAIN"
        assert "INV-9001" in ingress_result.task.prompt

        # Simulate MANIFOLD decision → DecisionReceipt
        receipt = _make_receipt("use_tool", task_id="inv-task")

        # Translate back to LangChain
        egress_result = translator.translate(receipt, "LANGCHAIN")
        assert egress_result.payload["observation"] == "use_tool"

    def test_autogpt_to_manifold_to_autogpt(self) -> None:
        ingress = ForeignPayloadIngress()
        translator = EgressTranslator()

        ag_payload = {
            "command": {"name": "browse_web", "args": {"task": "find MANIFOLD docs"}}
        }
        ingress_result = ingress.ingest(ag_payload)
        assert "MANIFOLD docs" in ingress_result.task.prompt

        receipt = _make_receipt("retrieve")
        egress = translator.translate(receipt, "AUTOGPT")
        assert egress.payload["command"]["name"] == "retrieve"
