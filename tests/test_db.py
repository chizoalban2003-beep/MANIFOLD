"""Tests for ManifoldDB — async SQLite persistence layer."""

from __future__ import annotations

import time

import pytest
import pytest_asyncio

from manifold.db import ManifoldDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db():
    """Yield an in-memory ManifoldDB, connected and cleaned up."""
    instance = ManifoldDB(":memory:")
    await instance.connect()
    yield instance
    await instance.close()


# ---------------------------------------------------------------------------
# save_task_outcome / get_domain_stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_retrieve_task_outcome():
    db = ManifoldDB(":memory:")
    await db.connect()
    await db.save_task_outcome(
        task=None,
        action="escalate",
        outcome={"resolved": True, "latency_ms": 120},
        task_hash="abc123",
        domain="billing",
        stakes=0.85,
    )
    stats = await db.get_domain_stats("billing")
    assert stats["total_tasks"] == 1
    assert stats["escalation_rate"] == pytest.approx(1.0)
    await db.close()


@pytest.mark.asyncio
async def test_domain_stats_multiple_actions(db: ManifoldDB):
    for action in ("escalate", "escalate", "refuse", "answer"):
        await db.save_task_outcome(
            task=None,
            action=action,
            outcome={},
            domain="finance",
            stakes=0.7,
        )
    stats = await db.get_domain_stats("finance")
    assert stats["total_tasks"] == 4
    assert stats["escalation_rate"] == pytest.approx(0.5)
    assert stats["refusal_rate"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_domain_stats_empty_domain(db: ManifoldDB):
    stats = await db.get_domain_stats("nonexistent")
    assert stats["total_tasks"] == 0
    assert stats["escalation_rate"] == 0.0
    assert stats["refusal_rate"] == 0.0


@pytest.mark.asyncio
async def test_domain_stats_cost_risk_asset(db: ManifoldDB):
    await db.save_task_outcome(
        task=None,
        action="answer",
        outcome={"cost": 0.2, "risk": 0.1, "asset": 0.8},
        domain="support",
    )
    stats = await db.get_domain_stats("support")
    assert stats["avg_cost"] == pytest.approx(0.2)
    assert stats["avg_risk"] == pytest.approx(0.1)
    assert stats["avg_asset"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager():
    async with ManifoldDB(":memory:") as db:
        await db.save_task_outcome(
            task=None, action="answer", outcome={}, domain="general"
        )
        stats = await db.get_domain_stats("general")
    assert stats["total_tasks"] == 1


# ---------------------------------------------------------------------------
# save_tool_reputation / get_tool_history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_get_tool_reputation(db: ManifoldDB):
    now = time.time()
    await db.save_tool_reputation("gpt-4o", 0.92, timestamp=now)
    history = await db.get_tool_history("gpt-4o")
    assert len(history) == 1
    assert history[0]["tool_name"] == "gpt-4o"
    assert history[0]["score"] == pytest.approx(0.92)
    assert history[0]["recorded_at"] == pytest.approx(now)


@pytest.mark.asyncio
async def test_tool_history_respects_limit(db: ManifoldDB):
    base_ts = time.time()
    for i in range(10):
        await db.save_tool_reputation("tool-x", 0.5 + i * 0.01, timestamp=base_ts + i)
    history = await db.get_tool_history("tool-x", limit=5)
    assert len(history) == 5


@pytest.mark.asyncio
async def test_tool_history_empty(db: ManifoldDB):
    history = await db.get_tool_history("unknown-tool")
    assert history == []


@pytest.mark.asyncio
async def test_tool_history_ordered_most_recent_first(db: ManifoldDB):
    base = time.time()
    for i in range(3):
        await db.save_tool_reputation("claude-3", 0.8 + i * 0.05, timestamp=base + i)
    history = await db.get_tool_history("claude-3")
    recorded_ats = [h["recorded_at"] for h in history]
    assert recorded_ats == sorted(recorded_ats, reverse=True)


# ---------------------------------------------------------------------------
# save_gossip_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_gossip_event(db: ManifoldDB):
    await db.save_gossip_event(
        source_id="org-a",
        target_id="org-b",
        risk_vector=[0.1, 0.2, 0.3, 0.4],
    )
    # Verify via direct SQL
    rows = await db._fetchall("SELECT source_id, target_id, risk_vector FROM gossip_events")
    assert len(rows) == 1
    assert rows[0][0] == "org-a"
    assert rows[0][1] == "org-b"
    import json
    assert json.loads(rows[0][2]) == pytest.approx([0.1, 0.2, 0.3, 0.4])


@pytest.mark.asyncio
async def test_save_multiple_gossip_events(db: ManifoldDB):
    for i in range(3):
        await db.save_gossip_event(f"src-{i}", f"tgt-{i}", [0.1, 0.2, 0.3, 0.4])
    rows = await db._fetchall("SELECT COUNT(*) FROM gossip_events")
    assert int(rows[0][0]) == 3


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_count(db: ManifoldDB):
    assert await db.task_count() == 0
    await db.save_task_outcome(task=None, action="answer", outcome={}, domain="x")
    await db.save_task_outcome(task=None, action="escalate", outcome={}, domain="x")
    assert await db.task_count() == 2


@pytest.mark.asyncio
async def test_escalation_count(db: ManifoldDB):
    for action in ("escalate", "escalate", "answer", "refuse"):
        await db.save_task_outcome(task=None, action=action, outcome={}, domain="y")
    assert await db.escalation_count() == 2


@pytest.mark.asyncio
async def test_refusal_count(db: ManifoldDB):
    for action in ("refuse", "answer", "escalate"):
        await db.save_task_outcome(task=None, action=action, outcome={}, domain="z")
    assert await db.refusal_count() == 1


# ---------------------------------------------------------------------------
# BrainTask integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_task_outcome_with_brain_task(db: ManifoldDB):
    from manifold.brain import BrainTask

    task = BrainTask(
        prompt="Is this safe?",
        domain="medical",
        stakes=0.95,
    )
    await db.save_task_outcome(task=task, action="escalate", outcome={"ok": True})
    stats = await db.get_domain_stats("medical")
    assert stats["total_tasks"] == 1
    assert stats["escalation_rate"] == pytest.approx(1.0)
