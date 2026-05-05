"""Tests for Phase 60: Swarm MapReduce (manifold/mapreduce.py)."""

from __future__ import annotations

import pytest

from manifold.b2b import AgentEconomyLedger
from manifold.clearing import ClearingEngine
from manifold.mapreduce import (
    Chunk,
    ChunkDistributor,
    ChunkResult,
    JobResult,
    JobState,
    JobTracker,
    MapReduceJob,
    SwarmAggregator,
)
from manifold.sandbox import ASTValidator, BudgetedExecutor
from manifold.sharding import ShardRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine() -> ClearingEngine:
    return ClearingEngine(ledger=AgentEconomyLedger())


def _make_executor() -> BudgetedExecutor:
    return BudgetedExecutor(max_instructions=50_000, validator=ASTValidator())


def _make_tracker(shard_router: ShardRouter | None = None) -> JobTracker:
    return JobTracker(
        shard_router=shard_router or ShardRouter("test"),
        clearing_engine=_make_engine(),
        executor=_make_executor(),
    )


DOUBLE_MAP = "def map_item(item):\n    return item * 2\n"
SUM_REDUCE = "def reduce_results(results):\n    return sum(results)\n"
UPPER_MAP = "def map_item(item):\n    return item.upper()\n"
CONCAT_REDUCE = "def reduce_results(results):\n    return ''.join(results)\n"


def _make_job(
    dataset: tuple = (1, 2, 3, 4, 5),
    map_func: str = DOUBLE_MAP,
    reduce_func: str = SUM_REDUCE,
    chunk_size: int = 3,
    job_id: str = "test-job",
) -> MapReduceJob:
    return MapReduceJob(
        job_id=job_id,
        dataset=dataset,
        map_func=map_func,
        reduce_func=reduce_func,
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# MapReduceJob
# ---------------------------------------------------------------------------


class TestMapReduceJob:
    def test_defaults(self) -> None:
        job = MapReduceJob(
            job_id="j1",
            dataset=(1, 2, 3),
            map_func=DOUBLE_MAP,
            reduce_func=SUM_REDUCE,
        )
        assert job.chunk_size == 100
        assert job.timeout_seconds == 30.0
        assert job.max_retries == 2
        assert job.submitter_org_id == "local"

    def test_frozen(self) -> None:
        job = _make_job()
        with pytest.raises((AttributeError, TypeError)):
            job.job_id = "other"  # type: ignore[misc]

    def test_chunk_count_exact(self) -> None:
        job = MapReduceJob(
            job_id="j", dataset=tuple(range(10)), map_func=DOUBLE_MAP,
            reduce_func=SUM_REDUCE, chunk_size=5,
        )
        assert job.chunk_count() == 2

    def test_chunk_count_remainder(self) -> None:
        job = MapReduceJob(
            job_id="j", dataset=tuple(range(11)), map_func=DOUBLE_MAP,
            reduce_func=SUM_REDUCE, chunk_size=5,
        )
        assert job.chunk_count() == 3

    def test_chunk_count_empty(self) -> None:
        job = MapReduceJob(
            job_id="j", dataset=(), map_func=DOUBLE_MAP, reduce_func=SUM_REDUCE,
        )
        assert job.chunk_count() == 0

    def test_chunk_count_single(self) -> None:
        job = MapReduceJob(
            job_id="j", dataset=(42,), map_func=DOUBLE_MAP, reduce_func=SUM_REDUCE,
            chunk_size=100,
        )
        assert job.chunk_count() == 1


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_creation(self) -> None:
        chunk = Chunk(
            chunk_id="job/chunk/0",
            job_id="job",
            index=0,
            items=(1, 2, 3),
        )
        assert chunk.status == "pending"
        assert chunk.retry_count == 0

    def test_is_timed_out_when_running_and_expired(self) -> None:
        import time
        chunk = Chunk("id", "job", 0, (1,), status="running", assigned_at=time.time() - 100)
        assert chunk.is_timed_out(timeout_seconds=30.0)

    def test_is_not_timed_out_when_pending(self) -> None:
        import time
        chunk = Chunk("id", "job", 0, (1,), status="pending", assigned_at=time.time() - 100)
        assert not chunk.is_timed_out(timeout_seconds=30.0)

    def test_is_not_timed_out_when_fresh(self) -> None:
        import time
        chunk = Chunk("id", "job", 0, (1,), status="running", assigned_at=time.time())
        assert not chunk.is_timed_out(timeout_seconds=30.0)


# ---------------------------------------------------------------------------
# ChunkResult
# ---------------------------------------------------------------------------


class TestChunkResult:
    def test_to_dict(self) -> None:
        result = ChunkResult(
            chunk_id="j/0",
            peer_org_id="peer-a",
            success=True,
            output=(2, 4, 6),
            error="",
            instructions_used=100,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == [2, 4, 6]
        assert d["error"] == ""

    def test_frozen(self) -> None:
        r = ChunkResult("j/0", "peer", True, (), "", 0)
        with pytest.raises((AttributeError, TypeError)):
            r.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ChunkDistributor
# ---------------------------------------------------------------------------


class TestChunkDistributor:
    def test_distributes_correct_count(self) -> None:
        job = _make_job(dataset=tuple(range(10)), chunk_size=3)
        dist = ChunkDistributor(shard_router=ShardRouter("dist"))
        chunks = dist.distribute(job)
        # ceil(10 / 3) = 4
        assert len(chunks) == 4

    def test_chunk_ids_are_unique(self) -> None:
        job = _make_job(dataset=tuple(range(9)), chunk_size=3)
        dist = ChunkDistributor(shard_router=ShardRouter("dist"))
        chunks = dist.distribute(job)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_items_cover_full_dataset(self) -> None:
        data = tuple(range(7))
        job = _make_job(dataset=data, chunk_size=3)
        dist = ChunkDistributor(shard_router=ShardRouter("dist"))
        chunks = dist.distribute(job)
        all_items: tuple = ()
        for c in chunks:
            all_items += c.items
        assert sorted(all_items) == sorted(data)

    def test_fallback_to_local_when_no_peers(self) -> None:
        job = _make_job(dataset=(1, 2, 3), chunk_size=3)
        dist = ChunkDistributor(shard_router=ShardRouter("dist"), local_peer_id="local-node")
        chunks = dist.distribute(job)
        assert chunks[0].assigned_peer == "local-node"

    def test_assigns_peer_from_dht(self) -> None:
        router = ShardRouter("dist")
        router.add_peer("peer-a", "http://peer-a:8080")
        job = _make_job(dataset=(1, 2, 3), chunk_size=3)
        dist = ChunkDistributor(shard_router=router, local_peer_id="local")
        chunks = dist.distribute(job)
        # Should have a non-local peer assignment
        assert chunks[0].assigned_peer != ""

    def test_empty_dataset_produces_no_chunks(self) -> None:
        job = MapReduceJob(
            job_id="empty", dataset=(), map_func=DOUBLE_MAP, reduce_func=SUM_REDUCE,
        )
        dist = ChunkDistributor(shard_router=ShardRouter("dist"))
        chunks = dist.distribute(job)
        assert chunks == []

    def test_chunk_indices_are_sequential(self) -> None:
        job = _make_job(dataset=tuple(range(9)), chunk_size=3)
        dist = ChunkDistributor(shard_router=ShardRouter("dist"))
        chunks = dist.distribute(job)
        assert [c.index for c in chunks] == [0, 1, 2]

    def test_last_chunk_has_remainder_items(self) -> None:
        job = _make_job(dataset=tuple(range(7)), chunk_size=3)
        dist = ChunkDistributor(shard_router=ShardRouter("dist"))
        chunks = dist.distribute(job)
        assert len(chunks[-1].items) == 1  # 7 % 3 == 1


# ---------------------------------------------------------------------------
# SwarmAggregator — map_chunk
# ---------------------------------------------------------------------------


class TestSwarmAggregatorMap:
    def test_map_doubles(self) -> None:
        agg = SwarmAggregator(executor=_make_executor())
        job = _make_job(dataset=(1, 2, 3), chunk_size=3)
        chunk = Chunk("j/0", "j", 0, (1, 2, 3), assigned_peer="local")
        result = agg.map_chunk(chunk, job)
        assert result.success
        assert result.output == (2, 4, 6)
        assert result.error == ""

    def test_map_strings(self) -> None:
        agg = SwarmAggregator(executor=_make_executor())
        job = MapReduceJob(
            job_id="s", dataset=("hello", "world"),
            map_func=UPPER_MAP, reduce_func=CONCAT_REDUCE, chunk_size=2,
        )
        chunk = Chunk("s/0", "s", 0, ("hello", "world"), assigned_peer="local")
        result = agg.map_chunk(chunk, job)
        assert result.success
        assert result.output == ("HELLO", "WORLD")

    def test_map_invalid_code_returns_failure(self) -> None:
        bad_map = "import os\ndef map_item(item):\n    return os.getcwd()\n"
        job = MapReduceJob(
            job_id="bad", dataset=(1,), map_func=bad_map, reduce_func=SUM_REDUCE,
        )
        agg = SwarmAggregator(executor=_make_executor())
        chunk = Chunk("bad/0", "bad", 0, (1,), assigned_peer="local")
        result = agg.map_chunk(chunk, job)
        assert not result.success
        assert "ASTValidator" in result.error

    def test_map_chunk_id_preserved(self) -> None:
        agg = SwarmAggregator(executor=_make_executor())
        job = _make_job(dataset=(5,), chunk_size=1)
        chunk = Chunk("specific-id", "j", 0, (5,), assigned_peer="peer-x")
        result = agg.map_chunk(chunk, job)
        assert result.chunk_id == "specific-id"
        assert result.peer_org_id == "peer-x"

    def test_map_runtime_error_returns_failure(self) -> None:
        error_map = "def map_item(item):\n    return 1 / 0\n"
        agg = SwarmAggregator(executor=_make_executor())
        job = MapReduceJob(
            job_id="err", dataset=(1,), map_func=error_map, reduce_func=SUM_REDUCE,
        )
        chunk = Chunk("err/0", "err", 0, (1,), assigned_peer="local")
        result = agg.map_chunk(chunk, job)
        assert not result.success


# ---------------------------------------------------------------------------
# SwarmAggregator — reduce
# ---------------------------------------------------------------------------


class TestSwarmAggregatorReduce:
    def test_reduce_sum(self) -> None:
        agg = SwarmAggregator(executor=_make_executor())
        job = _make_job()
        success, output, err = agg.reduce(job, [2, 4, 6, 8, 10])
        assert success
        assert output == 30
        assert err == ""

    def test_reduce_concat(self) -> None:
        agg = SwarmAggregator(executor=_make_executor())
        job = MapReduceJob(
            job_id="c", dataset=("a", "b"),
            map_func=UPPER_MAP, reduce_func=CONCAT_REDUCE,
        )
        success, output, err = agg.reduce(job, ["A", "B", "C"])
        assert success
        assert output == "ABC"

    def test_reduce_invalid_code(self) -> None:
        bad_reduce = "import sys\ndef reduce_results(r):\n    return sys.exit(0)\n"
        agg = SwarmAggregator(executor=_make_executor())
        job = MapReduceJob(
            job_id="bad", dataset=(1,), map_func=DOUBLE_MAP, reduce_func=bad_reduce,
        )
        success, output, err = agg.reduce(job, [1])
        assert not success
        assert "ASTValidator" in err

    def test_reduce_empty_list(self) -> None:
        agg = SwarmAggregator(executor=_make_executor())
        job = _make_job()
        success, output, _ = agg.reduce(job, [])
        assert success
        assert output == 0


# ---------------------------------------------------------------------------
# JobTracker — submit
# ---------------------------------------------------------------------------


class TestJobTrackerSubmit:
    def test_submit_registers_job(self) -> None:
        tracker = _make_tracker()
        job = _make_job()
        job_id = tracker.submit(job)
        assert job_id == job.job_id
        assert tracker.active_job_count() == 1

    def test_submit_empty_dataset_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(ValueError, match="empty"):
            tracker.submit(MapReduceJob(
                job_id="empty", dataset=(), map_func=DOUBLE_MAP, reduce_func=SUM_REDUCE,
            ))

    def test_submit_invalid_map_func_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(ValueError, match="map_func"):
            tracker.submit(MapReduceJob(
                job_id="bad", dataset=(1,),
                map_func="import os\ndef map_item(x): return x",
                reduce_func=SUM_REDUCE,
            ))

    def test_submit_invalid_reduce_func_raises(self) -> None:
        tracker = _make_tracker()
        with pytest.raises(ValueError, match="reduce_func"):
            tracker.submit(MapReduceJob(
                job_id="bad", dataset=(1,),
                map_func=DOUBLE_MAP,
                reduce_func="import os\ndef reduce_results(r): return r",
            ))

    def test_submit_creates_chunks(self) -> None:
        tracker = _make_tracker()
        job = _make_job(dataset=tuple(range(10)), chunk_size=3)
        tracker.submit(job)
        state = tracker.get_state(job.job_id)
        assert state is not None
        assert len(state.chunks) == 4  # ceil(10/3)


# ---------------------------------------------------------------------------
# JobTracker — run (end-to-end)
# ---------------------------------------------------------------------------


class TestJobTrackerRun:
    def test_run_double_and_sum(self) -> None:
        tracker = _make_tracker()
        job = _make_job(dataset=(1, 2, 3, 4, 5), chunk_size=3)
        result = tracker.run(job)
        assert result.success
        assert result.reduce_output == 30  # sum of [2,4,6,8,10]
        assert result.job_id == job.job_id

    def test_run_single_item(self) -> None:
        tracker = _make_tracker()
        job = MapReduceJob(
            job_id="single", dataset=(7,),
            map_func=DOUBLE_MAP, reduce_func=SUM_REDUCE,
        )
        result = tracker.run(job)
        assert result.success
        assert result.reduce_output == 14

    def test_run_string_dataset(self) -> None:
        tracker = _make_tracker()
        job = MapReduceJob(
            job_id="str", dataset=("hello", "world"),
            map_func=UPPER_MAP, reduce_func=CONCAT_REDUCE,
            chunk_size=1,
        )
        result = tracker.run(job)
        assert result.success
        assert set(result.reduce_output) == set("HELLOWORLD")

    def test_run_mints_tokens(self) -> None:
        engine = _make_engine()
        tracker = JobTracker(
            shard_router=ShardRouter("t"),
            clearing_engine=engine,
            executor=_make_executor(),
            local_peer_id="local-node",
        )
        job = _make_job(dataset=(1, 2, 3), chunk_size=3)
        result = tracker.run(job)
        assert result.success
        assert result.tokens_minted > 0
        # Local node should have been rewarded
        assert engine.trust_balance("local-node") > 0

    def test_run_chunks_total_matches(self) -> None:
        tracker = _make_tracker()
        job = _make_job(dataset=tuple(range(10)), chunk_size=3)
        result = tracker.run(job)
        assert result.chunks_total == 4

    def test_run_all_chunks_succeeded(self) -> None:
        tracker = _make_tracker()
        job = _make_job(dataset=(1, 2, 3, 4, 5), chunk_size=2)
        result = tracker.run(job)
        assert result.chunks_succeeded == 3
        assert result.chunks_failed == 0

    def test_run_duration_positive(self) -> None:
        tracker = _make_tracker()
        job = _make_job()
        result = tracker.run(job)
        assert result.duration_seconds >= 0.0

    def test_job_result_to_dict(self) -> None:
        tracker = _make_tracker()
        job = _make_job(dataset=(1, 2), chunk_size=2)
        result = tracker.run(job)
        d = result.to_dict()
        assert "job_id" in d
        assert "reduce_output" in d
        assert "chunks_total" in d

    def test_failed_map_func_marks_chunks_failed(self) -> None:
        tracker = _make_tracker()
        error_map = "def map_item(item):\n    x = None\n    return x.attribute\n"
        job = MapReduceJob(
            job_id="fail-map", dataset=(1, 2),
            map_func=error_map, reduce_func=SUM_REDUCE,
            chunk_size=2, max_retries=0,
        )
        result = tracker.run(job)
        assert not result.success

    def test_run_large_dataset(self) -> None:
        tracker = _make_tracker()
        data = tuple(range(1, 51))  # 50 items
        job = MapReduceJob(
            job_id="large", dataset=data,
            map_func=DOUBLE_MAP, reduce_func=SUM_REDUCE,
            chunk_size=10,
        )
        result = tracker.run(job)
        assert result.success
        expected = sum(x * 2 for x in data)
        assert result.reduce_output == expected

    def test_summary_updates(self) -> None:
        tracker = _make_tracker()
        tracker.run(_make_job(job_id="s1"))
        tracker.run(_make_job(job_id="s2"))
        summary = tracker.summary()
        assert summary["total_jobs"] == 2


# ---------------------------------------------------------------------------
# JobState
# ---------------------------------------------------------------------------


class TestJobState:
    def test_all_done_when_terminal(self) -> None:
        job = _make_job(dataset=(1,), chunk_size=1)
        state = JobState(job=job, chunks=[
            Chunk("j/0", "j", 0, (1,), status="done"),
        ])
        assert state.all_done()

    def test_not_all_done_when_running(self) -> None:
        job = _make_job(dataset=(1,), chunk_size=1)
        state = JobState(job=job, chunks=[
            Chunk("j/0", "j", 0, (1,), status="running"),
        ])
        assert not state.all_done()

    def test_pending_or_running_returns_non_terminal(self) -> None:
        job = _make_job(dataset=(1, 2), chunk_size=1)
        state = JobState(job=job, chunks=[
            Chunk("j/0", "j", 0, (1,), status="done"),
            Chunk("j/1", "j", 1, (2,), status="pending"),
        ])
        pending = state.pending_or_running_chunks()
        assert len(pending) == 1
        assert pending[0].chunk_id == "j/1"
