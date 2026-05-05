"""Phase 60: Swarm MapReduce — Distributed Compute over the MANIFOLD Network.

Allows the Swarm to process massive datasets collaboratively.  Work is split
into chunks, dispatched to the peer closest (XOR distance) to each chunk's
key via the DHT (:mod:`~manifold.sharding`), executed inside an AST-sandboxed
``BudgetedExecutor``, and economically settled through the
:class:`~manifold.clearing.ClearingEngine`.

Architecture
------------
::

    Dataset → ChunkDistributor → [chunk₀, chunk₁, …, chunkₙ]
                                       │
                            ShardRouter (DHT peer lookup)
                                       │
                              BudgetedExecutor (map_func per chunk)
                                       │
                            SwarmAggregator (reduce_func over results)
                                       │
                              ClearingEngine.mint_for_canary_success
                               (Trust Tokens → contributing peers)

Key classes
-----------
``MapReduceJob``
    Immutable descriptor: dataset, ``map_func`` source, ``reduce_func`` source.
``Chunk``
    One slice of the dataset plus routing metadata.
``ChunkResult``
    The outcome of processing a single chunk (mapped output or error).
``JobState``
    Mutable in-flight tracking state for a running job.
``ChunkDistributor``
    Splits the dataset and assigns chunks to DHT-routed peers.
``SwarmAggregator``
    Collects ``ChunkResult`` objects and runs the final reduce pass.
``JobTracker``
    Orchestrates end-to-end: distribute → (re-)assign on timeout → aggregate.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .clearing import ClearingEngine
from .sandbox import ASTValidator, BudgetedExecutor, SandboxTimeoutError
from .sharding import ShardRouter


# ---------------------------------------------------------------------------
# MapReduceJob
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MapReduceJob:
    """Immutable descriptor for a distributed MapReduce computation.

    Parameters
    ----------
    job_id:
        Unique identifier for this job.
    dataset:
        A list of items to process.  Each item is passed as the sole
        argument to ``map_func``.  Items must be JSON-serialisable.
    map_func:
        Python source for the mapper.  Must define a top-level function
        ``def map_item(item): ...`` that returns a JSON-serialisable value.
        Executed inside :class:`~manifold.sandbox.BudgetedExecutor`.
    reduce_func:
        Python source for the reducer.  Must define a top-level function
        ``def reduce_results(results): ...`` where ``results`` is the
        list of all mapper outputs.  Executed inside the sandbox.
    chunk_size:
        Number of dataset items per chunk.  Default: ``100``.
    timeout_seconds:
        Seconds before a chunk is considered failed and reassigned.
        Default: ``30.0``.
    max_retries:
        Maximum reassignments per chunk before the job is marked failed.
        Default: ``2``.
    submitter_org_id:
        Org ID of the submitting node (used for Trust Token accounting).
    """

    job_id: str
    dataset: tuple[Any, ...]
    map_func: str
    reduce_func: str
    chunk_size: int = 100
    timeout_seconds: float = 30.0
    max_retries: int = 2
    submitter_org_id: str = "local"

    def chunk_count(self) -> int:
        """Return the number of chunks this dataset produces."""
        if not self.dataset:
            return 0
        return (len(self.dataset) + self.chunk_size - 1) // self.chunk_size


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """One slice of a :class:`MapReduceJob` dataset.

    Attributes
    ----------
    chunk_id:
        ``"<job_id>/<index>"`` unique within the job.
    job_id:
        Parent job identifier.
    index:
        Zero-based chunk position.
    items:
        The subset of dataset items assigned to this chunk.
    assigned_peer:
        Peer org ID currently responsible for this chunk (``""`` = unassigned).
    status:
        One of ``"pending"``, ``"running"``, ``"done"``, ``"failed"``.
    assigned_at:
        POSIX timestamp of the most recent assignment.
    retry_count:
        Number of times this chunk has been reassigned.
    """

    chunk_id: str
    job_id: str
    index: int
    items: tuple[Any, ...]
    assigned_peer: str = ""
    status: str = "pending"
    assigned_at: float = 0.0
    retry_count: int = 0

    def is_timed_out(self, timeout_seconds: float) -> bool:
        """Return ``True`` if the chunk has been running past *timeout_seconds*."""
        return (
            self.status == "running"
            and self.assigned_at > 0
            and (time.time() - self.assigned_at) > timeout_seconds
        )


# ---------------------------------------------------------------------------
# ChunkResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkResult:
    """Outcome of processing a single :class:`Chunk`.

    Attributes
    ----------
    chunk_id:
        The ``Chunk.chunk_id`` this result belongs to.
    peer_org_id:
        The peer that produced this result.
    success:
        ``True`` if the map function ran without error.
    output:
        The return value of ``map_func`` applied to the chunk's items
        (one entry per item).  Empty on failure.
    error:
        Human-readable error message on failure.
    instructions_used:
        Opcode budget consumed by the sandboxed execution.
    """

    chunk_id: str
    peer_org_id: str
    success: bool
    output: tuple[Any, ...]
    error: str
    instructions_used: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "chunk_id": self.chunk_id,
            "peer_org_id": self.peer_org_id,
            "success": self.success,
            "output": list(self.output),
            "error": self.error,
            "instructions_used": self.instructions_used,
        }


# ---------------------------------------------------------------------------
# JobResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobResult:
    """Final outcome of a completed :class:`MapReduceJob`.

    Attributes
    ----------
    job_id:
        The parent job.
    success:
        ``True`` if all chunks completed and the reduce pass succeeded.
    reduce_output:
        The return value of ``reduce_func`` over all mapper outputs.
    chunks_total:
        Number of chunks the dataset was split into.
    chunks_succeeded:
        Chunks that completed without error.
    chunks_failed:
        Chunks that failed even after retries.
    tokens_minted:
        Trust Tokens minted across all contributing peers.
    error:
        Human-readable description if ``success`` is ``False``.
    duration_seconds:
        Wall-clock time for the full job.
    """

    job_id: str
    success: bool
    reduce_output: Any
    chunks_total: int
    chunks_succeeded: int
    chunks_failed: int
    tokens_minted: float
    error: str
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "job_id": self.job_id,
            "success": self.success,
            "reduce_output": self.reduce_output,
            "chunks_total": self.chunks_total,
            "chunks_succeeded": self.chunks_succeeded,
            "chunks_failed": self.chunks_failed,
            "tokens_minted": self.tokens_minted,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 4),
        }


# ---------------------------------------------------------------------------
# JobState
# ---------------------------------------------------------------------------


@dataclass
class JobState:
    """Mutable in-flight tracking state for a running :class:`MapReduceJob`.

    Attributes
    ----------
    job:
        The originating :class:`MapReduceJob`.
    chunks:
        Ordered list of :class:`Chunk` objects.
    results:
        ``chunk_id → ChunkResult`` map for completed chunks.
    started_at:
        POSIX timestamp when :meth:`JobTracker.run` was called.
    status:
        ``"running"`` | ``"done"`` | ``"failed"``.
    """

    job: MapReduceJob
    chunks: list[Chunk] = field(default_factory=list)
    results: dict[str, ChunkResult] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    status: str = "running"

    def pending_or_running_chunks(self) -> list[Chunk]:
        """Return chunks that still need processing."""
        return [c for c in self.chunks if c.status in ("pending", "running")]

    def all_done(self) -> bool:
        """``True`` if every chunk is in a terminal state."""
        return all(c.status in ("done", "failed") for c in self.chunks)


# ---------------------------------------------------------------------------
# ChunkDistributor
# ---------------------------------------------------------------------------


@dataclass
class ChunkDistributor:
    """Splits a dataset into chunks and assigns each to a DHT peer.

    Parameters
    ----------
    shard_router:
        The :class:`~manifold.sharding.ShardRouter` used to look up the
        XOR-closest peer for each chunk key.
    local_peer_id:
        Fallback peer org ID when no remote peer is available.

    Example
    -------
    ::

        distributor = ChunkDistributor(shard_router=_SHARD_ROUTER)
        chunks = distributor.distribute(job)
    """

    shard_router: ShardRouter
    local_peer_id: str = "local"

    def distribute(self, job: MapReduceJob) -> list[Chunk]:
        """Split *job* into :class:`Chunk` objects and assign peers.

        For each chunk index ``i`` the DHT key is
        ``"<job_id>/chunk/<i>"`` and the closest peer is resolved via
        :meth:`~manifold.sharding.ShardRouter.shard_for`.  If the routing
        table is empty the local node is used as the fallback peer.

        Parameters
        ----------
        job:
            The :class:`MapReduceJob` to split.

        Returns
        -------
        list[Chunk]
            One :class:`Chunk` per slice, in order.
        """
        chunks: list[Chunk] = []
        data = list(job.dataset)
        for i in range(0, len(data), job.chunk_size):
            idx = i // job.chunk_size
            items = tuple(data[i: i + job.chunk_size])
            chunk_id = f"{job.job_id}/chunk/{idx}"
            dht_key = chunk_id
            peer_endpoint = self.shard_router.shard_for(dht_key)
            # Derive a peer org ID from the endpoint or fall back to local
            if peer_endpoint:
                peer_id = hashlib.sha256(
                    peer_endpoint.encode("utf-8")
                ).hexdigest()[:8]
            else:
                peer_id = self.local_peer_id
            chunk = Chunk(
                chunk_id=chunk_id,
                job_id=job.job_id,
                index=idx,
                items=items,
                assigned_peer=peer_id,
                status="pending",
                assigned_at=0.0,
                retry_count=0,
            )
            chunks.append(chunk)
        return chunks


# ---------------------------------------------------------------------------
# SwarmAggregator
# ---------------------------------------------------------------------------


@dataclass
class SwarmAggregator:
    """Runs sandboxed map and reduce functions over a dataset.

    Each chunk's items are passed individually through ``map_func`` using
    :class:`~manifold.sandbox.BudgetedExecutor`, then all mapped outputs
    are combined by ``reduce_func``.

    Parameters
    ----------
    executor:
        The :class:`~manifold.sandbox.BudgetedExecutor` used to run code.
    validator:
        The :class:`~manifold.sandbox.ASTValidator` used to pre-validate
        code before execution.

    Example
    -------
    ::

        aggregator = SwarmAggregator(executor=_SANDBOX_EXECUTOR)
        result = aggregator.map_chunk(chunk, job)
        final = aggregator.reduce(job, mapped_outputs)
    """

    executor: BudgetedExecutor
    validator: ASTValidator = field(default_factory=ASTValidator)

    def map_chunk(self, chunk: Chunk, job: MapReduceJob) -> ChunkResult:
        """Apply ``job.map_func`` to every item in *chunk*.

        The function is validated with :class:`~manifold.sandbox.ASTValidator`
        once then each item is executed in a fresh sandbox context.  If any
        item fails the whole chunk is marked failed.

        Parameters
        ----------
        chunk:
            The chunk to process.
        job:
            The parent job (provides ``map_func`` source and metadata).

        Returns
        -------
        ChunkResult
        """
        # Validate the map function once
        violations = self.validator.validate(job.map_func)
        if violations:
            return ChunkResult(
                chunk_id=chunk.chunk_id,
                peer_org_id=chunk.assigned_peer,
                success=False,
                output=(),
                error=f"ASTValidator rejected map_func: {violations[0].description}",
                instructions_used=0,
            )

        outputs: list[Any] = []
        total_instructions = 0
        for item in chunk.items:
            # Build a self-contained snippet that calls map_item(item)
            snippet = f"{job.map_func}\n_result = map_item({item!r})\n"
            try:
                exec_result = self.executor.execute(snippet)
                total_instructions += exec_result.instructions_used
                if exec_result.success:
                    outputs.append(exec_result.output_env.get("_result"))
                else:
                    return ChunkResult(
                        chunk_id=chunk.chunk_id,
                        peer_org_id=chunk.assigned_peer,
                        success=False,
                        output=(),
                        error=exec_result.error or "map_func execution failed",
                        instructions_used=total_instructions,
                    )
            except SandboxTimeoutError as exc:
                return ChunkResult(
                    chunk_id=chunk.chunk_id,
                    peer_org_id=chunk.assigned_peer,
                    success=False,
                    output=(),
                    error=f"SandboxTimeoutError: {exc}",
                    instructions_used=exc.instructions_used,
                )

        return ChunkResult(
            chunk_id=chunk.chunk_id,
            peer_org_id=chunk.assigned_peer,
            success=True,
            output=tuple(outputs),
            error="",
            instructions_used=total_instructions,
        )

    def reduce(
        self,
        job: MapReduceJob,
        mapped_outputs: list[Any],
    ) -> tuple[bool, Any, str]:
        """Apply ``job.reduce_func`` to all *mapped_outputs*.

        Parameters
        ----------
        job:
            The parent job (provides ``reduce_func`` source).
        mapped_outputs:
            Flat list of all individual mapper return values.

        Returns
        -------
        tuple[bool, Any, str]
            ``(success, result_value, error_message)``
        """
        violations = self.validator.validate(job.reduce_func)
        if violations:
            return False, None, f"ASTValidator rejected reduce_func: {violations[0].description}"

        snippet = f"{job.reduce_func}\n_result = reduce_results({mapped_outputs!r})\n"
        try:
            exec_result = self.executor.execute(snippet)
            if exec_result.success:
                return True, exec_result.output_env.get("_result"), ""
            return False, None, exec_result.error or "reduce_func execution failed"
        except SandboxTimeoutError as exc:
            return False, None, f"SandboxTimeoutError in reduce: {exc}"


# ---------------------------------------------------------------------------
# JobTracker
# ---------------------------------------------------------------------------


@dataclass
class JobTracker:
    """Orchestrates the lifecycle of a :class:`MapReduceJob`.

    Workflow
    --------
    1. :meth:`submit` — validate and register a new job.
    2. :meth:`run` — distribute, map (with timeout/retry), reduce, settle.
    3. Trust Tokens are minted for each peer that returned results via
       :meth:`~manifold.clearing.ClearingEngine.mint_for_canary_success`.

    Parameters
    ----------
    shard_router:
        DHT used to locate peers for chunks.
    clearing_engine:
        Economy engine for Trust Token minting.
    executor:
        Sandboxed Python executor.
    local_peer_id:
        Fallback peer ID used when no DHT peer is available.
    token_reward_per_chunk:
        Trust Tokens minted per successfully completed chunk.
        Default: ``1.0``.

    Example
    -------
    ::

        tracker = JobTracker(shard_router=_SHARD_ROUTER,
                             clearing_engine=_CLEARING_ENGINE)
        result = tracker.run(job)
        if result.success:
            print("MapReduce output:", result.reduce_output)
    """

    shard_router: ShardRouter
    clearing_engine: ClearingEngine
    executor: BudgetedExecutor = field(default_factory=lambda: BudgetedExecutor(max_instructions=50_000))
    local_peer_id: str = "local"
    token_reward_per_chunk: float = 1.0

    _jobs: dict[str, JobState] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def submit(self, job: MapReduceJob) -> str:
        """Register *job* and return its ``job_id``.

        Parameters
        ----------
        job:
            The job to register.

        Returns
        -------
        str
            The ``job_id`` from the submitted :class:`MapReduceJob`.

        Raises
        ------
        ValueError
            If the job's ``map_func`` or ``reduce_func`` contain AST
            violations, or if the dataset is empty.
        """
        if not job.dataset:
            raise ValueError(f"job_id={job.job_id!r}: dataset must not be empty")

        validator = ASTValidator()
        map_violations = validator.validate(job.map_func)
        if map_violations:
            raise ValueError(
                f"job_id={job.job_id!r}: map_func AST violation: "
                f"{map_violations[0].description}"
            )
        reduce_violations = validator.validate(job.reduce_func)
        if reduce_violations:
            raise ValueError(
                f"job_id={job.job_id!r}: reduce_func AST violation: "
                f"{reduce_violations[0].description}"
            )

        distributor = ChunkDistributor(
            shard_router=self.shard_router,
            local_peer_id=self.local_peer_id,
        )
        chunks = distributor.distribute(job)
        state = JobState(job=job, chunks=chunks, started_at=time.time())
        with self._lock:
            self._jobs[job.job_id] = state
        return job.job_id

    def run(self, job: MapReduceJob) -> JobResult:
        """Submit *job* (if not already submitted) and execute it synchronously.

        Chunks are processed locally using the :class:`SwarmAggregator`.
        Failed chunks are retried up to ``job.max_retries`` times.  After all
        chunks finish (or fail), the reduce pass runs over the successful
        outputs.  Participating peers receive Trust Token rewards.

        Parameters
        ----------
        job:
            The job to execute.

        Returns
        -------
        JobResult
        """
        t_start = time.monotonic()

        # Register if not already known
        with self._lock:
            if job.job_id not in self._jobs:
                self.submit(job)
            state = self._jobs[job.job_id]

        aggregator = SwarmAggregator(
            executor=self.executor,
            validator=ASTValidator(),
        )

        # Process each chunk (with timeout/retry simulation)
        for chunk in state.chunks:
            self._process_chunk_with_retry(chunk, job, state, aggregator)

        # Gather successful mapped outputs
        mapped_outputs: list[Any] = []
        for chunk in state.chunks:
            if chunk.status == "done":
                cr = state.results.get(chunk.chunk_id)
                if cr and cr.success:
                    mapped_outputs.extend(cr.output)

        chunks_succeeded = sum(1 for c in state.chunks if c.status == "done")
        chunks_failed = sum(1 for c in state.chunks if c.status == "failed")

        # Mint Trust Tokens for contributing peers
        tokens_minted = 0.0
        rewarded: set[str] = set()
        for chunk in state.chunks:
            if chunk.status == "done" and chunk.assigned_peer not in rewarded:
                self.clearing_engine.mint_for_canary_success(chunk.assigned_peer)
                tokens_minted += self.token_reward_per_chunk
                rewarded.add(chunk.assigned_peer)

        # Run reduce pass
        if chunks_failed > 0 and not mapped_outputs:
            state.status = "failed"
            return JobResult(
                job_id=job.job_id,
                success=False,
                reduce_output=None,
                chunks_total=len(state.chunks),
                chunks_succeeded=chunks_succeeded,
                chunks_failed=chunks_failed,
                tokens_minted=tokens_minted,
                error=f"{chunks_failed} chunk(s) failed with no successful output",
                duration_seconds=time.monotonic() - t_start,
            )

        success, reduce_output, reduce_error = aggregator.reduce(job, mapped_outputs)
        state.status = "done" if success else "failed"

        return JobResult(
            job_id=job.job_id,
            success=success,
            reduce_output=reduce_output,
            chunks_total=len(state.chunks),
            chunks_succeeded=chunks_succeeded,
            chunks_failed=chunks_failed,
            tokens_minted=tokens_minted,
            error=reduce_error,
            duration_seconds=time.monotonic() - t_start,
        )

    def get_state(self, job_id: str) -> JobState | None:
        """Return the :class:`JobState` for *job_id*, or ``None``."""
        with self._lock:
            return self._jobs.get(job_id)

    def active_job_count(self) -> int:
        """Return the number of jobs currently tracked."""
        with self._lock:
            return len(self._jobs)

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of tracker activity."""
        with self._lock:
            by_status: dict[str, int] = {}
            for state in self._jobs.values():
                by_status[state.status] = by_status.get(state.status, 0) + 1
        return {
            "total_jobs": sum(by_status.values()),
            "by_status": by_status,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_chunk_with_retry(
        self,
        chunk: Chunk,
        job: MapReduceJob,
        state: JobState,
        aggregator: SwarmAggregator,
    ) -> None:
        """Attempt to process *chunk*, retrying on failure up to ``job.max_retries``."""
        max_attempts = job.max_retries + 1
        for attempt in range(max_attempts):
            chunk.status = "running"
            chunk.assigned_at = time.time()

            result = aggregator.map_chunk(chunk, job)
            state.results[chunk.chunk_id] = result

            if result.success:
                chunk.status = "done"
                return

            # Failure: reassign if retries remain
            chunk.retry_count += 1
            if attempt < max_attempts - 1:
                # Reassign to a different peer by rotating the DHT key
                retry_key = f"{chunk.chunk_id}/retry/{attempt}"
                peer_endpoint = self.shard_router.shard_for(retry_key)
                if peer_endpoint:
                    chunk.assigned_peer = hashlib.sha256(
                        peer_endpoint.encode("utf-8")
                    ).hexdigest()[:8]
                chunk.status = "pending"

        chunk.status = "failed"
