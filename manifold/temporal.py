"""Phase 63: Temporal Forking Engine — Parallel Reality Branching.

When the OS encounters a high-risk :class:`~manifold.dag.DAGNode`, it forks
the execution context into *N* independent branches.  Each branch receives a
deep-copied :class:`~manifold.gridmapper.GridState` and
:class:`~manifold.b2b.AgentEconomyLedger` so mutations in one branch cannot
contaminate another.  After all branches complete (or fail),
:class:`TimelineCollapse` selects the branch with the highest net yield and
returns a :class:`CollapseResult` for the caller to commit.

Architecture
------------
::

    ┌──────── ParallelTimeline ────────┐
    │  source_state  source_ledger    │
    │       │               │         │
    │  StateForker.fork_both()         │
    │       │               │         │
    │  Branch 0  Branch 1  Branch N   │
    │  (deep copies — no shared ref)  │
    └──────────────────────────────────┘
                     │
             TimelineCollapse
                     │
    T_optimal = argmax  Asset(t)
                      ──────────────
                      Cost(t) + Risk(t)

Key classes
-----------
``ForkSpec``
    Immutable descriptor for a single branch (branch_id + label).
``BranchResult``
    Mutable outcome of executing one branch.
``CollapseResult``
    Immutable summary of the collapse operation (winner + all branches).
``StateForker``
    Creates isolated deep-copies of grid state and economy ledger.
``ParallelTimeline``
    Manages *N* branches and runs them via a caller-supplied executor.
``TimelineCollapse``
    Picks the winning branch and returns a ``CollapseResult``.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .b2b import AgentEconomyLedger
from .gridmapper import GridState


# ---------------------------------------------------------------------------
# ForkSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForkSpec:
    """Immutable descriptor for one branch of a :class:`ParallelTimeline`.

    Attributes
    ----------
    branch_id:
        UUID string uniquely identifying this branch within a fork session.
    label:
        Human-readable name (e.g. ``"optimistic"`` or ``"conservative"``).
    """

    branch_id: str
    label: str


# ---------------------------------------------------------------------------
# BranchResult
# ---------------------------------------------------------------------------


@dataclass
class BranchResult:
    """Mutable result produced by executing one timeline branch.

    Attributes
    ----------
    branch_id:
        Must match the :attr:`ForkSpec.branch_id` of the branch that produced
        this result.
    label:
        Human-readable branch name (copied from :class:`ForkSpec`).
    asset:
        Economic asset gained by this execution path.
    cost:
        Economic cost incurred.
    risk:
        Residual risk realised.
    success:
        ``True`` if the branch completed without an unhandled exception.
    error:
        Non-empty string describing the exception when ``success`` is
        ``False``.
    duration_s:
        Wall-clock execution time in seconds (set by :class:`ParallelTimeline`
        after the executor returns).
    """

    branch_id: str
    label: str
    asset: float
    cost: float
    risk: float
    success: bool
    error: str = ""
    duration_s: float = 0.0

    def yield_score(self) -> float:
        """Return Asset / (Cost + Risk).

        The denominator is floored to a small positive value to avoid
        division by zero when both cost and risk are zero.
        """
        denom = self.cost + self.risk
        if denom <= 0.0:
            denom = 1e-9
        return self.asset / denom

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "branch_id": self.branch_id,
            "label": self.label,
            "asset": self.asset,
            "cost": self.cost,
            "risk": self.risk,
            "success": self.success,
            "error": self.error,
            "duration_s": self.duration_s,
            "yield_score": self.yield_score(),
        }


# ---------------------------------------------------------------------------
# CollapseResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollapseResult:
    """Immutable summary of a :meth:`TimelineCollapse.collapse` operation.

    Attributes
    ----------
    winning_branch_id:
        ``branch_id`` of the branch with the highest yield score.
    winning_label:
        Human-readable label of the winning branch.
    winning_yield:
        Yield score of the winning branch.
    all_branches:
        Tuple of all :class:`BranchResult` objects (winner + losers).
    collapsed_at:
        POSIX timestamp when the collapse was performed.
    fork_id:
        Identifier of the :class:`ParallelTimeline` that produced these
        branches.
    """

    winning_branch_id: str
    winning_label: str
    winning_yield: float
    all_branches: tuple[BranchResult, ...]
    collapsed_at: float
    fork_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "winning_branch_id": self.winning_branch_id,
            "winning_label": self.winning_label,
            "winning_yield": self.winning_yield,
            "all_branches": [b.to_dict() for b in self.all_branches],
            "collapsed_at": self.collapsed_at,
            "fork_id": self.fork_id,
        }


# ---------------------------------------------------------------------------
# StateForker
# ---------------------------------------------------------------------------

class StateForker:
    """Creates isolated deep-copies of MANIFOLD state objects.

    All methods are static; ``StateForker`` carries no instance state.

    The copies are guaranteed to share no references with the originals, so
    mutations inside a branch cannot contaminate the parent timeline.

    Example
    -------
    ::

        forked_state, forked_ledger = StateForker.fork_both(state, ledger)
        # modify forked_state freely — original state is untouched
    """

    @staticmethod
    def fork_grid_state(state: GridState) -> GridState:
        """Return an independent deep copy of *state*."""
        return copy.deepcopy(state)

    @staticmethod
    def fork_economy_ledger(ledger: AgentEconomyLedger) -> AgentEconomyLedger:
        """Return an independent deep copy of *ledger*."""
        return copy.deepcopy(ledger)

    @staticmethod
    def fork_both(
        state: GridState, ledger: AgentEconomyLedger
    ) -> tuple[GridState, AgentEconomyLedger]:
        """Return independent deep copies of *both* state and ledger.

        Equivalent to calling :meth:`fork_grid_state` and
        :meth:`fork_economy_ledger` separately but slightly cheaper as only
        one :func:`copy.deepcopy` traversal is needed.
        """
        pair = copy.deepcopy((state, ledger))
        return pair[0], pair[1]


# ---------------------------------------------------------------------------
# TimelineExecutor type alias
# ---------------------------------------------------------------------------

#: Callable signature for branch executors passed to :meth:`ParallelTimeline.run`.
#: The executor receives the branch_id, a forked GridState, and a forked
#: AgentEconomyLedger, and must return a :class:`BranchResult`.
TimelineExecutor = Callable[[str, GridState, AgentEconomyLedger], "BranchResult"]


# ---------------------------------------------------------------------------
# ParallelTimeline
# ---------------------------------------------------------------------------


@dataclass
class ParallelTimeline:
    """Routes the same task down multiple execution paths simultaneously.

    Each branch receives an independent deep-copy of the source
    :class:`~manifold.gridmapper.GridState` and
    :class:`~manifold.b2b.AgentEconomyLedger` so mutations in one branch
    cannot bleed into another.

    Usage
    -----
    ::

        timeline = ParallelTimeline(state, ledger)
        timeline.add_branch("optimistic")
        timeline.add_branch("conservative")
        timeline.add_branch("aggressive")
        results = timeline.run(my_executor)
        collapse = TimelineCollapse.collapse(results, fork_id=timeline.fork_id)
    """

    _source_state: GridState = field(repr=False)
    _source_ledger: AgentEconomyLedger = field(repr=False)
    _fork_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _specs: list[ForkSpec] = field(default_factory=list, init=False, repr=False)
    _results: list[BranchResult] = field(default_factory=list, init=False, repr=False)

    def __init__(
        self,
        state: GridState,
        ledger: AgentEconomyLedger,
        *,
        fork_id: str | None = None,
    ) -> None:
        self._source_state = state
        self._source_ledger = ledger
        self._fork_id = fork_id or str(uuid.uuid4())
        self._specs = []
        self._results = []

    @property
    def fork_id(self) -> str:
        """Unique identifier for this fork session."""
        return self._fork_id

    @property
    def branch_count(self) -> int:
        """Number of branches registered so far."""
        return len(self._specs)

    def add_branch(self, label: str) -> ForkSpec:
        """Register a new branch and return its :class:`ForkSpec`.

        Parameters
        ----------
        label:
            Human-readable name for this branch.

        Returns
        -------
        ForkSpec
            The spec created for this branch.
        """
        spec = ForkSpec(branch_id=str(uuid.uuid4()), label=label)
        self._specs.append(spec)
        return spec

    def run(self, executor: TimelineExecutor) -> list[BranchResult]:
        """Execute all branches sequentially and collect results.

        Each branch runs in the same thread (zero external dependencies —
        no ``concurrent.futures`` or ``threading``).  For every branch:

        1. A deep-copy of the source state and ledger is made.
        2. The executor is called with ``(branch_id, forked_state, forked_ledger)``.
        3. If the executor raises, a failure :class:`BranchResult` is
           synthesised automatically.

        Parameters
        ----------
        executor:
            A callable matching :data:`TimelineExecutor`.

        Returns
        -------
        list[BranchResult]
            One result per registered branch.
        """
        results: list[BranchResult] = []
        for spec in self._specs:
            forked_state, forked_ledger = StateForker.fork_both(
                self._source_state, self._source_ledger
            )
            t0 = time.monotonic()
            try:
                result = executor(spec.branch_id, forked_state, forked_ledger)
                result.duration_s = time.monotonic() - t0
            except Exception as exc:  # noqa: BLE001
                result = BranchResult(
                    branch_id=spec.branch_id,
                    label=spec.label,
                    asset=0.0,
                    cost=0.0,
                    risk=1.0,
                    success=False,
                    error=str(exc),
                    duration_s=time.monotonic() - t0,
                )
            results.append(result)
        self._results = results
        return results

    def results(self) -> list[BranchResult]:
        """Return the results from the most recent :meth:`run` call."""
        return list(self._results)


# ---------------------------------------------------------------------------
# TimelineCollapse
# ---------------------------------------------------------------------------


class TimelineCollapse:
    """Collapses parallel timelines and commits the one with highest net yield.

    Selection criterion (from the problem spec):

        T_optimal = argmax  Asset(t) / (Cost(t) + Risk(t))

    If no branch succeeded, the algorithm falls back to choosing among all
    branches (including failures) to guarantee a result is always returned.

    All methods are static; ``TimelineCollapse`` carries no instance state.

    Example
    -------
    ::

        collapse = TimelineCollapse.collapse(results, fork_id=timeline.fork_id)
        print(collapse.winning_label, collapse.winning_yield)
    """

    @staticmethod
    def collapse(
        branches: list[BranchResult],
        *,
        fork_id: str = "",
    ) -> CollapseResult:
        """Collapse *branches* and return the optimal :class:`CollapseResult`.

        Parameters
        ----------
        branches:
            List of :class:`BranchResult` objects from
            :meth:`ParallelTimeline.run`.
        fork_id:
            The fork session identifier (pass ``timeline.fork_id``).

        Returns
        -------
        CollapseResult

        Raises
        ------
        ValueError
            If *branches* is empty.
        """
        if not branches:
            raise ValueError("Cannot collapse an empty list of branches")

        successful = [b for b in branches if b.success]
        candidates = successful if successful else branches
        winner = max(candidates, key=lambda b: b.yield_score())

        return CollapseResult(
            winning_branch_id=winner.branch_id,
            winning_label=winner.label,
            winning_yield=winner.yield_score(),
            all_branches=tuple(branches),
            collapsed_at=time.time(),
            fork_id=fork_id,
        )
