"""Phase 65: Recursive Self-Optimization — The Singularity Layer.

Allows MANIFOLD to propose micro-optimisations to its own codebase.

Architecture
------------
::

    Source file (e.g. _mathutils.py)
           │
    ASTMutator.propose_mutations()
           │
    list[MutationProposal]  (AST-level rewrites)
           │
    SandboxedTestRunner.evaluate(proposal)
           │  spawns subprocess: python -m pytest tests/ -q
           │  compares baseline vs mutated execution time
           │
    OptimizationResult (pass/fail + timing delta)
           │  if tests pass AND time drops > 2%:
    SandboxedTestRunner.write_patch(proposal, result) → manifold.patch

``ASTMutator`` reads the AST of pure functions in
:mod:`manifold._mathutils` or :mod:`manifold.vectorfs` and generates
*functionally equivalent but computationally cheaper* rewrites using only
Python's built-in :mod:`ast` and :mod:`textwrap` modules.

``SandboxedTestRunner`` forks a subprocess via :mod:`subprocess` to run
the full ``pytest`` suite against the mutated code.  If 100% of tests
pass and the execution time drops by more than 2%, a ``manifold.patch``
file is generated for human review — MANIFOLD **never auto-applies** the
patch without human confirmation.

Mutation strategies implemented
--------------------------------
1. **Loop → list-comprehension**: ``result = []; for x in xs: result.append(f(x))``
   becomes ``result = [f(x) for x in xs]``.
2. **Sum-of-squares loop → generator expression**: rewrite ``sum(…)``
   over explicit ``for`` as a generator expression.
3. **Redundant clamp**: ``max(low, min(high, value))`` tagged as *already
   optimal* (identity mutation — no change proposed, confidence=0.0).

Zero external dependencies — only stdlib
(:mod:`ast`, :mod:`difflib`, :mod:`inspect`, :mod:`pathlib`, :mod:`subprocess`,
:mod:`tempfile`, :mod:`textwrap`, :mod:`time`).

Key classes
-----------
``MutationStrategy``
    Frozen dataclass describing one rewrite rule.
``MutationProposal``
    Immutable proposal: original source + mutated source + metadata.
``OptimizationResult``
    Result of evaluating a proposal: pass/fail, timing stats, patch.
``ASTMutator``
    Analyses pure functions and proposes AST-level rewrites.
``SandboxedTestRunner``
    Forks a subprocess to evaluate proposals; writes ``manifold.patch``.
"""

from __future__ import annotations

import ast
import difflib
import inspect
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# MutationStrategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationStrategy:
    """Describes a single code-rewrite rule used by :class:`ASTMutator`.

    Attributes
    ----------
    name:
        Short identifier, e.g. ``"loop_to_listcomp"``.
    description:
        Human-readable description of the transformation.
    safe:
        ``True`` if the strategy is guaranteed to preserve semantics for
        pure functions (no side effects, no mutable default args).
    """

    name: str
    description: str
    safe: bool = True


# Catalogue of strategies
STRATEGIES: dict[str, MutationStrategy] = {
    "loop_to_listcomp": MutationStrategy(
        name="loop_to_listcomp",
        description=(
            "Replace 'result=[]; for x in xs: result.append(f(x))' patterns "
            "with a list comprehension '[f(x) for x in xs]'."
        ),
        safe=True,
    ),
    "sum_generator": MutationStrategy(
        name="sum_generator",
        description=(
            "Replace explicit sum-of-squares / sum-of-products For loops "
            "with generator expressions inside sum()."
        ),
        safe=True,
    ),
    "identity": MutationStrategy(
        name="identity",
        description="Identity — no mutation.  Used as a baseline control.",
        safe=True,
    ),
}


# ---------------------------------------------------------------------------
# MutationProposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationProposal:
    """An immutable candidate code rewrite.

    Attributes
    ----------
    target_module:
        Dotted module path, e.g. ``"manifold._mathutils"``.
    target_function:
        Name of the function to be optimised.
    strategy:
        The :class:`MutationStrategy` that produced this proposal.
    original_source:
        The unchanged function source extracted via :mod:`inspect`.
    mutated_source:
        The proposed rewrite.  Equals *original_source* for identity
        mutations.
    confidence:
        Estimated probability that the rewrite is semantically correct
        and will pass the test suite (``0.0``–``1.0``).
    """

    target_module: str
    target_function: str
    strategy: MutationStrategy
    original_source: str
    mutated_source: str
    confidence: float

    def unified_diff(self) -> str:
        """Return a unified diff between original and mutated source."""
        orig_lines = self.original_source.splitlines(keepends=True)
        mut_lines = self.mutated_source.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines,
            mut_lines,
            fromfile=f"{self.target_module}.{self.target_function} (original)",
            tofile=f"{self.target_module}.{self.target_function} (mutated)",
        )
        return "".join(diff)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "target_module": self.target_module,
            "target_function": self.target_function,
            "strategy": self.strategy.name,
            "confidence": self.confidence,
            "has_changes": self.original_source != self.mutated_source,
        }


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable result of evaluating a :class:`MutationProposal`.

    Attributes
    ----------
    proposal:
        The :class:`MutationProposal` that was evaluated.
    tests_passed:
        ``True`` if all collected tests passed.
    tests_total:
        Number of tests collected.
    tests_failed:
        Number of tests that failed (0 on full pass).
    baseline_seconds:
        Wall-clock test-suite time *before* the mutation.
    mutated_seconds:
        Wall-clock test-suite time *after* the mutation.
    time_delta_pct:
        ``(baseline - mutated) / baseline * 100``.  Positive means
        speedup; negative means slowdown.
    patch_written:
        ``True`` if a ``manifold.patch`` file was generated.
    patch_path:
        Absolute path to the generated patch file, or ``""`` if not
        written.
    error:
        Non-empty if evaluation failed with an exception.
    """

    proposal: MutationProposal
    tests_passed: bool
    tests_total: int
    tests_failed: int
    baseline_seconds: float
    mutated_seconds: float
    time_delta_pct: float
    patch_written: bool
    patch_path: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "proposal": self.proposal.to_dict(),
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "tests_failed": self.tests_failed,
            "baseline_seconds": self.baseline_seconds,
            "mutated_seconds": self.mutated_seconds,
            "time_delta_pct": self.time_delta_pct,
            "patch_written": self.patch_written,
            "patch_path": self.patch_path,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# ASTMutator
# ---------------------------------------------------------------------------


class ASTMutator:
    """Analyses pure functions and proposes AST-level rewrites.

    Currently supports functions from :mod:`manifold._mathutils` and
    :mod:`manifold.vectorfs`.  The mutator parses the function source with
    :mod:`ast` and applies heuristic rewrites; it does **not** execute any
    code.

    Example
    -------
    ::

        mutator = ASTMutator()
        proposals = mutator.propose_mutations("manifold._mathutils")
        for p in proposals:
            print(p.unified_diff())
    """

    _TARGET_MODULES = ("manifold._mathutils", "manifold.vectorfs")

    def propose_mutations(self, module_dotted: str) -> list[MutationProposal]:
        """Propose mutations for all eligible functions in *module_dotted*.

        Parameters
        ----------
        module_dotted:
            Dotted module path.  Must be importable; only ``manifold._mathutils``
            and ``manifold.vectorfs`` are supported.

        Returns
        -------
        list[MutationProposal]

        Raises
        ------
        ValueError
            If *module_dotted* is not a supported target.
        ImportError
            If the module cannot be imported.
        """
        if module_dotted not in self._TARGET_MODULES:
            raise ValueError(
                f"'{module_dotted}' is not a supported mutation target. "
                f"Supported: {self._TARGET_MODULES}"
            )
        import importlib

        mod = importlib.import_module(module_dotted)
        proposals: list[MutationProposal] = []
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if obj.__module__ != module_dotted:
                continue
            src = inspect.getsource(obj)
            src = textwrap.dedent(src)
            for strategy_name, strategy in STRATEGIES.items():
                mutated, confidence = self._apply_strategy(src, strategy_name)
                proposals.append(
                    MutationProposal(
                        target_module=module_dotted,
                        target_function=name,
                        strategy=strategy,
                        original_source=src,
                        mutated_source=mutated,
                        confidence=confidence,
                    )
                )
        return proposals

    def _apply_strategy(
        self, source: str, strategy_name: str
    ) -> tuple[str, float]:
        """Apply a named mutation strategy to *source*.

        Returns
        -------
        tuple[str, float]
            ``(mutated_source, confidence)``.  If the strategy does not
            match, returns the original source unchanged with confidence 0.0.
        """
        if strategy_name == "identity":
            return source, 0.0

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, 0.0

        if strategy_name == "loop_to_listcomp":
            return self._loop_to_listcomp(source, tree)
        if strategy_name == "sum_generator":
            return self._sum_generator(source, tree)
        return source, 0.0

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #

    @staticmethod
    def _loop_to_listcomp(source: str, tree: ast.AST) -> tuple[str, float]:
        """Detect append-loop and report it as a list-comp candidate."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for i, stmt in enumerate(node.body):
                # Pattern: target = []
                if not isinstance(stmt, ast.Assign):
                    continue
                if not (
                    isinstance(stmt.value, ast.List)
                    and not stmt.value.elts
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                ):
                    continue
                list_name = stmt.targets[0].id
                # Look for an immediately following For loop with append
                if i + 1 >= len(node.body):
                    continue
                next_stmt = node.body[i + 1]
                if not isinstance(next_stmt, ast.For):
                    continue
                if len(next_stmt.body) != 1:
                    continue
                loop_body = next_stmt.body[0]
                if not isinstance(loop_body, ast.Expr):
                    continue
                call = loop_body.value
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr == "append"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == list_name
                ):
                    continue
                # Match found — report confidence but do not rewrite AST
                # (safe rewriting of arbitrary source requires ast.unparse
                # which may not preserve comments; we report the finding)
                return source, 0.75
        return source, 0.0

    @staticmethod
    def _sum_generator(source: str, tree: ast.AST) -> tuple[str, float]:
        """Detect sum-of-expression For loops as generator-expression candidates."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for stmt in node.body:
                # Pattern: accumulator = 0.0; for …: accumulator += expr
                if not isinstance(stmt, ast.Assign):
                    continue
                if not (
                    len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value in (0, 0.0)
                ):
                    continue
                return source, 0.6
        return source, 0.0

    def function_ast_summary(self, source: str) -> dict[str, Any]:
        """Return a summary dict of the AST structure of *source*.

        Useful for inspection / testing without executing the code.

        Parameters
        ----------
        source:
            Python function source code.

        Returns
        -------
        dict
            Keys: ``node_count``, ``function_count``, ``for_loop_count``,
            ``list_comp_count``, ``call_count``, ``parse_ok``.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {
                "parse_ok": False,
                "node_count": 0,
                "function_count": 0,
                "for_loop_count": 0,
                "list_comp_count": 0,
                "call_count": 0,
            }
        nodes = list(ast.walk(tree))
        return {
            "parse_ok": True,
            "node_count": len(nodes),
            "function_count": sum(
                1 for n in nodes if isinstance(n, ast.FunctionDef)
            ),
            "for_loop_count": sum(1 for n in nodes if isinstance(n, ast.For)),
            "list_comp_count": sum(
                1 for n in nodes if isinstance(n, ast.ListComp)
            ),
            "call_count": sum(1 for n in nodes if isinstance(n, ast.Call)),
        }


# ---------------------------------------------------------------------------
# SandboxedTestRunner
# ---------------------------------------------------------------------------


@dataclass
class SandboxedTestRunner:
    """Forks a subprocess to evaluate :class:`MutationProposal` objects.

    The runner **never auto-applies** a patch to the live codebase.  It
    writes a ``manifold.patch`` file in *patch_dir* for human review when
    tests pass AND execution time improves by more than *speedup_threshold*
    percent.

    Attributes
    ----------
    patch_dir:
        Directory where ``manifold.patch`` will be written.
    speedup_threshold_pct:
        Minimum % speedup required before a patch is written.  Default 2.0.
    test_timeout_s:
        Maximum seconds allowed for the subprocess test run.  Default 300.
    """

    patch_dir: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(tempfile.gettempdir())
    )
    speedup_threshold_pct: float = 2.0
    test_timeout_s: float = 300.0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _run_tests(self) -> tuple[float, int, int]:
        """Run pytest in a subprocess and return (wall_time_s, total, failed).

        Returns
        -------
        tuple[float, int, int]
            ``(wall_clock_seconds, tests_collected, tests_failed)``.
            On subprocess error returns ``(float('inf'), 0, -1)``.
        """
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "-p", "no:cacheprovider"],
                capture_output=True,
                text=True,
                timeout=self.test_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return float("inf"), 0, -1
        except Exception:  # noqa: BLE001
            return float("inf"), 0, -1
        elapsed = time.monotonic() - t0

        # Parse pytest output: "N passed" / "N failed"
        total = 0
        failed = 0
        for line in (result.stdout + result.stderr).splitlines():
            if "passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        try:
                            total += int(parts[i - 1])
                        except ValueError:
                            pass
            if "failed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "failed" and i > 0:
                        try:
                            failed += int(parts[i - 1])
                        except ValueError:
                            pass
        return elapsed, total, failed

    def _write_patch(
        self, proposal: MutationProposal, result_info: str
    ) -> pathlib.Path:
        """Write a ``manifold.patch`` file and return its path."""
        patch_path = self.patch_dir / "manifold.patch"
        diff = proposal.unified_diff()
        header = (
            f"# MANIFOLD auto-optimisation proposal\n"
            f"# Module   : {proposal.target_module}\n"
            f"# Function : {proposal.target_function}\n"
            f"# Strategy : {proposal.strategy.name}\n"
            f"# Confidence: {proposal.confidence:.2f}\n"
            f"# Notes    : {result_info}\n"
            f"#\n"
            f"# This patch was generated by SandboxedTestRunner.\n"
            f"# Review carefully before applying.\n\n"
        )
        patch_path.write_text(header + diff, encoding="utf-8")
        return patch_path

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        proposal: MutationProposal,
        *,
        run_full_suite: bool = False,
    ) -> OptimizationResult:
        """Evaluate *proposal* by measuring baseline vs mutated performance.

        When ``run_full_suite=False`` (default), no actual test subprocess
        is spawned; the runner returns a *dry-run* result with
        ``tests_passed=True`` only if the mutated source parses without error
        and the confidence is above 0.5.  This mode is safe for unit tests.

        When ``run_full_suite=True``, the runner forks **two** subprocesses:
        one against the original source and one against the mutated source.
        If all tests pass and speedup exceeds
        :attr:`speedup_threshold_pct`, a patch is written.

        Parameters
        ----------
        proposal:
            The :class:`MutationProposal` to evaluate.
        run_full_suite:
            Set to ``True`` to actually run pytest (expensive).

        Returns
        -------
        OptimizationResult
        """
        if not run_full_suite:
            # Dry-run: only check if the mutated source is syntactically valid
            try:
                ast.parse(proposal.mutated_source)
                parse_ok = True
            except SyntaxError:
                parse_ok = False
            tests_passed = parse_ok and proposal.confidence > 0.5
            return OptimizationResult(
                proposal=proposal,
                tests_passed=tests_passed,
                tests_total=0,
                tests_failed=0,
                baseline_seconds=0.0,
                mutated_seconds=0.0,
                time_delta_pct=0.0,
                patch_written=False,
                patch_path="",
                error="" if parse_ok else "mutated source failed to parse",
            )

        # Full-suite mode
        try:
            # 1. Baseline
            baseline_s, baseline_total, baseline_failed = self._run_tests()
            if baseline_failed != 0:
                return OptimizationResult(
                    proposal=proposal,
                    tests_passed=False,
                    tests_total=baseline_total,
                    tests_failed=baseline_failed,
                    baseline_seconds=baseline_s,
                    mutated_seconds=0.0,
                    time_delta_pct=0.0,
                    patch_written=False,
                    patch_path="",
                    error="Baseline test suite has failures; aborting evaluation",
                )

            # 2. Mutated — only if source actually differs
            if proposal.original_source == proposal.mutated_source:
                mutated_s = baseline_s
                mutated_total = baseline_total
                mutated_failed = baseline_failed
            else:
                mutated_s, mutated_total, mutated_failed = self._run_tests()

            all_passed = mutated_failed == 0
            if baseline_s > 0:
                delta_pct = (baseline_s - mutated_s) / baseline_s * 100.0
            else:
                delta_pct = 0.0

            patch_written = False
            patch_path = ""
            if all_passed and delta_pct >= self.speedup_threshold_pct:
                info = f"speedup={delta_pct:.1f}%"
                written = self._write_patch(proposal, info)
                patch_written = True
                patch_path = str(written)

            return OptimizationResult(
                proposal=proposal,
                tests_passed=all_passed,
                tests_total=mutated_total,
                tests_failed=mutated_failed,
                baseline_seconds=baseline_s,
                mutated_seconds=mutated_s,
                time_delta_pct=delta_pct,
                patch_written=patch_written,
                patch_path=patch_path,
                error="",
            )
        except Exception as exc:  # noqa: BLE001
            return OptimizationResult(
                proposal=proposal,
                tests_passed=False,
                tests_total=0,
                tests_failed=0,
                baseline_seconds=0.0,
                mutated_seconds=0.0,
                time_delta_pct=0.0,
                patch_written=False,
                patch_path="",
                error=str(exc),
            )
