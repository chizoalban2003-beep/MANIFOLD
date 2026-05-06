"""Tests for Phase 65: Recursive Self-Optimization (manifold/singularity.py)."""

from __future__ import annotations

import ast
import pathlib
import tempfile

import pytest

from manifold.singularity import (
    ASTMutator,
    MutationProposal,
    MutationStrategy,
    OptimizationResult,
    STRATEGIES,
    SandboxedTestRunner,
)


# ---------------------------------------------------------------------------
# MutationStrategy
# ---------------------------------------------------------------------------


class TestMutationStrategy:
    def test_fields(self) -> None:
        s = MutationStrategy(name="test_strat", description="A test", safe=True)
        assert s.name == "test_strat"
        assert s.safe is True

    def test_frozen(self) -> None:
        s = MutationStrategy(name="x", description="y", safe=True)
        with pytest.raises((AttributeError, TypeError)):
            s.name = "z"  # type: ignore[misc]

    def test_catalogue_contains_known_strategies(self) -> None:
        for name in ("loop_to_listcomp", "sum_generator", "identity"):
            assert name in STRATEGIES

    def test_catalogue_values_are_strategy_instances(self) -> None:
        for strat in STRATEGIES.values():
            assert isinstance(strat, MutationStrategy)

    def test_identity_strategy_is_safe(self) -> None:
        assert STRATEGIES["identity"].safe is True


# ---------------------------------------------------------------------------
# ASTMutator — unit tests
# ---------------------------------------------------------------------------


class TestASTMutator:
    def setup_method(self) -> None:
        self.mutator = ASTMutator()

    def test_propose_mutations_mathutils(self) -> None:
        proposals = self.mutator.propose_mutations("manifold._mathutils")
        assert len(proposals) > 0

    def test_propose_mutations_vectorfs(self) -> None:
        proposals = self.mutator.propose_mutations("manifold.vectorfs")
        assert len(proposals) > 0

    def test_propose_mutations_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="not a supported mutation target"):
            self.mutator.propose_mutations("manifold.brain")

    def test_proposals_have_required_fields(self) -> None:
        proposals = self.mutator.propose_mutations("manifold._mathutils")
        for p in proposals:
            assert p.target_module == "manifold._mathutils"
            assert isinstance(p.target_function, str)
            assert len(p.target_function) > 0
            assert isinstance(p.strategy, MutationStrategy)
            assert isinstance(p.original_source, str)
            assert isinstance(p.mutated_source, str)
            assert 0.0 <= p.confidence <= 1.0

    def test_identity_mutation_unchanged(self) -> None:
        proposals = self.mutator.propose_mutations("manifold._mathutils")
        identity = [p for p in proposals if p.strategy.name == "identity"]
        for p in identity:
            assert p.original_source == p.mutated_source
            assert p.confidence == 0.0

    def test_all_mutated_sources_are_valid_python(self) -> None:
        """Every mutated source must parse without SyntaxError."""
        proposals = self.mutator.propose_mutations("manifold._mathutils")
        for p in proposals:
            try:
                ast.parse(p.mutated_source)
            except SyntaxError as e:
                pytest.fail(f"SyntaxError in mutated source for {p.target_function}: {e}")

    def test_vectorfs_mutated_sources_are_valid_python(self) -> None:
        proposals = self.mutator.propose_mutations("manifold.vectorfs")
        for p in proposals:
            try:
                ast.parse(p.mutated_source)
            except SyntaxError as e:
                pytest.fail(f"SyntaxError in {p.target_function}: {e}")

    def test_proposal_to_dict_keys(self) -> None:
        proposals = self.mutator.propose_mutations("manifold._mathutils")
        assert len(proposals) > 0
        d = proposals[0].to_dict()
        for key in ("target_module", "target_function", "strategy", "confidence", "has_changes"):
            assert key in d

    def test_identity_proposal_has_no_changes(self) -> None:
        proposals = self.mutator.propose_mutations("manifold._mathutils")
        identity = [p for p in proposals if p.strategy.name == "identity"]
        for p in identity:
            assert p.to_dict()["has_changes"] is False

    def test_unified_diff_identity_is_empty(self) -> None:
        proposals = self.mutator.propose_mutations("manifold._mathutils")
        identity = [p for p in proposals if p.strategy.name == "identity"]
        for p in identity:
            diff = p.unified_diff()
            assert diff == ""

    def test_function_ast_summary_valid_code(self) -> None:
        src = "def add(a, b):\n    return a + b\n"
        summary = self.mutator.function_ast_summary(src)
        assert summary["parse_ok"] is True
        assert summary["function_count"] == 1
        assert summary["node_count"] > 0

    def test_function_ast_summary_invalid_code(self) -> None:
        src = "def broken(a, b\n    return a +"
        summary = self.mutator.function_ast_summary(src)
        assert summary["parse_ok"] is False
        assert summary["node_count"] == 0

    def test_function_ast_summary_for_loop_detected(self) -> None:
        src = "def f(xs):\n    result = []\n    for x in xs:\n        result.append(x)\n    return result\n"
        summary = self.mutator.function_ast_summary(src)
        assert summary["for_loop_count"] >= 1

    def test_function_ast_summary_listcomp_detected(self) -> None:
        src = "def f(xs):\n    return [x * 2 for x in xs]\n"
        summary = self.mutator.function_ast_summary(src)
        assert summary["list_comp_count"] >= 1

    def test_loop_to_listcomp_detects_append_pattern(self) -> None:
        src = (
            "def f(xs):\n"
            "    result = []\n"
            "    for x in xs:\n"
            "        result.append(x * 2)\n"
            "    return result\n"
        )
        tree = ast.parse(src)
        _, confidence = ASTMutator._loop_to_listcomp(src, tree)
        assert confidence > 0.5

    def test_loop_to_listcomp_no_match(self) -> None:
        src = "def f(xs):\n    return xs\n"
        tree = ast.parse(src)
        _, confidence = ASTMutator._loop_to_listcomp(src, tree)
        assert confidence == 0.0

    def test_sum_generator_detects_accumulator(self) -> None:
        src = (
            "def f(xs):\n"
            "    total = 0.0\n"
            "    for x in xs:\n"
            "        total += x * x\n"
            "    return total\n"
        )
        tree = ast.parse(src)
        _, confidence = ASTMutator._sum_generator(src, tree)
        assert confidence > 0.0

    def test_sum_generator_no_match(self) -> None:
        src = "def f(x):\n    return x + 1\n"
        tree = ast.parse(src)
        _, confidence = ASTMutator._sum_generator(src, tree)
        assert confidence == 0.0

    def test_proposal_count_equals_strategies_times_functions(self) -> None:
        import manifold._mathutils as mm
        import inspect
        fn_count = sum(
            1 for name, obj in inspect.getmembers(mm, inspect.isfunction)
            if obj.__module__ == "manifold._mathutils"
        )
        proposals = self.mutator.propose_mutations("manifold._mathutils")
        assert len(proposals) == fn_count * len(STRATEGIES)

    def test_apply_strategy_unknown_returns_original(self) -> None:
        src = "def f(x):\n    return x\n"
        mutated, confidence = self.mutator._apply_strategy(src, "unknown_strategy_xyz")
        assert mutated == src
        assert confidence == 0.0

    def test_apply_strategy_syntax_error_returns_original(self) -> None:
        bad_src = "def f(x\n    return"
        mutated, confidence = self.mutator._apply_strategy(bad_src, "loop_to_listcomp")
        assert mutated == bad_src
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# SandboxedTestRunner — unit tests (dry-run mode only)
# ---------------------------------------------------------------------------


class TestSandboxedTestRunner:
    def _make_proposal(
        self,
        original: str = "def f(x):\n    return x\n",
        mutated: str | None = None,
        confidence: float = 0.8,
    ) -> MutationProposal:
        from manifold.singularity import STRATEGIES
        if mutated is None:
            mutated = original
        return MutationProposal(
            target_module="manifold._mathutils",
            target_function="f",
            strategy=STRATEGIES["identity"],
            original_source=original,
            mutated_source=mutated,
            confidence=confidence,
        )

    def test_dry_run_valid_source_high_confidence_passes(self) -> None:
        runner = SandboxedTestRunner()
        proposal = self._make_proposal(confidence=0.8)
        result = runner.evaluate(proposal, run_full_suite=False)
        assert result.tests_passed is True
        assert result.patch_written is False

    def test_dry_run_low_confidence_fails(self) -> None:
        runner = SandboxedTestRunner()
        proposal = self._make_proposal(confidence=0.3)
        result = runner.evaluate(proposal, run_full_suite=False)
        assert result.tests_passed is False

    def test_dry_run_invalid_mutated_source_fails(self) -> None:
        runner = SandboxedTestRunner()
        proposal = self._make_proposal(mutated="def f(x\n    return", confidence=0.9)
        result = runner.evaluate(proposal, run_full_suite=False)
        assert result.tests_passed is False
        assert "parse" in result.error.lower()

    def test_dry_run_returns_optimization_result(self) -> None:
        runner = SandboxedTestRunner()
        proposal = self._make_proposal()
        result = runner.evaluate(proposal, run_full_suite=False)
        assert isinstance(result, OptimizationResult)

    def test_dry_run_no_patch_written(self) -> None:
        runner = SandboxedTestRunner()
        proposal = self._make_proposal(confidence=0.9)
        result = runner.evaluate(proposal, run_full_suite=False)
        assert result.patch_written is False
        assert result.patch_path == ""

    def test_optimization_result_to_dict(self) -> None:
        runner = SandboxedTestRunner()
        proposal = self._make_proposal()
        result = runner.evaluate(proposal, run_full_suite=False)
        d = result.to_dict()
        for key in ("proposal", "tests_passed", "tests_total", "tests_failed",
                    "baseline_seconds", "mutated_seconds", "time_delta_pct",
                    "patch_written", "patch_path", "error"):
            assert key in d

    def test_optimization_result_is_frozen(self) -> None:
        runner = SandboxedTestRunner()
        proposal = self._make_proposal()
        result = runner.evaluate(proposal, run_full_suite=False)
        with pytest.raises((AttributeError, TypeError)):
            result.tests_passed = not result.tests_passed  # type: ignore[misc]

    def test_speedup_threshold_default(self) -> None:
        runner = SandboxedTestRunner()
        assert runner.speedup_threshold_pct == 2.0

    def test_custom_patch_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = SandboxedTestRunner(patch_dir=pathlib.Path(tmpdir))
            assert runner.patch_dir == pathlib.Path(tmpdir)

    def test_write_patch_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = SandboxedTestRunner(patch_dir=pathlib.Path(tmpdir))
            proposal = self._make_proposal(
                original="def f(x):\n    return x + 1\n",
                mutated="def f(x):\n    return x + 1  # optimised\n",
                confidence=0.9,
            )
            patch_path = runner._write_patch(proposal, "speedup=5.0%")
            assert patch_path.exists()
            content = patch_path.read_text(encoding="utf-8")
            assert "MANIFOLD" in content
            assert proposal.target_function in content

    def test_write_patch_contains_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = SandboxedTestRunner(patch_dir=pathlib.Path(tmpdir))
            proposal = self._make_proposal(confidence=0.9)
            path = runner._write_patch(proposal, "test")
            assert "identity" in path.read_text(encoding="utf-8")

    def test_full_suite_baseline_failure_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate baseline having failures — should abort gracefully."""
        runner = SandboxedTestRunner()

        def fake_run_tests() -> tuple[float, int, int]:
            return 1.0, 10, 2  # 2 failures

        monkeypatch.setattr(runner, "_run_tests", fake_run_tests)
        proposal = self._make_proposal(confidence=0.9)
        result = runner.evaluate(proposal, run_full_suite=True)
        assert result.tests_passed is False
        assert "Baseline" in result.error

    def test_full_suite_all_pass_no_speedup_no_patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = SandboxedTestRunner(speedup_threshold_pct=2.0)
        call_count = [0]

        def fake_run_tests() -> tuple[float, int, int]:
            call_count[0] += 1
            return 10.0, 100, 0  # same time both runs

        monkeypatch.setattr(runner, "_run_tests", fake_run_tests)
        proposal = self._make_proposal(
            original="def f(x):\n    return x\n",
            mutated="def f(x):\n    return x  # slightly changed\n",
            confidence=0.9,
        )
        result = runner.evaluate(proposal, run_full_suite=True)
        assert result.tests_passed is True
        assert result.patch_written is False

    def test_full_suite_speedup_triggers_patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = SandboxedTestRunner(
                patch_dir=pathlib.Path(tmpdir),
                speedup_threshold_pct=2.0,
            )
            call_count = [0]

            def fake_run_tests() -> tuple[float, int, int]:
                call_count[0] += 1
                if call_count[0] == 1:
                    return 10.0, 100, 0  # baseline
                return 9.0, 100, 0  # mutated: 10% faster

            monkeypatch.setattr(runner, "_run_tests", fake_run_tests)
            proposal = self._make_proposal(
                original="def f(x):\n    return x\n",
                mutated="def f(x):\n    return x  # faster\n",
                confidence=0.9,
            )
            result = runner.evaluate(proposal, run_full_suite=True)
            assert result.tests_passed is True
            assert result.patch_written is True
            assert pathlib.Path(result.patch_path).exists()

    def test_full_suite_identity_mutation_skips_second_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Identity mutation (original==mutated) should only run pytest once."""
        runner = SandboxedTestRunner()
        call_count = [0]

        def fake_run_tests() -> tuple[float, int, int]:
            call_count[0] += 1
            return 5.0, 50, 0

        monkeypatch.setattr(runner, "_run_tests", fake_run_tests)
        proposal = self._make_proposal(confidence=0.9)  # original == mutated
        runner.evaluate(proposal, run_full_suite=True)
        assert call_count[0] == 1  # only baseline run
