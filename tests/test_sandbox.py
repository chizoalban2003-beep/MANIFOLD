"""Tests for Phase 44: AST Execution Sandbox (manifold/sandbox.py)."""

from __future__ import annotations

import pytest

from manifold.sandbox import (
    ASTValidator,
    BudgetedExecutor,
    ExecutionResult,
    SandboxTimeoutError,
    SandboxViolation,
)


# ---------------------------------------------------------------------------
# SandboxViolation
# ---------------------------------------------------------------------------


class TestSandboxViolation:
    def test_to_dict(self) -> None:
        v = SandboxViolation(rule="test_rule", description="desc", lineno=5)
        d = v.to_dict()
        assert d["rule"] == "test_rule"
        assert d["description"] == "desc"
        assert d["lineno"] == 5

    def test_defaults(self) -> None:
        v = SandboxViolation(rule="r", description="d")
        assert v.lineno == 0

    def test_frozen(self) -> None:
        v = SandboxViolation(rule="r", description="d")
        with pytest.raises((AttributeError, TypeError)):
            v.rule = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SandboxTimeoutError
# ---------------------------------------------------------------------------


class TestSandboxTimeoutError:
    def test_attributes(self) -> None:
        err = SandboxTimeoutError(12000, 10000)
        assert err.instructions_used == 12000
        assert err.max_instructions == 10000

    def test_is_runtime_error(self) -> None:
        err = SandboxTimeoutError(1, 1)
        assert isinstance(err, RuntimeError)

    def test_message(self) -> None:
        err = SandboxTimeoutError(50, 10)
        assert "50" in str(err)
        assert "10" in str(err)


# ---------------------------------------------------------------------------
# ASTValidator — safe code
# ---------------------------------------------------------------------------


class TestASTValidatorSafe:
    def test_empty_string_is_safe(self) -> None:
        v = ASTValidator()
        assert v.validate("") == []

    def test_simple_arithmetic(self) -> None:
        v = ASTValidator()
        assert v.validate("x = 1 + 2 * 3") == []

    def test_function_def_is_safe(self) -> None:
        v = ASTValidator()
        assert v.validate("def add(a, b):\n    return a + b") == []

    def test_list_comprehension(self) -> None:
        v = ASTValidator()
        assert v.validate("result = [x*2 for x in range(10)]") == []

    def test_string_ops(self) -> None:
        v = ASTValidator()
        assert v.validate("s = 'hello ' + 'world'") == []

    def test_is_safe_true(self) -> None:
        v = ASTValidator()
        assert v.is_safe("x = 42") is True

    def test_is_safe_false_for_import(self) -> None:
        v = ASTValidator()
        assert v.is_safe("import os") is False

    def test_conditional_expression(self) -> None:
        v = ASTValidator()
        assert v.validate("result = 'yes' if True else 'no'") == []

    def test_dict_literal(self) -> None:
        v = ASTValidator()
        assert v.validate("d = {'key': 'value', 'n': 42}") == []

    def test_nested_function_call_safe(self) -> None:
        v = ASTValidator()
        # max and len are not forbidden
        assert v.validate("result = max(len([1,2,3]), 5)") == []


# ---------------------------------------------------------------------------
# ASTValidator — forbidden imports
# ---------------------------------------------------------------------------


class TestASTValidatorImports:
    def test_import_blocked(self) -> None:
        v = ASTValidator()
        violations = v.validate("import os")
        assert len(violations) >= 1
        assert violations[0].rule == "import_not_allowed"

    def test_from_import_blocked(self) -> None:
        v = ASTValidator()
        violations = v.validate("from os import path")
        assert len(violations) >= 1
        assert violations[0].rule == "import_not_allowed"

    def test_import_sys_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("import sys")

    def test_import_as_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("import subprocess as sp")

    def test_multiple_imports_blocked(self) -> None:
        v = ASTValidator()
        violations = v.validate("import os\nimport sys")
        rules = [vi.rule for vi in violations]
        assert rules.count("import_not_allowed") >= 2

    def test_lineno_recorded(self) -> None:
        v = ASTValidator()
        violations = v.validate("x = 1\nimport os\n")
        import_v = [vi for vi in violations if vi.rule == "import_not_allowed"]
        assert import_v[0].lineno == 2


# ---------------------------------------------------------------------------
# ASTValidator — forbidden calls
# ---------------------------------------------------------------------------


class TestASTValidatorForbiddenCalls:
    def test_eval_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("eval('1+1')")

    def test_exec_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("exec('x=1')")

    def test_open_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("open('/etc/passwd')")

    def test_dunder_import_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("__import__('os')")

    def test_compile_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("compile('x=1', '<s>', 'exec')")

    def test_getattr_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("getattr(object, '__class__')")

    def test_vars_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("vars()")

    def test_globals_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("globals()")

    def test_locals_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("locals()")

    def test_method_named_eval_blocked(self) -> None:
        # Attribute-style call: obj.eval() — "eval" is a forbidden attr name
        v = ASTValidator()
        assert not v.is_safe("obj.eval('something')")


# ---------------------------------------------------------------------------
# ASTValidator — forbidden attributes
# ---------------------------------------------------------------------------


class TestASTValidatorForbiddenAttributes:
    def test_dunder_class_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("x = (1).__class__")

    def test_dunder_builtins_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("x = {}.__builtins__")

    def test_dunder_dict_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("x = obj.__dict__")

    def test_dunder_subclasses_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("x = int.__subclasses__()")

    def test_dunder_mro_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("x = int.__mro__")

    def test_dunder_globals_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("x = fn.__globals__")


# ---------------------------------------------------------------------------
# ASTValidator — builtins overwrite
# ---------------------------------------------------------------------------


class TestASTValidatorBuiltinsOverwrite:
    def test_builtins_assignment_blocked(self) -> None:
        v = ASTValidator()
        violations = v.validate("__builtins__ = {}")
        rules = [vi.rule for vi in violations]
        assert "builtins_overwrite" in rules

    def test_builtins_reassignment_blocked(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("__builtins__ = None")


# ---------------------------------------------------------------------------
# ASTValidator — syntax errors
# ---------------------------------------------------------------------------


class TestASTValidatorSyntaxErrors:
    def test_syntax_error_recorded(self) -> None:
        v = ASTValidator()
        violations = v.validate("def(: bad code !!!")
        assert len(violations) >= 1
        assert violations[0].rule == "syntax_error"

    def test_syntax_error_returns_early(self) -> None:
        v = ASTValidator()
        # Should not raise, just return violations
        result = v.validate("!!!BAD!!!")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# ASTValidator — bypass attempts
# ---------------------------------------------------------------------------


class TestASTValidatorBypassAttempts:
    def test_chained_getattr_bypass(self) -> None:
        v = ASTValidator()
        # Attempt: ().__class__.__bases__[0].__subclasses__()
        code = "().__class__.__bases__[0].__subclasses__()"
        assert not v.is_safe(code)

    def test_nested_eval(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("x = eval(eval('1+1'))")

    def test_import_in_function(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("def f():\n    import os\n    return os.getcwd()")

    def test_open_in_class(self) -> None:
        v = ASTValidator()
        assert not v.is_safe("class A:\n    f = open('/etc/passwd')")

    def test_exec_via_alias_blocked(self) -> None:
        # Can't alias exec without import, but exec('...') direct should fail
        v = ASTValidator()
        assert not v.is_safe("exec('import os; os.system(\"ls\")')")


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_to_dict(self) -> None:
        r = ExecutionResult(
            success=True, output_env={"x": 42}, instructions_used=10, error=""
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["instructions_used"] == 10
        assert d["error"] == ""
        assert "x" in d["output_keys"]

    def test_to_dict_no_env_values(self) -> None:
        r = ExecutionResult(
            success=False, output_env={}, instructions_used=5, error="oops"
        )
        d = r.to_dict()
        assert d["output_keys"] == []

    def test_frozen(self) -> None:
        r = ExecutionResult(success=True, output_env={}, instructions_used=0, error="")
        with pytest.raises((AttributeError, TypeError)):
            r.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BudgetedExecutor — success paths
# ---------------------------------------------------------------------------


class TestBudgetedExecutorSuccess:
    def test_simple_assignment(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("x = 1 + 2")
        assert result.success
        assert result.output_env["x"] == 3

    def test_string_operation(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("s = 'hello'[::-1]")
        assert result.success
        assert result.output_env["s"] == "olleh"

    def test_list_comprehension(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("nums = [i*i for i in range(5)]")
        assert result.success
        assert result.output_env["nums"] == [0, 1, 4, 9, 16]

    def test_function_definition_and_call(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("def sq(n): return n*n\nresult = sq(7)")
        assert result.success
        assert result.output_env["result"] == 49

    def test_instructions_counted(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("x = 1")
        assert result.instructions_used > 0

    def test_output_env_strips_dunder_keys(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("x = 99")
        assert all(not k.startswith("__") for k in result.output_env)

    def test_empty_code(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("")
        assert result.success

    def test_multiple_variables(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("a = 1\nb = 2\nc = a + b")
        assert result.success
        assert result.output_env["c"] == 3


# ---------------------------------------------------------------------------
# BudgetedExecutor — validation failures
# ---------------------------------------------------------------------------


class TestBudgetedExecutorValidationFailure:
    def test_import_rejected(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("import os")
        assert not result.success
        assert "Validation failed" in result.error

    def test_eval_rejected(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("eval('1+1')")
        assert not result.success

    def test_open_rejected(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("f = open('/etc/passwd')")
        assert not result.success

    def test_instructions_zero_on_validation_failure(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("import os")
        assert result.instructions_used == 0


# ---------------------------------------------------------------------------
# BudgetedExecutor — runtime errors
# ---------------------------------------------------------------------------


class TestBudgetedExecutorRuntimeErrors:
    def test_division_by_zero(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("x = 1 / 0")
        assert not result.success
        assert "ZeroDivisionError" in result.error

    def test_name_error(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("x = undefined_variable")
        assert not result.success
        assert "NameError" in result.error

    def test_type_error(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("x = 1 + 'a'")
        assert not result.success

    def test_index_error(self) -> None:
        ex = BudgetedExecutor()
        result = ex.execute("lst = []\nx = lst[100]")
        assert not result.success


# ---------------------------------------------------------------------------
# BudgetedExecutor — budget exceeded
# ---------------------------------------------------------------------------


class TestBudgetedExecutorBudget:
    def test_infinite_loop_raises_timeout(self) -> None:
        ex = BudgetedExecutor(max_instructions=500)
        with pytest.raises(SandboxTimeoutError):
            ex.execute("x = 0\nwhile True:\n    x += 1")

    def test_small_budget_exceeded(self) -> None:
        ex = BudgetedExecutor(max_instructions=5)
        with pytest.raises(SandboxTimeoutError):
            ex.execute("x = 0\nfor i in range(1000):\n    x += i")

    def test_large_budget_not_exceeded(self) -> None:
        ex = BudgetedExecutor(max_instructions=100_000)
        result = ex.execute("x = sum(range(100))")
        assert result.success

    def test_timeout_error_has_correct_max(self) -> None:
        ex = BudgetedExecutor(max_instructions=10)
        try:
            ex.execute("x = 0\nwhile True:\n    x += 1")
        except SandboxTimeoutError as exc:
            assert exc.max_instructions == 10
        else:
            pytest.fail("Expected SandboxTimeoutError")
