"""Phase 44: AST Execution Sandbox — Safe LLM-Generated Code Runner.

Agents often produce Python code snippets to solve tasks.  This module
provides a two-stage pipeline that first *statically validates* the code
via the ``ast`` module and then *budgeted-executes* it inside a fully
isolated ``exec()`` context.

Pipeline
--------
1. :class:`ASTValidator` scans the AST for forbidden constructs
   (imports, dangerous built-in calls, attribute access to private
   internals, ``__builtins__`` manipulation).
2. :class:`BudgetedExecutor` compiles and executes the sanitised code
   with ``sys.settrace()`` counting every opcode.  If the budget is
   exceeded a :class:`SandboxTimeoutError` is raised.

Key classes
-----------
``SandboxViolation``
    Describes a single forbidden pattern found in the source.
``SandboxTimeoutError``
    Raised when the execution budget is exceeded.
``ASTValidator``
    Static analyser that whitelists safe AST node types.
``ExecutionResult``
    Immutable result of a :class:`BudgetedExecutor` run.
``BudgetedExecutor``
    Runs sanitised code under an opcode budget.
"""

from __future__ import annotations

import ast
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# SandboxViolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxViolation:
    """A single forbidden pattern detected by :class:`ASTValidator`.

    Attributes
    ----------
    rule:
        Short rule name (e.g. ``"import_not_allowed"``).
    description:
        Human-readable description of the violation.
    lineno:
        Source line number (1-based) where the violation was found,
        or ``0`` if not applicable.
    """

    rule: str
    description: str
    lineno: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {"rule": self.rule, "description": self.description, "lineno": self.lineno}


# ---------------------------------------------------------------------------
# SandboxTimeoutError
# ---------------------------------------------------------------------------


class SandboxTimeoutError(RuntimeError):
    """Raised when code execution exceeds the allowed opcode budget.

    Attributes
    ----------
    instructions_used:
        Number of opcodes executed before the budget was exhausted.
    max_instructions:
        The configured limit.
    """

    def __init__(self, instructions_used: int, max_instructions: int) -> None:
        self.instructions_used = instructions_used
        self.max_instructions = max_instructions
        super().__init__(
            f"Execution budget exceeded: {instructions_used} opcodes used "
            f"(max={max_instructions})"
        )


# ---------------------------------------------------------------------------
# ASTValidator
# ---------------------------------------------------------------------------

# Built-in names that must never be called inside sandboxed code.
_FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "input",
        "breakpoint",
        "vars",
        "globals",
        "locals",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    }
)

# Attribute names that may expose interpreter internals.
_FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__func__",
        "__self__",
        "__dict__",
        "__module__",
        "__qualname__",
        "__reduce__",
        "__reduce_ex__",
        "__init_subclass__",
    }
)

# Safe subset of Python built-ins provided to sandboxed code.
# Dangerous callables (eval, exec, open, __import__, etc.) are excluded.
_SAFE_BUILTINS: dict[str, object] = {
    # Numeric / math
    "abs": abs,
    "divmod": divmod,
    "hash": hash,
    "hex": hex,
    "max": max,
    "min": min,
    "oct": oct,
    "pow": pow,
    "round": round,
    "sum": sum,
    # Type constructors
    "bool": bool,
    "bytes": bytes,
    "complex": complex,
    "dict": dict,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "list": list,
    "object": object,
    "set": set,
    "str": str,
    "tuple": tuple,
    # Iteration / sequence helpers
    "all": all,
    "any": any,
    "enumerate": enumerate,
    "filter": filter,
    "iter": iter,
    "len": len,
    "map": map,
    "next": next,
    "range": range,
    "reversed": reversed,
    "slice": slice,
    "sorted": sorted,
    "zip": zip,
    # Repr / formatting
    "bin": bin,
    "chr": chr,
    "format": format,
    "ord": ord,
    "repr": repr,
    # Type introspection (safe read-only)
    "callable": callable,
    "id": id,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "type": type,
    # Exceptions / error handling
    "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "AttributeError": AttributeError,
    "EOFError": EOFError,
    "Exception": Exception,
    "FloatingPointError": FloatingPointError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "LookupError": LookupError,
    "MemoryError": MemoryError,
    "NameError": NameError,
    "NotImplementedError": NotImplementedError,
    "OverflowError": OverflowError,
    "RecursionError": RecursionError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "SyntaxError": SyntaxError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
    # Constants
    "None": None,
    "True": True,
    "False": False,
    "NotImplemented": NotImplemented,
    "Ellipsis": ...,
    # print is useful and not dangerous in this context
    "print": print,
}


@dataclass
class ASTValidator(ast.NodeVisitor):
    """Static analyser that walks an AST and blocks forbidden constructs.

    A piece of code is **valid** if and only if :meth:`validate` returns an
    empty list.

    Forbidden patterns
    ------------------
    * Any ``import`` or ``from … import`` statement.
    * Calls to functions in :data:`_FORBIDDEN_CALLS`.
    * Access to dunder attributes in :data:`_FORBIDDEN_ATTRS`.
    * Direct assignment to ``__builtins__``.

    Example
    -------
    ::

        validator = ASTValidator()
        violations = validator.validate("x = 1 + 2")
        assert violations == []

        violations = validator.validate("import os")
        assert violations[0].rule == "import_not_allowed"
    """

    _violations: list[SandboxViolation] = field(
        default_factory=list, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, source: str) -> list[SandboxViolation]:
        """Parse *source* and return a list of :class:`SandboxViolation`.

        Parameters
        ----------
        source:
            Python source code to analyse.

        Returns
        -------
        list[SandboxViolation]
            Empty list if the code is safe; one entry per violation found.
        """
        self._violations = []
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            self._violations.append(
                SandboxViolation(
                    rule="syntax_error",
                    description=str(exc),
                    lineno=getattr(exc, "lineno", 0) or 0,
                )
            )
            return list(self._violations)
        self.visit(tree)
        return list(self._violations)

    def is_safe(self, source: str) -> bool:
        """Return ``True`` if *source* has no violations."""
        return len(self.validate(source)) == 0

    # ------------------------------------------------------------------
    # AST visitor overrides
    # ------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self._violations.append(
            SandboxViolation(
                rule="import_not_allowed",
                description=f"import statement is forbidden: {ast.unparse(node)}",
                lineno=node.lineno,
            )
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self._violations.append(
            SandboxViolation(
                rule="import_not_allowed",
                description=f"from-import statement is forbidden: {ast.unparse(node)}",
                lineno=node.lineno,
            )
        )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        name: str | None = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr

        if name in _FORBIDDEN_CALLS:
            self._violations.append(
                SandboxViolation(
                    rule="forbidden_call",
                    description=f"call to {name!r} is not allowed",
                    lineno=node.lineno,
                )
            )
        # Recurse into arguments, but not into the function itself
        # (already handled above).
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr in _FORBIDDEN_ATTRS:
            self._violations.append(
                SandboxViolation(
                    rule="forbidden_attribute",
                    description=f"access to attribute {node.attr!r} is not allowed",
                    lineno=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__builtins__":
                self._violations.append(
                    SandboxViolation(
                        rule="builtins_overwrite",
                        description="assignment to __builtins__ is not allowed",
                        lineno=node.lineno,
                    )
                )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionResult:
    """Immutable result of a :class:`BudgetedExecutor` run.

    Attributes
    ----------
    success:
        ``True`` if execution completed without error.
    output_env:
        The ``locals`` dict produced by ``exec()`` (contains any variables
        the code created).
    instructions_used:
        Number of opcodes counted by ``sys.settrace``.
    error:
        Human-readable error description (empty string on success).
    """

    success: bool
    output_env: dict[str, Any]
    instructions_used: int
    error: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (output_env values omitted)."""
        return {
            "success": self.success,
            "instructions_used": self.instructions_used,
            "error": self.error,
            "output_keys": list(self.output_env.keys()),
        }


# ---------------------------------------------------------------------------
# BudgetedExecutor
# ---------------------------------------------------------------------------


@dataclass
class BudgetedExecutor:
    """Execute sanitised Python code with an opcode budget.

    The executor uses :func:`sys.settrace` to count the number of line-trace
    events (each corresponds to roughly one statement or expression
    evaluation) and raises :class:`SandboxTimeoutError` if the count exceeds
    *max_instructions*.

    Parameters
    ----------
    max_instructions:
        Maximum number of trace events before execution is aborted.
        Default: ``10_000``.
    validator:
        :class:`ASTValidator` instance to use for pre-execution validation.
        A fresh instance is created if ``None``.

    Example
    -------
    ::

        executor = BudgetedExecutor(max_instructions=500)
        result = executor.execute("x = sum(range(10))")
        assert result.success
        assert result.output_env["x"] == 45
    """

    max_instructions: int = 10_000
    validator: ASTValidator = field(default_factory=ASTValidator)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, source: str) -> ExecutionResult:
        """Validate and execute *source*.

        Parameters
        ----------
        source:
            Python source code to run.

        Returns
        -------
        ExecutionResult
            On success, ``output_env`` contains the variables created by
            *source*.  On failure (validation error, runtime exception, or
            budget exceeded), ``success`` is ``False`` and ``error``
            describes the problem.

        Raises
        ------
        SandboxTimeoutError
            If the opcode count exceeds :attr:`max_instructions`.
        """
        # Stage 1: static validation
        violations = self.validator.validate(source)
        if violations:
            msg = "; ".join(v.description for v in violations)
            return ExecutionResult(
                success=False,
                output_env={},
                instructions_used=0,
                error=f"Validation failed: {msg}",
            )

        # Stage 2: budgeted execution
        counter: list[int] = [0]
        max_instrs = self.max_instructions

        def _tracer(frame: Any, event: str, arg: Any) -> Any:  # noqa: ARG001
            counter[0] += 1
            if counter[0] > max_instrs:
                raise SandboxTimeoutError(counter[0], max_instrs)
            return _tracer

        env: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        old_trace = sys.gettrace()
        try:
            code = compile(source, "<sandbox>", "exec")
            sys.settrace(_tracer)
            exec(code, env)  # noqa: S102
        except SandboxTimeoutError:
            raise
        except Exception:  # noqa: BLE001
            # Stop tracing immediately before formatting the traceback so
            # that Python's internal suggestion/distance machinery does not
            # consume the remaining budget and trigger a spurious timeout.
            sys.settrace(None)
            tb = traceback.format_exc()
            return ExecutionResult(
                success=False,
                output_env={},
                instructions_used=counter[0],
                error=tb,
            )
        finally:
            sys.settrace(old_trace)

        # Strip internal keys from the output env
        clean_env = {k: v for k, v in env.items() if not k.startswith("__")}
        return ExecutionResult(
            success=True,
            output_env=clean_env,
            instructions_used=counter[0],
            error="",
        )
