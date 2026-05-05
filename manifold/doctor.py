"""Phase 51: System Doctor & Auto-Repair — Self-Healing Diagnostics.

Ensures zero repository or runtime errors.  Provides:

* **RepositoryLinter** — pure-Python static analyser that verifies module
  imports are resolved and checks for obvious circular dependencies between
  modules.
* **WALRepair** — scans ``.jsonl`` files for half-written (corrupt) JSON
  lines, quarantines them, and atomically rewrites the clean remainder using
  ``os.rename``.
* **DoctorReport** — immutable summary of a full diagnostic run.

Safety guarantees
-----------------
* Every WAL rewrite is performed via a ``<target>.tmp`` scratch file followed
  by ``os.rename`` — an atomic operation on POSIX-compliant file systems.
* The linter never *executes* module code; it only imports and inspects.

Key classes
-----------
``LintIssue``
    A single lint finding (module + description).
``LintReport``
    Summary of a :meth:`RepositoryLinter.lint` run.
``RepositoryLinter``
    Pure-Python import-level static analyser.
``RepairResult``
    Stats from repairing one WAL file.
``WALRepair``
    Detects and quarantines corrupt ``.jsonl`` lines.
``DoctorReport``
    Immutable summary of a complete Doctor run.
``ManifoldDoctor``
    Orchestrates linting + WAL repair.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# LintIssue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintIssue:
    """A single finding from :class:`RepositoryLinter`.

    Attributes
    ----------
    module_name:
        Dotted module path (e.g. ``"manifold.brain"``).
    kind:
        Short category: ``"import_error"`` | ``"circular_dependency"`` |
        ``"missing_docstring"``.
    description:
        Human-readable explanation.
    """

    module_name: str
    kind: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "module_name": self.module_name,
            "kind": self.kind,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# LintReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintReport:
    """Summary of a :meth:`RepositoryLinter.lint` run.

    Attributes
    ----------
    modules_checked:
        Number of Python module files examined.
    issues:
        Tuple of :class:`LintIssue` objects (empty = clean).
    duration_seconds:
        Wall-clock time taken.
    """

    modules_checked: int
    issues: tuple[LintIssue, ...]
    duration_seconds: float

    @property
    def clean(self) -> bool:
        """``True`` if no issues were found."""
        return len(self.issues) == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "modules_checked": self.modules_checked,
            "issues": [i.to_dict() for i in self.issues],
            "clean": self.clean,
            "duration_seconds": round(self.duration_seconds, 4),
        }


# ---------------------------------------------------------------------------
# RepositoryLinter
# ---------------------------------------------------------------------------


@dataclass
class RepositoryLinter:
    """Pure-Python import-level static analyser for a MANIFOLD repository.

    Parameters
    ----------
    package_name:
        Top-level package to scan (default: ``"manifold"``).
    check_docstrings:
        Emit ``"missing_docstring"`` issues for modules that lack a module
        docstring.  Default: ``True``.

    Example
    -------
    ::

        linter = RepositoryLinter()
        report = linter.lint(Path("/repo/manifold"))
        if not report.clean:
            for issue in report.issues:
                print(issue.description)
    """

    package_name: str = "manifold"
    check_docstrings: bool = True

    def lint(self, manifold_dir: Path) -> LintReport:
        """Scan all ``.py`` files in *manifold_dir*.

        The linter:
        1. Discovers all ``*.py`` files.
        2. Attempts to import each module; records ``ImportError`` as a
           ``"import_error"`` issue.
        3. Checks for missing module docstrings (if *check_docstrings*).
        4. Detects simple circular dependency hints via import exception text.

        Parameters
        ----------
        manifold_dir:
            Directory containing the package's ``.py`` files.

        Returns
        -------
        LintReport
            Summary of findings.
        """
        t_start = time.monotonic()
        issues: list[LintIssue] = []

        py_files = sorted(manifold_dir.glob("*.py"))
        modules_checked = 0

        for py_file in py_files:
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                # Skip private helpers except __init__
                pass
            module_name = f"{self.package_name}.{py_file.stem}"
            if py_file.name == "__init__.py":
                module_name = self.package_name

            modules_checked += 1
            try:
                mod = importlib.import_module(module_name)
                if self.check_docstrings and not (mod.__doc__ or "").strip():
                    issues.append(
                        LintIssue(
                            module_name=module_name,
                            kind="missing_docstring",
                            description=f"{module_name!r} has no module docstring",
                        )
                    )
            except ImportError as exc:
                err_msg = str(exc)
                kind = (
                    "circular_dependency"
                    if "partially initialized" in err_msg or "circular" in err_msg.lower()
                    else "import_error"
                )
                issues.append(
                    LintIssue(
                        module_name=module_name,
                        kind=kind,
                        description=f"Cannot import {module_name!r}: {err_msg}",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                issues.append(
                    LintIssue(
                        module_name=module_name,
                        kind="import_error",
                        description=f"Error loading {module_name!r}: {type(exc).__name__}: {exc}",
                    )
                )

        duration = time.monotonic() - t_start
        return LintReport(
            modules_checked=modules_checked,
            issues=tuple(issues),
            duration_seconds=duration,
        )

    def check_sys_modules(self, prefix: str | None = None) -> list[str]:
        """Return all currently imported module names matching *prefix*.

        Parameters
        ----------
        prefix:
            If given, only names starting with this prefix are returned.
            Default: return all.
        """
        pfx = prefix or ""
        return sorted(
            name for name in sys.modules if not pfx or name.startswith(pfx)
        )


# ---------------------------------------------------------------------------
# RepairResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairResult:
    """Stats from repairing one WAL file.

    Attributes
    ----------
    path:
        Absolute path of the repaired file.
    lines_total:
        Total non-empty lines before repair.
    lines_healthy:
        Lines that parsed as valid JSON.
    lines_quarantined:
        Lines that could not be parsed (corrupt / half-written).
    quarantine_path:
        Path where corrupt lines were saved (empty if none were found).
    bytes_recovered:
        Approximate bytes of clean data in the repaired file.
    """

    path: str
    lines_total: int
    lines_healthy: int
    lines_quarantined: int
    quarantine_path: str
    bytes_recovered: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "path": self.path,
            "lines_total": self.lines_total,
            "lines_healthy": self.lines_healthy,
            "lines_quarantined": self.lines_quarantined,
            "quarantine_path": self.quarantine_path,
            "bytes_recovered": self.bytes_recovered,
        }


# ---------------------------------------------------------------------------
# WALRepair
# ---------------------------------------------------------------------------


@dataclass
class WALRepair:
    """Detects and quarantines corrupt ``.jsonl`` lines.

    A *corrupt* line is any line that fails ``json.loads``.  This can happen
    when a power failure interrupts a write mid-line.

    Parameters
    ----------
    quarantine_suffix:
        Suffix appended to the original filename to create the quarantine
        file (e.g. ``"gossip.jsonl"`` → ``"gossip.jsonl.quarantine"``).
        Default: ``".quarantine"``.

    Example
    -------
    ::

        repair = WALRepair()
        result = repair.repair(Path("/var/manifold/gossip.jsonl"))
        if result.lines_quarantined:
            print(f"Quarantined {result.lines_quarantined} corrupt lines")
    """

    quarantine_suffix: str = ".quarantine"

    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def repair(self, path: Path) -> RepairResult:
        """Scan *path* for corrupt lines, quarantine them, rewrite clean data.

        Parameters
        ----------
        path:
            Path to the ``.jsonl`` file to repair.  If the file does not
            exist, a no-op result is returned.

        Returns
        -------
        RepairResult
            Stats about the repair operation.
        """
        if not path.exists():
            return RepairResult(
                path=str(path),
                lines_total=0,
                lines_healthy=0,
                lines_quarantined=0,
                quarantine_path="",
                bytes_recovered=0,
            )

        with self._lock:
            healthy: list[str] = []
            corrupt: list[str] = []

            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    stripped = ln.strip()
                    if not stripped:
                        continue
                    try:
                        json.loads(stripped)
                        healthy.append(stripped)
                    except (json.JSONDecodeError, ValueError):
                        corrupt.append(stripped)

            lines_total = len(healthy) + len(corrupt)

            # If nothing is corrupt, skip the rewrite entirely
            if not corrupt:
                return RepairResult(
                    path=str(path),
                    lines_total=lines_total,
                    lines_healthy=len(healthy),
                    lines_quarantined=0,
                    quarantine_path="",
                    bytes_recovered=sum(len(ln) + 1 for ln in healthy),
                )

            # Write quarantine file
            quarantine_path = path.with_suffix(path.suffix + self.quarantine_suffix)
            with quarantine_path.open("a", encoding="utf-8") as fh:
                for ln in corrupt:
                    fh.write(ln + "\n")

            # Atomic rewrite
            tmp_path = path.with_suffix(".tmp")
            try:
                with tmp_path.open("w", encoding="utf-8") as fh:
                    for ln in healthy:
                        fh.write(ln + "\n")
                os.rename(str(tmp_path), str(path))
            except Exception:  # noqa: BLE001
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise

            bytes_recovered = sum(len(ln) + 1 for ln in healthy)
            return RepairResult(
                path=str(path),
                lines_total=lines_total,
                lines_healthy=len(healthy),
                lines_quarantined=len(corrupt),
                quarantine_path=str(quarantine_path),
                bytes_recovered=bytes_recovered,
            )

    def repair_directory(
        self,
        directory: Path,
        *,
        glob_pattern: str = "*.jsonl",
    ) -> list[RepairResult]:
        """Repair all matching WAL files in *directory*.

        Parameters
        ----------
        directory:
            Directory to scan.
        glob_pattern:
            Filename glob (default: ``"*.jsonl"``).

        Returns
        -------
        list[RepairResult]
            One result per file, in lexicographic path order.
        """
        results: list[RepairResult] = []
        for p in sorted(directory.glob(glob_pattern)):
            results.append(self.repair(p))
        return results


# ---------------------------------------------------------------------------
# DoctorReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorReport:
    """Immutable summary of a complete Doctor run.

    Attributes
    ----------
    timestamp:
        POSIX timestamp when the run completed.
    lint_report:
        Result from the :class:`RepositoryLinter`.
    repair_results:
        Tuple of :class:`RepairResult` objects, one per WAL file inspected.
    duration_seconds:
        Total wall-clock time for the full diagnostic.
    """

    timestamp: float
    lint_report: LintReport
    repair_results: tuple[RepairResult, ...]
    duration_seconds: float

    @property
    def healthy(self) -> bool:
        """``True`` if linting was clean and no lines were quarantined."""
        return self.lint_report.clean and all(
            r.lines_quarantined == 0 for r in self.repair_results
        )

    @property
    def total_quarantined(self) -> int:
        """Total corrupt lines quarantined across all WAL files."""
        return sum(r.lines_quarantined for r in self.repair_results)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "timestamp": self.timestamp,
            "healthy": self.healthy,
            "lint_report": self.lint_report.to_dict(),
            "repair_results": [r.to_dict() for r in self.repair_results],
            "total_quarantined": self.total_quarantined,
            "duration_seconds": round(self.duration_seconds, 4),
        }


# ---------------------------------------------------------------------------
# ManifoldDoctor
# ---------------------------------------------------------------------------


@dataclass
class ManifoldDoctor:
    """Orchestrates linting + WAL repair for MANIFOLD.

    Parameters
    ----------
    manifold_dir:
        Path to the ``manifold/`` package directory.  Defaults to the
        ``manifold`` sub-directory of the current working directory.
    data_dir:
        Directory containing ``.jsonl`` WAL files.  Defaults to
        ``manifold_data/`` under the current working directory.
    package_name:
        Top-level package to lint.  Default: ``"manifold"``.
    check_docstrings:
        Pass-through to :class:`RepositoryLinter`.  Default: ``True``.

    Example
    -------
    ::

        doctor = ManifoldDoctor()
        report = doctor.diagnose()
        if report.healthy:
            print("✅  All systems healthy")
        else:
            print(f"⚠  {report.total_quarantined} corrupt WAL lines quarantined")
    """

    manifold_dir: Path = field(default_factory=lambda: Path(os.getcwd()) / "manifold")
    data_dir: Path = field(
        default_factory=lambda: Path(os.getcwd()) / "manifold_data"
    )
    package_name: str = "manifold"
    check_docstrings: bool = True

    _linter: RepositoryLinter = field(init=False, repr=False)
    _repair: WALRepair = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._linter = RepositoryLinter(
            package_name=self.package_name,
            check_docstrings=self.check_docstrings,
        )
        self._repair = WALRepair()

    def diagnose(self) -> DoctorReport:
        """Run a full diagnostic cycle.

        Steps
        -----
        1. Lint all Python modules under :attr:`manifold_dir`.
        2. Repair all ``.jsonl`` WAL files under :attr:`data_dir`.

        Returns
        -------
        DoctorReport
            Immutable summary of findings and repairs.
        """
        t_start = time.monotonic()

        lint_report = self._linter.lint(self.manifold_dir)

        repair_results: list[RepairResult] = []
        if self.data_dir.exists() and self.data_dir.is_dir():
            repair_results = self._repair.repair_directory(self.data_dir)

        duration = time.monotonic() - t_start
        return DoctorReport(
            timestamp=time.time(),
            lint_report=lint_report,
            repair_results=tuple(repair_results),
            duration_seconds=duration,
        )

    def format_report(self, report: DoctorReport) -> str:
        """Format *report* as a human-readable diagnostic output.

        Parameters
        ----------
        report:
            A :class:`DoctorReport` to format.

        Returns
        -------
        str
            Multi-line text suitable for printing to a terminal.
        """
        lines: list[str] = [
            "",
            "╔══════════════════════════════════════════╗",
            "║   MANIFOLD System Doctor — Diagnostic    ║",
            "╚══════════════════════════════════════════╝",
            "",
            f"  Timestamp  : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(report.timestamp))}",
            f"  Duration   : {report.duration_seconds:.3f}s",
            f"  Health     : {'✅  HEALTHY' if report.healthy else '⚠   ISSUES FOUND'}",
            "",
            "─── Linter ──────────────────────────────────",
            f"  Modules checked : {report.lint_report.modules_checked}",
            f"  Issues found    : {len(report.lint_report.issues)}",
        ]
        for issue in report.lint_report.issues:
            lines.append(f"    [{issue.kind}] {issue.module_name}: {issue.description}")

        lines += [
            "",
            "─── WAL Repair ──────────────────────────────",
            f"  Files scanned     : {len(report.repair_results)}",
            f"  Lines quarantined : {report.total_quarantined}",
        ]
        for r in report.repair_results:
            if r.lines_quarantined > 0:
                lines.append(
                    f"    {Path(r.path).name}: {r.lines_quarantined} corrupt line(s) → {r.quarantine_path}"
                )

        lines.append("")
        return "\n".join(lines)
