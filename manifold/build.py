"""Phase 52: Manifold Compiler & Release Bundler — Production Build System.

Automates the complete release pipeline:

1. Run the full ``pytest`` test suite.
2. Execute the :class:`~manifold.doctor.RepositoryLinter`.
3. Compile a ``manifold.pyz`` single-file executable (via :mod:`zipapp`).
4. Generate a ``release.manifest.json`` with SHA-256 checksums and an
   HMAC-SHA256 signature of the binary (Ed25519-style, pure stdlib).

Only proceeds to the next step when all checks pass.

Key classes
-----------
``ReleaseConfig``
    Configuration for the build pipeline.
``ReleaseStepResult``
    Outcome of one pipeline step.
``ReleaseManifest``
    Cryptographically signed release artefact descriptor.
``ReleaseBuilder``
    Orchestrates the full release pipeline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ReleaseConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseConfig:
    """Configuration for :class:`ReleaseBuilder`.

    Parameters
    ----------
    source_dir:
        Root of the MANIFOLD repository.  Defaults to ``os.getcwd()``.
    output_dir:
        Directory where build artefacts are written.  Defaults to
        ``<source_dir>/dist``.
    pyz_name:
        Filename for the compiled ``zipapp`` binary.
        Default: ``"manifold.pyz"``.
    signing_key:
        Hex-encoded 32-byte HMAC secret used to sign the binary.  If empty,
        a random ephemeral key is generated.
    run_tests:
        Whether to run the ``pytest`` suite before building.  Default: ``True``.
    run_lint:
        Whether to run the :class:`~manifold.doctor.RepositoryLinter` before
        compiling.  Default: ``True``.
    test_timeout_seconds:
        Maximum seconds allowed for the test suite.  Default: ``600``.
    """

    source_dir: str = field(default_factory=os.getcwd)
    output_dir: str = ""
    pyz_name: str = "manifold.pyz"
    signing_key: str = ""
    run_tests: bool = True
    run_lint: bool = True
    test_timeout_seconds: float = 600.0

    def resolved_output_dir(self) -> Path:
        """Return the resolved output directory Path."""
        if self.output_dir:
            return Path(self.output_dir)
        return Path(self.source_dir) / "dist"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (key redacted)."""
        return {
            "source_dir": self.source_dir,
            "output_dir": str(self.resolved_output_dir()),
            "pyz_name": self.pyz_name,
            "signing_key": "<redacted>" if self.signing_key else "<ephemeral>",
            "run_tests": self.run_tests,
            "run_lint": self.run_lint,
            "test_timeout_seconds": self.test_timeout_seconds,
        }


# ---------------------------------------------------------------------------
# ReleaseStepResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseStepResult:
    """Outcome of one pipeline step.

    Attributes
    ----------
    step_name:
        Short step identifier (e.g. ``"tests"``, ``"lint"``, ``"compile"``).
    passed:
        Whether the step succeeded.
    duration_seconds:
        Wall-clock time for this step.
    output:
        Captured stdout/stderr text (truncated to 4096 chars).
    """

    step_name: str
    passed: bool
    duration_seconds: float
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "step_name": self.step_name,
            "passed": self.passed,
            "duration_seconds": round(self.duration_seconds, 3),
            "output": self.output[:4096],
        }


# ---------------------------------------------------------------------------
# ReleaseManifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseManifest:
    """Signed descriptor of a MANIFOLD release artefact.

    Attributes
    ----------
    version:
        Release version string (e.g. ``"3.0.0"``).
    pyz_path:
        Absolute path of the compiled ``.pyz`` binary.
    sha256:
        Hex-encoded SHA-256 digest of the binary.
    signature:
        HMAC-SHA256 hex signature of the digest, keyed with the signing key.
    key_id:
        Short identifier for the signing key (first 8 hex chars of its hash).
    timestamp:
        POSIX timestamp when the manifest was created.
    steps:
        Tuple of :class:`ReleaseStepResult` objects in pipeline order.
    """

    version: str
    pyz_path: str
    sha256: str
    signature: str
    key_id: str
    timestamp: float
    steps: tuple[ReleaseStepResult, ...]

    @property
    def all_passed(self) -> bool:
        """``True`` if every pipeline step passed."""
        return all(s.passed for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "version": self.version,
            "pyz_path": self.pyz_path,
            "sha256": self.sha256,
            "signature": self.signature,
            "key_id": self.key_id,
            "timestamp": self.timestamp,
            "all_passed": self.all_passed,
            "steps": [s.to_dict() for s in self.steps],
        }

    def save(self, path: Path) -> None:
        """Write the manifest as ``release.manifest.json`` to *path*.

        Parameters
        ----------
        path:
            Target file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# ReleaseBuilder
# ---------------------------------------------------------------------------


@dataclass
class ReleaseBuilder:
    """Orchestrates the full MANIFOLD release pipeline.

    Parameters
    ----------
    config:
        Build configuration.

    Example
    -------
    ::

        builder = ReleaseBuilder(ReleaseConfig(source_dir="/repo"))
        manifest = builder.build(version="3.0.0")
        if manifest.all_passed:
            print(f"✅  Release {manifest.version} built: {manifest.sha256[:12]}…")
        else:
            print("❌  Build failed")
    """

    config: ReleaseConfig = field(default_factory=ReleaseConfig)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def build(self, version: str = "3.0.0") -> ReleaseManifest:
        """Run the full release pipeline.

        Steps (in order):
        1. ``tests``   — ``pytest tests/ -q`` (if *run_tests*).
        2. ``lint``    — :class:`~manifold.doctor.RepositoryLinter` (if *run_lint*).
        3. ``compile`` — :mod:`zipapp` bundle.
        4. ``sign``    — SHA-256 + HMAC-SHA256 signature.

        The pipeline halts at the first failing step.

        Parameters
        ----------
        version:
            Version string to embed in the manifest.

        Returns
        -------
        ReleaseManifest
            Contains step results and cryptographic checksums.
        """
        output_dir = self.config.resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        steps: list[ReleaseStepResult] = []

        # Step 1: Tests
        if self.config.run_tests:
            step = self._run_tests()
            steps.append(step)
            if not step.passed:
                return self._make_manifest(version, steps, output_dir)

        # Step 2: Lint
        if self.config.run_lint:
            step = self._run_lint()
            steps.append(step)
            if not step.passed:
                return self._make_manifest(version, steps, output_dir)

        # Step 3: Compile
        pyz_path = output_dir / self.config.pyz_name
        step = self._compile(pyz_path)
        steps.append(step)
        if not step.passed:
            return self._make_manifest(version, steps, output_dir)

        # Step 4: Sign
        sign_step = self._sign(pyz_path)
        steps.append(sign_step)

        return self._make_manifest(version, steps, output_dir, pyz_path=pyz_path)

    # ------------------------------------------------------------------
    # Private step runners
    # ------------------------------------------------------------------

    def _run_tests(self) -> ReleaseStepResult:
        """Run ``pytest`` and capture output."""
        t = time.monotonic()
        try:
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "pytest",
                    "tests/",
                    "-q",
                    "--tb=short",
                    "--no-header",
                ],
                cwd=self.config.source_dir,
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout_seconds,
            )
            passed = result.returncode == 0
            output = (result.stdout + result.stderr).strip()
        except subprocess.TimeoutExpired:
            passed = False
            output = f"pytest timed out after {self.config.test_timeout_seconds}s"
        except Exception as exc:  # noqa: BLE001
            passed = False
            output = f"pytest failed to start: {exc}"

        return ReleaseStepResult(
            step_name="tests",
            passed=passed,
            duration_seconds=time.monotonic() - t,
            output=output,
        )

    def _run_lint(self) -> ReleaseStepResult:
        """Run the RepositoryLinter."""
        from .doctor import RepositoryLinter

        t = time.monotonic()
        try:
            manifold_dir = Path(self.config.source_dir) / "manifold"
            linter = RepositoryLinter(check_docstrings=False)
            report = linter.lint(manifold_dir)
            passed = report.clean
            output_lines = [f"Modules checked: {report.modules_checked}"]
            for issue in report.issues:
                output_lines.append(f"  [{issue.kind}] {issue.module_name}: {issue.description}")
            output = "\n".join(output_lines)
        except Exception as exc:  # noqa: BLE001
            passed = False
            output = f"Linter error: {exc}"

        return ReleaseStepResult(
            step_name="lint",
            passed=passed,
            duration_seconds=time.monotonic() - t,
            output=output,
        )

    def _compile(self, pyz_path: Path) -> ReleaseStepResult:
        """Bundle the manifold package as a zipapp."""
        import shutil
        import tempfile
        import zipapp

        t = time.monotonic()
        try:
            src = Path(self.config.source_dir)
            # Build a clean staging directory
            with tempfile.TemporaryDirectory() as tmp:
                staging = Path(tmp) / "manifold"
                shutil.copytree(
                    src / "manifold",
                    staging,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                )
                # Copy __main__.py to staging root for the zipapp entrypoint
                main_src = src / "manifold" / "__main__.py"
                if main_src.exists():
                    shutil.copy(main_src, Path(tmp) / "__main__.py")

                zipapp.create_archive(
                    source=tmp,
                    target=str(pyz_path),
                    interpreter="/usr/bin/env python3",
                    compressed=True,
                )
            output = f"Compiled → {pyz_path} ({pyz_path.stat().st_size} bytes)"
            passed = True
        except Exception as exc:  # noqa: BLE001
            output = f"Compile error: {exc}"
            passed = False

        return ReleaseStepResult(
            step_name="compile",
            passed=passed,
            duration_seconds=time.monotonic() - t,
            output=output,
        )

    def _sign(self, pyz_path: Path) -> ReleaseStepResult:
        """Generate SHA-256 + HMAC-SHA256 signature for the binary."""
        t = time.monotonic()
        try:
            if not pyz_path.exists():
                raise FileNotFoundError(f"{pyz_path} not found")
            data = pyz_path.read_bytes()
            sha256 = hashlib.sha256(data).hexdigest()
            output = f"SHA-256: {sha256}"
            passed = True
        except Exception as exc:  # noqa: BLE001
            output = f"Sign error: {exc}"
            passed = False

        return ReleaseStepResult(
            step_name="sign",
            passed=passed,
            duration_seconds=time.monotonic() - t,
            output=output,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_sha256(self, path: Path) -> str:
        """Return hex SHA-256 of the file at *path*."""
        if not path.exists():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _get_signing_key(self) -> bytes:
        """Return the 32-byte signing key (or generate an ephemeral one)."""
        if self.config.signing_key:
            raw = bytes.fromhex(self.config.signing_key)
            return raw[:32] if len(raw) >= 32 else raw.ljust(32, b"\x00")
        return secrets.token_bytes(32)

    def _make_manifest(
        self,
        version: str,
        steps: list[ReleaseStepResult],
        output_dir: Path,
        pyz_path: Path | None = None,
    ) -> ReleaseManifest:
        """Assemble a :class:`ReleaseManifest` from completed steps."""
        if pyz_path is None:
            pyz_path = output_dir / self.config.pyz_name

        sha256 = self._compute_sha256(pyz_path) if pyz_path.exists() else ""

        key = self._get_signing_key()
        key_id = hashlib.sha256(key).hexdigest()[:8]
        sig = (
            hmac.new(key, sha256.encode(), hashlib.sha256).hexdigest()
            if sha256
            else ""
        )

        manifest = ReleaseManifest(
            version=version,
            pyz_path=str(pyz_path),
            sha256=sha256,
            signature=sig,
            key_id=key_id,
            timestamp=time.time(),
            steps=tuple(steps),
        )

        if all(s.passed for s in steps) and sha256:
            manifest.save(output_dir / "release.manifest.json")

        return manifest
