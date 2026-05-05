"""Phase 43: Pure-Python Packaging — Single-File Executable Compiler.

``ZipAppCompiler`` uses Python's built-in :mod:`zipapp` module to bundle
the ``manifold/`` directory and a ``__main__.py`` entrypoint into a
single ``manifold.pyz`` binary that can be invoked with:

.. code-block:: shell

    python manifold.pyz --port 8080 --genesis --daemon

Key classes
-----------
``PackerConfig``
    Configuration for the ZipApp build.
``BuildResult``
    Outcome of a :meth:`ZipAppCompiler.build` call.
``ZipAppCompiler``
    Orchestrates the :mod:`zipapp` bundle creation.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import zipapp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# PackerConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackerConfig:
    """Configuration for :class:`ZipAppCompiler`.

    Parameters
    ----------
    source_dir:
        Root directory of the MANIFOLD project (must contain ``manifold/``).
        Defaults to the current working directory.
    output_path:
        Path for the compiled ``.pyz`` archive.
        Default: ``"manifold.pyz"`` (relative to *source_dir*).
    interpreter:
        Shebang interpreter line.  Default: ``"/usr/bin/env python3"``.
    compressed:
        Whether to apply ZIP compression.  Default: ``True``.
    """

    source_dir: str = field(default_factory=os.getcwd)
    output_path: str = "manifold.pyz"
    interpreter: str = "/usr/bin/env python3"
    compressed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "source_dir": self.source_dir,
            "output_path": self.output_path,
            "interpreter": self.interpreter,
            "compressed": self.compressed,
        }


# ---------------------------------------------------------------------------
# BuildResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildResult:
    """Outcome of a :meth:`ZipAppCompiler.build` call.

    Attributes
    ----------
    success:
        ``True`` if the archive was created successfully.
    output_path:
        Absolute path to the produced ``.pyz`` file (empty on failure).
    file_size_bytes:
        Size of the produced archive in bytes (``0`` on failure).
    elapsed_seconds:
        Wall-clock time taken for the build.
    error:
        Human-readable error message on failure (empty on success).
    """

    success: bool
    output_path: str
    file_size_bytes: int
    elapsed_seconds: float
    error: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "success": self.success,
            "output_path": self.output_path,
            "file_size_bytes": self.file_size_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# ZipAppCompiler
# ---------------------------------------------------------------------------


@dataclass
class ZipAppCompiler:
    """Bundles the MANIFOLD project into a self-contained ``.pyz`` archive.

    Parameters
    ----------
    config:
        :class:`PackerConfig` controlling build options.

    Example
    -------
    ::

        compiler = ZipAppCompiler()
        result = compiler.build()
        if result.success:
            print(f"Built {result.output_path} ({result.file_size_bytes} bytes)")
    """

    config: PackerConfig = field(default_factory=PackerConfig)

    def build(self) -> BuildResult:
        """Compile the project into a ``.pyz`` archive.

        The compiler:

        1. Creates a temporary staging directory.
        2. Copies the ``manifold/`` package into it.
        3. Writes a ``__main__.py`` entrypoint into the staging dir.
        4. Calls :func:`zipapp.create_archive` to produce the ``.pyz``.
        5. Returns a :class:`BuildResult` with the outcome.

        Returns
        -------
        BuildResult
        """
        t0 = time.time()
        source_dir = Path(self.config.source_dir).resolve()
        manifold_pkg = source_dir / "manifold"
        output_path = Path(self.config.output_path)
        if not output_path.is_absolute():
            output_path = source_dir / output_path

        if not manifold_pkg.is_dir():
            return BuildResult(
                success=False,
                output_path="",
                file_size_bytes=0,
                elapsed_seconds=time.time() - t0,
                error=f"manifold package directory not found: {manifold_pkg}",
            )

        tmp_dir: str | None = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="manifold_packer_")
            staging = Path(tmp_dir)

            # Copy manifold package into staging area
            shutil.copytree(str(manifold_pkg), str(staging / "manifold"))

            # Write the __main__.py entrypoint
            main_src = self._entrypoint_source()
            (staging / "__main__.py").write_text(main_src, encoding="utf-8")

            # Build the archive using zipapp
            zipapp.create_archive(
                str(staging),
                str(output_path),
                interpreter=self.config.interpreter,
                compressed=self.config.compressed,
            )

            size = output_path.stat().st_size if output_path.exists() else 0
            return BuildResult(
                success=True,
                output_path=str(output_path),
                file_size_bytes=size,
                elapsed_seconds=time.time() - t0,
                error="",
            )
        except Exception as exc:  # noqa: BLE001
            return BuildResult(
                success=False,
                output_path="",
                file_size_bytes=0,
                elapsed_seconds=time.time() - t0,
                error=str(exc),
            )
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _entrypoint_source() -> str:
        """Return the ``__main__.py`` source for the standalone archive."""
        return (
            '"""MANIFOLD v2.0.0 — Autonomic OS Entrypoint.\n\n'
            "Usage\n-----\n"
            "    python manifold.pyz [--port PORT] [--genesis] [--daemon]\n"
            '"""\n'
            "from __future__ import annotations\n\n"
            "import argparse\n\n\n"
            "def _build_parser() -> argparse.ArgumentParser:\n"
            '    parser = argparse.ArgumentParser(\n'
            '        prog="manifold",\n'
            '        description="MANIFOLD v2.0.0 — Autonomic Trust OS",\n'
            "    )\n"
            '    parser.add_argument("--port", type=int, default=8080)\n'
            '    parser.add_argument("--host", default="0.0.0.0")\n'
            '    parser.add_argument("--genesis", action="store_true")\n'
            '    parser.add_argument("--daemon", action="store_true")\n'
            "    return parser\n\n\n"
            "def main() -> int:\n"
            "    args = _build_parser().parse_args()\n"
            "    if args.genesis:\n"
            "        from manifold.genesis import GenesisMint, GenesisConfig\n"
            "        mint = GenesisMint(GenesisConfig())\n"
            '        allocs = mint.mint({"bootstrap-a": 1.0, "bootstrap-b": 2.0})\n'
            "        if not args.daemon:\n"
            '            print(f"[genesis] Minted {len(allocs)} allocations")\n'
            "    if args.daemon:\n"
            "        from manifold.watchdog import ProcessWatchdog\n"
            "        wd = ProcessWatchdog()\n"
            "        wd.start()\n"
            "    from manifold.server import run_server\n"
            "    try:\n"
            "        run_server(port=args.port, host=args.host)\n"
            "    except KeyboardInterrupt:\n"
            "        pass\n"
            "    return 0\n\n\n"
            'if __name__ == "__main__":\n'
            "    raise SystemExit(main())\n"
        )
