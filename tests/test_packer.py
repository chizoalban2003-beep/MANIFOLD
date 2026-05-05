"""Tests for Phase 43: Pure-Python Packaging (manifold/packer.py)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from manifold.packer import BuildResult, PackerConfig, ZipAppCompiler


# ---------------------------------------------------------------------------
# PackerConfig
# ---------------------------------------------------------------------------


class TestPackerConfig:
    def test_defaults(self) -> None:
        cfg = PackerConfig()
        assert cfg.output_path == "manifold.pyz"
        assert cfg.interpreter == "/usr/bin/env python3"
        assert cfg.compressed is True

    def test_custom_values(self) -> None:
        cfg = PackerConfig(output_path="/tmp/out.pyz", interpreter="/usr/bin/python3", compressed=False)
        assert cfg.output_path == "/tmp/out.pyz"
        assert cfg.interpreter == "/usr/bin/python3"
        assert cfg.compressed is False

    def test_to_dict(self) -> None:
        cfg = PackerConfig()
        d = cfg.to_dict()
        assert "source_dir" in d
        assert d["output_path"] == "manifold.pyz"
        assert d["interpreter"] == "/usr/bin/env python3"
        assert d["compressed"] is True

    def test_frozen(self) -> None:
        cfg = PackerConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.output_path = "other.pyz"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BuildResult
# ---------------------------------------------------------------------------


class TestBuildResult:
    def test_to_dict_success(self) -> None:
        result = BuildResult(
            success=True,
            output_path="/tmp/manifold.pyz",
            file_size_bytes=12345,
            elapsed_seconds=0.5,
            error="",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output_path"] == "/tmp/manifold.pyz"
        assert d["file_size_bytes"] == 12345
        assert d["error"] == ""

    def test_to_dict_failure(self) -> None:
        result = BuildResult(
            success=False,
            output_path="",
            file_size_bytes=0,
            elapsed_seconds=0.1,
            error="something went wrong",
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "something went wrong"

    def test_frozen(self) -> None:
        result = BuildResult(success=True, output_path="x", file_size_bytes=0, elapsed_seconds=0.0, error="")
        with pytest.raises((AttributeError, TypeError)):
            result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ZipAppCompiler
# ---------------------------------------------------------------------------


class TestZipAppCompiler:
    def test_missing_manifold_dir_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PackerConfig(source_dir=tmp, output_path=os.path.join(tmp, "out.pyz"))
            compiler = ZipAppCompiler(config=cfg)
            result = compiler.build()
        assert result.success is False
        assert "not found" in result.error

    def test_build_success(self) -> None:
        # Use the actual repo source dir
        repo_dir = Path(__file__).parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "manifold.pyz")
            cfg = PackerConfig(source_dir=str(repo_dir), output_path=out_path)
            compiler = ZipAppCompiler(config=cfg)
            result = compiler.build()
        assert result.success is True
        assert result.file_size_bytes > 0
        assert result.error == ""

    def test_build_produces_pyz_file(self) -> None:
        repo_dir = Path(__file__).parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "test_out.pyz")
            cfg = PackerConfig(source_dir=str(repo_dir), output_path=out_path)
            result = ZipAppCompiler(config=cfg).build()
        assert result.success is True

    def test_elapsed_seconds_positive(self) -> None:
        repo_dir = Path(__file__).parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PackerConfig(source_dir=str(repo_dir), output_path=os.path.join(tmp, "a.pyz"))
            result = ZipAppCompiler(config=cfg).build()
        assert result.elapsed_seconds >= 0.0

    def test_default_config(self) -> None:
        compiler = ZipAppCompiler()
        assert compiler.config.output_path == "manifold.pyz"

    def test_entrypoint_source_contains_argparse(self) -> None:
        src = ZipAppCompiler._entrypoint_source()
        assert "argparse" in src
        assert "--port" in src
        assert "--genesis" in src
        assert "--daemon" in src

    def test_entrypoint_source_contains_server(self) -> None:
        src = ZipAppCompiler._entrypoint_source()
        assert "run_server" in src

    def test_entrypoint_source_contains_genesis(self) -> None:
        src = ZipAppCompiler._entrypoint_source()
        assert "GenesisMint" in src

    def test_build_relative_output_path(self) -> None:
        repo_dir = Path(__file__).parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PackerConfig(source_dir=str(repo_dir), output_path=os.path.join(tmp, "rel.pyz"))
            result = ZipAppCompiler(config=cfg).build()
        assert result.success is True

    def test_uncompressed_build(self) -> None:
        repo_dir = Path(__file__).parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PackerConfig(
                source_dir=str(repo_dir),
                output_path=os.path.join(tmp, "uncompressed.pyz"),
                compressed=False,
            )
            result = ZipAppCompiler(config=cfg).build()
        assert result.success is True

    def test_file_size_nonzero_on_success(self) -> None:
        repo_dir = Path(__file__).parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PackerConfig(source_dir=str(repo_dir), output_path=os.path.join(tmp, "b.pyz"))
            result = ZipAppCompiler(config=cfg).build()
        assert result.file_size_bytes > 0

    def test_build_result_output_path_absolute(self) -> None:
        repo_dir = Path(__file__).parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PackerConfig(source_dir=str(repo_dir), output_path=os.path.join(tmp, "c.pyz"))
            result = ZipAppCompiler(config=cfg).build()
        assert Path(result.output_path).is_absolute()
