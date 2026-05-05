"""Tests for Phase 59: Native Swarm Orchestrator — SSH Wrapper (deploy/ssh.py).

All tests use ``unittest.mock`` to intercept ``subprocess.run`` calls so no
real SSH connections are required.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from manifold.deploy.ssh import (
    RemoteCommandResult,
    SSHClient,
    SecureCopy,
    TransferResult,
)


# ---------------------------------------------------------------------------
# RemoteCommandResult
# ---------------------------------------------------------------------------


class TestRemoteCommandResult:
    def test_ok_when_returncode_zero(self) -> None:
        r = RemoteCommandResult("host", "cmd", 0, "out", "", 0.1)
        assert r.ok is True

    def test_not_ok_when_returncode_nonzero(self) -> None:
        r = RemoteCommandResult("host", "cmd", 1, "", "err", 0.1)
        assert r.ok is False

    def test_to_dict_keys(self) -> None:
        r = RemoteCommandResult("h", "cmd", 0, "out", "err", 1.23)
        d = r.to_dict()
        assert d["host"] == "h"
        assert d["command"] == "cmd"
        assert d["returncode"] == 0
        assert d["stdout"] == "out"
        assert d["stderr"] == "err"
        assert d["ok"] is True
        assert "elapsed_seconds" in d

    def test_to_dict_not_ok(self) -> None:
        r = RemoteCommandResult("h", "cmd", 2, "", "fail", 0.5)
        assert r.to_dict()["ok"] is False

    def test_to_dict_elapsed_rounded(self) -> None:
        r = RemoteCommandResult("h", "cmd", 0, "", "", 1.123456789)
        d = r.to_dict()
        assert isinstance(d["elapsed_seconds"], float)
        assert len(str(d["elapsed_seconds"])) <= 10  # reasonable rounding


# ---------------------------------------------------------------------------
# TransferResult
# ---------------------------------------------------------------------------


class TestTransferResult:
    def test_ok_property(self) -> None:
        r = TransferResult("host", "/local/f", "/remote/f", 0, "", 0.1)
        assert r.ok is True

    def test_not_ok_property(self) -> None:
        r = TransferResult("host", "/local/f", "/remote/f", 1, "err", 0.1)
        assert r.ok is False

    def test_to_dict_keys(self) -> None:
        r = TransferResult("h", "/a", "/b", 0, "", 2.5)
        d = r.to_dict()
        assert d["host"] == "h"
        assert d["local_path"] == "/a"
        assert d["remote_path"] == "/b"
        assert d["returncode"] == 0
        assert d["ok"] is True
        assert "elapsed_seconds" in d


# ---------------------------------------------------------------------------
# SSHClient — construction
# ---------------------------------------------------------------------------


class TestSSHClientConstruction:
    def test_defaults(self) -> None:
        c = SSHClient("10.0.0.1")
        assert c.host == "10.0.0.1"
        assert c.user == "root"
        assert c.port == 22
        assert c.key_path is None
        assert c.connect_timeout == 10
        assert c.strict_host_checking is False

    def test_custom_params(self) -> None:
        c = SSHClient("h", user="ubuntu", key_path="/k", port=2222, connect_timeout=5)
        assert c.user == "ubuntu"
        assert c.key_path == "/k"
        assert c.port == 2222
        assert c.connect_timeout == 5

    def test_base_args_no_key(self) -> None:
        c = SSHClient("h")
        args = c._base_args()
        assert "ssh" in args
        assert "-i" not in args
        assert "StrictHostKeyChecking=no" in " ".join(args)

    def test_base_args_with_key(self) -> None:
        c = SSHClient("h", key_path="/mykey")
        args = c._base_args()
        assert "-i" in args
        idx = args.index("-i")
        assert args[idx + 1] == "/mykey"

    def test_base_args_strict_host_checking_yes(self) -> None:
        c = SSHClient("h", strict_host_checking=True)
        args = c._base_args()
        assert "StrictHostKeyChecking=yes" in " ".join(args)

    def test_base_args_port_included(self) -> None:
        c = SSHClient("h", port=2222)
        args = c._base_args()
        assert str(2222) in args


# ---------------------------------------------------------------------------
# SSHClient.run — mock subprocess
# ---------------------------------------------------------------------------


class TestSSHClientRun:
    def _mock_proc(self, returncode=0, stdout="", stderr=""):
        proc = MagicMock()
        proc.returncode = returncode
        proc.stdout = stdout
        proc.stderr = stderr
        return proc

    def test_run_success(self) -> None:
        client = SSHClient("10.0.0.1")
        with patch("subprocess.run", return_value=self._mock_proc(0, "hello", "")) as mock_run:
            result = client.run("echo hello")
        mock_run.assert_called_once()
        assert result.ok is True
        assert result.stdout == "hello"
        assert result.returncode == 0
        assert result.host == "10.0.0.1"
        assert result.command == "echo hello"

    def test_run_failure(self) -> None:
        client = SSHClient("10.0.0.1")
        with patch("subprocess.run", return_value=self._mock_proc(1, "", "error")) as mock_run:
            result = client.run("bad command")
        mock_run.assert_called_once()
        assert result.ok is False
        assert result.returncode == 1
        assert result.stderr == "error"

    def test_run_includes_host_in_args(self) -> None:
        client = SSHClient("10.0.0.1", user="ubuntu")
        captured_args = []

        def fake_run(args, **kwargs):
            captured_args.extend(args)
            return self._mock_proc()

        with patch("subprocess.run", side_effect=fake_run):
            client.run("ls")

        assert "ubuntu@10.0.0.1" in captured_args
        assert "ls" in captured_args

    def test_run_timeout_returns_error_result(self) -> None:
        client = SSHClient("10.0.0.1")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["ssh"], 5)):
            result = client.run("slow_cmd", timeout=5)
        assert result.ok is False
        assert result.returncode == -1
        assert "Timeout" in result.stderr

    def test_run_ssh_not_found(self) -> None:
        client = SSHClient("10.0.0.1")
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = client.run("cmd")
        assert result.ok is False
        assert result.returncode == -2
        assert "not found" in result.stderr

    def test_run_passes_timeout_kwarg(self) -> None:
        client = SSHClient("10.0.0.1")
        received = {}

        def fake_run(args, **kwargs):
            received.update(kwargs)
            return self._mock_proc()

        with patch("subprocess.run", side_effect=fake_run):
            client.run("cmd", timeout=30)

        assert received.get("timeout") == 30

    def test_run_elapsed_positive(self) -> None:
        client = SSHClient("10.0.0.1")
        with patch("subprocess.run", return_value=self._mock_proc()):
            result = client.run("cmd")
        assert result.elapsed_seconds >= 0.0

    def test_run_key_passed_to_ssh(self) -> None:
        client = SSHClient("10.0.0.1", key_path="/my/key")
        captured_args = []

        def fake_run(args, **kwargs):
            captured_args.extend(args)
            return self._mock_proc()

        with patch("subprocess.run", side_effect=fake_run):
            client.run("cmd")

        assert "-i" in captured_args
        assert "/my/key" in captured_args

    def test_run_many_calls_run_for_each_command(self) -> None:
        client = SSHClient("10.0.0.1")
        call_count = []

        def fake_run(args, **kwargs):
            call_count.append(1)
            return self._mock_proc()

        commands = ["cmd1", "cmd2", "cmd3"]
        with patch("subprocess.run", side_effect=fake_run):
            results = client.run_many(commands)

        assert len(results) == 3
        assert len(call_count) == 3

    def test_run_many_returns_list_of_results(self) -> None:
        client = SSHClient("10.0.0.1")
        with patch("subprocess.run", return_value=self._mock_proc(0, "ok", "")):
            results = client.run_many(["a", "b"])
        assert all(isinstance(r, RemoteCommandResult) for r in results)
        assert all(r.ok for r in results)

    def test_run_many_empty_list(self) -> None:
        client = SSHClient("10.0.0.1")
        with patch("subprocess.run", return_value=self._mock_proc()):
            results = client.run_many([])
        assert results == []


# ---------------------------------------------------------------------------
# SecureCopy — construction
# ---------------------------------------------------------------------------


class TestSecureCopyConstruction:
    def test_defaults(self) -> None:
        scp = SecureCopy("10.0.0.1")
        assert scp.host == "10.0.0.1"
        assert scp.user == "root"
        assert scp.port == 22
        assert scp.remote_base_dir == "/opt/manifold"

    def test_custom_remote_base_dir(self) -> None:
        scp = SecureCopy("h", remote_base_dir="/home/ubuntu/manifold")
        assert scp.remote_base_dir == "/home/ubuntu/manifold"

    def test_base_args_uses_capital_P(self) -> None:
        scp = SecureCopy("h", port=2222)
        args = scp._base_args()
        assert "-P" in args
        assert "2222" in args


# ---------------------------------------------------------------------------
# SecureCopy.upload — mock subprocess
# ---------------------------------------------------------------------------


class TestSecureCopyUpload:
    def _mock_proc(self, returncode=0, stderr=""):
        proc = MagicMock()
        proc.returncode = returncode
        proc.stderr = stderr
        return proc

    def test_upload_success(self) -> None:
        scp = SecureCopy("10.0.0.1")
        with patch("subprocess.run", return_value=self._mock_proc(0)) as mock_run:
            result = scp.upload("/local/file.pyz")
        mock_run.assert_called_once()
        assert result.ok is True
        assert result.local_path == "/local/file.pyz"
        assert result.host == "10.0.0.1"

    def test_upload_default_remote_path_uses_base_dir(self) -> None:
        scp = SecureCopy("10.0.0.1", remote_base_dir="/opt/manifold")
        captured_args = []

        def fake_run(args, **kwargs):
            captured_args.extend(args)
            return self._mock_proc()

        with patch("subprocess.run", side_effect=fake_run):
            result = scp.upload("/local/manifold.pyz")

        assert result.remote_path == "/opt/manifold/manifold.pyz"
        # The scp target should contain the remote path
        target_args = [a for a in captured_args if "manifold.pyz" in a and "@" in a]
        assert target_args, "Expected scp target in args"

    def test_upload_explicit_remote_path(self) -> None:
        scp = SecureCopy("10.0.0.1")
        with patch("subprocess.run", return_value=self._mock_proc()):
            result = scp.upload("/local/f.pyz", "/custom/path.pyz")
        assert result.remote_path == "/custom/path.pyz"

    def test_upload_failure(self) -> None:
        scp = SecureCopy("10.0.0.1")
        with patch("subprocess.run", return_value=self._mock_proc(1, "permission denied")):
            result = scp.upload("/local/f")
        assert result.ok is False
        assert result.returncode == 1
        assert "permission denied" in result.stderr

    def test_upload_timeout(self) -> None:
        scp = SecureCopy("10.0.0.1")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["scp"], 10)):
            result = scp.upload("/f")
        assert result.ok is False
        assert result.returncode == -1
        assert "Timeout" in result.stderr

    def test_upload_scp_not_found(self) -> None:
        scp = SecureCopy("10.0.0.1")
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = scp.upload("/f")
        assert result.ok is False
        assert result.returncode == -2

    def test_upload_manifold_bundle_all_files(self) -> None:
        scp = SecureCopy("10.0.0.1")
        call_count = []

        def fake_run(args, **kwargs):
            call_count.append(1)
            return self._mock_proc()

        with patch("subprocess.run", side_effect=fake_run):
            results = scp.upload_manifold_bundle(
                "/dist/manifold.pyz",
                "/dist/release.manifest.json",
                "/dist/genesis_config.json",
            )

        assert len(results) == 3
        assert len(call_count) == 3
        assert all(isinstance(r, TransferResult) for r in results)

    def test_upload_manifold_bundle_optional_files_none(self) -> None:
        scp = SecureCopy("10.0.0.1")
        call_count = []

        def fake_run(args, **kwargs):
            call_count.append(1)
            return self._mock_proc()

        with patch("subprocess.run", side_effect=fake_run):
            results = scp.upload_manifold_bundle("/dist/manifold.pyz")

        assert len(results) == 1
        assert len(call_count) == 1

    def test_upload_includes_key_in_args(self) -> None:
        scp = SecureCopy("10.0.0.1", key_path="/my/key")
        captured_args = []

        def fake_run(args, **kwargs):
            captured_args.extend(args)
            return self._mock_proc()

        with patch("subprocess.run", side_effect=fake_run):
            scp.upload("/f")

        assert "-i" in captured_args
        assert "/my/key" in captured_args

    def test_upload_elapsed_positive(self) -> None:
        scp = SecureCopy("10.0.0.1")
        with patch("subprocess.run", return_value=self._mock_proc()):
            result = scp.upload("/f")
        assert result.elapsed_seconds >= 0.0
