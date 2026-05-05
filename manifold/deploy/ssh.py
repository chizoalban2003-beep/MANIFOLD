"""Phase 59 — SSH Wrapper (`manifold/deploy/ssh.py`).

Provides a secure, native wrapper around the system ``ssh`` and ``scp``
commands using only the :mod:`subprocess` module (zero external dependencies).

Classes
-------
``RemoteCommandResult``
    Structured result of a remote command execution.
``TransferResult``
    Structured result of a file-transfer operation.
``SSHClient``
    Executes remote commands via ``ssh``.
``SecureCopy``
    Transfers files to a remote host via ``scp``.
"""

from __future__ import annotations

import dataclasses
import subprocess
import time
from typing import Sequence

__all__ = [
    "RemoteCommandResult",
    "SSHClient",
    "SecureCopy",
    "TransferResult",
]

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RemoteCommandResult:
    """Outcome of a single remote command execution."""

    host: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        """Return ``True`` when the remote process exited with code 0."""
        return self.returncode == 0

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "ok": self.ok,
        }


@dataclasses.dataclass(frozen=True)
class TransferResult:
    """Outcome of a single ``scp`` file-transfer operation."""

    host: str
    local_path: str
    remote_path: str
    returncode: int
    stderr: str
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "local_path": self.local_path,
            "remote_path": self.remote_path,
            "returncode": self.returncode,
            "stderr": self.stderr,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "ok": self.ok,
        }


# ---------------------------------------------------------------------------
# SSHClient
# ---------------------------------------------------------------------------


class SSHClient:
    """Execute commands on a remote host via ``ssh``.

    Parameters
    ----------
    host:
        Hostname or IP address of the remote machine.
    user:
        Remote username (default ``"root"``).
    key_path:
        Path to the private SSH key file.  When ``None`` the system's default
        key is used.
    port:
        Remote SSH port (default 22).
    connect_timeout:
        Seconds before the connection attempt is aborted (default 10).
    strict_host_checking:
        When ``False`` (default) ``StrictHostKeyChecking=no`` is passed to
        ``ssh`` so deployment can proceed against freshly provisioned nodes.
    """

    def __init__(
        self,
        host: str,
        user: str = "root",
        key_path: str | None = None,
        port: int = 22,
        connect_timeout: int = 10,
        strict_host_checking: bool = False,
    ) -> None:
        self.host = host
        self.user = user
        self.key_path = key_path
        self.port = port
        self.connect_timeout = connect_timeout
        self.strict_host_checking = strict_host_checking

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_args(self) -> list[str]:
        """Return the common ``ssh`` flag list shared by all invocations."""
        args = [
            "ssh",
            "-p", str(self.port),
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", f"StrictHostKeyChecking={'yes' if self.strict_host_checking else 'no'}",
            "-o", "BatchMode=yes",
        ]
        if self.key_path:
            args += ["-i", self.key_path]
        return args

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        command: str,
        *,
        timeout: int | None = 60,
    ) -> RemoteCommandResult:
        """Execute *command* on the remote host and return the result.

        Parameters
        ----------
        command:
            Shell command string to execute remotely.
        timeout:
            Maximum seconds to wait for the remote process to complete.
            ``None`` disables the timeout.
        """
        target = f"{self.user}@{self.host}"
        args: list[str] = self._base_args() + [target, command]

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            stderr = f"Timeout after {timeout}s"
            if exc.stderr:
                stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr)
            return RemoteCommandResult(
                host=self.host,
                command=command,
                returncode=-1,
                stdout="",
                stderr=stderr,
                elapsed_seconds=elapsed,
            )
        except FileNotFoundError:
            elapsed = time.monotonic() - t0
            return RemoteCommandResult(
                host=self.host,
                command=command,
                returncode=-2,
                stdout="",
                stderr="ssh executable not found",
                elapsed_seconds=elapsed,
            )

        elapsed = time.monotonic() - t0
        return RemoteCommandResult(
            host=self.host,
            command=command,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_seconds=elapsed,
        )

    def run_many(
        self,
        commands: Sequence[str],
        *,
        timeout: int | None = 60,
    ) -> list[RemoteCommandResult]:
        """Execute each command in *commands* sequentially and return all results."""
        return [self.run(cmd, timeout=timeout) for cmd in commands]


# ---------------------------------------------------------------------------
# SecureCopy
# ---------------------------------------------------------------------------


class SecureCopy:
    """Transfer files to a remote host via ``scp``.

    Parameters
    ----------
    host:
        Hostname or IP address of the remote machine.
    user:
        Remote username (default ``"root"``).
    key_path:
        Path to the private SSH key file.
    port:
        Remote SSH port (default 22).
    connect_timeout:
        Connection timeout in seconds (default 10).
    strict_host_checking:
        When ``False`` (default) ``StrictHostKeyChecking=no`` is passed.
    remote_base_dir:
        Base directory on the remote host where files will be placed
        (default ``"/opt/manifold"``).
    """

    def __init__(
        self,
        host: str,
        user: str = "root",
        key_path: str | None = None,
        port: int = 22,
        connect_timeout: int = 10,
        strict_host_checking: bool = False,
        remote_base_dir: str = "/opt/manifold",
    ) -> None:
        self.host = host
        self.user = user
        self.key_path = key_path
        self.port = port
        self.connect_timeout = connect_timeout
        self.strict_host_checking = strict_host_checking
        self.remote_base_dir = remote_base_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_args(self) -> list[str]:
        args = [
            "scp",
            "-P", str(self.port),
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", f"StrictHostKeyChecking={'yes' if self.strict_host_checking else 'no'}",
            "-o", "BatchMode=yes",
        ]
        if self.key_path:
            args += ["-i", self.key_path]
        return args

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(
        self,
        local_path: str,
        remote_path: str | None = None,
        *,
        timeout: int | None = 120,
    ) -> TransferResult:
        """Upload *local_path* to the remote host.

        Parameters
        ----------
        local_path:
            Path of the local file to upload.
        remote_path:
            Destination path on the remote host.  When ``None`` the file is
            placed in :attr:`remote_base_dir` with the same basename.
        timeout:
            Maximum seconds to wait for the transfer to complete.
        """
        import os

        if remote_path is None:
            remote_path = f"{self.remote_base_dir}/{os.path.basename(local_path)}"

        target = f"{self.user}@{self.host}:{remote_path}"
        args: list[str] = self._base_args() + [local_path, target]

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            stderr = f"Timeout after {timeout}s"
            if exc.stderr:
                stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr)
            return TransferResult(
                host=self.host,
                local_path=local_path,
                remote_path=remote_path,
                returncode=-1,
                stderr=stderr,
                elapsed_seconds=elapsed,
            )
        except FileNotFoundError:
            elapsed = time.monotonic() - t0
            return TransferResult(
                host=self.host,
                local_path=local_path,
                remote_path=remote_path,
                returncode=-2,
                stderr="scp executable not found",
                elapsed_seconds=elapsed,
            )

        elapsed = time.monotonic() - t0
        return TransferResult(
            host=self.host,
            local_path=local_path,
            remote_path=remote_path,
            returncode=proc.returncode,
            stderr=proc.stderr,
            elapsed_seconds=elapsed,
        )

    def upload_manifold_bundle(
        self,
        pyz_path: str,
        manifest_path: str | None = None,
        genesis_config_path: str | None = None,
        *,
        timeout: int | None = 120,
    ) -> list[TransferResult]:
        """Upload the MANIFOLD binary bundle to the remote host.

        Transfers *pyz_path* (the compiled ``manifold.pyz``), and
        optionally *manifest_path* (``release.manifest.json``) and
        *genesis_config_path* (a ``GenesisConfig`` JSON file) to
        :attr:`remote_base_dir`.

        Returns a list of :class:`TransferResult` objects, one per file.
        """
        results: list[TransferResult] = []
        for path in filter(None, [pyz_path, manifest_path, genesis_config_path]):
            results.append(self.upload(path, timeout=timeout))
        return results
