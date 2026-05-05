"""Phase 59 — Network Bootstrapper (`manifold/deploy/bootstrapper.py`).

Automates the "Cold Start" sequence of a multi-node MANIFOLD network using
only the Python standard library (``subprocess``, ``json``, ``time``,
``concurrent.futures``).

Architecture
------------
1. The :class:`SwarmDeployer` accepts a list of :class:`NodeSpec` objects, each
   describing an SSH-accessible machine.
2. It bootstraps the **genesis node** first: uploads the binary, starts the
   daemon with ``--genesis --daemon``, and captures the ``NodeID`` from the
   process output.
3. Worker nodes are bootstrapped concurrently using a
   :class:`~concurrent.futures.ThreadPoolExecutor`.  Each worker receives the
   genesis address as a ``--peer <genesis_ip>:<port>`` flag so it joins the
   Trust Economy DHT automatically.
4. If any node fails to boot the error is logged and the deployer continues
   with the remaining nodes.

Classes
-------
``NodeSpec``
    Description of a target deployment host.
``NodeResult``
    Outcome of deploying to one node.
``SwarmDeployer``
    Orchestrates a multi-node cold-start sequence.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import re
import time
from typing import Callable, Sequence

from manifold.deploy.ssh import SSHClient, SecureCopy, TransferResult

__all__ = [
    "NodeResult",
    "NodeSpec",
    "SwarmDeployer",
]

# ---------------------------------------------------------------------------
# NodeSpec
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NodeSpec:
    """Description of a single deployment target.

    Parameters
    ----------
    ip:
        IP address or hostname of the remote machine.
    user:
        SSH username (default ``"root"``).
    key_path:
        Path to the SSH private key.
    port:
        SSH port (default 22).
    manifold_port:
        Port the MANIFOLD daemon should listen on (default 8080).
    """

    ip: str
    user: str = "root"
    key_path: str | None = None
    port: int = 22
    manifold_port: int = 8080

    @classmethod
    def from_dict(cls, d: dict) -> "NodeSpec":
        """Create a :class:`NodeSpec` from a plain dictionary."""
        return cls(
            ip=d["ip"],
            user=d.get("user", "root"),
            key_path=d.get("key_path"),
            port=int(d.get("port", 22)),
            manifold_port=int(d.get("manifold_port", 8080)),
        )

    def to_dict(self) -> dict:
        return {
            "ip": self.ip,
            "user": self.user,
            "key_path": self.key_path,
            "port": self.port,
            "manifold_port": self.manifold_port,
        }


# ---------------------------------------------------------------------------
# NodeResult
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NodeResult:
    """Outcome of deploying MANIFOLD to a single node.

    Parameters
    ----------
    spec:
        The target :class:`NodeSpec`.
    success:
        ``True`` when all bootstrap steps completed without error.
    node_id:
        The ``NodeID`` extracted from the daemon's startup output (genesis
        node only; ``None`` for worker nodes or on failure).
    dashboard_url:
        URL of the node's HTTP dashboard.
    error:
        Human-readable error message when *success* is ``False``.
    transfer_results:
        List of :class:`~manifold.deploy.ssh.TransferResult` objects from the
        file upload stage.
    elapsed_seconds:
        Total time spent bootstrapping this node.
    """

    spec: NodeSpec
    success: bool
    node_id: str | None = None
    dashboard_url: str | None = None
    error: str | None = None
    transfer_results: list[TransferResult] = dataclasses.field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ip": self.spec.ip,
            "success": self.success,
            "node_id": self.node_id,
            "dashboard_url": self.dashboard_url,
            "error": self.error,
            "transfer_results": [r.to_dict() for r in self.transfer_results],
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


# ---------------------------------------------------------------------------
# SwarmDeployer
# ---------------------------------------------------------------------------


class SwarmDeployer:
    """Orchestrate a cold-start deployment of a MANIFOLD swarm.

    Parameters
    ----------
    pyz_path:
        Local path to the compiled ``manifold.pyz`` binary.
    manifest_path:
        Local path to ``release.manifest.json`` (optional).
    genesis_config_path:
        Local path to the ``GenesisConfig`` JSON file (optional).
    remote_install_dir:
        Directory on remote hosts where files are placed
        (default ``"/opt/manifold"``).
    max_workers:
        Maximum concurrent worker-node deployments (default 8).
    startup_wait_seconds:
        How long to wait after launching the daemon before querying it
        (default 3.0).
    progress_callback:
        Optional callable receiving ``(step: str, node_ip: str, detail: str)``
        messages for real-time progress reporting.
    """

    # Regex to extract NodeID from genesis stdout, e.g. "NodeID: abc123"
    _NODE_ID_RE = re.compile(r"NodeID[:\s]+([0-9a-fA-F]{8,64})", re.IGNORECASE)
    # Regex for [node_id] style output
    _NODE_ID_BRACKET_RE = re.compile(r"\[node_id\]\s+([0-9a-fA-F]{8,64})", re.IGNORECASE)

    def __init__(
        self,
        pyz_path: str,
        manifest_path: str | None = None,
        genesis_config_path: str | None = None,
        remote_install_dir: str = "/opt/manifold",
        max_workers: int = 8,
        startup_wait_seconds: float = 3.0,
        progress_callback: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.pyz_path = pyz_path
        self.manifest_path = manifest_path
        self.genesis_config_path = genesis_config_path
        self.remote_install_dir = remote_install_dir
        self.max_workers = max_workers
        self.startup_wait_seconds = startup_wait_seconds
        self._progress = progress_callback or (lambda step, ip, detail: None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_ssh(self, spec: NodeSpec) -> SSHClient:
        return SSHClient(
            host=spec.ip,
            user=spec.user,
            key_path=spec.key_path,
            port=spec.port,
        )

    def _make_scp(self, spec: NodeSpec) -> SecureCopy:
        return SecureCopy(
            host=spec.ip,
            user=spec.user,
            key_path=spec.key_path,
            port=spec.port,
            remote_base_dir=self.remote_install_dir,
        )

    def _upload_bundle(self, spec: NodeSpec) -> tuple[list[TransferResult], str | None]:
        """Upload the MANIFOLD bundle to *spec* and return (results, error)."""
        scp = self._make_scp(spec)
        self._progress("upload", spec.ip, "Uploading MANIFOLD bundle…")
        results = scp.upload_manifold_bundle(
            self.pyz_path,
            self.manifest_path,
            self.genesis_config_path,
        )
        failed = [r for r in results if not r.ok]
        if failed:
            msgs = "; ".join(f"{r.local_path}: {r.stderr}" for r in failed)
            return results, f"Upload failed: {msgs}"
        return results, None

    def _ensure_remote_dir(self, spec: NodeSpec) -> str | None:
        """Create the remote install directory and return error or None."""
        ssh = self._make_ssh(spec)
        result = ssh.run(f"mkdir -p {self.remote_install_dir}")
        if not result.ok:
            return f"mkdir failed: {result.stderr}"
        return None

    def _extract_node_id(self, output: str) -> str | None:
        """Try to extract a NodeID from *output*."""
        for pattern in (self._NODE_ID_RE, self._NODE_ID_BRACKET_RE):
            m = pattern.search(output)
            if m:
                return m.group(1)
        return None

    def _launch_genesis(self, spec: NodeSpec) -> tuple[str | None, str | None]:
        """Launch the genesis daemon and return (node_id, error)."""
        ssh = self._make_ssh(spec)
        pyz = f"{self.remote_install_dir}/manifold.pyz"
        cmd = (
            f"nohup python3 {pyz} --genesis --daemon "
            f"--port {spec.manifold_port} "
            f"> {self.remote_install_dir}/genesis.log 2>&1 & "
            f"sleep {self.startup_wait_seconds:.1f}; "
            f"head -40 {self.remote_install_dir}/genesis.log"
        )
        self._progress("launch", spec.ip, "Launching genesis daemon…")
        result = ssh.run(cmd, timeout=30 + int(self.startup_wait_seconds))
        if not result.ok and result.returncode not in (-1, 1):
            return None, f"Launch command failed (rc={result.returncode}): {result.stderr}"
        node_id = self._extract_node_id(result.stdout + result.stderr)
        return node_id, None

    def _launch_worker(
        self,
        spec: NodeSpec,
        genesis_ip: str,
        genesis_port: int,
    ) -> str | None:
        """Launch a worker node pointing at the genesis peer.  Returns error or None."""
        ssh = self._make_ssh(spec)
        pyz = f"{self.remote_install_dir}/manifold.pyz"
        peer = f"{genesis_ip}:{genesis_port}"
        cmd = (
            f"nohup python3 {pyz} --daemon "
            f"--port {spec.manifold_port} "
            f"--peer {peer} "
            f"> {self.remote_install_dir}/node.log 2>&1 &"
        )
        self._progress("launch", spec.ip, f"Launching worker (peer={peer})…")
        result = ssh.run(cmd, timeout=30)
        if not result.ok and result.returncode not in (-1, 1):
            return f"Worker launch failed (rc={result.returncode}): {result.stderr}"
        return None

    # ------------------------------------------------------------------
    # Bootstrap steps
    # ------------------------------------------------------------------

    def _bootstrap_genesis(self, spec: NodeSpec) -> NodeResult:
        t0 = time.monotonic()
        self._progress("start", spec.ip, "Bootstrapping genesis node…")

        err = self._ensure_remote_dir(spec)
        if err:
            return NodeResult(spec=spec, success=False, error=err, elapsed_seconds=time.monotonic() - t0)

        transfers, err = self._upload_bundle(spec)
        if err:
            return NodeResult(spec=spec, success=False, error=err, transfer_results=transfers, elapsed_seconds=time.monotonic() - t0)

        node_id, err = self._launch_genesis(spec)
        if err:
            return NodeResult(spec=spec, success=False, error=err, transfer_results=transfers, elapsed_seconds=time.monotonic() - t0)

        dashboard = f"http://{spec.ip}:{spec.manifold_port}/dashboard"
        self._progress("done", spec.ip, f"Genesis online — dashboard: {dashboard}")
        return NodeResult(
            spec=spec,
            success=True,
            node_id=node_id,
            dashboard_url=dashboard,
            transfer_results=transfers,
            elapsed_seconds=time.monotonic() - t0,
        )

    def _bootstrap_worker(
        self,
        spec: NodeSpec,
        genesis_ip: str,
        genesis_port: int,
    ) -> NodeResult:
        t0 = time.monotonic()
        self._progress("start", spec.ip, "Bootstrapping worker node…")

        err = self._ensure_remote_dir(spec)
        if err:
            return NodeResult(spec=spec, success=False, error=err, elapsed_seconds=time.monotonic() - t0)

        transfers, err = self._upload_bundle(spec)
        if err:
            return NodeResult(spec=spec, success=False, error=err, transfer_results=transfers, elapsed_seconds=time.monotonic() - t0)

        err = self._launch_worker(spec, genesis_ip, genesis_port)
        if err:
            return NodeResult(spec=spec, success=False, error=err, transfer_results=transfers, elapsed_seconds=time.monotonic() - t0)

        dashboard = f"http://{spec.ip}:{spec.manifold_port}/dashboard"
        self._progress("done", spec.ip, f"Worker online — dashboard: {dashboard}")
        return NodeResult(
            spec=spec,
            success=True,
            dashboard_url=dashboard,
            transfer_results=transfers,
            elapsed_seconds=time.monotonic() - t0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deploy(self, nodes: Sequence[NodeSpec]) -> list[NodeResult]:
        """Bootstrap a swarm from *nodes*.

        The first element of *nodes* is treated as the **genesis node**; all
        remaining nodes are bootstrapped concurrently as workers.

        Parameters
        ----------
        nodes:
            Ordered list of :class:`NodeSpec` objects.  Must contain at least
            one element.

        Returns
        -------
        list[NodeResult]
            One :class:`NodeResult` per node, in the same order as *nodes*.
        """
        if not nodes:
            return []

        genesis_spec, *worker_specs = nodes
        results: list[NodeResult] = []

        # 1. Bootstrap genesis
        genesis_result = self._bootstrap_genesis(genesis_spec)
        results.append(genesis_result)

        if not worker_specs:
            return results

        # 2. Bootstrap workers concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(
                    self._bootstrap_worker,
                    spec,
                    genesis_spec.ip,
                    genesis_spec.manifold_port,
                )
                for spec in worker_specs
            ]
            # Collect results in submission order
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:  # pragma: no cover
                    # Extract spec from the corresponding position
                    idx = len(results) - 1  # one result per genesis already
                    spec = worker_specs[idx]
                    results.append(NodeResult(
                        spec=spec,
                        success=False,
                        error=str(exc),
                    ))

        return results

    @classmethod
    def from_cluster_json(cls, path: str, **kwargs) -> tuple["SwarmDeployer", list[NodeSpec]]:
        """Load a ``cluster.json`` file and return ``(deployer, nodes)``.

        The JSON file must contain a top-level ``"nodes"`` list.  Each element
        is a node dict accepted by :meth:`NodeSpec.from_dict`.  The ``"pyz"``
        key (required) specifies the local path to ``manifold.pyz``.

        Example cluster.json::

            {
                "pyz": "dist/manifold.pyz",
                "manifest": "release.manifest.json",
                "nodes": [
                    {"ip": "10.0.0.1", "user": "ubuntu", "key_path": "~/.ssh/id_ed25519"},
                    {"ip": "10.0.0.2", "user": "ubuntu", "key_path": "~/.ssh/id_ed25519"}
                ]
            }
        """
        with open(path, encoding="utf-8") as fh:
            data: dict = json.load(fh)

        nodes = [NodeSpec.from_dict(n) for n in data["nodes"]]
        deployer = cls(
            pyz_path=data["pyz"],
            manifest_path=data.get("manifest"),
            genesis_config_path=data.get("genesis_config"),
            **kwargs,
        )
        return deployer, nodes
