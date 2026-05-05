"""Tests for Phase 59: Native Swarm Orchestrator — Bootstrapper (deploy/bootstrapper.py).

All subprocess calls are intercepted via ``unittest.mock`` to avoid real SSH
connections.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch, call

import pytest

from manifold.deploy.bootstrapper import NodeResult, NodeSpec, SwarmDeployer
from manifold.deploy.ssh import RemoteCommandResult, TransferResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ssh_result(returncode=0, stdout="", stderr="", host="10.0.0.1", cmd=""):
    return RemoteCommandResult(host, cmd, returncode, stdout, stderr, 0.01)


def _make_transfer_result(returncode=0, local="/f", remote="/r", stderr=""):
    return TransferResult("10.0.0.1", local, remote, returncode, stderr, 0.01)


def _make_deployer(pyz="/dist/manifold.pyz", **kwargs) -> SwarmDeployer:
    return SwarmDeployer(pyz_path=pyz, **kwargs)


def _make_spec(ip="10.0.0.1", **kwargs) -> NodeSpec:
    return NodeSpec(ip=ip, **kwargs)


# ---------------------------------------------------------------------------
# NodeSpec
# ---------------------------------------------------------------------------


class TestNodeSpec:
    def test_defaults(self) -> None:
        spec = NodeSpec(ip="10.0.0.1")
        assert spec.user == "root"
        assert spec.key_path is None
        assert spec.port == 22
        assert spec.manifold_port == 8080

    def test_from_dict_required_only(self) -> None:
        spec = NodeSpec.from_dict({"ip": "1.2.3.4"})
        assert spec.ip == "1.2.3.4"
        assert spec.user == "root"

    def test_from_dict_full(self) -> None:
        spec = NodeSpec.from_dict({
            "ip": "5.6.7.8",
            "user": "ubuntu",
            "key_path": "/k",
            "port": 2222,
            "manifold_port": 9000,
        })
        assert spec.user == "ubuntu"
        assert spec.key_path == "/k"
        assert spec.port == 2222
        assert spec.manifold_port == 9000

    def test_to_dict_round_trip(self) -> None:
        spec = NodeSpec(ip="1.1.1.1", user="ec2-user", port=22, manifold_port=8080)
        d = spec.to_dict()
        assert d["ip"] == "1.1.1.1"
        assert d["user"] == "ec2-user"


# ---------------------------------------------------------------------------
# NodeResult
# ---------------------------------------------------------------------------


class TestNodeResult:
    def test_success_to_dict(self) -> None:
        spec = _make_spec()
        r = NodeResult(spec=spec, success=True, node_id="abc123", dashboard_url="http://x/dashboard")
        d = r.to_dict()
        assert d["ip"] == "10.0.0.1"
        assert d["success"] is True
        assert d["node_id"] == "abc123"
        assert d["dashboard_url"] == "http://x/dashboard"
        assert d["error"] is None

    def test_failure_to_dict(self) -> None:
        spec = _make_spec()
        r = NodeResult(spec=spec, success=False, error="Upload failed")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "Upload failed"

    def test_transfer_results_serialized(self) -> None:
        spec = _make_spec()
        tr = _make_transfer_result()
        r = NodeResult(spec=spec, success=True, transfer_results=[tr])
        d = r.to_dict()
        assert len(d["transfer_results"]) == 1
        assert "ok" in d["transfer_results"][0]


# ---------------------------------------------------------------------------
# SwarmDeployer construction
# ---------------------------------------------------------------------------


class TestSwarmDeployerConstruction:
    def test_defaults(self) -> None:
        d = _make_deployer()
        assert d.pyz_path == "/dist/manifold.pyz"
        assert d.manifest_path is None
        assert d.max_workers == 8
        assert d.remote_install_dir == "/opt/manifold"

    def test_custom_params(self) -> None:
        d = SwarmDeployer(
            pyz_path="/my/pyz",
            manifest_path="/my/manifest.json",
            remote_install_dir="/srv/manifold",
            max_workers=4,
        )
        assert d.manifest_path == "/my/manifest.json"
        assert d.remote_install_dir == "/srv/manifold"
        assert d.max_workers == 4


# ---------------------------------------------------------------------------
# SwarmDeployer._extract_node_id
# ---------------------------------------------------------------------------


class TestExtractNodeId:
    def test_extracts_nodeid_label(self) -> None:
        d = _make_deployer()
        nid = d._extract_node_id("NodeID: deadbeef1234")
        assert nid == "deadbeef1234"

    def test_extracts_bracket_form(self) -> None:
        d = _make_deployer()
        nid = d._extract_node_id("[node_id] aabbcc1122334455")
        assert nid == "aabbcc1122334455"

    def test_returns_none_when_absent(self) -> None:
        d = _make_deployer()
        assert d._extract_node_id("Starting MANIFOLD…") is None

    def test_case_insensitive(self) -> None:
        d = _make_deployer()
        nid = d._extract_node_id("NODEID: ABCDEF01234567890")
        assert nid is not None


# ---------------------------------------------------------------------------
# SwarmDeployer.deploy — empty list
# ---------------------------------------------------------------------------


class TestSwarmDeployerDeployEmpty:
    def test_returns_empty_list_for_no_nodes(self) -> None:
        d = _make_deployer()
        assert d.deploy([]) == []


# ---------------------------------------------------------------------------
# SwarmDeployer.deploy — happy path via mocks
# ---------------------------------------------------------------------------


class TestSwarmDeployerDeployHappyPath:
    """Patch SSHClient and SecureCopy to avoid real SSH."""

    def _make_mock_ssh(self, *, node_id="abc123", rc=0):
        mock = MagicMock()
        mock.run.return_value = _make_ssh_result(
            returncode=rc,
            stdout=f"NodeID: {node_id}\nstarting...",
        )
        return mock

    def _make_mock_scp(self, *, rc=0):
        mock = MagicMock()
        mock.upload.return_value = _make_transfer_result(returncode=rc)
        mock.upload_manifold_bundle.return_value = [_make_transfer_result(returncode=rc)]
        return mock

    def test_single_genesis_node_success(self) -> None:
        deployer = _make_deployer()
        spec = _make_spec("10.0.0.1")

        with (
            patch.object(deployer, "_make_ssh", return_value=self._make_mock_ssh()),
            patch.object(deployer, "_make_scp", return_value=self._make_mock_scp()),
        ):
            results = deployer.deploy([spec])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].spec is spec
        assert "dashboard" in (results[0].dashboard_url or "")

    def test_genesis_plus_worker_success(self) -> None:
        deployer = _make_deployer()
        genesis = _make_spec("10.0.0.1")
        worker = _make_spec("10.0.0.2")

        with (
            patch.object(deployer, "_make_ssh", return_value=self._make_mock_ssh()),
            patch.object(deployer, "_make_scp", return_value=self._make_mock_scp()),
        ):
            results = deployer.deploy([genesis, worker])

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_results_order_matches_nodes(self) -> None:
        deployer = _make_deployer()
        nodes = [_make_spec(f"10.0.0.{i}") for i in range(1, 5)]

        with (
            patch.object(deployer, "_make_ssh", return_value=self._make_mock_ssh()),
            patch.object(deployer, "_make_scp", return_value=self._make_mock_scp()),
        ):
            results = deployer.deploy(nodes)

        assert [r.spec.ip for r in results] == [n.ip for n in nodes]

    def test_genesis_result_has_node_id(self) -> None:
        deployer = _make_deployer()
        spec = _make_spec("10.0.0.1")

        with (
            patch.object(deployer, "_make_ssh", return_value=self._make_mock_ssh(node_id="deadbeef1234")),
            patch.object(deployer, "_make_scp", return_value=self._make_mock_scp()),
        ):
            results = deployer.deploy([spec])

        assert results[0].node_id == "deadbeef1234"

    def test_dashboard_url_format(self) -> None:
        deployer = _make_deployer()
        spec = NodeSpec(ip="1.2.3.4", manifold_port=9090)

        with (
            patch.object(deployer, "_make_ssh", return_value=self._make_mock_ssh()),
            patch.object(deployer, "_make_scp", return_value=self._make_mock_scp()),
        ):
            results = deployer.deploy([spec])

        assert results[0].dashboard_url == "http://1.2.3.4:9090/dashboard"

    def test_progress_callback_called(self) -> None:
        calls_log: list[tuple] = []

        def cb(step, ip, detail):
            calls_log.append((step, ip, detail))

        deployer = _make_deployer(progress_callback=cb)
        spec = _make_spec("10.0.0.1")

        with (
            patch.object(deployer, "_make_ssh", return_value=self._make_mock_ssh()),
            patch.object(deployer, "_make_scp", return_value=self._make_mock_scp()),
        ):
            deployer.deploy([spec])

        assert len(calls_log) >= 1
        steps = [c[0] for c in calls_log]
        assert "start" in steps
        assert "done" in steps


# ---------------------------------------------------------------------------
# SwarmDeployer — failure handling
# ---------------------------------------------------------------------------


class TestSwarmDeployerFailureHandling:
    def _make_failing_ssh(self, rc=1, stderr="mkdir: permission denied"):
        mock = MagicMock()
        mock.run.return_value = _make_ssh_result(returncode=rc, stderr=stderr)
        return mock

    def _make_failing_scp(self):
        mock = MagicMock()
        mock.upload_manifold_bundle.return_value = [
            _make_transfer_result(returncode=1, stderr="transfer failed")
        ]
        return mock

    def test_mkdir_failure_marks_node_failed(self) -> None:
        deployer = _make_deployer()
        spec = _make_spec("10.0.0.1")

        with patch.object(deployer, "_make_ssh", return_value=self._make_failing_ssh()):
            results = deployer.deploy([spec])

        assert results[0].success is False
        assert results[0].error is not None

    def test_upload_failure_marks_node_failed(self) -> None:
        deployer = _make_deployer()
        spec = _make_spec("10.0.0.1")

        good_ssh = MagicMock()
        good_ssh.run.return_value = _make_ssh_result(returncode=0)

        with (
            patch.object(deployer, "_make_ssh", return_value=good_ssh),
            patch.object(deployer, "_make_scp", return_value=self._make_failing_scp()),
        ):
            results = deployer.deploy([spec])

        assert results[0].success is False
        assert "Upload failed" in (results[0].error or "")

    def test_worker_failure_does_not_abort_other_workers(self) -> None:
        """One worker failing must not prevent others from being deployed."""
        deployer = _make_deployer(max_workers=4)
        genesis = _make_spec("10.0.0.1")
        workers = [_make_spec(f"10.0.0.{i}") for i in range(2, 6)]

        call_count = [0]
        good_ssh = MagicMock()
        good_ssh.run.return_value = _make_ssh_result(returncode=0, stdout="NodeID: abc")

        bad_scp = MagicMock()
        bad_scp.upload_manifold_bundle.return_value = [
            _make_transfer_result(returncode=1, stderr="error")
        ]
        good_scp = MagicMock()
        good_scp.upload_manifold_bundle.return_value = [_make_transfer_result(returncode=0)]

        def ssh_factory(spec):
            return good_ssh

        def scp_factory(spec):
            call_count[0] += 1
            # First call is genesis (good), 2nd is a bad worker, rest are good
            if spec.ip == "10.0.0.2":
                return bad_scp
            return good_scp

        with (
            patch.object(deployer, "_make_ssh", side_effect=ssh_factory),
            patch.object(deployer, "_make_scp", side_effect=scp_factory),
        ):
            results = deployer.deploy([genesis] + workers)

        assert len(results) == 5  # genesis + 4 workers
        # Not all should fail — only the one bad worker
        successes = [r for r in results if r.success]
        assert len(successes) >= 4

    def test_all_results_present_even_on_partial_failure(self) -> None:
        deployer = _make_deployer()
        nodes = [_make_spec(f"10.0.0.{i}") for i in range(1, 4)]

        ssh_mock = MagicMock()
        ssh_mock.run.return_value = _make_ssh_result(returncode=1, stderr="fail")

        with patch.object(deployer, "_make_ssh", return_value=ssh_mock):
            results = deployer.deploy(nodes)

        assert len(results) == 3
        assert all(not r.success for r in results)


# ---------------------------------------------------------------------------
# SwarmDeployer.from_cluster_json
# ---------------------------------------------------------------------------


class TestFromClusterJson:
    def test_load_valid_cluster(self, tmp_path) -> None:
        cluster = {
            "pyz": "/dist/manifold.pyz",
            "manifest": "/dist/release.manifest.json",
            "nodes": [
                {"ip": "10.0.0.1", "user": "ubuntu"},
                {"ip": "10.0.0.2"},
            ],
        }
        p = tmp_path / "cluster.json"
        p.write_text(json.dumps(cluster))
        deployer, nodes = SwarmDeployer.from_cluster_json(str(p))
        assert deployer.pyz_path == "/dist/manifold.pyz"
        assert deployer.manifest_path == "/dist/release.manifest.json"
        assert len(nodes) == 2
        assert nodes[0].ip == "10.0.0.1"
        assert nodes[0].user == "ubuntu"

    def test_load_minimal_cluster(self, tmp_path) -> None:
        cluster = {
            "pyz": "/p.pyz",
            "nodes": [{"ip": "1.2.3.4"}],
        }
        p = tmp_path / "c.json"
        p.write_text(json.dumps(cluster))
        deployer, nodes = SwarmDeployer.from_cluster_json(str(p))
        assert len(nodes) == 1
        assert deployer.manifest_path is None

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            SwarmDeployer.from_cluster_json(str(tmp_path / "nonexistent.json"))

    def test_missing_pyz_key_raises(self, tmp_path) -> None:
        cluster = {"nodes": [{"ip": "1.2.3.4"}]}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(cluster))
        with pytest.raises(KeyError):
            SwarmDeployer.from_cluster_json(str(p))

    def test_extra_kwargs_passed_to_deployer(self, tmp_path) -> None:
        cluster = {"pyz": "/p.pyz", "nodes": [{"ip": "1.2.3.4"}]}
        p = tmp_path / "c.json"
        p.write_text(json.dumps(cluster))
        deployer, _ = SwarmDeployer.from_cluster_json(str(p), max_workers=2, remote_install_dir="/custom")
        assert deployer.max_workers == 2
        assert deployer.remote_install_dir == "/custom"
