"""Phase 59: Native Swarm Orchestrator — deploy sub-package.

Exports the top-level public API so callers can do::

    from manifold.deploy import SSHClient, SecureCopy, SwarmDeployer
"""

from manifold.deploy.ssh import RemoteCommandResult, SSHClient, SecureCopy, TransferResult
from manifold.deploy.bootstrapper import NodeResult, NodeSpec, SwarmDeployer

__all__ = [
    "RemoteCommandResult",
    "SSHClient",
    "SecureCopy",
    "TransferResult",
    "NodeResult",
    "NodeSpec",
    "SwarmDeployer",
]
