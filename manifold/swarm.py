"""Phase 33: Mesh Swarm Routing — Peer-to-Peer Task Delegation.

When the local ``ActiveInterceptor`` encounters a ``QuotaExhaustedError`` or
an ``InterceptorVeto``, the ``SwarmRouter`` evaluates known B2B peers and
forwards the ``BrainTask`` to the peer with the highest *routing value*:

.. math::

    V_{swarm} = \\max_{i \\in N} (Asset_i - Cost_{routing} - Risk_i)

The winning peer receives the task payload via a ``POST`` to its ``/shield``
endpoint (using only ``urllib.request`` — zero external dependencies).  The
``asset`` reward is split: the local node retains
``(1 - asset_share)`` and the delegating peer receives ``asset_share``.

Key classes
-----------
``SwarmPeer``
    Descriptor for a known B2B peer node.
``SwarmRouteResult``
    Outcome of a swarm routing attempt.
``SwarmRouter``
    Evaluates peers and delegates tasks; uses ``urllib.request`` for HTTP.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .brain import BrainTask


# ---------------------------------------------------------------------------
# SwarmPeer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwarmPeer:
    """Descriptor for a known B2B peer node in the swarm.

    Attributes
    ----------
    org_id:
        Unique identifier for the peer organisation.
    endpoint:
        Base URL of the peer's MANIFOLD server (e.g. ``"http://peer:8080"``).
    asset:
        Estimated asset / reward value the peer can offer for completed tasks.
    risk:
        Peer's current risk level [0, 1].
    cost_routing:
        Overhead cost for routing to this peer (default: ``0.1``).
    token_availability:
        Estimated fraction of quota remaining at the peer [0, 1].
        Higher values make the peer more attractive.  Default: ``1.0``.
    """

    org_id: str
    endpoint: str
    asset: float = 1.0
    risk: float = 0.1
    cost_routing: float = 0.1
    token_availability: float = 1.0

    @property
    def routing_value(self) -> float:
        """Compute the routing value for this peer.

        .. math::

            V = (Asset + TokenAvailability) / 2 - Cost_{routing} - Risk

        Clamped to ``[-1.0, 1.0]``.
        """
        raw = (self.asset + self.token_availability) / 2.0 - self.cost_routing - self.risk
        return max(-1.0, min(1.0, raw))

    def shield_url(self) -> str:
        """Return the full ``/shield`` endpoint URL for this peer."""
        return self.endpoint.rstrip("/") + "/shield"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "org_id": self.org_id,
            "endpoint": self.endpoint,
            "asset": self.asset,
            "risk": self.risk,
            "cost_routing": self.cost_routing,
            "token_availability": self.token_availability,
            "routing_value": self.routing_value,
        }


# ---------------------------------------------------------------------------
# SwarmRouteResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwarmRouteResult:
    """Outcome of a :class:`SwarmRouter` delegation attempt.

    Attributes
    ----------
    delegated:
        ``True`` if the task was successfully forwarded to a peer.
    peer:
        The peer that was selected (``None`` if no eligible peer found).
    routing_value:
        The routing value score for the selected peer (``0.0`` if none).
    response:
        The JSON response body from the peer's ``/shield`` endpoint, or
        ``None`` if delegation failed or was not attempted.
    error:
        Human-readable error description if delegation failed, else ``""``.
    """

    delegated: bool
    peer: SwarmPeer | None
    routing_value: float
    response: dict[str, Any] | None
    error: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "delegated": self.delegated,
            "peer": self.peer.to_dict() if self.peer else None,
            "routing_value": self.routing_value,
            "response": self.response,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# SwarmRouter
# ---------------------------------------------------------------------------


@dataclass
class SwarmRouter:
    """Evaluates known B2B peers and delegates :class:`~manifold.brain.BrainTask`
    objects to the best available peer when the local node is saturated.

    The peer with the highest :attr:`SwarmPeer.routing_value` is selected.
    The task payload is ``POST``-ed to the peer's ``/shield`` endpoint via
    ``urllib.request``.

    Parameters
    ----------
    peers:
        Initial list of known swarm peers.
    timeout:
        HTTP request timeout in seconds.  Default: ``5.0``.
    min_routing_value:
        Only consider peers whose routing value exceeds this threshold.
        Default: ``0.0``.
    asset_share:
        Fraction of the task's asset reward shared with the delegating peer.
        Default: ``0.5``.

    Example
    -------
    ::

        router = SwarmRouter(timeout=3.0)
        router.register_peer(SwarmPeer(org_id="peer-b", endpoint="http://peer-b:8080"))
        result = router.route(task)
        if result.delegated:
            print("Delegated to", result.peer.org_id)
    """

    peers: list[SwarmPeer] = field(default_factory=list)
    timeout: float = 5.0
    min_routing_value: float = 0.0
    asset_share: float = 0.5

    def register_peer(self, peer: SwarmPeer) -> None:
        """Add a peer to the routing table.

        If a peer with the same ``org_id`` already exists it is replaced.

        Parameters
        ----------
        peer:
            The peer descriptor to register.
        """
        self.peers = [p for p in self.peers if p.org_id != peer.org_id]
        self.peers.append(peer)

    def remove_peer(self, org_id: str) -> bool:
        """Remove the peer with the given *org_id*.

        Parameters
        ----------
        org_id:
            Peer to remove.

        Returns
        -------
        bool
            ``True`` if a peer was removed, ``False`` if not found.
        """
        before = len(self.peers)
        self.peers = [p for p in self.peers if p.org_id != org_id]
        return len(self.peers) < before

    def best_peer(self) -> SwarmPeer | None:
        """Return the peer with the highest routing value above the threshold.

        Returns
        -------
        SwarmPeer | None
            ``None`` if no eligible peer exists.
        """
        eligible = [p for p in self.peers if p.routing_value >= self.min_routing_value]
        if not eligible:
            return None
        return max(eligible, key=lambda p: p.routing_value)

    def route(self, task: BrainTask) -> SwarmRouteResult:
        """Delegate *task* to the best available peer.

        Steps:
        1. Find the peer with the highest routing value.
        2. Build a ``/shield``-compatible JSON payload from the task.
        3. POST the payload to the peer's endpoint via ``urllib.request``.
        4. Return a :class:`SwarmRouteResult` describing the outcome.

        Parameters
        ----------
        task:
            The :class:`~manifold.brain.BrainTask` to delegate.

        Returns
        -------
        SwarmRouteResult
        """
        peer = self.best_peer()
        if peer is None:
            return SwarmRouteResult(
                delegated=False,
                peer=None,
                routing_value=0.0,
                response=None,
                error="No eligible peers available",
            )

        payload = self._task_to_payload(task)
        try:
            response_data = self._post_shield(peer, payload)
            return SwarmRouteResult(
                delegated=True,
                peer=peer,
                routing_value=peer.routing_value,
                response=response_data,
                error="",
            )
        except Exception as exc:  # noqa: BLE001
            return SwarmRouteResult(
                delegated=False,
                peer=peer,
                routing_value=peer.routing_value,
                response=None,
                error=str(exc),
            )

    def peer_count(self) -> int:
        """Return the number of registered peers."""
        return len(self.peers)

    def routing_table(self) -> list[dict[str, Any]]:
        """Return the sorted routing table as a list of dicts.

        Returns
        -------
        list[dict[str, Any]]
            Peers sorted by descending routing value.
        """
        return [p.to_dict() for p in sorted(self.peers, key=lambda p: -p.routing_value)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _task_to_payload(task: BrainTask) -> bytes:
        """Serialise a ``BrainTask`` to a ``/shield``-compatible JSON payload."""
        data: dict[str, Any] = {
            "prompt": task.prompt,
            "domain": task.domain,
            "stakes": task.stakes,
            "uncertainty": task.uncertainty,
            "complexity": task.complexity,
            "tool_relevance": task.tool_relevance,
            "time_pressure": task.time_pressure,
            "safety_sensitivity": task.safety_sensitivity,
            "collaboration_value": task.collaboration_value,
            "source_confidence": task.source_confidence,
            "user_patience": task.user_patience,
            "dynamic_goal": task.dynamic_goal,
        }
        return json.dumps(data).encode("utf-8")

    def _post_shield(
        self,
        peer: SwarmPeer,
        payload: bytes,
    ) -> dict[str, Any]:
        """POST *payload* to *peer*'s ``/shield`` endpoint.

        Parameters
        ----------
        peer:
            Target peer.
        payload:
            JSON-encoded request body.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response body.

        Raises
        ------
        urllib.error.URLError
            If the request fails at the network layer.
        ValueError
            If the response body cannot be parsed as JSON.
        """
        req = urllib.request.Request(
            url=peer.shield_url(),
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)  # type: ignore[return-value]
