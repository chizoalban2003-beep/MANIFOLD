"""Phase 22: Fleet Dashboard — CI/CD Telemetry and B2B Economy Visualisation.

The Streamlit dashboard previously visualised a single MANIFOLD node.  Phase 22
elevates it to a **Fleet Command Center** by introducing structured data helpers
that aggregate CI/CD history and the inter-org economy into dashboard-ready
objects.

Key ideas
---------
* **CI/CD Telemetry** — ``CIBuildRecord`` captures the outcome of each
  ``ManifoldCICheck`` run.  ``CIBuildHistory`` stores the full history and
  provides analytics (pass rate, most-risky tools, trend).
* **B2B Economy Map** — ``B2BEconomySnapshot`` summarises the
  ``AgentEconomyLedger`` of one or more ``B2BRouter`` instances into a
  dashboard-friendly structure.
* **FleetDashboardData** — top-level container that bundles CI history +
  economy snapshot, suitable for a single ``st.session_state`` value.
* **FleetPanelRenderer** — pure-Python renderer that converts
  ``FleetDashboardData`` into a list of human-readable text blocks (used by
  tests; the Streamlit panel consumes the same helpers).

Key classes
-----------
``CIBuildRecord``
    A single CI build outcome with timestamp, pass/fail, and risk deltas.
``CIBuildHistory``
    Ordered collection of ``CIBuildRecord`` objects with analytics helpers.
``B2BEconomySnapshot``
    Summarised view of one or more ``AgentEconomyLedger`` instances.
``FleetDashboardData``
    Top-level container for fleet-wide state.
``FleetPanelRenderer``
    Converts fleet data into formatted text blocks (for tests + CLI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from .b2b import AgentEconomyLedger, B2BRouter, EconomyEntry
from .gitops import CIRiskReport


# ---------------------------------------------------------------------------
# CIBuildRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CIBuildRecord:
    """A single CI build outcome.

    Attributes
    ----------
    build_id:
        Human-readable build identifier (e.g. ``"build-042"`` or a git SHA).
    passed:
        ``True`` if the CI check passed.
    risk_report:
        The ``CIRiskReport`` produced by ``ManifoldCICheck``.
    timestamp:
        UTC timestamp of the build.  Defaults to *now*.
    branch:
        Git branch name, if known.
    commit_sha:
        Git commit SHA, if known.

    Example
    -------
    ::

        record = CIBuildRecord(
            build_id="build-001",
            passed=ci_report.passed,
            risk_report=ci_report,
        )
    """

    build_id: str
    passed: bool
    risk_report: CIRiskReport
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    branch: str = "main"
    commit_sha: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict."""
        flagged = sum(1 for d in self.risk_report.deltas if d.vetoed)
        max_delta = max((d.delta for d in self.risk_report.deltas), default=0.0)
        return {
            "build_id": self.build_id,
            "passed": self.passed,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "timestamp": self.timestamp.isoformat(),
            "total_tools": len(self.risk_report.deltas),
            "flagged_tools": flagged,
            "max_delta": round(max_delta, 4),
        }


# ---------------------------------------------------------------------------
# CIBuildHistory
# ---------------------------------------------------------------------------


class CIBuildHistory:
    """Ordered collection of ``CIBuildRecord`` objects with analytics.

    Records are stored in insertion order (oldest first).

    Example
    -------
    ::

        history = CIBuildHistory()
        history.add(record)
        print(history.pass_rate())
        print(history.most_risky_tools(top_n=3))
    """

    def __init__(self) -> None:
        self._records: list[CIBuildRecord] = []

    def add(self, record: CIBuildRecord) -> None:
        """Append a build record.

        Parameters
        ----------
        record:
            The build record to add.
        """
        self._records.append(record)

    def records(self) -> list[CIBuildRecord]:
        """Return all records (oldest first)."""
        return list(self._records)

    def latest(self, n: int = 10) -> list[CIBuildRecord]:
        """Return the *n* most recent records (newest first).

        Parameters
        ----------
        n:
            Number of records to return.

        Returns
        -------
        list[CIBuildRecord]
        """
        return list(reversed(self._records[-n:]))

    def pass_rate(self) -> float:
        """Return the fraction of builds that passed [0, 1].

        Returns ``1.0`` if there are no records.
        """
        if not self._records:
            return 1.0
        return sum(1 for r in self._records if r.passed) / len(self._records)

    def total_builds(self) -> int:
        """Return total number of recorded builds."""
        return len(self._records)

    def failed_builds(self) -> list[CIBuildRecord]:
        """Return all failed build records."""
        return [r for r in self._records if not r.passed]

    def most_risky_tools(self, top_n: int = 5) -> list[tuple[str, float]]:
        """Return the *top_n* tools sorted by their maximum risk delta.

        Parameters
        ----------
        top_n:
            Number of tools to return.

        Returns
        -------
        list[tuple[str, float]]
            ``(tool_name, max_delta)`` pairs, sorted descending by delta.
        """
        tool_max: dict[str, float] = {}
        for record in self._records:
            for delta in record.risk_report.deltas:
                prev = tool_max.get(delta.tool_name, 0.0)
                if delta.delta > prev:
                    tool_max[delta.tool_name] = delta.delta
        return sorted(tool_max.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def avg_flagged_per_build(self) -> float:
        """Return average number of flagged tools per build."""
        if not self._records:
            return 0.0
        return sum(
            sum(1 for d in r.risk_report.deltas if d.vetoed)
            for r in self._records
        ) / len(self._records)

    def summary(self) -> dict[str, object]:
        """Return a summary dict of build history analytics.

        Returns
        -------
        dict
            Keys: ``total_builds``, ``passed``, ``failed``, ``pass_rate``,
            ``avg_flagged_per_build``.
        """
        passed = sum(1 for r in self._records if r.passed)
        return {
            "total_builds": self.total_builds(),
            "passed": passed,
            "failed": self.total_builds() - passed,
            "pass_rate": round(self.pass_rate(), 4),
            "avg_flagged_per_build": round(self.avg_flagged_per_build(), 4),
        }


# ---------------------------------------------------------------------------
# B2BEconomySnapshot
# ---------------------------------------------------------------------------


@dataclass
class B2BEconomySnapshot:
    """Summarised view of one or more ``AgentEconomyLedger`` instances.

    Aggregates ledger data from multiple ``B2BRouter`` instances into a
    single dashboard-friendly structure.

    Attributes
    ----------
    entries:
        All ``EconomyEntry`` objects collected from every ledger.
    org_labels:
        Optional display labels mapped from ``org_id`` → human-readable name.

    Example
    -------
    ::

        snapshot = B2BEconomySnapshot.from_routers([router_a, router_b])
        print(snapshot.total_trust_cost())
    """

    entries: list[EconomyEntry] = field(default_factory=list)
    org_labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_ledgers(
        cls,
        ledgers: Sequence[AgentEconomyLedger],
        org_labels: dict[str, str] | None = None,
    ) -> "B2BEconomySnapshot":
        """Build a snapshot from multiple ``AgentEconomyLedger`` objects.

        Parameters
        ----------
        ledgers:
            Ledgers to aggregate.
        org_labels:
            Optional mapping of org_id → display name.

        Returns
        -------
        B2BEconomySnapshot
        """
        all_entries: list[EconomyEntry] = []
        for ledger in ledgers:
            all_entries.extend(ledger.entries())
        return cls(entries=all_entries, org_labels=org_labels or {})

    @classmethod
    def from_routers(
        cls,
        routers: Sequence[B2BRouter],
        org_labels: dict[str, str] | None = None,
    ) -> "B2BEconomySnapshot":
        """Build a snapshot from multiple ``B2BRouter`` instances.

        Parameters
        ----------
        routers:
            Routers whose ledgers will be aggregated.
        org_labels:
            Optional mapping of org_id → display name.

        Returns
        -------
        B2BEconomySnapshot
        """
        return cls.from_ledgers([r.ledger for r in routers], org_labels=org_labels)

    def total_trust_cost(self) -> float:
        """Return total trust cost across all entries."""
        return sum(e.net_trust_cost for e in self.entries)

    def block_rate(self) -> float:
        """Return fraction of calls that were blocked."""
        if not self.entries:
            return 0.0
        blocked = sum(1 for e in self.entries if not e.allowed)
        return blocked / len(self.entries)

    def org_costs(self) -> dict[str, float]:
        """Return total trust cost grouped by remote org ID."""
        costs: dict[str, float] = {}
        for e in self.entries:
            costs[e.remote_org_id] = costs.get(e.remote_org_id, 0.0) + e.net_trust_cost
        return costs

    def org_block_rates(self) -> dict[str, float]:
        """Return block rate grouped by remote org ID."""
        total: dict[str, int] = {}
        blocked: dict[str, int] = {}
        for e in self.entries:
            total[e.remote_org_id] = total.get(e.remote_org_id, 0) + 1
            if not e.allowed:
                blocked[e.remote_org_id] = blocked.get(e.remote_org_id, 0) + 1
        return {org: blocked.get(org, 0) / total[org] for org in total}

    def top_partners(self, top_n: int = 5) -> list[tuple[str, float]]:
        """Return the *top_n* remote orgs by total trust cost (descending).

        Parameters
        ----------
        top_n:
            Number of orgs to return.

        Returns
        -------
        list[tuple[str, float]]
        """
        costs = self.org_costs()
        return sorted(costs.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def avg_reputation(self) -> float:
        """Return average remote-org reputation across all calls."""
        if not self.entries:
            return 0.0
        return sum(e.reputation_score for e in self.entries) / len(self.entries)

    def summary(self) -> dict[str, object]:
        """Return a summary dict.

        Returns
        -------
        dict
            Keys: ``total_calls``, ``block_rate``, ``total_trust_cost``,
            ``avg_reputation``, ``unique_remote_orgs``.
        """
        return {
            "total_calls": len(self.entries),
            "block_rate": round(self.block_rate(), 4),
            "total_trust_cost": round(self.total_trust_cost(), 4),
            "avg_reputation": round(self.avg_reputation(), 4),
            "unique_remote_orgs": len(set(e.remote_org_id for e in self.entries)),
        }


# ---------------------------------------------------------------------------
# FleetDashboardData
# ---------------------------------------------------------------------------


@dataclass
class FleetDashboardData:
    """Top-level container for fleet-wide MANIFOLD state.

    Bundles CI build history and the inter-org economy snapshot into a
    single object, suitable for ``st.session_state`` or CLI rendering.

    Attributes
    ----------
    ci_history:
        CI/CD build history for this fleet node.
    economy:
        B2B economy snapshot aggregating all router ledgers.
    node_id:
        Identifier for this fleet node.
    version:
        MANIFOLD version string.

    Example
    -------
    ::

        data = FleetDashboardData(
            ci_history=history,
            economy=snapshot,
            node_id="prod-node-1",
        )
        renderer = FleetPanelRenderer(data)
        print(renderer.ci_summary_text())
    """

    ci_history: CIBuildHistory = field(default_factory=CIBuildHistory)
    economy: B2BEconomySnapshot = field(default_factory=B2BEconomySnapshot)
    node_id: str = "manifold-node"
    version: str = "1.1.0"

    def to_summary_dict(self) -> dict[str, object]:
        """Return a combined summary dict for serialisation or display."""
        return {
            "node_id": self.node_id,
            "version": self.version,
            "ci": self.ci_history.summary(),
            "economy": self.economy.summary(),
        }


# ---------------------------------------------------------------------------
# FleetPanelRenderer
# ---------------------------------------------------------------------------


class FleetPanelRenderer:
    """Converts ``FleetDashboardData`` into formatted text blocks.

    This class is used by tests and CLI tools.  The Streamlit panel in
    ``app.py`` consumes the same underlying data helpers.

    Parameters
    ----------
    data:
        Fleet dashboard data to render.

    Example
    -------
    ::

        renderer = FleetPanelRenderer(data)
        print(renderer.ci_summary_text())
        print(renderer.economy_summary_text())
    """

    def __init__(self, data: FleetDashboardData) -> None:
        self.data = data

    def ci_summary_text(self) -> str:
        """Return a human-readable CI history summary."""
        s = self.data.ci_history.summary()
        lines = [
            f"=== CI/CD Telemetry — {self.data.node_id} ===",
            f"Total builds : {s['total_builds']}",
            f"Passed       : {s['passed']}",
            f"Failed       : {s['failed']}",
            f"Pass rate    : {s['pass_rate']:.1%}",
            f"Avg flagged  : {s['avg_flagged_per_build']:.2f} tools/build",
        ]
        risky = self.data.ci_history.most_risky_tools(top_n=3)
        if risky:
            lines.append("Top risky tools:")
            for name, delta in risky:
                lines.append(f"  {name}: Δrisk={delta:.4f}")
        return "\n".join(lines)

    def economy_summary_text(self) -> str:
        """Return a human-readable B2B economy summary."""
        s = self.data.economy.summary()
        lines = [
            "=== B2B Economy Map ===",
            f"Total calls        : {s['total_calls']}",
            f"Block rate         : {s['block_rate']:.1%}",
            f"Total trust cost   : {s['total_trust_cost']:.4f}",
            f"Avg reputation     : {s['avg_reputation']:.4f}",
            f"Unique remote orgs : {s['unique_remote_orgs']}",
        ]
        partners = self.data.economy.top_partners(top_n=3)
        if partners:
            lines.append("Top trading partners (by cost):")
            for org_id, cost in partners:
                label = self.data.economy.org_labels.get(org_id, org_id)
                lines.append(f"  {label}: {cost:.4f} trust units")
        return "\n".join(lines)

    def full_report_text(self) -> str:
        """Return a combined report covering CI + economy."""
        return self.ci_summary_text() + "\n\n" + self.economy_summary_text()
