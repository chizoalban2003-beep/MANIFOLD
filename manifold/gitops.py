"""Phase 19: GitOps Loop — CI/CD for Trust.

MANIFOLD can now act as a gate in your CI/CD pipeline.  Every pull request or
deployment commit passes through a lightweight risk analysis before merging.

Key ideas
---------
* ``ManifoldCICheck`` — analyses the diff between the old and new set of
  tool profiles (or ``BrainTask`` workloads) and produces a ``CIRiskReport``.
  If any tool's risk score spikes above the configured threshold the build
  **fails**, just like a linting or security check.

* ``AutonomousPRProposal`` — when ``AutoRuleDiscovery`` detects that the live
  system needs a penalty adjustment, it can auto-draft a "Pull Request"
  description with the math, rationale, and a YAML diff for the compliance
  officer to review in GitHub/GitLab.

Key classes
-----------
``CIRiskDelta``
    Risk change for a single tool between two code revisions.
``CIRiskReport``
    Summary of all risk deltas found in one CI run.  ``passed`` is ``True``
    only if no delta exceeds the veto threshold.
``ManifoldCICheck``
    Runs a risk-gate check against a set of tool profiles.
``AutonomousPRProposal``
    Generates a PR description (markdown + YAML diff) from live discovery
    data that can be opened against the repository's ``manifold.yaml``.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Sequence

from .autodiscovery import AutoRuleDiscovery, PenaltyProposal
from .brain import ToolProfile
from .connector import ConnectorRegistry
from .policy import ManifoldPolicy, RuleDiff


# ---------------------------------------------------------------------------
# CIRiskDelta
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CIRiskDelta:
    """Risk change for a single tool between two code revisions.

    Attributes
    ----------
    tool_name:
        Name of the tool that changed.
    old_risk:
        Risk score before the change.  ``None`` if the tool is new.
    new_risk:
        Risk score after the change.
    delta:
        ``new_risk - old_risk`` (positive = risk increased).
    old_reliability:
        Reliability before the change.  ``None`` if new.
    new_reliability:
        Reliability after the change.
    reliability_delta:
        ``new_reliability - old_reliability`` (negative = degraded).
    vetoed:
        ``True`` if ``new_risk`` exceeds the configured threshold.
    """

    tool_name: str
    old_risk: float | None
    new_risk: float
    delta: float
    old_reliability: float | None
    new_reliability: float
    reliability_delta: float
    vetoed: bool


# ---------------------------------------------------------------------------
# CIRiskReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CIRiskReport:
    """Summary of all risk deltas found in one CI check run.

    Attributes
    ----------
    passed:
        ``True`` when no tool's new risk score exceeds the veto threshold *and*
        no reliability degradation exceeds *max_reliability_drop*.
    deltas:
        Per-tool risk changes.
    vetoed_tools:
        Names of tools that triggered a veto.
    risk_veto_threshold:
        The threshold used for this run.
    max_reliability_drop:
        Maximum allowed reliability drop before failing the build.
    summary:
        Human-readable one-liner.
    """

    passed: bool
    deltas: tuple[CIRiskDelta, ...]
    vetoed_tools: tuple[str, ...]
    risk_veto_threshold: float
    max_reliability_drop: float
    summary: str

    def as_markdown(self) -> str:
        """Render the report as a GitHub/GitLab CI annotation in Markdown."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"## MANIFOLD Trust Gate — {status}",
            "",
            f"Risk veto threshold: `{self.risk_veto_threshold}`  "
            f"Max reliability drop: `{self.max_reliability_drop}`",
            "",
        ]
        if not self.deltas:
            lines.append("_No tool profiles changed._")
        else:
            lines.append("| Tool | Old Risk | New Risk | Δ Risk | Old Rel | New Rel | Δ Rel | Status |")
            lines.append("|------|----------|----------|--------|---------|---------|-------|--------|")
            for d in self.deltas:
                old_r = f"{d.old_risk:.3f}" if d.old_risk is not None else "new"
                old_rel = f"{d.old_reliability:.3f}" if d.old_reliability is not None else "new"
                veto = "🚫 VETO" if d.vetoed else "✅ OK"
                lines.append(
                    f"| {d.tool_name} | {old_r} | {d.new_risk:.3f} | "
                    f"{d.delta:+.3f} | {old_rel} | {d.new_reliability:.3f} | "
                    f"{d.reliability_delta:+.3f} | {veto} |"
                )
        if self.vetoed_tools:
            lines.append("")
            names = ", ".join(f"`{n}`" for n in self.vetoed_tools)
            lines.append(f"**Vetoed tools:** {names}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ManifoldCICheck
# ---------------------------------------------------------------------------


@dataclass
class ManifoldCICheck:
    """Runs a risk-gate check against tool profiles in a CI pipeline.

    Parameters
    ----------
    risk_veto_threshold:
        Maximum new risk score a tool may have without failing the build.
        Default: ``0.45``.
    max_risk_delta:
        Maximum *increase* in a tool's risk score allowed in one PR.
        Default: ``0.15``.
    max_reliability_drop:
        Maximum *decrease* in a tool's reliability score allowed in one PR.
        Default: ``0.10``.
    policy:
        Optional ``ManifoldPolicy`` to derive thresholds from.  When
        provided, the first domain's ``risk_tolerance`` overrides
        *risk_veto_threshold*.

    Example
    -------
    ::

        checker = ManifoldCICheck(risk_veto_threshold=0.4)
        report = checker.check(old_registry, new_registry)
        print(report.as_markdown())
        if not report.passed:
            raise SystemExit(1)
    """

    risk_veto_threshold: float = 0.45
    max_risk_delta: float = 0.15
    max_reliability_drop: float = 0.10
    policy: ManifoldPolicy | None = None

    def __post_init__(self) -> None:
        if self.policy is not None and self.policy.domains:
            self.risk_veto_threshold = self.policy.domains[0].risk_tolerance

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def check(
        self,
        old_profiles: Sequence[ToolProfile],
        new_profiles: Sequence[ToolProfile],
    ) -> CIRiskReport:
        """Compare old and new tool profiles and produce a ``CIRiskReport``.

        Parameters
        ----------
        old_profiles:
            Profiles from the *base* branch (before the PR).
        new_profiles:
            Profiles from the *head* branch (after the PR).

        Returns
        -------
        CIRiskReport
        """
        old_map = {p.name: p for p in old_profiles}
        new_map = {p.name: p for p in new_profiles}

        deltas: list[CIRiskDelta] = []
        vetoed_tools: list[str] = []

        for name, new_p in new_map.items():
            old_p = old_map.get(name)
            old_risk = old_p.risk if old_p is not None else None
            old_rel = old_p.reliability if old_p is not None else None

            delta = new_p.risk - (old_risk if old_risk is not None else new_p.risk)
            rel_delta = new_p.reliability - (old_rel if old_rel is not None else new_p.reliability)

            # Fail conditions
            risk_exceeded = new_p.risk > self.risk_veto_threshold
            delta_exceeded = delta > self.max_risk_delta
            rel_drop_exceeded = rel_delta < -self.max_reliability_drop

            vetoed = risk_exceeded or delta_exceeded or rel_drop_exceeded

            d = CIRiskDelta(
                tool_name=name,
                old_risk=old_risk,
                new_risk=new_p.risk,
                delta=delta,
                old_reliability=old_rel,
                new_reliability=new_p.reliability,
                reliability_delta=rel_delta,
                vetoed=vetoed,
            )
            deltas.append(d)
            if vetoed:
                vetoed_tools.append(name)

        passed = len(vetoed_tools) == 0
        summary = (
            f"PASSED — {len(deltas)} tool(s) checked, none vetoed."
            if passed
            else f"FAILED — {len(vetoed_tools)} tool(s) vetoed: {', '.join(vetoed_tools)}"
        )
        return CIRiskReport(
            passed=passed,
            deltas=tuple(deltas),
            vetoed_tools=tuple(vetoed_tools),
            risk_veto_threshold=self.risk_veto_threshold,
            max_reliability_drop=self.max_reliability_drop,
            summary=summary,
        )

    def check_registry(
        self,
        old_registry: ConnectorRegistry,
        new_registry: ConnectorRegistry,
    ) -> CIRiskReport:
        """Convenience wrapper: compare two ``ConnectorRegistry`` instances.

        Parameters
        ----------
        old_registry:
            Registry from the base branch.
        new_registry:
            Registry from the head branch.

        Returns
        -------
        CIRiskReport
        """
        old_profiles = [
            old_registry.get(n).refreshed_profile()  # type: ignore[union-attr]
            for n in old_registry.names()
            if old_registry.get(n) is not None
        ]
        new_profiles = [
            new_registry.get(n).refreshed_profile()  # type: ignore[union-attr]
            for n in new_registry.names()
            if new_registry.get(n) is not None
        ]
        return self.check(old_profiles, new_profiles)

    def check_policy_diff(self, policy: ManifoldPolicy) -> CIRiskReport:
        """Fail the build if any *pending* ``RuleDiff`` contains a risk spike.

        A diff is considered a risk spike when ``new_value > risk_veto_threshold``
        for a ``risk_tolerance`` field or when ``new_value < (1 - max_reliability_drop)``
        for a ``min_tool_reliability`` field.

        Parameters
        ----------
        policy:
            The ``ManifoldPolicy`` whose pending diffs are examined.

        Returns
        -------
        CIRiskReport
        """
        vetoed_tools: list[str] = []
        deltas: list[CIRiskDelta] = []

        for diff in policy.pending_diffs:
            if diff.status == "pending":
                risk_delta = diff.new_value - diff.old_value if diff.field == "risk_tolerance" else 0.0
                rel_delta = diff.new_value - diff.old_value if diff.field == "min_tool_reliability" else 0.0
                new_risk = diff.new_value if diff.field == "risk_tolerance" else 0.0
                new_rel = diff.new_value if diff.field == "min_tool_reliability" else 1.0

                risk_exceeded = diff.field == "risk_tolerance" and diff.new_value > self.risk_veto_threshold
                delta_exceeded = diff.field == "risk_tolerance" and risk_delta > self.max_risk_delta
                rel_drop = diff.field == "min_tool_reliability" and rel_delta < -self.max_reliability_drop
                vetoed = risk_exceeded or delta_exceeded or rel_drop

                d = CIRiskDelta(
                    tool_name=f"policy:{diff.rule_name}",
                    old_risk=diff.old_value if diff.field == "risk_tolerance" else None,
                    new_risk=new_risk,
                    delta=risk_delta,
                    old_reliability=diff.old_value if diff.field == "min_tool_reliability" else None,
                    new_reliability=new_rel,
                    reliability_delta=rel_delta,
                    vetoed=vetoed,
                )
                deltas.append(d)
                if vetoed:
                    vetoed_tools.append(d.tool_name)

        passed = len(vetoed_tools) == 0
        summary = (
            f"PASSED — {len(deltas)} diff(s) checked."
            if passed
            else f"FAILED — {len(vetoed_tools)} diff(s) breached threshold."
        )
        return CIRiskReport(
            passed=passed,
            deltas=tuple(deltas),
            vetoed_tools=tuple(vetoed_tools),
            risk_veto_threshold=self.risk_veto_threshold,
            max_reliability_drop=self.max_reliability_drop,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# AutonomousPRProposal
# ---------------------------------------------------------------------------


@dataclass
class AutonomousPRProposal:
    """Auto-drafts a Pull Request description for a ``manifold.yaml`` change.

    When ``AutoRuleDiscovery`` finds a new required penalty, MANIFOLD can
    auto-generate a PR description (title, body, YAML diff) ready to open
    against the repository.

    Parameters
    ----------
    discovery:
        Live ``AutoRuleDiscovery`` instance.
    policy:
        Current ``ManifoldPolicy``.
    author:
        Author tag embedded in the PR body.  Default: ``"manifold-bot"``.
    min_confidence:
        Only include proposals with confidence above this threshold.
        Default: ``0.6``.

    Example
    -------
    ::

        proposer = AutonomousPRProposal(discovery=ard, policy=policy)
        pr = proposer.draft()
        print(pr.title)
        print(pr.body)
    """

    discovery: AutoRuleDiscovery
    policy: ManifoldPolicy
    author: str = "manifold-bot"
    min_confidence: float = 0.6

    _drafts: list["PRDraft"] = field(default_factory=list, init=False, repr=False)

    def draft(self) -> "PRDraft":
        """Generate a ``PRDraft`` from pending penalty proposals.

        Returns
        -------
        PRDraft
        """
        proposals = [
            p for p in self.discovery.suggest_penalty_updates()
            if p.confidence >= self.min_confidence
        ]
        diffs = _proposals_to_rule_diffs(proposals, domain="general")

        # Build a YAML diff string showing what changes
        yaml_diff = _render_yaml_diff(self.policy, diffs)
        title = self._pr_title(proposals)
        body = self._pr_body(proposals, diffs, yaml_diff)

        pr = PRDraft(
            title=title,
            body=body,
            yaml_diff=yaml_diff,
            proposals=tuple(proposals),
            diffs=tuple(diffs),
        )
        self._drafts.append(pr)
        return pr

    def draft_from_diffs(self, diffs: list[RuleDiff]) -> "PRDraft":
        """Generate a ``PRDraft`` directly from pre-computed ``RuleDiff`` objects.

        Parameters
        ----------
        diffs:
            List of diffs to include in the PR.

        Returns
        -------
        PRDraft
        """
        yaml_diff = _render_yaml_diff(self.policy, diffs)
        title = f"[manifold-bot] Policy update: {len(diffs)} rule(s) changed"
        body = self._pr_body([], diffs, yaml_diff)
        pr = PRDraft(
            title=title,
            body=body,
            yaml_diff=yaml_diff,
            proposals=(),
            diffs=tuple(diffs),
        )
        self._drafts.append(pr)
        return pr

    def drafts(self) -> list["PRDraft"]:
        """Return all drafted PRs."""
        return list(self._drafts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pr_title(self, proposals: list[PenaltyProposal]) -> str:
        if not proposals:
            return "[manifold-bot] No penalty changes detected"
        rules = ", ".join(p.rule_name for p in proposals[:3])
        suffix = f" (+{len(proposals) - 3} more)" if len(proposals) > 3 else ""
        return f"[manifold-bot] Update policy penalties: {rules}{suffix}"

    def _pr_body(
        self,
        proposals: list[PenaltyProposal],
        diffs: list[RuleDiff],
        yaml_diff: str,
    ) -> str:
        lines = [
            "## 🤖 Autonomous Policy PR — generated by MANIFOLD",
            "",
            f"**Author:** `{self.author}`  ",
            f"**Min confidence:** `{self.min_confidence}`  ",
            f"**Proposals:** {len(proposals)}  ",
            f"**Diffs:** {len(diffs)}",
            "",
        ]
        if proposals:
            lines += [
                "### Penalty Proposals",
                "",
                "| Rule | Trigger | Current | Proposed | Confidence | Rationale |",
                "|------|---------|---------|----------|------------|-----------|",
            ]
            for p in proposals:
                lines.append(
                    f"| `{p.rule_name}` | `{p.trigger}` | "
                    f"`{p.current_penalty:.2f}` | `{p.proposed_penalty:.2f}` | "
                    f"`{p.confidence:.2f}` | {p.rationale} |"
                )
            lines.append("")

        if diffs:
            lines += [
                "### Policy YAML Diff",
                "",
                "```yaml",
                yaml_diff.strip(),
                "```",
                "",
            ]

        lines += [
            "---",
            "_This PR was auto-generated by the MANIFOLD Autonomous Policy Engine._",
            "_Review the math above and merge/reject via the compliance workflow._",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PRDraft
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PRDraft:
    """An auto-generated Pull Request ready to open against the policy file.

    Attributes
    ----------
    title:
        PR title string.
    body:
        Full PR body in GitHub-flavoured Markdown.
    yaml_diff:
        YAML showing the proposed changes to ``manifold.yaml``.
    proposals:
        The ``PenaltyProposal`` objects that drove this PR.
    diffs:
        The ``RuleDiff`` objects encoded in the PR.
    """

    title: str
    body: str
    yaml_diff: str
    proposals: tuple[PenaltyProposal, ...]
    diffs: tuple[RuleDiff, ...]


# ---------------------------------------------------------------------------
# GitHub Actions YAML template
# ---------------------------------------------------------------------------

GITHUB_ACTION_YAML = textwrap.dedent("""\
    # .github/workflows/manifold-trust.yml
    # Auto-generated by MANIFOLD Phase 19 GitOps Loop
    name: MANIFOLD Trust Gate

    on:
      pull_request:
        branches: [main, master]
      push:
        branches: [main, master]

    jobs:
      manifold-ci-check:
        name: MANIFOLD Risk Gate
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
            with:
              fetch-depth: 0

          - name: Set up Python
            uses: actions/setup-python@v5
            with:
              python-version: "3.12"

          - name: Install MANIFOLD
            run: pip install manifold-ai

          - name: Run MANIFOLD Trust Audit
            run: python deploy_shadow.py --ci-mode --policy manifold.yaml

          - name: Upload Trust Report
            if: always()
            uses: actions/upload-artifact@v4
            with:
              name: manifold-trust-report
              path: manifold_report.json
""")


def generate_github_action() -> str:
    """Return the GitHub Actions YAML template for the trust gate.

    Returns
    -------
    str
        A YAML string suitable for saving as
        ``.github/workflows/manifold-trust.yml``.
    """
    return GITHUB_ACTION_YAML


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _proposals_to_rule_diffs(
    proposals: list[PenaltyProposal],
    domain: str = "general",
) -> list[RuleDiff]:
    """Convert ``PenaltyProposal`` objects to ``RuleDiff`` objects."""
    diffs: list[RuleDiff] = []
    for p in proposals:
        diffs.append(
            RuleDiff(
                rule_name=p.rule_name,
                trigger=p.trigger,
                domain=domain,
                field="risk_tolerance",
                old_value=p.current_penalty,
                new_value=p.proposed_penalty,
                rationale=p.rationale,
                confidence=p.confidence,
                status="pending",
            )
        )
    return diffs


def _render_yaml_diff(policy: ManifoldPolicy, diffs: list[RuleDiff]) -> str:
    """Produce a simple YAML diff string showing proposed changes."""
    if not diffs:
        return "# No changes proposed\n"
    lines = ["# Proposed changes to manifold.yaml", ""]
    for diff in diffs:
        lines.append(f"# Rule: {diff.rule_name}  Trigger: {diff.trigger}")
        lines.append(f"# Confidence: {diff.confidence:.2f}  Domain: {diff.domain}")
        lines.append(f"-   {diff.field}: {diff.old_value}")
        lines.append(f"+   {diff.field}: {diff.new_value}")
        lines.append(f"#   Rationale: {diff.rationale}")
        lines.append("")
    return "\n".join(lines)
