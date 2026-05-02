"""Trust Audit experiments for MANIFOLD Brain.

TrustAudit is the product-facing diagnostic layer for enterprise agent logs. It
answers the practical question: "Would MANIFOLD Brain have reduced regret,
repeated failures, or bad-tool reliance compared with my current routing policy?"
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from .brain import BrainConfig, BrainOutcome, BrainTask, ManifoldBrain, ToolProfile
from .brainbench import (
    BrainLabelledTask,
    BrainPolicyScore,
    run_brain_benchmark,
    sample_brain_tasks,
)


@dataclass(frozen=True)
class TrustAuditConfig:
    """Economic assumptions for support-style trust audits."""

    summary_note_cost: float = 0.06
    repeated_failure_penalty: float = 0.80
    bad_tool_failure_count: int = 4
    benchmark_config: BrainConfig = BrainConfig(generations=3, population_size=16, grid_size=5)


@dataclass(frozen=True)
class TrustAuditFinding:
    name: str
    baseline_cost: float
    manifold_cost: float
    improvement: float
    interpretation: str


@dataclass(frozen=True)
class TrustAuditReport:
    findings: tuple[TrustAuditFinding, ...]
    recommendations: tuple[str, ...]

    @property
    def average_improvement(self) -> float:
        if not self.findings:
            return 0.0
        return fmean(finding.improvement for finding in self.findings)


def run_support_trust_audit(
    tasks: list[BrainLabelledTask] | None = None,
    config: TrustAuditConfig | None = None,
) -> TrustAuditReport:
    """Run a synthetic-but-actionable customer support trust audit."""

    config = config or TrustAuditConfig()
    tasks = tasks or sample_support_tasks()
    findings = (
        regret_reduction_audit(tasks, config),
        gossip_summary_audit(tasks, config),
        bad_tool_memory_audit(config),
    )
    return TrustAuditReport(
        findings=findings,
        recommendations=compile_trust_audit_recommendations(findings),
    )


def regret_reduction_audit(
    tasks: list[BrainLabelledTask], config: TrustAuditConfig
) -> TrustAuditFinding:
    """Compare MANIFOLD Brain utility against an always-answer baseline."""

    report = run_brain_benchmark(tasks, config.benchmark_config)
    scores = {score.name: score for score in report.scores}
    brain = scores["manifold_brain"]
    baseline = scores["always_answer"]
    baseline_cost = regret_cost(baseline)
    manifold_cost = regret_cost(brain)
    return TrustAuditFinding(
        name="regret_reduction",
        baseline_cost=baseline_cost,
        manifold_cost=manifold_cost,
        improvement=relative_improvement(baseline_cost, manifold_cost),
        interpretation=(
            "Compares the cost of always answering against MANIFOLD Brain's routed policy. "
            "This is the headline BrainBench sales metric."
        ),
    )


def gossip_summary_audit(
    tasks: list[BrainLabelledTask], config: TrustAuditConfig
) -> TrustAuditFinding:
    """Model gossip as a summary note that prevents repeated support failures."""

    repeated_issue_count = max(0, len(tasks) - len({task.task.domain for task in tasks}))
    without_gossip = repeated_issue_count * config.repeated_failure_penalty
    note_count = max(1, len({task.task.domain for task in tasks}))
    with_gossip = note_count * config.summary_note_cost
    return TrustAuditFinding(
        name="gossip_summary_notes",
        baseline_cost=without_gossip,
        manifold_cost=with_gossip,
        improvement=relative_improvement(without_gossip, with_gossip),
        interpretation=(
            "Models gossip as the token/latency cost of writing a summary note. "
            "The note is useful when repeated tickets would otherwise hit the same penalty."
        ),
    )


def bad_tool_memory_audit(config: TrustAuditConfig) -> TrustAuditFinding:
    """Check whether failed tool outcomes reduce future reliance on that tool."""

    flaky_tool = ToolProfile(
        "order_lookup",
        cost=0.05,
        latency=0.05,
        reliability=0.90,
        risk=0.04,
        asset=0.85,
        domain="support",
    )
    brain = ManifoldBrain(config.benchmark_config, [flaky_tool])
    task = BrainTask(
        "Check order status before answering",
        domain="support",
        uncertainty=0.35,
        complexity=0.35,
        stakes=0.55,
        source_confidence=0.75,
        tool_relevance=0.95,
        time_pressure=0.6,
        safety_sensitivity=0.05,
        user_patience=0.8,
    )
    before = brain.decide(task)
    before_uses_tool = 1.0 if before.action == "use_tool" else 0.0
    for _ in range(config.bad_tool_failure_count):
        brain.learn(
            task,
            before,
            BrainOutcome(
                success=False,
                cost_paid=0.20,
                risk_realized=0.75,
                asset_gained=0.0,
                rule_violations=1,
            ),
        )
    after = brain.decide(task)
    after_uses_tool = 1.0 if after.action == "use_tool" else 0.0
    return TrustAuditFinding(
        name="bad_tool_memory",
        baseline_cost=before_uses_tool,
        manifold_cost=after_uses_tool,
        improvement=max(0.0, before_uses_tool - after_uses_tool),
        interpretation=(
            "Injects repeated failures for a high-relevance support tool. "
            "A positive improvement means BrainMemory reduced tool reliance."
        ),
    )


def regret_cost(score: BrainPolicyScore) -> float:
    return max(0.0, score.average_action_cost + score.average_risk_penalty - score.utility)


def relative_improvement(baseline_cost: float, manifold_cost: float) -> float:
    if baseline_cost <= 0.0:
        return 0.0
    return max(0.0, (baseline_cost - manifold_cost) / baseline_cost)


def compile_trust_audit_recommendations(
    findings: tuple[TrustAuditFinding, ...]
) -> tuple[str, ...]:
    lookup = {finding.name: finding for finding in findings}
    recommendations: list[str] = []
    if lookup["regret_reduction"].improvement > 0.10:
        recommendations.append("Lead with BrainBench regret reduction in the sales narrative.")
    else:
        recommendations.append("Collect richer labels; the current task map does not show strong regret reduction.")
    if lookup["gossip_summary_notes"].improvement > 0.0:
        recommendations.append("Model gossip as summary-note cost in support workflows; it prevents repeated penalties.")
    if lookup["bad_tool_memory"].improvement > 0.0:
        recommendations.append("Enable BrainMemory before live routing so bad tools develop reputation scars.")
    else:
        recommendations.append("Bad-tool memory did not demote the tool; increase failure penalty or lower tool prior.")
    return tuple(recommendations)


def sample_support_tasks() -> list[BrainLabelledTask]:
    return [
        BrainLabelledTask(
            BrainTask("Where is my order?", "support", 0.25, 0.25, 0.45, 0.8, 0.9, 0.7, 0.0, user_patience=0.9),
            "use_tool",
        ),
        BrainLabelledTask(
            BrainTask("Refund policy is unclear", "support", 0.75, 0.4, 0.45, 0.7, 0.2, 0.4, 0.1, user_patience=0.8),
            "clarify",
        ),
        BrainLabelledTask(
            BrainTask("User threatens legal action", "support", 0.8, 0.7, 0.95, 0.5, 0.3, 0.5, 0.6, user_patience=0.7, dynamic_goal=True),
            "escalate",
            2.0,
        ),
        BrainLabelledTask(
            BrainTask("Small talk while waiting", "support", 0.1, 0.1, 0.1, 0.95, 0.1, 0.7, 0.0, user_patience=0.9),
            "answer",
        ),
        BrainLabelledTask(
            BrainTask("Second agent sees same delayed package", "support", 0.35, 0.3, 0.50, 0.65, 0.8, 0.6, 0.05, user_patience=0.75),
            "use_tool",
        ),
    ]


def format_trust_audit_report(report: TrustAuditReport) -> str:
    lines = ["MANIFOLD Trust Audit"]
    for finding in report.findings:
        lines.append(
            f"- {finding.name}: baseline={finding.baseline_cost:.3f}, "
            f"manifold={finding.manifold_cost:.3f}, improvement={finding.improvement:.1%}"
        )
        lines.append(f"  {finding.interpretation}")
    lines.append("Recommendations:")
    for recommendation in report.recommendations:
        lines.append(f"- {recommendation}")
    return "\n".join(lines)
