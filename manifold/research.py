"""Research probes for MANIFOLD Brain.

These experiments are intentionally modest. They do not prove general
intelligence. They test whether specific research claims are plausible:

- quality of the problem map matters,
- outcome feedback changes future decisions,
- MANIFOLD Brain behaves like an executive controller rather than a generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import fmean

from .brain import BrainConfig, BrainOutcome, BrainTask, ManifoldBrain, default_tools
from .brainbench import BrainLabelledTask, run_brain_benchmark, sample_brain_tasks


@dataclass(frozen=True)
class ResearchFinding:
    name: str
    metric: float
    passed: bool
    interpretation: str


@dataclass(frozen=True)
class ResearchReport:
    findings: tuple[ResearchFinding, ...]
    honest_summary: tuple[str, ...]


def run_research_suite(seed: int = 2500) -> ResearchReport:
    """Run bounded research probes against MANIFOLD Brain."""

    findings = (
        map_quality_sensitivity(seed),
        outcome_memory_adaptation(seed),
        baseline_competitiveness(seed),
    )
    return ResearchReport(
        findings=findings,
        honest_summary=(
            "MANIFOLD Brain is plausible as an agent executive/controller layer.",
            "It is not a replacement for LLM generation or neural reasoning.",
            "Its strongest use is deciding when an AI system should act, verify, use tools, escalate, or refuse.",
            "The approach depends on measurable task features and outcome labels; bad maps reduce value.",
        ),
    )


def map_quality_sensitivity(seed: int = 2500) -> ResearchFinding:
    """Check whether degraded task maps reduce benchmark utility."""

    clean_tasks = sample_brain_tasks()
    noisy_tasks = perturb_tasks(clean_tasks, seed=seed, noise=0.35)
    config = BrainConfig(generations=3, population_size=16, grid_size=5, seed=seed)
    clean = run_brain_benchmark(clean_tasks, config)
    noisy = run_brain_benchmark(noisy_tasks, config)
    clean_score = next(score for score in clean.scores if score.name == "manifold_brain")
    noisy_score = next(score for score in noisy.scores if score.name == "manifold_brain")
    drop = clean_score.utility - noisy_score.utility
    return ResearchFinding(
        name="map_quality_sensitivity",
        metric=drop,
        passed=drop >= 0.02,
        interpretation=(
            "Utility falls when task features are perturbed, so MANIFOLD is not magic: "
            "it needs reasonably measured cost/risk/asset inputs."
        ),
    )


def outcome_memory_adaptation(seed: int = 2500) -> ResearchFinding:
    """Check whether repeated bad outcomes make a domain more cautious."""

    brain = ManifoldBrain(BrainConfig(generations=2, population_size=12, grid_size=5, seed=seed), default_tools())
    task = BrainTask(
        "Ambiguous regulated incident",
        domain="regulated",
        uncertainty=0.65,
        complexity=0.6,
        stakes=0.75,
        source_confidence=0.45,
        tool_relevance=0.5,
        safety_sensitivity=0.55,
        dynamic_goal=True,
    )
    before = brain.decide(task)
    for _ in range(4):
        brain.learn(
            task,
            before,
            BrainOutcome(
                success=False,
                cost_paid=0.4,
                risk_realized=0.8,
                asset_gained=0.0,
                rule_violations=1,
            ),
        )
    after = brain.decide(task)
    delta = after.risk_score - before.risk_score
    cautious_actions = {"clarify", "retrieve", "verify", "use_tool", "delegate", "plan", "escalate", "refuse"}
    return ResearchFinding(
        name="outcome_memory_adaptation",
        metric=delta,
        passed=delta > 0 and after.action in cautious_actions,
        interpretation=(
            "Negative outcomes increase domain risk pressure and keep the future policy in a cautious action band."
        ),
    )


def baseline_competitiveness(seed: int = 2500) -> ResearchFinding:
    """Check whether Brain beats naive agentic baselines on sample tasks."""

    report = run_brain_benchmark(
        sample_brain_tasks(),
        BrainConfig(generations=3, population_size=16, grid_size=5, seed=seed),
    )
    brain = next(score for score in report.scores if score.name == "manifold_brain")
    naive = [
        score
        for score in report.scores
        if score.name in {"always_answer", "react_style", "tool_first", "retrieve_first"}
    ]
    margin = brain.utility - max(score.utility for score in naive)
    return ResearchFinding(
        name="baseline_competitiveness",
        metric=margin,
        passed=margin > 0.05,
        interpretation=(
            "On the bundled benchmark, MANIFOLD Brain beats naive ReAct-like, tool-first, retrieve-first, "
            "and always-answer policies on utility."
        ),
    )


def perturb_tasks(
    tasks: list[BrainLabelledTask],
    seed: int,
    noise: float,
) -> list[BrainLabelledTask]:
    rng = random.Random(seed)
    perturbed: list[BrainLabelledTask] = []
    for labelled in tasks:
        task = labelled.task

        def jitter(value: float) -> float:
            return max(0.0, min(1.0, value + rng.uniform(-noise, noise)))

        perturbed.append(
            BrainLabelledTask(
                task=BrainTask(
                    prompt=task.prompt,
                    domain=task.domain,
                    uncertainty=jitter(task.uncertainty),
                    complexity=jitter(task.complexity),
                    stakes=jitter(task.stakes),
                    source_confidence=jitter(task.source_confidence),
                    tool_relevance=jitter(task.tool_relevance),
                    time_pressure=jitter(task.time_pressure),
                    safety_sensitivity=jitter(task.safety_sensitivity),
                    collaboration_value=jitter(task.collaboration_value),
                    user_patience=jitter(task.user_patience),
                    dynamic_goal=task.dynamic_goal,
                ),
                expected_action=labelled.expected_action,
                weight=labelled.weight,
            )
        )
    return perturbed


def format_research_report(report: ResearchReport) -> str:
    lines = ["MANIFOLD Brain research probes"]
    for finding in report.findings:
        status = "PASS" if finding.passed else "WARN"
        lines.append(f"- {status} {finding.name}: metric={finding.metric:.3f}")
        lines.append(f"  {finding.interpretation}")
    lines.append("Honest summary:")
    for item in report.honest_summary:
        lines.append(f"- {item}")
    return "\n".join(lines)
