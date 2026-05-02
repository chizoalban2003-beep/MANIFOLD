from manifold import BrainConfig
from manifold.brainbench import run_brain_benchmark, sample_brain_tasks


def test_brainbench_runs_against_agentic_baselines() -> None:
    report = run_brain_benchmark(
        sample_brain_tasks(),
        BrainConfig(generations=2, population_size=12, grid_size=5),
    )

    assert report.best_policy
    assert report.brain_rank >= 1
    assert {score.name for score in report.scores} >= {
        "manifold_brain",
        "react_style",
        "tool_first",
        "static_risk",
    }
    assert report.recommendations
