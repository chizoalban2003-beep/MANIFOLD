from manifold import BrainConfig
from manifold.brainbench import load_brain_tasks_csv, run_brain_benchmark, sample_brain_tasks


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


def test_brainbench_loads_real_agent_log_csv(tmp_path) -> None:
    path = tmp_path / "agent_logs.csv"
    path.write_text(
        "prompt,expected_action,domain,uncertainty,complexity,stakes,source_confidence,tool_relevance,time_pressure,safety_sensitivity,collaboration_value,user_patience,dynamic_goal,weight\n"
        "hello,answer,chat,0.05,0.1,0.1,0.95,0.1,0.5,0.0,0.2,0.9,false,1\n"
        "incident,use_tool,incident,0.65,0.6,0.8,0.25,0.8,0.9,0.25,0.2,0.6,true,2\n",
        encoding="utf-8",
    )

    tasks = load_brain_tasks_csv(str(path))
    report = run_brain_benchmark(
        tasks,
        BrainConfig(generations=2, population_size=12, grid_size=5),
    )

    assert len(tasks) == 2
    assert tasks[1].task.domain == "incident"
    assert tasks[1].expected_action == "use_tool"
    assert report.scores
