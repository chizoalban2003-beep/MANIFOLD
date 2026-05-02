from manifold import (
    DialogueTask,
    LabelledTask,
    TrustRouterConfig,
    load_labelled_tasks_csv,
    run_trust_benchmark,
    sample_trust_tasks,
)


def test_trustbench_runs_against_baselines() -> None:
    report = run_trust_benchmark(
        sample_trust_tasks(),
        TrustRouterConfig(generations=2, population_size=12, grid_size=5),
    )

    assert report.best_policy
    assert report.trustrouter_rank >= 1
    assert {score.name for score in report.scores} >= {
        "trustrouter",
        "always_answer",
        "risk_weighted_static",
    }
    assert report.recommendations


def test_trustbench_loads_labelled_csv(tmp_path) -> None:
    path = tmp_path / "tasks.csv"
    path.write_text(
        "prompt,expected_action,domain,uncertainty,complexity,stakes,source_confidence,user_patience,safety_sensitivity,dynamic_intent,weight\n"
        "hello,answer,chat,0.1,0.1,0.1,0.95,0.8,0,false,1\n"
        "unsafe,refuse,safety,0.9,0.8,0.9,0.2,0.5,0.95,true,2\n",
        encoding="utf-8",
    )

    tasks = load_labelled_tasks_csv(str(path))

    assert len(tasks) == 2
    assert tasks[1].expected_action == "refuse"
    assert tasks[1].weight == 2


def test_trustbench_scores_penalize_missed_verification() -> None:
    report = run_trust_benchmark(
        [
            LabelledTask(
                DialogueTask(
                    "danger",
                    uncertainty=0.9,
                    stakes=0.9,
                    source_confidence=0.2,
                    safety_sensitivity=0.8,
                ),
                "escalate",
            )
        ],
        TrustRouterConfig(generations=2, population_size=12, grid_size=5),
    )

    always_answer = next(score for score in report.scores if score.name == "always_answer")

    assert always_answer.missed_verification_rate == 1.0
    assert always_answer.average_risk_penalty > 0
