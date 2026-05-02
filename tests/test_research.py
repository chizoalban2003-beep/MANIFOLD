from manifold import run_research_suite


def test_research_suite_reports_bounded_claims() -> None:
    report = run_research_suite(seed=2500)

    assert len(report.findings) == 3
    assert report.honest_summary
    assert {finding.name for finding in report.findings} == {
        "map_quality_sensitivity",
        "outcome_memory_adaptation",
        "baseline_competitiveness",
    }
