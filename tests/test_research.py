from manifold import run_research_suite, run_gossip_research_suite


def test_research_suite_reports_bounded_claims() -> None:
    report = run_research_suite(seed=2500)

    assert len(report.findings) == 3
    assert report.honest_summary
    assert {finding.name for finding in report.findings} == {
        "map_quality_sensitivity",
        "outcome_memory_adaptation",
        "baseline_competitiveness",
    }


def test_gossip_research_suite_consensus_speed() -> None:
    report = run_gossip_research_suite(seed=2500, n_agents=20)

    names = {f.name for f in report.findings}
    assert names == {"consensus_speed", "sybil_resilience", "social_recovery"}
    assert report.honest_summary


def test_gossip_research_suite_all_probes_pass() -> None:
    report = run_gossip_research_suite(seed=2500, n_agents=20)

    for finding in report.findings:
        assert finding.passed, (
            f"Probe '{finding.name}' failed: metric={finding.metric:.3f}. "
            f"Interpretation: {finding.interpretation}"
        )


def test_gossip_research_suite_sybil_ratio_is_meaningful() -> None:
    report = run_gossip_research_suite(seed=2500, n_agents=20)

    sybil = next(f for f in report.findings if f.name == "sybil_resilience")
    # Unscreened notes should inflict more damage than flagged-scout notes.
    # Expected ratio ≈ 1/0.7 ≈ 1.43 (scout discount of 0.7x).
    assert sybil.metric >= 1.2, (
        f"Scout discount ratio {sybil.metric:.2f} too low — "
        "unscreened notes should outweigh flagged-scout notes by at least 1.2×"
    )


def test_gossip_research_suite_consensus_within_10_rounds() -> None:
    n_agents = 20
    report = run_gossip_research_suite(seed=2500, n_agents=n_agents)

    speed = next(f for f in report.findings if f.name == "consensus_speed")
    assert speed.metric <= 10, (
        f"Consensus took {speed.metric} rounds — expected ≤10 for a {n_agents}-agent network"
    )
