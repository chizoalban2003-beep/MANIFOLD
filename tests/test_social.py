from manifold import (
    SocialConfig,
    SocialGenome,
    SocialManifoldExperiment,
    compile_policy_audit,
    config_for_preset,
    run_social_experiment,
)
from manifold.social import load_grid_from_csv, recommended_prices, social_diversity


def test_gen_2000_seed_starts_near_trust_equilibrium() -> None:
    experiment = SocialManifoldExperiment(
        SocialConfig(population_size=80, generations=1, seed=2500)
    )

    deception = sum(agent.deception for agent in experiment.population) / len(experiment.population)
    verification = sum(agent.verification for agent in experiment.population) / len(experiment.population)
    gossip = sum(agent.gossip for agent in experiment.population) / len(experiment.population)

    assert 0.24 <= deception <= 0.40
    assert 0.42 <= verification <= 0.66
    assert 0.55 <= gossip <= 0.78


def test_verification_phase_transition_math() -> None:
    expected_loss = 0.45 * 0.50
    effective_check_cost = 0.30 * 0.70

    assert effective_check_cost < expected_loss


def test_social_engine_tracks_blacklist_and_forgiveness() -> None:
    config = SocialConfig(population_size=24, generations=3, seed=2500, life_steps=80)
    experiment = SocialManifoldExperiment(config)
    experiment.population = [
        SocialGenome(deception=0.95, verification=0.90, gossip=0.80, memory=0.10, energy=12.0)
        for _ in range(config.population_size)
    ]

    history = experiment.run()

    assert history[-1].blacklist_rate > 0
    assert history[-1].forgiveness_rate > 0


def test_domain_presets_price_verification_by_risk_ratio() -> None:
    trust = recommended_prices("trust")
    misinformation = recommended_prices("misinformation")
    compute = recommended_prices("compute")

    assert misinformation["verification_cost"] < trust["verification_cost"]
    assert compute["false_trust_penalty"] > trust["false_trust_penalty"]


def test_compute_preset_evolves_more_verification_than_deception() -> None:
    history = run_social_experiment(
        config_for_preset("compute", generations=5, population_size=36, seed=2500)
    )
    final = history[-1]

    assert final.average_verification > final.average_deception
    assert final.niche_counts.keys() == {
        "Scout",
        "Verifier",
        "Deceiver",
        "Gossip",
        "Pragmatist",
    }


def test_social_diversity_uses_all_five_genes() -> None:
    genomes = [
        SocialGenome(0.1, 0.2, 0.3, 0.4, 8.0),
        SocialGenome(0.9, 0.8, 0.7, 0.6, 16.0),
    ]

    assert social_diversity(genomes) > 1.0


def test_policy_audit_compiles_deployable_recommendations() -> None:
    config = config_for_preset("compute", generations=5, population_size=36, seed=2500)
    history = run_social_experiment(config)

    audit = compile_policy_audit(history, config)

    assert 0.0 < audit.verification_threshold <= 1.0
    assert audit.recommended_blacklist_after_lies == config.blacklist_after_lies
    assert audit.recommended_forgiveness_window > 0
    assert audit.robustness_score > 0
    assert "force random audits" in audit.monopoly_controls


def test_source_concentration_metrics_are_recorded() -> None:
    history = run_social_experiment(
        SocialConfig(
            population_size=24,
            generations=3,
            seed=2500,
            reputation_cap=0.2,
            random_audit_rate=0.2,
        )
    )
    final = history[-1]

    assert 0.0 <= final.top_source_share <= 1.0
    assert 0.0 <= final.source_hhi <= 1.0
    assert final.monopoly_pressure >= 0.0


def test_predatory_scouts_check_concentrated_reputation_sources() -> None:
    config = SocialConfig(
        population_size=10,
        generations=1,
        seed=2500,
        predatory_scouts=True,
        scout_source_share_trigger=0.10,
        scout_reputation_trigger=0.10,
        random_audit_rate=0.0,
    )
    experiment = SocialManifoldExperiment(config)
    experiment.population = [
        SocialGenome(deception=0.05, verification=0.9, gossip=0.1, memory=0.4, energy=12.0)
        for _ in range(config.population_size)
    ]
    experiment.reputation = [1.0 for _ in range(config.population_size)]

    history = experiment.run()

    assert history[-1].predatory_scout_rate > 0
    assert history[-1].niche_counts["Scout"] == config.population_size


def test_predation_threshold_is_evolved_and_audited() -> None:
    config = SocialConfig(population_size=24, generations=4, seed=2500)

    history = run_social_experiment(config)
    audit = compile_policy_audit(history, config)

    assert 0.55 <= history[-1].average_predation_threshold <= 0.98
    assert 0.55 <= audit.recommended_predation_threshold <= 0.98


def test_csv_grid_loader_maps_real_data(tmp_path) -> None:
    grid_path = tmp_path / "traffic.csv"
    grid_path.write_text(
        "row,col,cost,risk,asset,neutrality\n"
        "0,0,0.2,0.4,0.8,0.1\n"
        "1,1,0.1,0.2,0.3,\n",
        encoding="utf-8",
    )

    grid = load_grid_from_csv(str(grid_path), grid_size=3)
    history = run_social_experiment(
        SocialConfig(
            population_size=12,
            generations=2,
            grid_size=3,
            seed=2500,
            data_path=str(grid_path),
        )
    )

    assert grid[0][0].asset == 0.8
    assert grid[1][1].neutrality > 0
    assert len(history) == 2
