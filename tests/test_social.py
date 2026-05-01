from manifold import SocialConfig, SocialGenome, SocialManifoldExperiment, config_for_preset, run_social_experiment
from manifold.social import recommended_prices, social_diversity


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
    assert final.niche_counts.keys() == {"Verifier", "Deceiver", "Gossip", "Pragmatist"}


def test_social_diversity_uses_all_five_genes() -> None:
    genomes = [
        SocialGenome(0.1, 0.2, 0.3, 0.4, 8.0),
        SocialGenome(0.9, 0.8, 0.7, 0.6, 16.0),
    ]

    assert social_diversity(genomes) > 1.0
