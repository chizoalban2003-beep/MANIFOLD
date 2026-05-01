from manifold import ManifoldExperiment, SimulationConfig, VectorGenome, run_experiment


def test_vector_spends_energy_to_cross_spike() -> None:
    experiment = ManifoldExperiment(
        SimulationConfig(population_size=4, generations=1, recharge_enabled=False)
    )
    experiment.teacher_spikes[(1, 2)] = 8.0
    genome = VectorGenome(
        risk_multiplier=0.2,
        max_risk=3.0,
        armor_bias=1.0,
        conserve_bias=0.0,
        recharge_bias=0.0,
    )

    result = experiment.evaluate_vector(genome, generation=0)

    assert result.energy_spent > 0
    assert result.energy_remaining < experiment.config.energy_max
    assert result.reached_goal


def test_recharge_bias_makes_subtarget_attractive() -> None:
    config = SimulationConfig(
        population_size=4,
        generations=1,
        recharge_enabled=True,
        recharge_cells=((1, 1),),
        recharge_amount=8.0,
        energy_max=5.0,
    )
    experiment = ManifoldExperiment(config)
    genome = VectorGenome(
        risk_multiplier=0.1,
        max_risk=2.0,
        armor_bias=1.0,
        conserve_bias=0.0,
        recharge_bias=1.0,
    )

    route = experiment.evaluate_vector(genome, generation=0).route

    assert (1, 1) in route


def test_run_experiment_returns_generation_history() -> None:
    history = run_experiment(SimulationConfig(population_size=12, generations=5, seed=99))

    assert len(history) == 5
    assert history[-1].generation == 4
    assert set(history[-1].niche_counts) == {"Tank", "Scout", "Hybrid"}
    assert history[-1].diversity > 0
