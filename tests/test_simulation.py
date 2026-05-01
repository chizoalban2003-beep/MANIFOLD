from manifold import ManifoldExperiment, SimulationConfig, VectorGenome, run_experiment, transfer_population
from manifold.simulation import boost_needed


def test_vector_spends_energy_to_cross_teacher_spike() -> None:
    experiment = ManifoldExperiment(
        SimulationConfig(population_size=4, generations=1, recharge_enabled=False)
    )
    genome = VectorGenome(
        risk_multiplier=0.2,
        max_risk=4.5,
        energy_aversion=1.0,
        charger_bias=0.0,
    )
    experiment.population = [genome]
    results, teacher_mutated = experiment.step(generation=0)
    result = results[0]

    assert boost_needed(9.0, genome.max_risk) == 9.0
    assert teacher_mutated
    assert result.energy_spent > 0
    assert result.energy_remaining < experiment.config.energy_max


def test_chargers_support_survival_under_low_battery() -> None:
    config = SimulationConfig(
        population_size=4,
        generations=1,
        recharge_enabled=True,
        recharge_amount=4.0,
        energy_max=8.0,
        teacher_mode="periodic",
    )
    experiment = ManifoldExperiment(config)
    genome = VectorGenome(
        risk_multiplier=0.1,
        max_risk=5.0,
        energy_aversion=0.5,
        charger_bias=1.0,
    )
    experiment.population = [genome]

    result = experiment.step(generation=0)[0][0]

    assert result.recharge_visits >= 1
    assert result.reached_goal


def test_dual_rate_mutation_keeps_body_slower_than_policy() -> None:
    config = SimulationConfig(max_r_sigma=0.02, aversion_sigma=0.12, policy_sigma=0.12)
    parent = VectorGenome(0.5, 5.0, 1.5, 0.5)
    experiment = ManifoldExperiment(SimulationConfig(population_size=4, generations=1, seed=8))

    children = [parent.mutated(experiment.rng, config) for _ in range(100)]
    max_r_delta = sum(abs(child.max_risk - parent.max_risk) for child in children) / len(children)
    aversion_delta = sum(abs(child.energy_aversion - parent.energy_aversion) for child in children) / len(children)

    assert aversion_delta > max_r_delta * 2


def test_communication_reports_signal_metrics() -> None:
    history = run_experiment(
        SimulationConfig(
            population_size=12,
            generations=5,
            seed=99,
            communication_enabled=True,
            teacher_mode="periodic",
        )
    )

    assert len(history) == 5
    assert history[-1].generation == 4
    assert set(history[-1].niche_counts) == {"Body", "Planner", "Hybrid"}
    assert -1.0 <= history[-1].signal_spike_correlation <= 1.0


def test_transfer_population_scales_world_and_chargers() -> None:
    source = ManifoldExperiment(SimulationConfig(population_size=8, generations=2, seed=4))
    source.run()

    target = transfer_population(source, target_grid_size=21)

    assert target.config.grid_size == 21
    assert target.config.start == (0, 10)
    assert target.config.goal == (20, 10)
    assert target.config.recharge_cells == ((7, 10), (14, 10))
