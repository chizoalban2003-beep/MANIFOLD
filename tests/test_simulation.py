from manifold import ManifoldSimulation, reshape_cells


def test_route_membership_recovers_center_weighting():
    membership = ManifoldSimulation.route_membership()
    weights = membership / 8.0

    assert weights[4] == 0.5
    assert weights[0] == 0.375
    assert weights[2] == 0.375
    assert weights[6] == 0.375
    assert weights[8] == 0.375


def test_ontogeny_scenario_uses_energy_and_teacher():
    simulation = ManifoldSimulation(seed=7, scenario="ontogeny", population_size=28)
    results = simulation.run(generations=50)
    summary = results["summary"]

    assert summary["teacher_events"] >= 1
    assert summary["peak_avg_energy"] > 0.0
    assert summary["cumulative_avg_energy"] > 0.0
    assert len(results["history"]) == 50


def test_geometry_scenario_keeps_teacher_disabled():
    simulation = ManifoldSimulation(seed=7, scenario="geometry", population_size=24)
    results = simulation.run(generations=20)

    assert all(snapshot["teacher_event"] is None for snapshot in results["history"])
    assert all(snapshot["avg_energy_used"] == 0.0 for snapshot in results["history"])


def test_cell_reshape_preserves_grid_layout():
    grid = reshape_cells(list(range(9)))

    assert grid.shape == (3, 3)
    assert grid[1, 1] == 4
