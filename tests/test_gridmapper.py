from manifold import AgentPopulation, DynamicTarget, GridWorld


def test_grid_world_compiles_dynamic_targets() -> None:
    world = GridWorld(size=5)
    world.set_cell(2, 2, cost=0.1, risk=0.2, neutrality=0.7, asset=0.0)
    world.add_dynamic_targets(
        [
            {"id": "order", "pos": (2, 2), "asset": 5.0, "moves": "static"},
            DynamicTarget(
                id="cycle",
                pos=(0, 0),
                asset=2.0,
                moves="cycle",
                path=((1, 1), (1, 2)),
            ),
        ]
    )

    tick_zero = world.compiled_grid(0)
    tick_one = world.compiled_grid(1)

    assert tick_zero[2][2].asset == 5.0
    assert tick_zero[1][1].asset == 2.0
    assert tick_one[1][2].asset == 2.0


def test_agent_population_optimizes_world_with_rules() -> None:
    world = GridWorld(size=5)
    world.set_cell(2, 2, cost=0.1, risk=0.1, neutrality=0.8, asset=1.0)
    world.add_dynamic_targets(
        [{"id": "target", "pos": (2, 2), "asset": 3.0, "moves": "static"}]
    )
    world.add_rule("late_delivery", penalty=1.0, triggers="miss_target")
    world.add_rule("false_report", penalty=0.5, triggers="trusted_lie")

    result = AgentPopulation(n=16, predators=0.1).optimize(world, generations=3)

    assert len(result.history) == 3
    assert result.verification > 0
    assert result.reputation_cap > 0
    assert result.rule_penalty_budget == 0.75
    assert result.target_snapshots[0] == ((2, 2),)


def test_grid_world_loads_csv_and_exports_heatmap(tmp_path) -> None:
    source = tmp_path / "mapper.csv"
    output = tmp_path / "heatmap.csv"
    source.write_text(
        "row,col,cost,risk,asset,neutrality\n"
        "0,0,0.2,0.4,0.8,0.1\n"
        "4,4,0.3,0.2,1.1,0.5\n",
        encoding="utf-8",
    )

    world = GridWorld.from_csv(str(source), size=5)
    world.add_dynamic_targets([{"id": "moving", "pos": (1, 1), "asset": 2.0}])
    world.export_heatmap_csv(str(output))

    assert world.cells[0][0].risk == 0.4
    assert output.exists()
    assert "row,col,cost,risk,neutrality,asset" in output.read_text(encoding="utf-8")
