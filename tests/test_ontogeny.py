from random import Random
import unittest

from manifold.ontogeny import (
    ENERGY_MAX,
    Vector,
    VectorGenome,
    armor_boost_for,
    clone_with_energy,
    default_routes,
    evolve,
    seed_population,
)


class OntogenyTests(unittest.TestCase):
    def test_armor_boost_respects_finite_energy_budget(self) -> None:
        genome = VectorGenome(
            risk_multiplier=1.0,
            max_risk=3.0,
            armor_threshold=2.0,
            armor_efficiency=1.0,
            conserve_bias=0.0,
        )

        boost = armor_boost_for(
            risk=7.0,
            genome=genome,
            energy_remaining=3.0,
            timesteps_remaining=6,
        )

        self.assertAlmostEqual(boost, 0.5)

    def test_vector_survives_spike_by_spending_energy(self) -> None:
        genome = VectorGenome(
            risk_multiplier=1.0,
            max_risk=4.0,
            armor_threshold=3.0,
            armor_efficiency=1.0,
            conserve_bias=0.0,
        )
        vector = Vector(genome=genome, energy_max=ENERGY_MAX)
        flicker_route = default_routes()[2]

        result = vector.traverse(flicker_route, generation=8)

        self.assertTrue(result.survived)
        self.assertGreater(result.energy_spent, 0.0)
        self.assertGreaterEqual(result.energy_remaining, 0.0)
        self.assertTrue(all(step.effective_risk <= genome.max_risk for step in result.steps))

    def test_vector_takes_detour_when_spike_is_too_expensive(self) -> None:
        genome = VectorGenome(
            risk_multiplier=2.0,
            max_risk=2.5,
            armor_threshold=2.0,
            armor_efficiency=0.5,
            conserve_bias=0.9,
        )
        vector = Vector(genome=genome, energy_max=5.0)

        result = vector.evaluate(default_routes(), generation=8)

        self.assertEqual(result.route_name, "scout")
        self.assertTrue(result.survived)
        self.assertAlmostEqual(result.energy_spent, 0.0)

    def test_seed_population_covers_physics_space(self) -> None:
        population = seed_population(32, Random(11))

        risk_multipliers = [vector.genome.risk_multiplier for vector in population]
        max_risks = [vector.genome.max_risk for vector in population]

        self.assertAlmostEqual(min(risk_multipliers), 0.1)
        self.assertAlmostEqual(max(risk_multipliers), 2.5)
        self.assertAlmostEqual(min(max_risks), 2.0)
        self.assertAlmostEqual(max(max_risks), 9.5)

    def test_evolution_returns_generation_stats(self) -> None:
        population = seed_population(20, Random(3))
        final_population, history = evolve(
            population,
            routes=default_routes(),
            generations=10,
            rng=Random(5),
        )

        self.assertEqual(len(final_population), 20)
        self.assertEqual(len(history), 10)
        self.assertTrue(all(stats.average_regret >= 0.0 for stats in history))
        self.assertTrue(all(0.0 <= stats.survival_rate <= 1.0 for stats in history))
        self.assertGreaterEqual(history[-1].average_energy_spent, 0.0)

    def test_clone_with_energy_validates_capacity(self) -> None:
        vector = seed_population(2, Random(1))[0]

        self.assertEqual(clone_with_energy(vector, 12.0).energy_max, 12.0)
        with self.assertRaises(ValueError):
            clone_with_energy(vector, 0.0)


if __name__ == "__main__":
    unittest.main()
