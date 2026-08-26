import unittest
import numpy as np

from backend.data_generator import generate_student_data
from backend.genetic_algorithm import GeneticAlgorithm


class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        self.df = generate_student_data(num_students=100, random_state=42)
        self.ga = GeneticAlgorithm(self.df)

    def test_random_chromosome(self):
        chrom = self.ga.random_chromosome()
        self.assertIn("features", chrom)
        self.assertEqual(len(chrom["features"]), self.ga.num_features)
        self.assertTrue(0.0 <= chrom["alpha_gene"] <= 1.0)
        self.assertTrue(0.0 <= chrom["l1_gene"] <= 1.0)
        self.assertGreaterEqual(sum(chrom["features"]), 1)

    def test_gene_decoders(self):
        self.assertAlmostEqual(self.ga.decode_alpha(0.0), 1e-4, places=6)
        self.assertAlmostEqual(self.ga.decode_alpha(1.0), 100.0, places=4)
        self.assertEqual(self.ga.decode_l1_ratio(0.0), 0.001)
        self.assertEqual(self.ga.decode_l1_ratio(0.5), 0.5)
        self.assertEqual(self.ga.decode_l1_ratio(1.0), 1.0)

    def test_deep_copy_independence(self):
        parent1 = self.ga.random_chromosome()
        parent2 = self.ga.random_chromosome()
        parent1_features_before = list(parent1["features"])

        child1, child2 = self.ga.crossover(parent1, parent2)
        self.ga.mutate(child1)

        # Parent features must remain untouched regardless of child mutations
        self.assertEqual(parent1["features"], parent1_features_before)

    def test_fitness_calculation(self):
        chrom = self.ga.random_chromosome()
        fit = self.ga.fitness(chrom)
        self.assertIsInstance(fit, float)
        self.assertFalse(np.isnan(fit))

    def test_ga_evolution(self):
        ga_short = GeneticAlgorithm(
            self.df,
            population_size=10,
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        best_info = ga_short.evolve()

        self.assertIsNotNone(best_info)
        self.assertIn("fitness", best_info)
        self.assertIn("alpha", best_info)
        self.assertIn("l1_ratio", best_info)
        self.assertIn("selected_features", best_info)
        self.assertEqual(len(ga_short.history), 5)


if __name__ == "__main__":
    unittest.main()
