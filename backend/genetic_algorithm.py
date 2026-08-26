import random
import copy
import warnings
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning

# Suppress ElasticNet convergence warnings during search sweeps
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class GeneticAlgorithm:

    def __init__(
        self,
        dataframe,
        target_column="Final_Marks",
        population_size=30,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        gamma=0.02,
        tournament_size=3,
        elitism_count=2,
    ):
        self.df = dataframe
        self.target = target_column

        if self.target not in dataframe.columns:
            raise ValueError(f"Target column '{self.target}' not found in DataFrame")

        self.feature_names = [
            c for c in dataframe.columns
            if c != self.target
        ]

        self.num_features = len(self.feature_names)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.gamma = gamma
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

        self.population = []
        self.history = []
        self.current_generation = 0
        self.is_running = False
        self.best_chromosome = None

    # --------------------------------------------------
    # Chromosome Creation & Helper Methods
    # --------------------------------------------------

    def random_chromosome(self):
        """Generate a random chromosome with binary feature selection and continuous hyperparameter genes."""
        features = [random.randint(0, 1) for _ in range(self.num_features)]
        # Ensure at least 1 feature is selected initially
        if sum(features) == 0 and self.num_features > 0:
            features[random.randint(0, self.num_features - 1)] = 1

        return {
            "features": features,
            "alpha_gene": random.random(),
            "l1_gene": random.random(),
            "fitness": None,
        }

    def decode_alpha(self, gene):
        """Map gene [0, 1] to alpha log scale [10^-4, 10^2] -> [0.0001, 100.0]."""
        return float(10 ** (-4 + gene * 6))

    def decode_l1_ratio(self, gene):
        """Map gene [0, 1] to l1_ratio [0.001, 1.0]. ElasticNet l1_ratio must be in (0, 1]."""
        return float(max(0.001, min(1.0, gene)))

    def get_chromosome_info(self, chromosome):
        """Decode chromosome into a human-readable dictionary."""
        if chromosome is None:
            return None

        selected_indices = [
            i for i, bit in enumerate(chromosome["features"])
            if bit == 1
        ]
        selected_features = [self.feature_names[i] for i in selected_indices]

        return {
            "fitness": float(chromosome["fitness"]) if chromosome["fitness"] is not None else -999.0,
            "alpha": self.decode_alpha(chromosome["alpha_gene"]),
            "l1_ratio": self.decode_l1_ratio(chromosome["l1_gene"]),
            "alpha_gene": float(chromosome["alpha_gene"]),
            "l1_ratio_gene": float(chromosome["l1_gene"]),
            "selected_features": selected_features,
            "feature_bits": list(chromosome["features"]),
            "num_selected": len(selected_features),
            "total_features": self.num_features,
        }

    # --------------------------------------------------
    # Population Initialization
    # --------------------------------------------------

    def initialize_population(self):
        """Initialize random population."""
        self.population = [
            self.random_chromosome()
            for _ in range(self.population_size)
        ]

    # --------------------------------------------------
    # Fitness Evaluation
    # --------------------------------------------------

    def fitness(self, chromosome):
        """Calculate fitness score based on 5-fold cross validated R2 score with L0 feature complexity penalty."""
        selected = [
            self.feature_names[i]
            for i, bit in enumerate(chromosome["features"])
            if bit == 1
        ]

        if len(selected) == 0:
            return -999.0

        X = self.df[selected]
        y = self.df[self.target]

        alpha = self.decode_alpha(chromosome["alpha_gene"])
        l1_ratio = self.decode_l1_ratio(chromosome["l1_gene"])

        try:
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=5000,
                random_state=42,
                tol=1e-3,
            )

            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="r2",
            )

            if np.isnan(cv_scores).any():
                score = -999.0
            else:
                score = float(np.mean(cv_scores))
        except Exception:
            score = -999.0

        # L0 feature ratio penalty to favor parsimonious models
        penalty = self.gamma * (len(selected) / self.num_features)
        return float(score - penalty)

    def evaluate_population(self):
        """Evaluate fitness for all individuals in population."""
        for chromosome in self.population:
            if chromosome["fitness"] is None:
                chromosome["fitness"] = self.fitness(chromosome)

    # --------------------------------------------------
    # Tournament Selection
    # --------------------------------------------------

    def tournament_selection(self):
        """Select best parent from a random tournament group."""
        tournament = random.sample(self.population, self.tournament_size)
        tournament.sort(key=lambda x: x["fitness"], reverse=True)
        return copy.deepcopy(tournament[0])

    # --------------------------------------------------
    # Crossover (Deep Copy)
    # --------------------------------------------------

    def crossover(self, parent1, parent2):
        """Perform 2-point crossover on binary features and uniform crossover on continuous genes."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        if random.random() < self.crossover_rate and self.num_features > 1:
            p1 = random.randint(0, self.num_features - 2)
            p2 = random.randint(p1 + 1, self.num_features - 1)

            child1["features"] = (
                parent1["features"][:p1]
                + parent2["features"][p1:p2]
                + parent1["features"][p2:]
            )
            child2["features"] = (
                parent2["features"][:p1]
                + parent1["features"][p1:p2]
                + parent2["features"][p2:]
            )

            child1["alpha_gene"] = (
                parent1["alpha_gene"] if random.random() < 0.5 else parent2["alpha_gene"]
            )
            child2["alpha_gene"] = (
                parent2["alpha_gene"] if random.random() < 0.5 else parent1["alpha_gene"]
            )

            child1["l1_gene"] = (
                parent1["l1_gene"] if random.random() < 0.5 else parent2["l1_gene"]
            )
            child2["l1_gene"] = (
                parent2["l1_gene"] if random.random() < 0.5 else parent1["l1_gene"]
            )

            # Reset fitness since genes changed
            child1["fitness"] = None
            child2["fitness"] = None

        return child1, child2

    # --------------------------------------------------
    # Mutation
    # --------------------------------------------------

    def mutate(self, chromosome):
        """Mutate binary features and apply Gaussian perturbation to hyperparameter genes."""
        mutated = False
        for i in range(self.num_features):
            if random.random() < self.mutation_rate:
                chromosome["features"][i] ^= 1
                mutated = True

        # Ensure at least 1 feature remains selected
        if sum(chromosome["features"]) == 0 and self.num_features > 0:
            chromosome["features"][random.randint(0, self.num_features - 1)] = 1
            mutated = True

        if random.random() < self.mutation_rate:
            chromosome["alpha_gene"] += float(np.random.normal(0, 0.1))
            chromosome["alpha_gene"] = float(np.clip(chromosome["alpha_gene"], 0.0, 1.0))
            mutated = True

        if random.random() < self.mutation_rate:
            chromosome["l1_gene"] += float(np.random.normal(0, 0.1))
            chromosome["l1_gene"] = float(np.clip(chromosome["l1_gene"], 0.0, 1.0))
            mutated = True

        if mutated:
            chromosome["fitness"] = None

    # --------------------------------------------------
    # Main Evolution Loop
    # --------------------------------------------------

    def evolve(self, progress_callback=None):
        """Run the full genetic algorithm optimization."""
        self.is_running = True
        self.history = []
        self.current_generation = 0

        self.initialize_population()
        self.evaluate_population()

        for generation in range(self.generations):
            self.current_generation = generation + 1

            self.population.sort(key=lambda x: x["fitness"], reverse=True)
            best_in_gen = copy.deepcopy(self.population[0])
            self.history.append(float(best_in_gen["fitness"]))

            if self.best_chromosome is None or best_in_gen["fitness"] > self.best_chromosome["fitness"]:
                self.best_chromosome = copy.deepcopy(best_in_gen)

            if progress_callback:
                progress_callback(
                    self.current_generation,
                    self.generations,
                    best_in_gen["fitness"],
                    self.get_chromosome_info(best_in_gen)
                )

            # Elitism: preserve top N candidates
            new_population = [
                copy.deepcopy(c)
                for c in self.population[: self.elitism_count]
            ]

            # Reproduce to fill population
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                child1, child2 = self.crossover(parent1, parent2)

                self.mutate(child1)
                self.mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population
            self.evaluate_population()

        # Final evaluation and sort
        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        final_best = copy.deepcopy(self.population[0])

        if self.best_chromosome is None or final_best["fitness"] > self.best_chromosome["fitness"]:
            self.best_chromosome = final_best

        self.is_running = False
        return self.get_chromosome_info(self.best_chromosome)