import logging
from typing import Callable, List

import numpy as np

from utils.utils import formatter_single

# Logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=formatter_single.FORMATTER)

class GeneticAlgorithm:
    def __init__(
        self,
        fitness_function: Callable,
        num_generations: int = 40,
        population_size: int = 100,
        num_features: int = 2,
        num_bits_per_feature: int = 22,
        min_value: float = -100,
        max_value: float = 100,
        crossover_rate: float = 0.65,
        mutation_rate: float = 0.008,
        round_decimals: int = 6
    ):

        self.num_generations = num_generations
        self.population_size = population_size
        self.num_features = num_features
        self.num_bits_per_feature = num_bits_per_feature
        self.min_value = min_value
        self.max_value = max_value
        self.fitness_function = fitness_function
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.round_decimals = round_decimals


    def init_individuals(self) -> None:
        """
        Creates an initial population based on a discrete uniform distribution
        considering all the range of possible integer values of a bit representation,
        before converting to floating-point.
        """
        # Sample integers from a discrete uniform distribution
        self.individuals = np.random.randint(0, 2**self.num_bits_per_feature, size=self.population_size*self.num_features)
        # Convert integers into their binary representations
        self.individuals = np.vectorize(np.binary_repr)(self.individuals, width=self.num_bits_per_feature)
        # Reshape into 'x' and 'y'
        self.individuals = self.individuals.reshape(self.population_size, self.num_features)


    def convert_bit2real(self) -> List[float]:
        """
        Converts a bit string in its real representation
        """
        bit2real = lambda bits: int(bits, 2)*((self.max_value - self.min_value)/(2**self.num_bits_per_feature - 1)) + self.min_value
        real_individuals = np.array(list(map(bit2real, self.individuals.reshape((self.population_size*self.num_features))))).reshape((self.population_size, self.num_features))

        return real_individuals


    def calculate_fitness(self) -> List[float]:
        """
        Calculate fitness based on a specific function
        """
        real_individuals = self.convert_bit2real()
        fitness = np.array(list(map(self.fitness_function, real_individuals)))

        return fitness


    def crossover(self, individual1: List[str], individual2: List[str]) -> List[str]:
        """
        Performs the crossover operation onto 2 individuals
        """
        individual1 = ''.join(individual1)
        individual2 = ''.join(individual2)
        sep = np.random.randint(0, self.num_bits_per_feature-1)

        offspring = str(individual1)[:sep]+str(individual2)[sep:]

        return np.array([offspring[self.num_bits_per_feature:], offspring[:self.num_bits_per_feature]])


    def mutation(self, individual: List[str], mutation_rate: float) -> List[str]:
        """
        Invert a bit based on a certain probability
        """
        # Join x and y into one single representation
        individual = ''.join(individual)
        probs = np.random.uniform(0, 1, len(individual))

        mutated_individual = []

        for bit, prob in zip(individual, probs):
            if prob < mutation_rate:
                if bit == '1':
                    mutated_individual.append('0')
                else:
                    mutated_individual.append('1')
            else:
                mutated_individual.append(bit)

        mutated_individual = ''.join(mutated_individual)

        return np.array([mutated_individual[self.num_bits_per_feature:], mutated_individual[:self.num_bits_per_feature]])


    def roulette_wheel_selection(self) -> int:
        """
        Selects an individual based on the roulette wheel strategy
        """
        fitness = self.calculate_fitness()
        fitness_cumulative_sum = np.cumsum(fitness)
        rand_num = np.random.uniform(0, np.max(fitness_cumulative_sum))

        selected_individual = np.argmax(fitness_cumulative_sum >= rand_num)

        return selected_individual


    def iterate(self) -> None:

        self.init_individuals()
        fitness = self.calculate_fitness()

        for gen in range(self.num_generations):
            idx_best_fitness = np.argmax(fitness)

            # Gets best individual in its real representation (x, y)
            bit2real = lambda bits: int(bits, 2)*((self.max_value - self.min_value)/(2**self.num_bits_per_feature - 1)) + self.min_value
            best_x = bit2real(self.individuals[idx_best_fitness][0])
            best_y = bit2real(self.individuals[idx_best_fitness][1])

            log.info(f"Generation: {gen+1}/{self.num_generations} | Avg Fitness: {np.round(np.mean(fitness), self.round_decimals)} | Best Fitness: {np.round(np.max(fitness), self.round_decimals)} | Best individual: ({np.round(best_x, self.round_decimals)}, {np.round(best_y, self.round_decimals)})")

            selected_individuals = []
            offsprings = []
            mutated_offsprings = []

            # Selection
            for _ in range(self.population_size):
                selected_individuals.append(self.roulette_wheel_selection())

            # Crossover
            mates = np.random.choice(selected_individuals, self.population_size, replace=False)
            for selected_individual, mate in zip(selected_individuals, mates):
                if np.random.uniform(0, 1) < self.crossover_rate:
                    offsprings.append(self.crossover(self.individuals[selected_individual], self.individuals[mate]))
                else:
                    offsprings.append(self.individuals[selected_individual])

            # Mutation
            for offspring in offsprings:
                mutated_offsprings.append(self.mutation(offspring, self.mutation_rate))

            # N-1 offsprings + parent with the best fitness
            rand_index_offsprings = np.random.randint(0, self.population_size)
            del mutated_offsprings[rand_index_offsprings]
            mutated_offsprings.append(self.individuals[idx_best_fitness])

            self.individuals = np.array(mutated_offsprings)
            fitness = self.calculate_fitness()