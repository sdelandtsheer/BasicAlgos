### Composing photo mosaics with genetic algorithm

import numpy as np
import pandas as pd
from random import choices, random
from heapq import nlargest
from statistics import mean

class Chromosome:
    def __init__(self, , , ):
        pass

    def __repr__(self):
        pass

    def fitness(self):
        pass

    def mutate(self):
        pass

    def random(self):
        pass

    def crossover(self):
        pass


class GeneticAlgorithm:
    def __init__(self, n_indiv=None,
                 problem_structure=None,
                 pop=None,
                 threshold=None,
                 max_gen=None,
                 mut_rate=None,
                 crossover_rate=None,
                 mode=None):
        if n_indiv is not None:
            self.n_indiv = n_indiv
        else:
            self.n_indiv = 1000
        self.problem_structure = problem_structure
        if pop is not None:
            self.pop = pop
        else:
            self.pop = self.initialize_pop()
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = 0.9
        if max_gen is not None:
            self.max_gen = max_gen
        else:
            self.max_gen = 1000
        if mut_rate is not None:
            self.mut_rate = mut_rate
        else:
            self.mut_rate = 0.01
        if crossover_rate is not None:
            self.crossover_rate = crossover_rate
        else:
            self.crossover_rate = 0.01
        if mode is not None:
            self.mode = mode
        else:
            self.mode = 'roulette'
        self.fitness_key = type(self.pop[0]).fitness

    def initialize_pop(self):
        """
        Initializes the population with random chromosomes given the structure of the problem
        """
        if self.problem_structure is not None:
            pass
        else:
            raise ValueError("The problem structure is not specified")

    def select(self):
        """
        Selects two parents according to a certain specified method
        """
        if self.mode == 'roulette':
            return self.select_roulette()

        elif self.mode == 'tournament':
            return self.select_tournament()

    def select_roulette(self):
        """
        Selects individuals from the population with probability proportional to
        their share of the total fitness of the population
        """
        fitnesses = [candidate.fitness() for candidate in self.pop]
        return tuple(choices(self.pop, weights=fitnesses, k=2))


    def select_tournament(self):
        """
        Selects individuals from the population with equal chances for all at first,
        then the fittest 2 outcompete the losers
        """
        k = min(2, np.sqrt(self.n_indiv))
        candidates = choices(self.pop, k=k)
        return tuple(nlargest(2, candidates, key=self.fitness_key))


    def reproduce(self):
        """
        Produces new individuals from sets of parents and generate a new population
        """
        new_pop = []
        while len(new_pop) < self.n_indiv:
            parents = self.select()
            if random() < self.crossover_rate:
                new_pop.append(parents[0].crossover(parents[1]))
            else:
                new_pop.append(parents)
        if len(new_pop) > self.n_indiv:
            new_pop.pop()
        self.pop = new_pop

    def mutate(self):
        """
        Mutates the complete population
        """
        for individual in self.pop:
            if random() < self.mut_rate:
                individual.mutate()

    def run(self):
        """
        Runs the simulation and returns the best individual found at each generation and its fitness
        """
        results = pd.DataFrame(columns=["best", "fitness"])
        best_so_far = max(self.pop, key=self.fitness_key)
        for gen in range(self.max_gen):
            if best_so_far.fitness() >= self.threshold:
                print(f"Threshold passed at generation {gen}: an individual has been found with fitness {best_so_far.fitness()}")
                return results
            print(f"Generation {gen}: best fitness = {best_so_far.fitness()}, population fitness = {mean(map(self.fitness_key, self.pop))}")
            self.reproduce()
            self.mutate()







