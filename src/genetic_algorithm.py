"""
Genetic Algorithm for License Plate Localization
==================================================
Mirrors Section 3.3 / 4.3 / 4.4 of the project report.

Chromosome representation:
    chromosome = [x, y, w, h]   (top-left corner + width/height of a
                                  candidate bounding box)

Fitness function:
    F = De * Sa * (1 + 0.7 * Vi)

    De  -> Edge density            = sum(edge pixels) / (w * h)
    Sa  -> Aspect ratio score      = exp(-((w/h - 4)^2) / (2 * sigma^2))
    Vi  -> Intensity variance      = var(region) / 255

GA operators: tournament-style top-K selection, one-point crossover,
random mutation, elitism, run until convergence / max generations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


Chromosome = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class GAConfig:
    population_size: int = 40
    generations: int = 60
    elite_count: int = 5
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    min_w: int = 60
    max_w: int = 300
    min_h: int = 20
    max_h: int = 100
    aspect_ratio_target: float = 4.0
    aspect_ratio_sigma: float = 0.5
    convergence_patience: int = 15
    convergence_eps: float = 1e-4
    random_seed: int | None = 42


@dataclass
class GAResult:
    best_box: Chromosome
    best_fitness: float
    fitness_history: List[float] = field(default_factory=list)
    generations_run: int = 0


class GeneticPlateLocator:
    """
    Genetic-Algorithm-based search for the bounding box most likely to
    contain a license plate, using edge density, aspect ratio and
    intensity variance as the fitness criteria described in the report.
    """

    def __init__(self, image_shape: Tuple[int, int], config: GAConfig | None = None):
        self.height, self.width = image_shape[:2]
        self.cfg = config or GAConfig()
        if self.cfg.random_seed is not None:
            random.seed(self.cfg.random_seed)
            np.random.seed(self.cfg.random_seed)

    # ------------------------------------------------------------------ #
    # Population handling
    # ------------------------------------------------------------------ #
    def _random_chromosome(self) -> Chromosome:
        w = random.randint(self.cfg.min_w, min(self.cfg.max_w, self.width - 1))
        h = random.randint(self.cfg.min_h, min(self.cfg.max_h, self.height - 1))
        x = random.randint(0, max(0, self.width - w - 1))
        y = random.randint(0, max(0, self.height - h - 1))
        return (x, y, w, h)

    def _init_population(self) -> List[Chromosome]:
        return [self._random_chromosome() for _ in range(self.cfg.population_size)]

    def _clip(self, chromosome: Chromosome) -> Chromosome:
        x, y, w, h = chromosome
        w = int(np.clip(w, self.cfg.min_w, self.cfg.max_w))
        h = int(np.clip(h, self.cfg.min_h, self.cfg.max_h))
        x = int(np.clip(x, 0, max(0, self.width - w - 1)))
        y = int(np.clip(y, 0, max(0, self.height - h - 1)))
        return (x, y, w, h)

    # ------------------------------------------------------------------ #
    # Fitness function  F = De * Sa * (1 + 0.7 * Vi)
    # ------------------------------------------------------------------ #
    def edge_density(self, edges: np.ndarray, chromosome: Chromosome) -> float:
        x, y, w, h = chromosome
        region = edges[y:y + h, x:x + w]
        if region.size == 0:
            return 0.0
        return float(np.count_nonzero(region)) / float(w * h)

    def aspect_ratio_score(self, chromosome: Chromosome) -> float:
        _, _, w, h = chromosome
        ratio = w / max(h, 1)
        sigma = self.cfg.aspect_ratio_sigma
        target = self.cfg.aspect_ratio_target
        return float(np.exp(-((ratio - target) ** 2) / (2 * sigma ** 2)))

    def intensity_variance(self, gray: np.ndarray, chromosome: Chromosome) -> float:
        x, y, w, h = chromosome
        region = gray[y:y + h, x:x + w]
        if region.size == 0:
            return 0.0
        return float(np.var(region)) / 255.0

    def fitness(self, gray: np.ndarray, edges: np.ndarray, chromosome: Chromosome) -> float:
        de = self.edge_density(edges, chromosome)
        sa = self.aspect_ratio_score(chromosome)
        vi = self.intensity_variance(gray, chromosome)
        return de * sa * (1 + 0.7 * vi)

    # ------------------------------------------------------------------ #
    # Genetic operators
    # ------------------------------------------------------------------ #
    def _select(self, population: List[Chromosome], scores: List[float]) -> List[Chromosome]:
        ranked = [c for _, c in sorted(zip(scores, population), key=lambda p: p[0], reverse=True)]
        return ranked[: max(self.cfg.elite_count, self.cfg.population_size // 2)]

    def _crossover(self, parent_a: Chromosome, parent_b: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if random.random() > self.cfg.crossover_rate:
            return parent_a, parent_b
        point = random.randint(1, 3)  # one-point crossover across [x, y, w, h]
        child_a = parent_a[:point] + parent_b[point:]
        child_b = parent_b[:point] + parent_a[point:]
        return self._clip(child_a), self._clip(child_b)

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        if random.random() > self.cfg.mutation_rate:
            return chromosome
        x, y, w, h = chromosome
        gene = random.choice(["x", "y", "w", "h"])
        jitter = random.randint(-15, 15)
        if gene == "x":
            x += jitter
        elif gene == "y":
            y += jitter
        elif gene == "w":
            w += jitter
        else:
            h += jitter
        return self._clip((x, y, w, h))

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, gray: np.ndarray, edges: np.ndarray) -> GAResult:
        population = self._init_population()
        history: List[float] = []
        best_box: Chromosome = population[0]
        best_fitness = -np.inf
        stagnant_generations = 0

        for gen in range(self.cfg.generations):
            scores = [self.fitness(gray, edges, c) for c in population]
            gen_best_idx = int(np.argmax(scores))
            gen_best_fitness = scores[gen_best_idx]

            if gen_best_fitness > best_fitness + self.cfg.convergence_eps:
                best_fitness = gen_best_fitness
                best_box = population[gen_best_idx]
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            history.append(best_fitness)

            # Elitism: carry the fittest individuals forward untouched
            survivors = self._select(population, scores)
            next_population = survivors[: self.cfg.elite_count]

            while len(next_population) < self.cfg.population_size:
                parent_a, parent_b = random.sample(survivors, 2)
                child_a, child_b = self._crossover(parent_a, parent_b)
                next_population.append(self._mutate(child_a))
                if len(next_population) < self.cfg.population_size:
                    next_population.append(self._mutate(child_b))

            population = next_population

            if stagnant_generations >= self.cfg.convergence_patience:
                break

        return GAResult(
            best_box=best_box,
            best_fitness=best_fitness,
            fitness_history=history,
            generations_run=len(history),
        )
