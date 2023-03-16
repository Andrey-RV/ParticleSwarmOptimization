import random
import numpy as np
from typing import Sequence, Callable, Any

number = int | float
function_or_number = int | float | Callable[[number], number]
sequence_of_tuples = Sequence[tuple[number, ...]]
numeric_array = np.ndarray[number, Any]


class Particle():
    def __init__(self, bounds: sequence_of_tuples) -> None:
        self.position = np.array([random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.array([0.0 for _ in range(len(bounds))])
        self.personal_best_position = self.position.copy()
        self.fitness = float('inf')
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position: numeric_array, inertia: function_or_number) -> None:
        num_dimensions = len(self.position)
        for curr_dim in range(num_dimensions):
            p, g = random.uniform(0, 1), random.uniform(0, 1)
            self.velocity[curr_dim] = inertia * (self.velocity[curr_dim]
                                                 + 2 * p * (self.personal_best_position[curr_dim] - self.position[curr_dim])
                                                 + 2 * g * (global_best_position[curr_dim] - self.position[curr_dim]))

    def update_position(self, bounds: sequence_of_tuples) -> None:
        num_dimensions = len(self.position)
        for curr_dim in range(num_dimensions):
            self.position[curr_dim] += self.velocity[curr_dim]
            self.position[curr_dim] = max(self.position[curr_dim], bounds[curr_dim][0])
            self.position[curr_dim] = min(self.position[curr_dim], bounds[curr_dim][1])

    def evaluate_fitness(self, function: Callable[[numeric_array], number]) -> None:
        self.fitness = function(self.position)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.personal_best_position = self.position


class ParticleSwarmOptimization():
    def __init__(self, function: Callable[[numeric_array], number], inertia_weight: function_or_number,
                 bounds: sequence_of_tuples, num_particles: int, max_iter: int) -> None:
        self.function = function
        self.inertia = inertia_weight
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.swarm = [Particle(bounds) for __ in range(num_particles)]
        self.global_best_position = self.swarm[0].position
        self.global_best_fitness = float('inf')

    def optimize(self) -> None:
        for iteration in range(self.max_iter):
            for particle in self.swarm:
                particle.evaluate_fitness(self.function)
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position
            inertia = self.inertia(iteration) if callable(self.inertia) else self.inertia

            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, inertia)
                particle.update_position(self.bounds)

    @property
    def get_best_position(self) -> numeric_array:
        return self.global_best_position

    @property
    def get_best_fitness(self) -> number:
        return self.global_best_fitness
