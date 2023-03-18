import random
import numpy as np
from typing import Sequence, Callable, Any

number = int | float
function_or_number = int | float | Callable[[number], number]
sequence_of_tuples = Sequence[tuple[number, number]]
numeric_array = np.ndarray[number, Any]


class Particle():
    def __init__(self, bounds: sequence_of_tuples) -> None:
        r"""Instantiate a single particle with a random initial position and initial velocity of 0.

        Args:
            bounds (Sequence[tuple[int or float, int or float]): The limits of the search space for each dimension, the sequence should be of the form $[({x_1}_{min}, {x_1}_{max}), ({x_2}_{min}, {x_2}_{max}), \cdots]$, each tuple representing the min and max value for each dimension.
        """
        self.position = np.array([random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.array([0.0 for _ in range(len(bounds))])
        self.personal_best_position = self.position.copy()
        self.fitness = float('inf')
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position: numeric_array, inertia: function_or_number) -> None:
        r"""Update the velocity accordingly to the following formula:
            $$v_{i+1} = \omega v_{i} + 2rand()({p_{best}}_{i} - p_{i}) + 2rand()({g_{best}}_{i} - p_{i})$$
            * $v_{i+1}$: the new velocity.
            * $v_{i}$: the current velocity.
            * $\omega$: the inertia weight.
            * ${p_{best}}_{i}$: the personal best position of the particle.
            * $p_{i}$: the current position of the particle.
            * ${g_{best}}_{i}$: the global best position of the swarm.
            * $rand()$: a random number between 0 and 1.

        Args:
            global_best_position (np.ndarray[int or float]): An array containing the global best position of the swarm for each dimension.
            inertia (int or float or Callable): The inertia weight, if a callable is passed, it should take the current iteration as an argument and return the inertia weight.
        """
        num_dimensions = len(self.position)
        for curr_dim in range(num_dimensions):
            p, g = random.uniform(0, 1), random.uniform(0, 1)
            self.velocity[curr_dim] = (
                                       inertia * (self.velocity[curr_dim])
                                       + 2 * p * (self.personal_best_position[curr_dim] - self.position[curr_dim])
                                       + 2 * g * (global_best_position[curr_dim] - self.position[curr_dim])
                                      )

    def update_position(self, bounds: sequence_of_tuples) -> None:
        r"""Update the position of the particle by adding the velocity to the current position.
        It also makes sure that the new position is within the bounds of the search space.

        Args:
            bounds (Sequence[tuple[int or float, int or float]): The limits of the search space for each dimension, the sequence should be of the form $[({x_1}_{min}, {x_1}_{max}), ({x_2}_{min}, {x_2}_{max}), \cdots]$, each tuple representing the min of max value for each dimension.
        """
        num_dimensions = len(self.position)
        for curr_dim in range(num_dimensions):
            self.position[curr_dim] += self.velocity[curr_dim]
            self.position[curr_dim] = max(self.position[curr_dim], bounds[curr_dim][0])
            self.position[curr_dim] = min(self.position[curr_dim], bounds[curr_dim][1])

    def evaluate_fitness(self, function: Callable[[numeric_array], number]) -> None:
        r"""Evaluate the fitness of the particle by passing its position to the objective function
        and updating the personal best position and fitness if the new fitness is better than the current one.

        Args:
            function (Callable[np.ndarray[int or float]], int or float]): The objective function. It should take an array containing the position of the particle for each dimension and return a single real number representing the fitness of the particle.
        """
        self.fitness = function(self.position)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.personal_best_position = self.position


class ParticleSwarmOptimization():
    def __init__(self, function: Callable[[numeric_array], number], inertia_weight: function_or_number,
                 bounds: sequence_of_tuples, num_particles: int, max_iter: int) -> None:
        r"""Instantiate a particle swarm optimization algorithm.

        Args:
            function (Callable[np.ndarray[int or float]], int or float]): The objective function. It should take an array containing the position of the particle for each dimension and return a single real number representing the fitness of the particle.
            inertia_weight (int or float or Callable): The inertia weight, if a callable is passed, it should take the current iteration as an argument and return the inertia weight.
            bounds (Sequence[tuple[int or float, int or float]): The limits of the search space for each dimension, the sequence should be of the form $[({x_1}_{min}, {x_1}_{max}), ({x_2}_{min}, {x_2}_{max}), \cdots]$, each tuple representing the min of max value for each dimension.
            num_particles (int): The number of particles in the swarm.
            max_iter (int): The maximum number of iterations.
        """
        self.function = function
        self.inertia = inertia_weight
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.swarm = [Particle(bounds) for __ in range(num_particles)]
        self.global_best_position = self.swarm[0].position
        self.global_best_fitness = float('inf')

    def optimize(self) -> None:
        """Run the optimization algorithm for the specified number of iterations. For each iteration, it updates the velocity and position of each particle and updates the global best position and fitness if the new fitness is better than the current one.
        """
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
        """Return the global best position of the swarm.

        Returns:
            np.ndarray[int or float]: An array containing the global best position of the swarm for each dimension.
        """
        return self.global_best_position

    @property
    def get_best_fitness(self) -> number:
        """_Return the global best fitness of the swarm. If the algorithm doesn't get stuck in a local minimum, this should be the global minimum of the objective function.

        Returns:
            int or float: The global best fitness of the swarm
        """
        return self.global_best_fitness
