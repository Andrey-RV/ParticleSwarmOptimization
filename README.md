# Particle Swarm Optimization algorithm 

This project is a simple implementation of the Particle Swarm Optimization algorithm, which is a heuristic optimization algorithm that is inspired by the social behavior of birds flocking or fish schooling. It is a population-based algorithm that optimizes a problem by having a population of candidate solutions, here called particles, and moving these particles around in the search-space according to simple mathematical formulae over the particle's position and velocity. Each particle's movement is influenced by its local best known position, but is also guided toward the best known positions in the search-space, which are updated as better positions are found by other particles. This is expected to move the swarm toward the best solutions.

The algorithm implemented here works as follows:  
 1. Initializes a swarm of particles with random initial positions and initial velocities of zero.
 2. Calculates the fitness of each particle by evaluating the objective function at the particle's position.
 3. If the particle's fitness is better than its best fitness, then the particle's best fitness is updated to the particle's current fitness, and its best position is updated to its current position.
 4. If the particle's best fitness is better than the best global fitness found so far, then the best global fitness found so far is updated to the particle's best fitness, and the best global position found so far is updated to the particle's best position.
 5. Updates the particle's velocity and position according to the following formulae:
    -   $\displaystyle v_{i+1} = \omega v_{i} + 2rand()({p_{best}}_{i} - p_{i}) + 2rand()({g_{best}}_{i} - p_{i})$
    -   $\displaystyle p_{i+1} = p_{i} + v_{i+1}$
 6. Repeat steps 2-5 until hitting the maximum given number of iterations.

## Requirements
 -   Python 3.10 to avoid errors about the use of the new type union operators | (PEP 604).
 -   Numpy;

You can run the following command to install the required packages:
```
pip install -r requirements.txt
```

## Usage

To start, download or clone the repository, and navigate to the project directory. Create a new  .py file, and import the ParticleSwarmOptimization class from the pso module.

```python
from pso import ParticleSwarmOptimization
```

To instantiate a ParticleSwarmOptimization object, you must pass the following arguments:
 1.   **An objective_function**: it must be a function that takes a numeric array as argument (representing the values for each possible variable), and returns an int or float. For example, for the function $f(x,y) = x^2 -4x + y^2 -4y + 4 + \sin(xy)$, the objective function would be:
```python
def fit_function(x: Sequence[int | float]) -> int | float:
    return x[0] ** 2 - 4 * x[0] + x[1] ** 2 - 4 * x[1] + 4 + np.sin(x[0] * x[1])
```
 2.   **The inertia weight**: it can be either a constant value, or a function that takes the current iteration as argument and returns a float. For this example, the inertia weight used is a decreasing exponential function. A closured is used to change the parameters of the function without having to change the function itself.

```python
def create_inertia_weight_function(start_weight: float, end_weight: float, alpha: float) -> Callable[[float], float]:
    def inertia_weight(t: float) -> float:
        return (start_weight - end_weight) * np.exp(-alpha * t) - start_weight
    return inertia_weight


inertia = create_inertia_weight_function(1, 0.1, 0.001)
```
 3.   **Bounds**: a sequence of tuples, where each tuple represents the lower and upper bounds for each variable. For this example, the bounds chosen for the given function are $x\in[-5, 5]$ and $y\in[-5, 5]$.
```python
BOUNDS = [(-5, 5), (-5, 5)]
```
 4.   **The number of particles**: an integer representing the number of particles in the swarm.
```python
NUM_PARTICLES = 20
```
 5.   **The number of iterations**: an integer representing the maximum number of iterations.
```python
MAX_ITER = 30
```

Those parameters are unique for each problem, and must be chosen according to the problem.

With all the parameters set, you can now instantiate the ParticleSwarmOptimization object and call the optimize() method. The best position found by the algorithm will be stored in the global_best_position attribute, and the best fitness found will be stored in the global_best_fitness attribute.

```python
pso = ParticleSwarmOptimization(function=fit_function, inertia_weight=inertia, bounds=BOUNDS, num_particles=NUM_PARTICLES, max_iter=MAX_ITER)

pso.optimize()

pso.global_best_position
#array([2.14168296, 2.14209593])

pso.global_best_fitness
#-4.951997506043826
```

For this example, the global minimum for the function is $(2, 2)$, with $f(2, 2) = -4.76$. The algorithm found a somewhat close solution, and didn't get stuck in one of the two local minimum points that the function has.

## Documentation
The documentation about the implementation can be found [here](https://andrey-rv.github.io/ParticleSwarmOptimization/)
