from particle_swarm_optimization import ParticleSwarmOptimization, fit_function, create_inertia_weight_function

BOUNDS = [(-5, 5), (-5, 5)]
NUM_PARTICLES = 30
MAX_ITER = 20
START_WEIGHT = 1
END_WEIGHT = 0.1
ALPHA = 0.001


def main():
    inertia = create_inertia_weight_function(start_weight=START_WEIGHT, end_weight=END_WEIGHT, alpha=ALPHA)
    pso = ParticleSwarmOptimization(function=fit_function, inertia_weight=inertia, bounds=BOUNDS, num_particles=NUM_PARTICLES, max_iter=MAX_ITER)
    pso.optimize()
    best_position = pso.global_best_position
    print(f'{best_position=}')


if __name__ == '__main__':
    main()
