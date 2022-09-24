from math import floor
import numpy as np
import random

from objective import calculate_weights, SSE
class Particle:
    global_best_clusters = None
    global_best_sse = None

    def __init__(self, dataset: np.array, n_clusters: int, sse_params: dict, constants: dict) -> None:
        # Position: an array of n_clusters centroids; each centroid is an n-dimensional point
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.sse_params = sse_params
        self.constants = constants

        # Determining bounds for each of the n dimensions 
        self.lower_bounds = np.amin(dataset, axis=0)
        self.upper_bounds = np.amax(dataset, axis=0)

        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))

        self.position = np.array([[random.uniform(bound[0], bound[1]) for bound in self.bounds] for _ in range(self.n_clusters)])
        self.current_value = self.calculate_value(self.position)

        self.velocity = np.array([[random.uniform(-(bound[1] - bound[0]), bound[1] - bound[0]) for bound in self.bounds] for _ in range(self.n_clusters)])

        self.best_position = self.position # Best local solution
        self.best_value = self.current_value # Value of best local solution

        if Particle.global_best_sse is None or self.best_value < Particle.global_best_sse:
            Particle.global_best_clusters = self.best_position
            Particle.global_best_sse = self.best_value
    
    
    def calculate_value(self, position):
        weights = calculate_weights(self.dataset, position, self.sse_params['p'], self.sse_params['dist'])
        return SSE(self.dataset, position, weights, self.sse_params['p'], self.sse_params['dist'])

    def update_position(self):
        self.position = self.position + self.velocity

        self.current_value = self.calculate_value(self.position)
        if self.current_value < self.best_value:
            self.best_value = self.current_value
            self.best_position = self.position.copy()
            if self.current_value < Particle.global_best_sse:
                Particle.global_best_clusters = self.position.copy()
                Particle.global_best_sse = self.current_value

    
    def update_velocity(self):
        random_local = random.random()
        random_global = random.random()

        inertia = self.constants['inertia']
        cognitive = self.constants['cognitive']
        social = self.constants['social']

        self.velocity = (inertia * self.velocity +
                        cognitive *  (self.best_position - self.position) * random_local +
                        social * (Particle.global_best_clusters - self.position) * random_global)

class PSO:
    def __init__(self, dataset, swarm_size, n_clusters, n_iters, sse_params, constants, eps=None):
        self.n_iters = n_iters
        # if eps is none, we will just perform all the iterations
        self.eps = eps
        self.dataset = dataset
        self.swarm_size = swarm_size
        self.n_clusters = n_clusters
        self.sse_params = sse_params
        self.constants = constants

    def init_swarm(self):
        self.swarm = [Particle(self.dataset, self.n_clusters, self.sse_params, self.constants) for _ in range(self.swarm_size)]

        
    def fit(self, verbose = False):
    #    self.swarm = [Particle(self.dataset, self.n_clusters, self.sse_params, self.constants) for _ in range(self.swarm_size)]
        best_values = []
        for it_ in range(self.n_iters):
            if verbose and it_ % (floor(self.n_iters / 10)) == 0:
                print(f'Iteration {it_} of {self.n_iters}')
#            self.constants['inertia'] = (0.5)*(self.n_iters - it_)/(self.n_iters) + 0
            for particle in self.swarm:
                particle.update_velocity()
                particle.update_position()
            best_values.append(Particle.global_best_sse)
            

        return best_values

        
            

    




