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
    




