from hashlib import new
from numpy import array, amin, amax
from objective import calculate_weights, SSE
import random

class Individual:
    def __init__(self, dataset: array, n_clusters: int, sse_params: dict):
        self.dataset = dataset
        self.sse_params = sse_params
        
        # Determining bounds for each of the n dimensions 
        lower_bounds = amin(dataset, axis=0)
        upper_bounds = amax(dataset, axis=0)

        self.bounds = list(zip(lower_bounds, upper_bounds))

        # Position: an array of n_clusters centroids; each centroid is an n-dimensional point
        self.position = array([[random.uniform(bound[0], bound[1]) for bound in self.bounds] for _ in range(n_clusters)])
        self.value = self.calculate_value()

    def calculate_value(self):
        weights = calculate_weights(self.dataset, self.position, self.sse_params['p'], self.sse_params['dist'])
        return SSE(self.dataset, self.position, weights, self.sse_params['p'], self.sse_params['dist'])
    
    def __lt__(self, other):
        return self.value < other.value
    
class GeneticAlgorithm():
    def __init__(self, dataset: array, n_clusters: int, sse_params: dict, population_size: int,
                 n_generations: int, mutation_prob: float, tournament_size: int, elitism_size: int):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.sse_params = sse_params
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size

        self.population_size = population_size
        self.n_generations = n_generations
    
    def generate_population(self):
        return [Individual(self.dataset, self.n_clusters, self.sse_params) for _ in range(self.population_size)]
        
    def selection(self):
        best_value = float('inf')
        
        best_index = -1
        for _ in range(self.tournament_size):
            selection_idx = random.randrange(self.population_size)
            if self.population[selection_idx].value < best_value:
                best_index = selection_idx
        
        return best_index

        

    def crossover(self, child1, child2, parent1, parent2):
        # Binary crossover
        breakpoint = random.randrange(len(parent1.position))
        child1.position[:breakpoint] = parent1.position[:breakpoint]
        child1.position[breakpoint:] = parent2.position[breakpoint:]

        child2.position[:breakpoint] = parent2.position[:breakpoint]
        child2.position[breakpoint:] = parent1.position[breakpoint:]

    def mutation(self, individual: Individual):
        for i in range(len(individual.position)):
            if random.random() < self.mutation_prob:
                print("MUTATION")
                j = random.randrange(len(individual.position[i]))
                individual.position[i][j] = random.uniform(individual.position[i][j] - 0.01, individual.position[i][j] - 0.01)



    def fit(self):
        best_individuals = []
        self.population = self.generate_population()
        new_population = self.generate_population()

        for gen in range(self.n_generations):
            print(f'Generation {gen} of {self.n_generations}')
            self.population.sort()
            new_population[:self.elitism_size] = self.population[:self.elitism_size]

            for ind in range(self.elitism_size, self.population_size, 2):
                parent1_index = self.selection()
                parent2_index = self.selection()

                self.crossover(parent1=self.population[parent1_index],
                               parent2=self.population[parent2_index],
                               child1=new_population[ind],
                               child2=new_population[ind+1])
                
                self.mutation(new_population[ind])
                self.mutation(new_population[ind+1])

                new_population[ind].value = new_population[ind].calculate_value()
                new_population[ind+1].value = new_population[ind+1].calculate_value()

            self.population = new_population
            best_individuals.append(min(new_population).value)


        return best_individuals