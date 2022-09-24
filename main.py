from PSO import PSO, Particle
from GA import GeneticAlgorithm
from objective import distance_functions
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import data_preparation as dp
import time as time
from json import dumps
from matplotlib import pyplot as plt
from random import sample

SWARM_SIZE = 20
N_ITERS = 200
EPS = 1e0-6
OPTIMAL_K = 15

def run_log(fpath: str, n_iters: int, swarm_size: int,  objective_params: dict,
 consts: dict, cv_runs: int, X: np.array, eps: float, adj_param = None, plot = False, init_pop = None):

    with open(fpath, "a") as log_file:
        log_string = 'TESTING PSO WITH:\n'
        log_string += f'\nn_iters: {n_iters}\nPSO constants: {dumps(consts)}\nSSE parameters: {dumps(objective_params)}\nn_articles: {X.shape[0]}\nswarm size: {swarm_size}' 
        log_string += "\n=========================\n\n"

        pso = PSO(X, swarm_size, OPTIMAL_K, n_iters, objective_params, consts, eps)
        if init_pop is not None:
            for i in range(pso.swarm_size):
                pso.swarm[i].position = init_pop[i]
        for cv in range(cv_runs):
            start = time.time()
            best_results = pso.fit(verbose=True)
            end = time.time()
            log_string += f'{cv+1}. PSO best result: {best_results[-1]}, elapsed time: {end-start}\n'

            if plot:
                plt.plot(range(n_iters), best_results, label=f'iteration {cv}')

        if plot:
            plt.legend(loc='upper left')
            plt.show()
        #print(log_string)
        log_file.write(log_string)

def prepare_data():
    articles = dp.load_data('./articles/')
    vectorizer = TfidfVectorizer(input='filename', tokenizer = dp.LemmaTokenizer())
    print("Generating TF-IDF...")
    
    X_tfidf = vectorizer.fit_transform(articles)
    X_trans = dp.reduce_dimensions(X_tfidf)
    return X_tfidf, X_trans


if __name__ == "__main__":
    sse_params = {
        'p': 2,
        'dist': distance_functions[0]
        }
    constants = {
        'inertia' : 0.5,
        'cognitive': 0.7,
        'social': 0.6
    }
    X_tfidf, X_trans = prepare_data()
    ga = GeneticAlgorithm(X_trans, n_clusters = 15, sse_params = sse_params, population_size = 50, n_generations = 50, 
                            mutation_prob = 0.05, tournament_size = 5, elitism_size = 10)
    pso = PSO(X_trans, 50, 15, 50, sse_params, constants, 0.005)
#    run_log("bare_pso_log", 200, 50, sse_params, constants, 5, X_trans, 0.005, None, True)
    for _ in range(5):
        individuals, best_vals = ga.fit()
        init_vals = [individual.position for individual in individuals]

        pso.init_swarm()
        for i in range(pso.swarm_size):
            pso.swarm[i].position = init_vals[i]
            pso.swarm[i].current_value = pso.swarm[i].calculate_value(pso.swarm[i].position)
            pso.swarm[i].best_value = pso.swarm[i].current_value

            
            if Particle.global_best_sse is None or pso.swarm[i].best_value < Particle.global_best_sse:
                Particle.global_best_clusters = pso.swarm[i].best_position
                Particle.global_best_sse = pso.swarm[i].best_value

        pso_vals =  pso.fit(True)

        plt.plot(range(100), best_vals+pso_vals)
    plt.show()

        

        

