from PSO import PSO
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
 consts: dict, cv_runs: int, X: np.array, eps: float, adj_params = False, plot = False):

    with open(fpath, "a") as log_file:
        log_string = 'TESTING PSO WITH:\n'
        log_string += f'\nn_iters: {n_iters}\nPSO constants: {dumps(consts)}\nSSE parameters: {dumps(objective_params)}\nn_articles: {X.shape[0]}\nswarm size: {swarm_size}' 
        log_string += "\n=========================\n\n"

#       TODO: DA SE MENJAJU PARAMETRI
        pso = PSO(X, swarm_size, OPTIMAL_K, n_iters, objective_params, consts, eps)
        for cv in range(cv_runs):
            best_results = pso.fit(verbose=True)
            log_string += f'{cv+1}. PSO best result: {best_results[-1]}\n'
            if plot:
                plt.plot(range(n_iters), best_results)

        if plot:
            plt.show()
        #print(log_string)
        log_file.write(log_string)

def prepare_data():
    articles = dp.load_data('./articles/')
    vectorizer = TfidfVectorizer(input='filename', tokenizer = dp.LemmaTokenizer())
    print("Generating TF-IDF...")
    
    X_tfidf = vectorizer.fit_transform(sample(articles, 200))
    X_trans = dp.reduce_dimensions(X_tfidf)
    return X_tfidf, X_trans


if __name__ == "__main__":
    sse_params = {
        'p': 2,
        'dist': distance_functions[0]
        }
    constants = {
        'inertia' : 0.5,
        'cognitive': 0.5,
        'social': 0.5
    }
    X_tfidf, X_trans = prepare_data()

    #dataset = np.array([[0.1, 3.2, 4.5, 2.3], [0.444, 1.76, 0.87, 6.34], [3.773, 2.789, 4.64, 0.0123], [12.44, 1.24, 2.4, 0.001], [0.4323, 1.112, 4.11, 5.12]])

    ga = GeneticAlgorithm(X_trans, n_clusters = 15, sse_params = sse_params, population_size = 50, n_generations = 50, 
                            mutation_prob = 0.05, tournament_size = 5, elitism_size = 10)

    vals = ga.fit()

    print(vals)
    plt.plot(range(50), vals)

    plt.show()
#    print(pop)
#    print(len(pop))
#    run_log("bare_pso_log.txt", N_ITERS, SWARM_SIZE, sse_params, constants, 5, X_trans, EPS, True, plot = True)

    
