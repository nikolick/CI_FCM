from math import floor
from regex import F
from sklearn.cluster import KMeans
from objective import SSE, calculate_weights, distance_functions
from PSO import Particle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import data_preparation as dp
import time as time
from json import dumps
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from gap_statistic import OptimalK
from random import sample


SWARM_SIZE = 20
N_ITERS = 200
EPS = 1e0-6
OPTIMAL_K = 15

def elbow_silhouette_graph(data_frame):
    sum_of_squared_distances = []
    K = range(5,20)
    for num_clusters in K :
        kmeans = KMeans(n_clusters=num_clusters, random_state=1)
        kmeans.fit(data_frame)
        sum_of_squared_distances.append(kmeans.inertia_)
    plt.plot(K,sum_of_squared_distances,'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel('Sum of squared distances/Inertia') 
    plt.title('Elbow Method For Optimal k')
    plt.show()

    range_n_clusters = range(5,20)
    silhouette_avg = []
    for num_clusters in range_n_clusters:
    # initialise kmeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=1)
        kmeans.fit(data_frame)
        cluster_labels = kmeans.labels_
        # silhouette score
        silhouette_avg.append(silhouette_score(data_frame, cluster_labels))

    plt.plot(range_n_clusters,silhouette_avg,'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel('Silhouette score') 
    plt.title('Silhouette analysis For Optimal k')
    plt.show()

    optimalK = OptimalK(n_jobs=4, parallel_backend='joblib')
    n_clusters = optimalK(data_frame, cluster_array=np.arange(5,20))
    print(f'Gap statistic: {n_clusters}')

def run_log_crossvalidate(fpath: str, n_iters: int, swarm_size: int,  objective_params: dict, consts: dict, cv_runs: int, X: np.array, eps: float, svd: bool):
    with open(fpath, "a") as log_file:
        log_string = 'TESTING PSO WITH:\n'
        log_string += f'using svd: {svd}\nn_iters: {n_iters}\nPSO constants: {dumps(consts)}\nn_articles: {X.shape[0]}\nswarm size: {swarm_size}' 
        log_string += "\n=========================\n\n"
            
#        print(log_string)

#       TODO: DA SE MENJAJU PARAMETRI
        for cv in range(cv_runs):
            best_values = []
            Particle.global_best_clusters = None
            Particle.global_best_sse = None
            # Run a single full PSO
            swarm = [Particle(X, OPTIMAL_K, objective_params, consts) for _ in range(swarm_size)]
            for it in range(n_iters):
                if it % (floor(n_iters / 10)) == 0:
                    print(f'cv {cv}; iteration {it}')
                for particle in swarm:
                    particle.update_velocity()
                    particle.update_position()
                best_values.append(Particle.global_best_sse)
            log_string += f'{cv+1}. Local optimum: {Particle.global_best_sse}\n'
            plt.plot(range(n_iters), best_values)
            
        plt.show()
        
        print(log_string)
        #log_file.write(log_string)

def prepare_data():
    articles = dp.load_data('./articles/')
    vectorizer = TfidfVectorizer(input='filename', tokenizer = dp.LemmaTokenizer())
    print("Generating TF-IDF...")
    
    X_tfidf = vectorizer.fit_transform(sample(articles, 100))
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
    run_log_crossvalidate("bare_pso_log.txt", N_ITERS, SWARM_SIZE, sse_params, constants, 5, X_trans, EPS, True)
    
    # particles = [Particle(X_trans, 15, sse_params, constants) for _ in range(SWARM_SIZE)]

    # best_vals = []

    # for iter in range(N_ITERS):
    #     if iter % 10 == 0:
    #         print("Iteration " + str(iter) + "...")
    #     for particle in particles:
    #         particle.update_velocity()
    #         particle.update_position()

    #     best_vals.append(Particle.global_best_sse)
    #     if len(best_vals) > 2 and best_vals[-1] - best_vals[-2] < EPS:
    #         break
    


    # print("Local optimum " + str(best_vals[-1]) + " reached in " + str(iter) + " iterations.")

    # plt.plot(range(N_ITERS), best_vals)
    # plt.show()
    
