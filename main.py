from regex import F
from sklearn.cluster import KMeans
from objective import SSE, calculate_weights, distance_functions
from PSO import Particle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import data_preparation as dp
import time as time
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from gap_statistic import OptimalK
from random import sample

SWARM_SIZE = 20
N_ITERS = 1000

def elbow_silhouette_graph(data_frame):
    Sum_of_squared_distances = []
    K = range(5,20)
    for num_clusters in K :
        kmeans = KMeans(n_clusters=num_clusters, random_state=1)
        kmeans.fit(data_frame)
        Sum_of_squared_distances.append(kmeans.inertia_)
    plt.plot(K,Sum_of_squared_distances,'bx-')
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

if __name__ == "__main__":
    articles = dp.load_data('./articles/')
    vectorizer = TfidfVectorizer(input='filename', tokenizer = dp.LemmaTokenizer())
    print("Generating TF-IDF...")
    
    X_tfidf = vectorizer.fit_transform(sample(articles, 100))
    print(X_tfidf.shape)
    X_trans = dp.reduce_dimensions(X_tfidf)
    print(X_trans.shape)

    sse_params = {
        'p': 2,
        'dist': distance_functions[0]
        }
    constants = {
        'inertia' : 0.5,
        'cognitive': 0.5,
        'social': 0.5
    }
    
    particles = [Particle(X_trans, 15, sse_params, constants) for _ in range(SWARM_SIZE)]

    

    print(Particle.global_best_sse)

    
