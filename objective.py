from scipy.spatial.distance import cdist
import numpy as np

# different distance functions to use
distance_functions = ['cosine', 'minkowski', 'chebyshev']

# function to minimize 

def SSE(X, C, w, p, dist) -> float:
    # X - dataset of n-dimensional datapoints
    # C - clusters represented by centroids
    # w - weights assigned to datapoints, representing their membership to each cluster
    # p - exponent determining the influence of the weights on the total error
    # dist - distance function

    k = len(C) # number of clusters
    m = len(X) # number of datapoints

    total_error = 0

    # w[i][j] - degree of membership of the i-th datapoint to the j-th cluster
    distances = cdist(X, C, metric=dist)
    for i in range(0, m):
        for j in range(0, k):
            total_error += w[i][j]**p * (distances[i,j])**2

    return total_error


def calculate_weights(X, C, p,  dist):
    k = len(C)
    m = len(X)

    w = np.array(np.zeros((m, k)))
    is_centroid = np.ones(m) * -1.0
    
    distances = cdist(X, C, metric=dist)
    
    for i in range(0, m):
        for j in range(0, k):
            distance = distances[i, j]
            if distance > 0:
                w[i][j] = (1.0 / distance) ** (1/(p - 1))
            else: # data point is equal to the centroid
                is_centroid[i] = j
    
    # Normalizing the distances
    for i in range(0, m):
        if is_centroid[i] == -1.0:
            w[i] /= sum(w[i])
        else:
            w[i] = [1.0 if k == is_centroid[i] else 0 for k in range(len(w[i]))]
    
    return w


    