from math import floor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gap_statistic import OptimalK
import numpy as np
import os
from matplotlib import pyplot as plt
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD

class LemmaTokenizer(object):
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
    
    def __call__(self, document: str) -> list:
        print('vectorizing...')
        lemmas = []

        # Remove punctuation
        translator_1 = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        document = document.translate(translator_1)

        # Remove numbers
        document = re.sub(r'\d+', ' ', document)

        # Remove special characters
        document = re.sub(r'[^a-zA-Z0-9]+', ' ', document)

        for token in word_tokenize(document):

            token = token.strip()

            token = self.lemmatizer.lemmatize(token)

            # Using default stopwords list from NLTK; in the future a custom stop word list could be used
            stop_words = set(stopwords.words('english'))
            
            if token not in stop_words and len(token) > 2:
                lemmas.append(token)
        
        return lemmas

def load_data(fpath: str) -> list:
    return [fpath + fname for fname in os.listdir(fpath)]

def reduce_dimensions(X_tfidf):
    transformer = TruncatedSVD(n_components=floor(min(X_tfidf.shape) * 0.9), algorithm='arpack', tol = 0.001, random_state=41)
    
    X_trans = transformer.fit_transform(X_tfidf)

    print("Explained variance: ")
    print(sum(transformer.explained_variance_))

    return X_trans

# Function to use elbow method, silhouette score and gap statistic to find optimal k
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



