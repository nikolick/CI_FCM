from math import floor
import os
from random import random
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



