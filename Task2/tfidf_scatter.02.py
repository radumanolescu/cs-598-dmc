# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 20:55:37 2020

@author: https://beckernick.github.io/law-clustering/

Does not seem to be working for me as it does for the author.
Needs more work, probably to understand the data structures used by the author.
"""

import os
import pickle
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import pandas as pd
import seaborn as sns
import sklearn.cluster as cluster
from gensim.models import LdaModel, HdpModel
from gensim.corpora.dictionary import Dictionary
import pyLDAvis.gensim
import time

import re
import nltk
from nltk.corpus import stopwords
#import bs4
#import urllib2
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

with open("cuisine_docs.pickle", 'rb') as f:
    docs = pickle.load(f)
    # list of length 49 of lists of words, e.g. ['may', 'be', 'one', 'of', 'middleton', ...]

def cwd():
    return os.getcwd().replace("\\", "/")

def get_cuisines(dir_name):
    return list(map(lambda x: x.replace(".txt", ""), os.listdir(dir_name)))

#abels = ['American_New', 'American_Traditional', 'Asian_Fusion', 'Barbeque', 'British', 'Burgers', 'Cajun_Creole', 'Caribbean', 'Cheesesteaks', 'Chicken_Wings', 'Chinese', 'Desserts', 'Diners', 'Fast_Food', 'Filipino', 'Fish_Chips', 'French', 'Gluten-Free', 'Greek', 'Hawaiian', 'Hot_Dogs', 'Ice_Cream_Frozen_Yogurt', 'Indian', 'Irish', 'Italian', 'Japanese', 'Juice_Bars_Smoothies', 'Korean', 'Latin_American', 'Mediterranean', 'Mexican', 'Middle_Eastern', 'Modern_European', 'Pakistani', 'Peruvian', 'Pizza', 'Sandwiches', 'Scottish', 'Seafood', 'Soul_Food', 'Soup', 'Steakhouses', 'Sushi_Bars', 'Tapas_Bars', 'Tapas_Small_Plates', 'Tex-Mex', 'Thai', 'Vegetarian', 'Vietnamese']
labels = get_cuisines("rvwByCsne")

dictionary = {}
for i in range(len(labels)):
    dictionary[labels[i]] = " ".join(docs[i])

n = len(dictionary)
data_path = cwd()

# Calculating TF-IDF Vectors
stemmer = PorterStemmer()

def stem_words(words_list, stemmer):
    return [stemmer.stem(word) for word in words_list]

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_words(tokens, stemmer)
    return stems

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(dictionary.values())

from scipy.sparse import csr_matrix

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

save_sparse_csr(data_path + 'yelp_tfidfs.npz', tfs)

from sklearn.neighbors import NearestNeighbors

model_tf_idf = NearestNeighbors(metric='cosine', algorithm='brute')
model_tf_idf.fit(tfs)

def print_nearest_neighbors(query_tf_idf, full_doc_dictionary, knn_model, k):
    """
    Inputs: a query tf_idf vector, the dictionary of docs, the knn model, and the number of neighbors
    Prints the k nearest neighbors
    """
    distances, indices = knn_model.kneighbors(query_tf_idf, n_neighbors = k+1)
    nearest_neighbors = [full_doc_dictionary.keys()[x] for x in indices.flatten()]
    
    for doc in range(len(nearest_neighbors)):
        if doc == 0:
            print('Cuisine: {0}\n'.format(nearest_neighbors[doc]))
        else:
            print('{0}: {1}\n'.format(doc, nearest_neighbors[doc]))

#bill_id = np.random.choice(tfs.shape[0])
#print_nearest_neighbors(tfs[bill_id], dictionary, model_tf_idf, k=5) # throws error!?

#tfs = load_sparse_csr(data_path + 'yelp_tfidfs.npz')



from sklearn.cluster import KMeans

k = 10
km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=5, verbose=1)
km.fit(tfs)


#plt.hist(km.labels_, bins=k)
#plt.show()


cluster_assignments_dict = {}

for i in set(km.labels_):
    current_cluster_bills = [dictionary.keys()[x] for x in np.where(km.labels_ == i)[0]]
    cluster_assignments_dict[i] = current_cluster_bills

cluster_pick = np.random.choice(len(set(km.labels_)))
print('Cluster {0}'.format(cluster_pick))
for cuisine_id in cluster_assignments_dict[cluster_pick]:
    print(labels[cuisine_id])

# Visualizing the Laws as TF-IDF Vectors using t-SNE
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

tfs_reduced = TruncatedSVD(n_components=k, random_state=0).fit_transform(tfs)
tfs_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(tfs_reduced)

fig = plt.figure(figsize = (10, 10))
ax = plt.axes()
plt.scatter(tfs_embedded[:, 0], tfs_embedded[:, 1], marker = "x", c = km.labels_)
plt.show()


