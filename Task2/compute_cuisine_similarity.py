# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:00:06 2020

@author: Radu

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
import time

#https://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/
with open("topics_by_document.pickle", 'rb') as f:
    model = pickle.load(f)
    corpus = pickle.load(f)

def get_cuisines(dir_name):
    return list(map(lambda x: x.replace(".txt", ""), os.listdir(dir_name)))

#abels = ['American_New', 'American_Traditional', 'Asian_Fusion', 'Barbeque', 'British', 'Burgers', 'Cajun_Creole', 'Caribbean', 'Cheesesteaks', 'Chicken_Wings', 'Chinese', 'Desserts', 'Diners', 'Fast_Food', 'Filipino', 'Fish_Chips', 'French', 'Gluten-Free', 'Greek', 'Hawaiian', 'Hot_Dogs', 'Ice_Cream_Frozen_Yogurt', 'Indian', 'Irish', 'Italian', 'Japanese', 'Juice_Bars_Smoothies', 'Korean', 'Latin_American', 'Mediterranean', 'Mexican', 'Middle_Eastern', 'Modern_European', 'Pakistani', 'Peruvian', 'Pizza', 'Sandwiches', 'Scottish', 'Seafood', 'Soul_Food', 'Soup', 'Steakhouses', 'Sushi_Bars', 'Tapas_Bars', 'Tapas_Small_Plates', 'Tex-Mex', 'Thai', 'Vegetarian', 'Vietnamese']
labels = get_cuisines("rvwByCsne")
n = len(corpus)

def get_max_word_id(cps):
    # cps is the corpus
    max_word_id = -1
    for doc_bow in corpus:
        # doc_bow: list[(word_id, count)], e.g. [(638, 3), (646, 1), (651, 1)]
        for word_id__count in doc_bow:
            max_word_id = max(max_word_id, word_id__count[0])
    return max_word_id

def get_bow_mtx(cps):
    # cps is the corpus
    max_word_id = get_max_word_id(cps)
    bow_mtx = np.zeros(shape=(n,max_word_id+1), dtype=float)
    cuisineID = 0
    for doc_bow in corpus:
        # doc_bow: list[(word_id, count)], e.g. [(638, 3), (646, 1), (651, 1)]
        for word_id__count in doc_bow:
            word_id = word_id__count[0]
            word_ct = word_id__count[1]
            bow_mtx[cuisineID, word_id] = word_ct
        cuisineID = cuisineID + 1
    return bow_mtx

def get_topic_mtx(cps):
    # row index: cuisine
    # col index: topic
    # a[row,col] = probability of topic for cuisine
    topic_mtx = np.zeros(shape=(n,49), dtype=float)
    cuisineID = 0
    for doc_bow in cps:
        # doc_bow: list[(word_id, count)], e.g. [(638, 3), (646, 1), (651, 1)]
        #print("---------- cuisineID:" + str(cuisineID))
        #print(doc_bow[128:138])
        #print(len(doc_bow))
        #pprint(model.get_document_topics(doc_bow))
        topicsForDoc = model.get_document_topics(doc_bow)
        # list[(topic_id, probability)], e.g. [(2, 0.60961086), (6, 0.08670142), (9, 0.30324253)]
        for topic_fraction in topicsForDoc:
            topic = topic_fraction[0]
            fraction = topic_fraction[1]
            #print(topic_fraction)
            #print("topic   :" + str(topic))
            #print("fraction:" + str(fraction))
            topic_mtx[cuisineID, topic] = fraction
        cuisineID = cuisineID + 1
    return topic_mtx

def plot_similarity_square(mtx, lbl):
    #https://stackoverflow.com/questions/29481485/creating-a-distance-matrix/45834105
    #https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.distance_matrix.html#scipy.spatial.distance_matrix
    dm = distance_matrix(mtx, mtx)
    print(dm.min())    
    print(dm.max())    
    # https://kapilddatascience.wordpress.com/2016/05/29/plotting-similarity-score-in-a-matrix/
    fig, ax = plt.subplots(figsize=(22,22))
    cax = ax.matshow(dm, interpolation='nearest')
    ax.grid(True)
    #lt.title('Cuisine Similarity matrix')
    plt.xticks(range(n), lbl, rotation=90);
    plt.yticks(range(n), lbl);
    fig.colorbar(cax, ticks=[697, 1394, 2091, 2788, 3485, 4182, 4879, 5227.5, 5576, 5924.5, 6273, 6621.5, 6970])
    plt.show()

def plot_similarity_triangular(mtx, lbl):
    #http://seaborn.pydata.org/examples/many_pairwise_correlations.html    
    sns.set(style="white")
    # Create DataFrame from matrix
    #https://stackoverflow.com/questions/20763012/creating-a-pandas-dataframe-from-a-numpy-array-how-do-i-specify-the-index-colum
    d = pd.DataFrame(data=mtx.T, columns=list(lbl))    
    # Compute the correlation matrix
    corr = d.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(22, 18))    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
#https://thispointer.com/pandas-how-to-create-an-empty-dataframe-and-append-rows-columns-to-it-in-python/
#https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
def top_correlations(mtx, lbl):
    d = pd.DataFrame(data=mtx.T, columns=list(lbl))    
    corr = d.corr()
    np_corr = corr.to_numpy()
    rc_corr = pd.DataFrame(columns=['r', 'c', 'corr'])
    for r in range(corr.shape[0]):
        for c in range(corr.shape[1]):
            mod_corr = np_corr[r,c] if np_corr[r,c] < 1 and r > c else 0
            rc_corr = rc_corr.append({'r': r, 'c': c, 'corr': mod_corr}, ignore_index=True)
    rc_corr = rc_corr.sort_values(by=['corr'], ascending=False)
    np_corr = rc_corr.to_numpy(dtype=float)
    #print(np_corr[0:22,])
    cui_cui_co = pd.DataFrame(columns=['cuisine1', 'cuisine2', 'corr'])
    for i in range(22):
        cui1 = lbl[int(np_corr[i,0])]
        cui2 = lbl[int(np_corr[i,1])]
        cui_cui_co = cui_cui_co.append({'cuisine1': cui1, 'cuisine2': cui2, 'corr': np_corr[i,2]}, ignore_index=True)
    print(cui_cui_co)
    return cui_cui_co
    
"""Reorder the lines of a matrix according to the values of an array

The array clu maps each row of mtx to a cluster, i.e.
clu[i] = k indicates that mtx[i,] belongs to cluster k.
We want the matrix mtx reordered such that
a. the rows appear in the order of their clusters
b. within each cluster, rows appear in the same order as they did in the input
(i.e. stable sorting)
"""
def reorder_mtx(mtx, clu):
    #print(mtx)
    #print(clu)
    n = mtx.shape[0]
    m = mtx.shape[1]
    assert n == len(clu)
    mtx_r = np.zeros(shape=(n,m), dtype=float)
    c2 = np.zeros(shape=(2,n), dtype=int)
    c2[0,] = clu
    c2[1,] = range(0,n)
    c2t = c2.T
    cols = [1,0]
    c2ts = c2t[np.lexsort(c2[cols])] # sort by column 1, then 0
    #print(c2ts)
    for i in range(n):
        mtx_r[i,] = mtx[c2ts[i,1], ]
    return mtx_r
    
def reorder_vct(vct, clu):
    print(vct)
    print(clu)
    n = len(vct)
    assert n == len(clu)
    vct_r = ['' for i in range(n)]
    c2 = np.zeros(shape=(2,n), dtype=int)
    c2[0,] = clu
    c2[1,] = range(0,n)
    c2t = c2.T
    cols = [1,0]
    c2ts = c2t[np.lexsort(c2[cols])] # sort by column 1, then 0
    #print(c2ts)
    for i in range(n):
        vct_r[i] = vct[c2ts[i,1]]
    print(vct_r)
    return vct_r

# For 2.1, represent cusines using BoW and plot similarity
bow_mtx = get_bow_mtx(corpus)
#plot_similarity_square(bow_mtx, labels)
#plot_similarity_triangular(bow_mtx, labels)

# For 2.2, represent cusines as topics
#"computing similarity of individual reviews"!?
#https://piazza.com/class/k5on5knznhh517?cid=31
topic_mtx = get_topic_mtx(corpus)
#plot_similarity_square(topic_mtx, labels)
#plot_similarity_triangular(topic_mtx, labels)

# ---------- ---------- ---------- ---------- ---------- ---------- #
#https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
#https://scikit-learn.org/stable/modules/clustering.html

data = topic_mtx

def plot_clusters(data, algorithm, args, kwds):
    cluster_ids = algorithm(*args, **kwds).fit_predict(data)
    # list of cluster labels for each row of data: [1 4 1 4 1 1..]
    return cluster_ids

def get_cluster_structure(cluster_ids):
    # cluster_ids: list of cluster labels for each cuisine: [1 4 1 4 1 1..]
    cluster_dict = {}
    for cuisine_id in range(len(cluster_ids)):
        label = cluster_ids[cuisine_id]
        if not label in cluster_dict:
            cluster_dict[label] = []
        cluster_list = cluster_dict[label]
        cluster_list.append(cuisine_id)
    # map {cluster_id -> [list of cuisine ids in this cluster]}
    return cluster_dict
    
def print_clusters(cluster_struct, labels):
    # cluster_struct: map {cluster_id -> [list of cuisine ids in this cluster]}
    print("---------- Cuisine Clusters ----------")
    for cluster_id in cluster_struct:
        names = [labels[cuisine_id] for cuisine_id in cluster_struct[cluster_id]]
        if 2 < len(names) and len(names) < 8:
            print(names)

# For 2.3, try different clusterings of the cuisines

#https://scikit-learn.org/stable/modules/clustering.html#k-means
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#clusters = plot_clusters(data, cluster.KMeans, (), {'n_clusters':20})

#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html
#clusters = plot_clusters(data, cluster.AffinityPropagation, (), {'damping':0.75}) # 0.75: Good combination

#https://scikit-learn.org/stable/modules/clustering.html#mean-shift
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
#clusters = plot_clusters(data, cluster.MeanShift, (0.8,), {'cluster_all':False}) # 0.1: Good combination

#https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
#clusters = plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters':25}) # 6 | 4: Good combinations

#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
clusters = plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':12, 'linkage':'ward'}) # 8: Good combination

#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
#clusters = plot_clusters(data, cluster.DBSCAN, (), {'eps':0.95})

topic_mtx_r = reorder_mtx(topic_mtx, clusters)
labels_r = reorder_vct(labels, clusters)
#print(labels_r)

plot_similarity_triangular(topic_mtx_r, labels_r)

cluster_struct = get_cluster_structure(clusters)
print_clusters(cluster_struct, labels)

#top_correlations(topic_mtx, labels)

#pprint(topic_mtx)
#pprint(topic_mtx_r)


