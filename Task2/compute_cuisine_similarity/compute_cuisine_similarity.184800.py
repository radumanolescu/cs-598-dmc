# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:00:06 2020

@author: Radu
https://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/
"""
import pickle
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import pandas as pd
import seaborn as sns

with open("topics_by_document.pickle", 'rb') as f:
    model = pickle.load(f)
    corpus = pickle.load(f)

# ToDo: read these labels as file names
labels = ['American_New', 'American_Traditional', 'Asian_Fusion', 'Barbeque', 'British', 'Burgers', 'Cajun_Creole', 'Caribbean', 'Cheesesteaks', 'Chicken_Wings', 'Chinese', 'Desserts', 'Diners', 'Fast_Food', 'Filipino', 'Fish_Chips', 'French', 'Gluten-Free', 'Greek', 'Hawaiian', 'Hot_Dogs', 'Ice_Cream_Frozen_Yogurt', 'Indian', 'Irish', 'Italian', 'Japanese', 'Juice_Bars_Smoothies', 'Korean', 'Latin_American', 'Mediterranean', 'Mexican', 'Middle_Eastern', 'Modern_European', 'Pakistani', 'Peruvian', 'Pizza', 'Sandwiches', 'Scottish', 'Seafood', 'Soul_Food', 'Soup', 'Steakhouses', 'Sushi_Bars', 'Tapas_Bars', 'Tapas_Small_Plates', 'Tex-Mex', 'Thai', 'Vegetarian', 'Vietnamese']
n = len(corpus)

# row index: cuisine
# vol index: topic
# a[row,col] = probability of topic for cuisine
a = np.zeros(shape=(n,10), dtype=float)

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

    
cuisineID = 0
for doc_bow in corpus:
    # doc_bow: list[(word_id, count)], e.g. [(638, 3), (646, 1), (651, 1)]
    #print("---------- cuisineID:" + str(cuisineID))
    #print(doc_bow[128:138])
    #print(len(doc_bow))
    #pprint(model.get_document_topics(doc_bow))
    topicsForDoc = model.get_document_topics(doc_bow)
    # array[(topic_id, probability)], e.g. [(2, 0.60961086), (6, 0.08670142), (9, 0.30324253)]
    for topic_fraction in topicsForDoc:
        topic = topic_fraction[0]
        fraction = topic_fraction[1]
        #print(topic_fraction)
        #print("topic   :" + str(topic))
        #print("fraction:" + str(fraction))
        a[cuisineID, topic] = fraction
    cuisineID = cuisineID + 1

def plot_similarity_square(a):
    #https://stackoverflow.com/questions/29481485/creating-a-distance-matrix/45834105
    #https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.distance_matrix.html#scipy.spatial.distance_matrix
    dm = distance_matrix(a, a)
    print(dm.shape)    
    # https://kapilddatascience.wordpress.com/2016/05/29/plotting-similarity-score-in-a-matrix/
    fig, ax = plt.subplots(figsize=(22,22))
    cax = ax.matshow(dm, interpolation='nearest')
    ax.grid(True)
    plt.title('Cuisine Similarity matrix')
    plt.xticks(range(n), labels, rotation=90);
    plt.yticks(range(n), labels);
    fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75,.8,.85,.90,.95,1])
    plt.show()

def plot_similarity_triangular(a):
    #http://seaborn.pydata.org/examples/many_pairwise_correlations.html    
    sns.set(style="white")
    # Create DataFrame from matrix
    #https://stackoverflow.com/questions/20763012/creating-a-pandas-dataframe-from-a-numpy-array-how-do-i-specify-the-index-colum
    d = pd.DataFrame(data=a.T, columns=list(labels))    
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

plot_similarity_square(a)
plot_similarity_triangular(a)

bow_mtx = get_bow_mtx(corpus)
plot_similarity_square(bow_mtx)
plot_similarity_triangular(bow_mtx)
