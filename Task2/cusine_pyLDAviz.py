# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:22:04 2020

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
from gensim.models import LdaModel, HdpModel
from gensim.corpora.dictionary import Dictionary
import pyLDAvis.gensim
import time

#https://github.com/bmabey/pyLDAvis/blob/master/tests/pyLDAvis/test_gensim_models.py

with open("dictionary_corpus.pickle", 'rb') as f:
    dictionary = pickle.load(f)
    corpus = pickle.load(f)

# Building reverse index.
for (token, uid) in dictionary.token2id.items():
    dictionary.id2token[uid] = token

def get_cuisines(dir_name):
    return list(map(lambda x: x.replace(".txt", ""), os.listdir(dir_name)))

#abels = ['American_New', 'American_Traditional', 'Asian_Fusion', 'Barbeque', 'British', 'Burgers', 'Cajun_Creole', 'Caribbean', 'Cheesesteaks', 'Chicken_Wings', 'Chinese', 'Desserts', 'Diners', 'Fast_Food', 'Filipino', 'Fish_Chips', 'French', 'Gluten-Free', 'Greek', 'Hawaiian', 'Hot_Dogs', 'Ice_Cream_Frozen_Yogurt', 'Indian', 'Irish', 'Italian', 'Japanese', 'Juice_Bars_Smoothies', 'Korean', 'Latin_American', 'Mediterranean', 'Mexican', 'Middle_Eastern', 'Modern_European', 'Pakistani', 'Peruvian', 'Pizza', 'Sandwiches', 'Scottish', 'Seafood', 'Soul_Food', 'Soup', 'Steakhouses', 'Sushi_Bars', 'Tapas_Bars', 'Tapas_Small_Plates', 'Tex-Mex', 'Thai', 'Vegetarian', 'Vietnamese']
labels = get_cuisines("rvwByCsne")
n = len(corpus)

def test_lda(corpus, dictionary):
    """Trains a LDA model and tests the html outputs."""
    lda = LdaModel(corpus=corpus, num_topics=49)
    data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(data, 'index_lda.html')
    #os.remove('index_lda.html')

def test_hdp(corpus, dictionary):
    """Trains a HDP model and tests the html outputs."""
    hdp = HdpModel(corpus, dictionary.id2token)
    data = pyLDAvis.gensim.prepare(hdp, corpus, dictionary)
    pyLDAvis.save_html(data, 'index_hdp.html')
    #os.remove('index_hdp.html')
    
test_lda(corpus, dictionary)
test_hdp(corpus, dictionary)
 