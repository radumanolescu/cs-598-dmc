# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:30:47 2020

@author: Radu
"""

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
# https://scikit-learn.org/stable/modules/neighbors.html#classification

import pandas as pd
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

with open("review_dense.pickle", 'rb') as f:
    rvw_dense = pickle.load(f)
    lbl_trn = pickle.load(f)

num_trn = len(lbl_trn)

rvw_trn = rvw_dense[0:num_trn]
rvw_tst = rvw_dense[num_trn:]

X = rvw_trn
Y = lbl_trn
clf = KNeighborsClassifier(10)
clf = clf.fit(X, Y)

lbl_tst = clf.predict(rvw_tst)

outF = open("radufm2_results.txt", "w")
outF.write("radufm2\n")
for label in lbl_tst:
    outF.write(label + "\n")
outF.close()

