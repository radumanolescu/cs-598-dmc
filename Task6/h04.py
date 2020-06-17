# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:40:56 2020

@author: Radu
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

with open("review_dense.pickle", 'rb') as f:
    rvw_dense = pickle.load(f)
    lbl_trn = pickle.load(f)

num_trn = len(lbl_trn)

rvw_trn = rvw_dense[0:num_trn]
rvw_tst = rvw_dense[num_trn:]

X = rvw_trn
Y = lbl_trn
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)

lbl_tst = clf.predict(rvw_tst)

outF = open("radufm2_results.txt", "w")
outF.write("radufm2\n")
for label in lbl_tst:
    outF.write(label + "\n")
outF.close()

# 40	radufm2	0.566	-inf	0.5646	-inf	0.568	-inf	2020-04-19 | 19:26:16	1
