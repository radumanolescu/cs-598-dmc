# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:23:10 2020

@author: Radu
"""


import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

import pprint

with open("review_dense.pickle", 'rb') as f:
    rvw_dense = pickle.load(f)
    lbl_trn = pickle.load(f)

num_trn = len(lbl_trn)

rvw_trn = rvw_dense[0:num_trn]
rvw_tst = rvw_dense[num_trn:]

X = rvw_trn
Y = lbl_trn

# ---------- #


# Create grid for optimization
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# ---------- #

rf_random = rf_random.fit(X, Y)

lbl_tst = rf_random.predict(rvw_tst)

outF = open("radufm2_results.txt", "w")
outF.write("radufm2\n")
for label in lbl_tst:
    outF.write(label + "\n")
outF.close()
