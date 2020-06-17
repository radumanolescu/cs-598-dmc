# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:05:45 2020

@author: Radu
"""

import pandas as pd
import pickle
import numpy as np

with open("review_docs.pickle", 'rb') as f:
    docs = pickle.load(f)
    dictionary = pickle.load(f)
    corpus = pickle.load(f)

num_words = len(dictionary)
num_trn = 546
idx_zip = num_words + 0
idx_rvw = num_words + 1
idx_rtg = num_words + 2

# Read the additional info file
addtl_path = "C:/Users/Radu/-/Study/MCS-DS/CS-598-DMC/Task6/Hygiene/hygiene.dat.additional"
addtl_df = pd.read_csv(addtl_path, header=None)

# Read the labels file
lbls_path = "C:/Users/Radu/-/Study/MCS-DS/CS-598-DMC/Task6/Hygiene/hygiene.dat.labels"
lbls_df = pd.read_csv(lbls_path, header=None)

lbls_trn = lbls_df[0][0:num_trn].tolist()

# iterate over corpus, transform sparse to dense representation
rvw_dense = []
for rvw_id in range(len(corpus)):
    rvw_sparse = corpus[rvw_id]
    # Allocate space for words from corpus, zip, num reviews, avg rating
    a = np.zeros(num_words+3)
    for id_count in rvw_sparse:
        a[id_count[0]] = id_count[1]        
    a_zip = addtl_df.iloc[rvw_id,1]
    a_rvw = addtl_df.iloc[rvw_id,2]
    a_rtg = addtl_df.iloc[rvw_id,3]        
    a[idx_zip] = a_zip
    a[idx_rvw] = a_rvw
    a[idx_rtg] = a_rtg
    a_lst = a.tolist()
    rvw_dense.append(a_lst)
    
rvw_trn = rvw_dense[0:num_trn]
rvw_tst = rvw_dense[num_trn:]

with open("review_dense.pickle", 'wb') as f:
    pickle.dump(rvw_dense, f)
    pickle.dump(lbls_trn, f)
