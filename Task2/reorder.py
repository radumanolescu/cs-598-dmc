# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:16:39 2020

@author: Radu
"""

import numpy as np

aaa = np.array([[11,12,12,14],[21,22,23,24],[31,32,33,34],[41,42,43,44]], dtype=int)
bbb = np.array(['11','21','31','41'], dtype=object)
clusters = np.array([2,2,1,0])

def reorder_mtx(mtx, clu):
    print(mtx)
    print(clu)
    n = mtx.shape[0]
    m = mtx.shape[1]
    assert n == len(clu)
    mtx_r = np.zeros(shape=(n,m), dtype=float)
    c2 = np.zeros(shape=(2,n), dtype=int)
    c2[0,] = clusters
    c2[1,] = range(0,n)
    c2t = c2.T
    cols = [1,0]
    c2ts = c2t[np.lexsort(c2[cols])] # sort by column 1, then 0
    print(c2ts)
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
    c2[0,] = clusters
    c2[1,] = range(0,n)
    c2t = c2.T
    cols = [1,0]
    c2ts = c2t[np.lexsort(c2[cols])] # sort by column 1, then 0
    print(c2ts)
    for i in range(n):
        vct_r[i] = vct[c2ts[i,1]]
    return vct_r

    
print(reorder_mtx(aaa, clusters))

print(reorder_vct(bbb, clusters))

#https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column/30623882
#https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.lexsort.html
#https://jakevdp.github.io/PythonDataScienceHandbook/02.08-sorting.html
# sorted_array = an_array[numpy.argsort(an_array[:, 1])]
# c2 = np.zeros(shape=(2,n), dtype=int)
# c2[0,] = clusters
# c2[1,] = range(0,n)
# c2t = c2.T
# c2ts = c2t[np.argsort(c2t[:, 0])]
# cols = [1,0]
# c2t[np.lexsort(c2t.T[cols])]




