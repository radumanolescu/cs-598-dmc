# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:40:14 2020

@author: https://stackoverflow.com/questions/53619851/set-seaborn-bar-color-based-on-values-not-included-in-the-plot
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({"x" : list("ABCDEFGH"),
                   "y" : [3,4,5,2,1,6,3,4],
                   "z" : [4,5,7,1,4,5,3,4]})

norm = plt.Normalize(df.z.min(), df.z.max())
cmap = plt.get_cmap("magma")

plt.bar(x="x", height="y", data=df, color=cmap(norm(df.z.values)))

plt.show()
