# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:44:55 2020

@author: https://www.pythonprogramming.in/vary-the-color-of-each-bar-in-bar-chart-using-particular-value-in-matplotlib.html
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from numpy.random import rand
 
data = [2, 3, 5, 6, 8, 12, 7, 5]
fig, ax = plt.subplots(1, 1)
 
# Get a color map
my_cmap = cm.get_cmap('jet')
 
# Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
my_norm = Normalize(vmin=0, vmax=8)
 
ax.bar(range(8), rand(8), color=my_cmap(my_norm(data)))
plt.show()

