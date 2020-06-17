# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:49:08 2020

@author: https://python-graph-gallery.com/2-horizontal-barplot/
"""


# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# Make fake dataset
height = [3, 12, 5, 18, 45]
bars = ('A', 'B', 'C', 'D', 'E')
y_pos = np.arange(len(bars))
 
# Create horizontal bars
plt.barh(y_pos, height)
 
# Create names on the y-axis
plt.yticks(y_pos, bars)
 
# Show graphic
plt.show()
