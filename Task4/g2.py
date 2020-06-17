# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:37:06 2020

@author: Radu
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Restaurant  Rating
restaurants = pd.read_csv('tandoori_chicken.csv', sep='\t').sort_values(by=['Rating'])

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

width = restaurants['Rating']
rest_names = restaurants['Restaurant']
y_pos = np.arange(len(rest_names))

norm = plt.Normalize(restaurants['Rating'].min(), restaurants['Rating'].max())
# lo=red /^ hi=green
cmap = plt.get_cmap("RdYlGn")

# Create horizontal bars
plt.barh(y_pos, width, color=cmap(norm(restaurants['Rating'].values)))

# Create names on the y-axis
plt.yticks(y_pos, rest_names)

# Show graphic
plt.show()

