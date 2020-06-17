# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:37:06 2020

@author: Radu
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dishes = pd.read_csv('DishRatings.csv', sep='\t').sort_values(by=['Frequency'])

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

width = dishes['Frequency']
dish_names = dishes['Dish']
y_pos = np.arange(len(dish_names))

norm = plt.Normalize(dishes['Rating'].min(), dishes['Rating'].max())
# lo=red /^ hi=green
cmap = plt.get_cmap("RdYlGn")

# Create horizontal bars
plt.barh(y_pos, width, color=cmap(norm(dishes['Rating'].values)))

# Create names on the y-axis
plt.yticks(y_pos, dish_names)

# Show graphic
plt.show()

