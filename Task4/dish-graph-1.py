# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:59:04 2020

@author: Radu
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

dishes = pd.read_csv('SimDishRankings.csv')

sns.set(style="whitegrid")
sns.set_color_codes("pastel")
#map = sns.cubehelix_palette(light=1, as_cmap=False)
#map = plt.get_cmap("viridis")
cmap = cm.get_cmap('jet')

sns.barplot(x="Frequency", y="Rating", data=dishes, color=1, palette=cmap)

# Does not work
#sns.barplot(x="Frequency", y="Rating", data=dishes, color=1, palette=cmap)

# Does not work
#sns.barplot(x="Frequency", y="Dish", data=dishes,  label="Rating", color=1, palette=cmap)

# Works, but no color change
#sns.barplot(x="Frequency", y="Dish", data=dishes,  label="Rating", color="b")

# Does not work
#sns.barplot(x="Frequency", y="Rating", data=dishes,  label="Dish", color=1, palette=cmap)
