# from https://python-graph-gallery.com/3-control-color-of-barplots/
# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# Make a fake dataset
height = [3, 12, 5, 18, 45]
bars = ('A', 'B', 'C', 'D', 'E')
y_pos = np.arange(len(bars))

def version1():
    # provide red, green and blue + the transparency and it returns a color.
    plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(y_pos, bars)
    plt.show()

def version2():
    # If you want to give different colors to each bar, just provide a list of color names to the color argument:
    plt.bar(y_pos, height, color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.xticks(y_pos, bars)
    plt.show()

def version3():
    # The edgecolor argument allows to color the borders of barplots.
    plt.bar(y_pos, height, color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue')
    plt.xticks(y_pos, bars)
    plt.show()

version3()