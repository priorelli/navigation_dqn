import numpy as np
import matplotlib.pyplot as plt
from grid_activations import *

with open('positions.txt','r') as file:
    lines = file.readlines()

pos = []
for line in lines:
    tokens = line.strip().split('\t')
    pos.append([float(tokens[0]), float(tokens[1])])

# Creates two subplots and unpacks the output array immediately
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

for x, y in pos:
    ax1.plot([y], [x], marker='o', markersize=3, color='red')
ax1.set_xlim((0, 16))
ax1.set_ylim((0, 16))
ticks = np.arange(0, 16, 1)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.invert_yaxis()
ax1.grid()

ax2.scatter(x, y)

plt.show()
