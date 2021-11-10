from pprint import pprint

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

colors = list(mcolors.TABLEAU_COLORS.values())

A = [1, 10, 23, 45, 24, 25, 55, 67, 73, 26, 13, 96, 53, 23, 24, 43, 90, 49]
B = [24, 23, 29, 'BW', 49, 59, 72, 'BW', 9,
     183, 17, 12, 2, 49, 'BW', 479, 18, 'BW']


index = [k for k, value in enumerate(B) if value == 'BW']
index = [-1] + index + [len(B)]

slopes = []

for k in range(len(index)-1):
    x = A[index[k]+1:index[k+1]]
    y = B[index[k]+1:index[k+1]]

    if len(x) == 0:
        continue

    [slope, offset] = np.polyfit(x, y, 1)
    slopes.append(slope)

    reg_x = np.linspace(min(x), max(x), 10)
    reg_y = slope*reg_x + offset

    plt.plot(x, y, 'o', color=colors[k], label=f'Group {k}')
    plt.plot(reg_x, reg_y, color=colors[k])

pprint(slopes)

plt.legend()
plt.show()
