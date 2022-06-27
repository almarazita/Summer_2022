# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 03:39:19 2022

@author: almar

Quiz 7 Code
"""

import numpy as np
import matplotlib.pyplot as plt

# Write weight matrix w(t) in terms of the eigenvectors of correlation
# matrix Q to solve differential equation
# w(t) = sigma ci(t)ei 
Q = np.array([[0.2, 0.1], [0.1, 0.3]])
eigenval, eigenvec = np.linalg.eig(Q)
print(eigenval, eigenvec)

# For large t, largest eigenvalue term dominates
# w(t) is proportional to e1, and the final weight is aligned with the
# principle eigenvector
e1 = eigenvec[np.argmax(eigenval)]

# Possible final weight vectors
weights = np.array([[0.8944, 1.7889], [-1.5155, -1.3051],
                    [-1.5764, -1.2308], [1.0515, 1.7013]])

# Determine which is closest to whole number multiple of e1
diff = 1
index = 0
for i, x in enumerate(weights):
    c = x/e1
    val = abs(c[0] - round(c[0])) + abs(c[1] - round(c[1]))
    if val < diff:
        diff = val
        index = i

print("Final weight vector:", weights[index])

# Plotting
fig, ax = plt.subplots()
ax.arrow(0, 0, e1[0], e1[1], color = 'red', head_width = 0.03)
ax.arrow(0, 0, weights[0, 0], weights[0, 1], color = 'blue', head_width = 0.03)
ax.arrow(0, 0, weights[1, 0], weights[1, 1], color = 'blue', head_width = 0.03)
ax.arrow(0, 0, weights[2, 0], weights[2, 1], color = 'blue', head_width = 0.03)
ax.arrow(0, 0, weights[3, 0], weights[3, 1], color = 'red', head_width = 0.03)

