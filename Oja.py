# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 03:15:36 2022

@author: almar

Online learning algorithm using Oja's Rule
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import pickle

# 100 (x, y) data points a neuron will learn from
with open('c10p1.pickle', 'rb') as f:
    data = pickle.load(f)
data = data['c10p1']

# Step 1: Zero-mean centering
u1 = data[:, 0]
u2 = data[:, 1]
avg_u1 = np.average(u1)
avg_u2 = np.average(u2)
u1 = [x - avg_u1 for x in u1]
u2 = [y - avg_u2 for y in u2]
plt.scatter(u1, u2)

# Step 2: Update using discrete implementation of Oja's Rule
# w_i+1 = w_i + delta_t*eta*(vu - av^2w)
eta = 1 # 1/tau_w, time constant
a = 1
delta_t = 0.01
w = rand(2) # Random starting weight vector w0
u1_w = [w[0]]
u2_w = [w[1]]
timesteps = 100000

# Euler method
for t in range(timesteps):
    
    # Select input u
    u = np.array([u1[t % 100], u2[t % 100]])
    
    # Find output firing rate v
    v = np.dot(u, w)
    
    # Update w
    w = w + delta_t*eta*(v*u - (a*v**2)*w)
    u1_w += [w[0]]
    u2_w += [w[1]]

# Plot
print("Final weight vector =", w)
plt.scatter(u1_w, u2_w, color = 'red')
plt.scatter(w[0], w[1], color = 'black', marker = 'X')

# Determine input correlation matrix, principal eigenvalue/eigenvector
X = np.corrcoef(data.T)
eigenval, eigenvec = np.linalg.eig(X)
print("Eigenvalues of input correlation matrix =", eigenval,
      "\n", "Eigenvectors =", eigenvec, "\n")
print("e1/sqrt(a) =", eigenval[0]/np.sqrt(a))
plt.arrow(0, 0, eigenvec[0, 0], eigenvec[0, 1], head_width = 0.03)

length = np.linalg.norm(w)
print("Length of w =", length)
print("1/sqrt(a) =", 1/np.sqrt(a))
