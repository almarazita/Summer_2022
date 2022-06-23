# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 03:18:03 2022

@author: almar

Quiz 6 code
"""

import numpy as np
import matplotlib.pyplot as plt


# Defines an object of a feedforward network of 5 neurons
class feedforward:
    def __init__(self):
        self.u = np.ones(5)
        self.W = np.ones((5, 5))
    
    def setInput(self, u):
        if len(u) == 5:
            self.u = u
    
    def _output(self):
        return np.dot(self.u, self.W)
    
    def blur(self):
        self.W = np.array([[0.33, 0.33, 0, 0, 0.33], [0.33, 0.33, 0.33, 0, 0],
                      [0, 0.33, 0.33, 0.33, 0], [0, 0, 0.33, 0.33, 0.33],
                      [0.33, 0, 0, 0.33, 0.33]])
        return feedforward._output(self)
    
    def darken(self):
        self.W = np.array([[0.5, 0, 0, 0, 0], [0, 0.5, 0, 0, 0], [0, 0, 0.5, 0, 0],
                      [0, 0, 0, 0.5, 0], [0, 0, 0, 0, 0.5]])
        return feedforward._output(self)
    
    def edges(self):
        self.W = np.array([[0.75, 0, 0, 0, -0.75], [-0.75, 0.75, 0, 0, 0],
                      [0, -0.75, 0.75, 0, 0], [0, 0, -0.75, 0.75, 0],
                      [0, 0, 0, -0.75, 0.75]])
        return feedforward._output(self)
    
    def pixelate(self):
        self.W = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]])
        return feedforward._output(self)


def steady_state(u, W, M):
    """ Determines the steady-state voltage of a recurrent network with 5
    input nodes defined by static input vector u, weight matrix W, and
    recurrent weight matrix M. """
    h = np.dot(W, u)  # input
    lambda_i, e_i = np.linalg.eig(M)

    # If all eigenvalues are less than 1, v(t) converges
    if max(lambda_i) < 1:
        vss = 0
        for i in range(len(lambda_i)):
            vss += ((np.dot(h, e_i[:,i]))/(1-lambda_i[i]))*e_i[:,i]
        return vss
    
    else:
        return "The network is unstable"

def main():
    vision = feedforward()
    u = np.array([1, 2, 3, 4, 5])
    vision.setInput(u)
    print("Input:", u)
    print("Blur:", vision.blur())
    print("Darken:", vision.darken())
    print("Edges:", vision.edges())
    print("Pixelate:", vision.pixelate())
    
    plt.scatter([1, 2, 3, 4, 5], [5, 5, 5, 5, 5], c = u, cmap = 'Blues', s = 1000)
    plt.scatter([1, 2, 3, 4, 5], [4, 4, 4, 4, 4], c = vision.blur(), cmap = 'Blues', s = 1000)
    plt.scatter([1, 2, 3, 4, 5], [3, 3, 3, 3, 3], c = vision.darken(), cmap = 'Blues', s = 1000)
    plt.scatter([1, 2, 3, 4, 5], [2, 2, 2, 2, 2], c = vision.edges(), cmap = 'Blues', s = 1000)
    plt.scatter([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], c = vision.pixelate(), cmap = 'Blues', s = 1000)
    
    u = np.array([0.6, 0.5, 0.6, 0.2, 0.1])
    W = np.eye(5)
    W[W > 0] = 0.6
    W[W == 0] = 0.1
    M = np.array([[-0.125, 0, 0.125, 0.125, 0], [0, -0.125, 0, 0.125, 0.125],
                 [0.125, 0, -0.125, 0, 0.125], [0.125, 0.125, 0.0, -0.125, 0],
                [0, 0.125, 0.125, 0, -0.125]])
    print("\nu =", u)
    print("W =", W)
    print("M =", M)
    print("Steady state voltage =", steady_state(u, W, M))
    
if __name__ == "__main__":
    main()