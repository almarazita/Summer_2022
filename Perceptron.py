# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:20:22 2022

@author: almar

Teaching a perceptron using 2D inputs
"""

import numpy as np
import matplotlib.pyplot as plt

# The perceptron as classifier
def output(w, mu, *u):
    """ Returns +1 for each input that exceeds threshold
    and -1 otherwise """
    
    weightedSum = [np.dot(u[i], w) for i in range(len(u))]
    
    v = []
    for s in weightedSum:
        if(s > mu):
            v += [1]
        else:
            v += [-1]
    
    return v


v = output([1, 3], 0, [-1, 1], [1, 1], [-1, -1], [1, -1])
print(v)


def rule(u):
    """ Determines the desired output neuron will learn """
    if u[1] > 0:
        return True
    else:
        return False

def train():
    """ Perceptron learning rule """
    
    # Generate random training data
    u = np.random.randn(10000, 2)
    size = len(u)
    
    # Choose random values for starting weights and threshold (-1, 1]
    w = (np.random.rand(2)) * 2 - 1
    mu = np.random.rand(1) * 2 - 1
    epsilon = 1 # Learning rate
    
    # Desired output according to rule
    v_d = [1 if rule(x) else -1 for x in u]
    
    # Optionally keep track of values over time
    v = []
    weights = [np.copy(w)] # List of numpy arrays
    delta_w = []
    thresholds = [np.copy(mu)] # List of numpy arrays
    delta_mu = []
    
    t = 0
    #timesteps = num
    
    # Option to run until no change for 50 iterations
    while t < 100 or (np.any(np.array(delta_w[-100:])) and np.any(np.array(delta_mu[-100:]))):
    #for t in range(timesteps):
        
        # Determine output
        v = 1 if np.dot(u[t%size], w) > mu else -1
        
        # Update weights
        w[0] = w[0] + (epsilon * (v_d[t%size] - v) * u[t%size, 0])
        w[1] = w[1] + (epsilon * (v_d[t%size] - v) * u[t%size, 1])
        
        # Update threshold
        mu = mu - (epsilon * (v_d[t%size] - v))
        
        # Keep track of change over time
        weights += [np.copy(w)]
        delta_w += [weights[t + 1] - weights[t]]
        thresholds += [np.copy(mu)]
        delta_mu += [thresholds[t + 1] - thresholds[t]]
        
        # Update learning rate so weights and threshold converge
        epsilon /= 1.01
        
        # Update counter
        t += 1
    
    print("Iterated", t, "times")
    plt.plot(delta_w, label='w')
    plt.plot(delta_mu, label='mu')
    plt.legend()
    plt.show()

    return w, mu

def test(w, mu):
    """ Plots classification of new test data by trained perceptron """
    
    # Generate 100 random inputs
    u = np.random.randn(100, 2)
    
    # Expected output
    plt.subplot(211)
    for x in u:
        
        if rule(x):
            plt.scatter(x[0], x[1], color = 'blue')
        else:
            plt.scatter(x[0], x[1], color = 'red')
        
        plt.suptitle("Expected")    
    
    # Experimental output
    plt.subplot(212)
    for x in u:
        
        if np.dot(x, w) > mu:
            plt.scatter(x[0], x[1], color = 'blue')
        else:
            plt.scatter(x[0], x[1], color = 'red')
          
    
    # "Hyperplane" that separates the two groups
    xaxis = np.arange(min(u[:, 0]), max(u[:, 0]), 0.01)
    yaxis = [(-w[0]/w[1])*x + (mu/w[1]) for x in xaxis]
    plt.plot(xaxis, yaxis, color = 'black')
    plt.axis([min(u[:, 0]), max(u[:, 0]), min(u[:, 1]), max(u[:, 1])])
    plt.show()


w, mu = train()
print("Weights:", w)
print("Threshold:", mu)
test(w, mu)