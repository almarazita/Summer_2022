# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 00:37:04 2022

@author: almar

Quiz 3 code
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Likelihood ratio test with asymmetric costs
def threshold(mu1 = 5, sigma1 = 0.5, p_s1 = 0.5, loss1 = 1, mu2 = 7, 
              sigma2 = 1, p_s2 = 0.5, loss2 = 2):
    """
    Parameters
    ----------
    mu1 : Integer or float, optional.
        Mean response to stimulus 1. The default is 5.
    sigma1 : Integer or float, optional.
        Standard deviation in response to stimulus 1. The default is 0.5.
    p_s1 : Float, optional.
        The prior probability of stimulus 1 occurring. The default is 0.5.
    loss1 : Integer or float, optional
        The penalty weight given to mistakenly predicting stimulus 1. The default is 1.
    mu2 : Integer or float, optional
        Mean response to stimulus 2. The default is 7.
    sigma2 : Integer or float, optional
        Standard deviation in response to stimulus 2. The default is 1.
    p_s2 : Float, optional
        The prior probability of stimulus 1 occurring. The default is 0.5.
    loss2 : Integer or float, optional
        The penalty weight given to mistakenly predicting stimulus 2. The default is 2.

    Returns
    -------
    None. Plots the Gaussian responses of the neuron to stimuli 1 and 2.
    Plots the liklihood ratio and the corresponding ratio of loss functions.
    Calculates, prints to the screen, and plots the thresehold value z
    that maximizes P[correct].

        """
    # Plot response, firing rate, r
    r = np.arange(min(mu1, mu2) - 3*max(sigma1, sigma2),
                  max(mu1, mu2) + 3*max(sigma1, sigma2), 0.001)
    
    # Neuron response to s1 (Gaussian)
    s1PDF = norm.pdf(r, mu1, sigma1) # Likelihood
    
    # Neuron response to s2 (Gaussian)
    s2PDF = norm.pdf(r, mu2, sigma2) # Likelihood
    
    # Likelihood ratio, the best estimate for the threshold z, maximizing p(correct)
    # Choose s1 when p(r|s2)/p(r|s1) < 1
    # Choose s2 when p(r|s2)/p(r|s1) > 1
    likelihoodRatio = s2PDF/s1PDF # Divides elementwise to create new array
    
    # Building in cost and the role of the prior
    # Loss s1 = loss1 * p(s2|r)
    # Loss s2 = loss2 * p(s1|r)
    # Rearrange with Bayes rule
    # When does p(r|s2)/p(r|s1) = loss2*p(s2)/loss1*p(s1)?
    lossRatio = np.ones(len(r))*((loss2*p_s1)/(loss1*p_s2))
    intersects = [x for x in likelihoodRatio if abs(lossRatio[0] - x) < 0.01]
    loc = list(likelihoodRatio).index(min(intersects[-2:])) # Index of array when ratio is closest
    z = r[loc] # Value of r when likelihoodRatio = lossRatio
    print("z =", z)
    
    # Plotting the normal distributions of responses to s1 and s2
    plt.subplot(211) # 1st subplot
    plt.plot(r, s1PDF, label = "p(r|s1)")
    plt.plot(r, s2PDF, label = "p(r|s2)")
    plt.axvline(z, color="black")
    plt.ylabel("p(r|s)")
    plt.suptitle("Decoding Neural Response to Stimuli 1 and 2")
    plt.legend()
    
    # Plotting the likelihood ratio and possible thresholds
    plt.subplot(212) # 2nd subplot
    plt.plot(r, likelihoodRatio, label = "p(r|s2)/p(r|s1)")
    plt.plot(lossRatio, label = "loss s2/loss s1")
    plt.axis([0, r[-1], 0, lossRatio[0] + 1])
    plt.xlabel("r (Hz)")
    plt.ylabel("Ratio")
    plt.legend()
    plt.show()

def main():
    
    # Generate distributions and threshold for two neural signals
    m1 = float(input("Stimulus 1\nMu: "))
    s1 = float(input("Sigma: "))
    p1 = float(input("Probability: "))
    l1 = float(input("Penalty: "))
    m2 = float(input("\nStimulus 2\nMu: "))
    s2 = float(input("Sigma: "))
    p2 = float(input("Probability: "))
    l2 = float(input("Penalty: "))
    threshold(m1, s1, p1, l1, m2, s2, p2, l2)

if __name__ == "__main__":
    main()