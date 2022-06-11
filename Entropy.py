# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 19:02:43 2022

@author: almar

Quiz 4 code
"""

import numpy as np
from matplotlib import pyplot as plt 

# Calculating the total response entropy of a neuron, average information
p_F = 0.1 # P(F = 1), P(r+)
q_F = 1 - p_F # P(F = 0), P(r-)

# Total entropy H[R] = -P(r+)log P(r+) - P(r-)log P(r-)
# Represents the total number of states the message can take regardless of stimulus
H = -((p_F * np.log2(p_F)) + (q_F * np.log2(q_F)))
print("Total entropy H(F) =", round(H, 4), "bits")

# Calculating mutual information MI(S,F)
p_S = 0.1 # P(S = 1), P(s+)
q_S = 1 - p_S # P(S = 0), P(s-)

p_SF = 0.5 # P(F = 1|S = 1), P(r+|+)
p_S_ = 0.5 # P(F = 0|S = 1), P(r-|+)

p___ = 17/18 # P(F = 0|S = 0), P(r-|-)
p__F = 1/18 # P(F = 1|S = 0), P(r+|-)

# Noise entropy
# Represents the total number of states the message can take after the stimulus occurs
noiseH0 = -((p___ * np.log2(p___)) + (p__F * np.log2(p__F)))
noiseH1 = -((p_S_ * np.log2(p_S_)) + (p_SF * np.log2(p_SF)))
meanNoiseH = (q_S * noiseH0) + (p_S * noiseH1)
print("Noise entropy =", round(meanNoiseH, 4), "bits")

# Mutual information = total - noise
# The amount of entropy that is used in coding the stimulus
# Quantifies how independent F and S are
MI = H - meanNoiseH
print("Mutual information MI(S,F) =", round(MI, 4), "bits")

# Plotting
x = np.array([0, 1])
y = np.array([p___, p__F])
plt.subplot(311)
plt.bar(x, y, color='red', width = 0.2, tick_label=x)
plt.axis([-0.5, 1.5, 0, 1])
plt.xlabel("F")
plt.ylabel("P(F|S-)")

y = np.array([q_F, p_F])
plt.subplot(312)
plt.bar(x, y, color='blue', width = 0.2, tick_label=x)
plt.axis([-0.5, 1.5, 0, 1])
plt.xlabel("F")
plt.ylabel("P(F)")

y = np.array([p_S_, p_SF])
plt.subplot(313)
plt.bar(x, y, color='green', width = 0.2, tick_label=x)
plt.axis([-0.5, 1.5, 0, 1])
plt.xlabel("F")
plt.ylabel("P(F|S+)")