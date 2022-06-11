# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 00:33:21 2022

@author: almar

Quiz 4 code
Poisson neuron models and population coding
"""

import numpy as np
from matplotlib import pyplot as plt
import statistics as st

# Load in data
# Array of stimulus velocities and neuron matrices of 100 trials
# per direction in stim
import pickle
with open('tuning_3.4.pickle', 'rb') as f:
    data = pickle.load(f)

stim = data['stim']
neuron1 = data['neuron1']
neuron2 = data['neuron2']
neuron3 = data['neuron3']
neuron4 = data['neuron4']

# Compute the average firing rates for each neuron for each direction in stim
avgRates = np.zeros((4, len(stim)))
for i in range(0, len(stim)):
    avgRates[0, i] = neuron1[:, i].sum()
    avgRates[1, i] = neuron2[:, i].sum()
    avgRates[2, i] = neuron3[:, i].sum()
    avgRates[3, i] = neuron4[:, i].sum()
avgRates = avgRates/100

# Find rmax for each neuron
rmax1 = max(avgRates[0])
rmax2 = max(avgRates[1])
rmax3 = max(avgRates[2])
rmax4 = max(avgRates[3])

d1 = stim[list(avgRates[0]).index(rmax1)]
d2 = stim[list(avgRates[1]).index(rmax2)]
d3 = stim[list(avgRates[2]).index(rmax3)]
d4 = stim[list(avgRates[3]).index(rmax4)]

print("Neuron 1:\n", "\trmax:", round(rmax1, 2), "Hz\n", "\tOrientation:", d1, "degrees\n")
print("Neuron 2:\n", "\trmax:", round(rmax2, 2), "Hz\n", "\tOrientation:", d2, "degrees\n")
print("Neuron 3:\n", "\trmax:", round(rmax3, 2), "Hz\n", "\tOrientation:", d3, "degrees\n")
print("Neuron 4:\n", "\trmax:", round(rmax4, 2), "Hz\n", "\tOrientation:", d4, "degrees\n")

# Plot the tuning curves
plt.plot(stim, avgRates[0], label="neuron 1")
plt.plot(stim, avgRates[1], label="neuron 2")
plt.plot(stim, avgRates[2], label="neuron 3")
plt.plot(stim, avgRates[3], label="neuron 4")
plt.xlabel("Velocity Direction (degrees)")
plt.ylabel("Mean Firing Rate (Hz)")
plt.suptitle("Tuning Curves of Cricket Cercal Cells")
plt.legend()
plt.show()

# Poisson firing test
def poisson(response):
    response = response*10
    mu = [st.mean(response[:,i]) for i in range(len(stim))]
    var = [st.pvariance(response[:,i]) for i in range(len(stim))]
    Fano = [var[i]/mu[i] for i in range(len(mu)) if not mu[i] == 0]
    Fano.sort(key=lambda x: abs(x-1))
    return(Fano[0])

Fanos = [poisson(neuron1), poisson(neuron2), poisson(neuron3), poisson(neuron4)]
print("Neuron", Fanos.index(max(Fanos))+1, "is not Poisson.")


# Load in data
# Population coding
# Responses to mystery stimulus for each neuron and their basis vectors
with open('pop_coding_3.4.pickle', 'rb') as f:
    data = pickle.load(f)

r1 = st.mean(data['r1'])
r2 = st.mean(data['r2'])
r3 = st.mean(data['r3'])
r4 = st.mean(data['r4'])
c1 = data['c1']
c2 = data['c2']
c3 = data['c3']
c4 = data['c4']

# Compute population vector
vpop = (r1/rmax1*c1) + (r2/rmax2*c2) + (r3/rmax3*c3) + (r4/rmax4*c4)
angle = np.arctan(vpop[1]/vpop[0]) # Convert from Cartesian
angle = angle*(180/np.pi) # Convert to degrees
angle = 90 - angle # 0 is positive y axis, 90 is positive x axis
print("The mystery stimulus is", round(angle), "degrees.")

# Plotting
# Preferred directions
xaxis = np.array([d1, d2, d3, d4])
# Normalized firing rates
yaxis = np.array([r1/rmax1, r2/rmax2, r3/rmax3, r4/rmax4])
# On average, r/rmax = cos(s - sa)
avgResponse = [np.cos((angle*np.pi/180)-x) for x in np.arange(np.pi/4, 7*np.pi/4, 0.01)]
# Replace impossible negative firing rates with 0
avgResponse = [0 if x < 0 else x for x in avgResponse]
plt.scatter(xaxis, yaxis, c=['blue', 'orange', 'green', 'red'])
plt.plot(np.linspace(45, 315, 472), avgResponse, color='black', ls='--', label="avg response")
plt.xlabel("Preferred Orientation")
plt.xticks([d1, angle, d2, d3, d4], [d1, 's', d2, d3, d4])
plt.ylabel("ra/rmax")
plt.suptitle("Population Response to Mystery Stimulus")
plt.legend()
plt.show()

# Population and basis vectors
fig, ax = plt.subplots()
ax.arrow(0, 0, np.sqrt(2)/2, np.sqrt(2)/2, color='blue', head_width=.03)
ax.arrow(0, 0, np.sqrt(2)/2, -np.sqrt(2)/2, color='orange', head_width=.03)
ax.arrow(0, 0, -np.sqrt(2)/2, -np.sqrt(2)/2, color='green', head_width=.03)
ax.arrow(0, 0, -np.sqrt(2)/2, np.sqrt(2)/2, color='red', head_width=.03)
ax.arrow(0, 0, (r1/rmax1*c1)[0], (r1/rmax1*c1)[1], head_width=.03)
ax.arrow(0, 0, (r2/rmax2*c2)[0], (r2/rmax2*c2)[1], head_width=.03)
ax.arrow(0, 0, vpop[0], vpop[1], head_width=.03)
ax.set_title('Population Vector')