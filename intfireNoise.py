# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 02:08:34 2022

@author: almar
"""

from __future__ import print_function
"""
Created on Wed Apr 22 16:02:53 2015

Basic integrate-and-fire neuron 
R Rao 2007

translated to Python by rkp 2015
"""

import numpy as np
import matplotlib.pyplot as plt


# input current
I = 1 # nA

# capacitance and leak resistance
C = 1 # nF
R = 40 # M ohms

# I & F implementation dV/dt = - V/RC + I/C
# Using h = 1 ms step size, Euler method

V = -70
tstop = 200
abs_ref = 5 # absolute refractory period 
ref = 0 # absolute refractory period counter
V_trace = []  # voltage trace for plotting
V_th = -55 # spike threshold
spiketimes = [] # list of spike times

# input current
noiseamp = 1 # amplitude of added noise
I += noiseamp*np.random.normal(0, 1, (tstop,)) # nA; Gaussian noise

for t in range(tstop):
  
   if not ref:
       V = V - (V/(R*C)) + (I[t]/C)
   else:
       ref -= 1
       V = -70 # reset voltage
   
   if V > V_th:
       V = 30 # emit spike
       spiketimes = np.append(spiketimes, t)
       ref = abs_ref # set refractory counter

   V_trace += [V]


plt.plot(V_trace)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.suptitle('Integrate-and-fire Neuron')
plt.show()

ISIs = np.diff(spiketimes)
plt.hist(ISIs)
plt.xlabel("Interspike Interval (ms)")
plt.suptitle("Interspike Interval Distribution")