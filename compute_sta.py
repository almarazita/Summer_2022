"""
Created on Wed Apr 22 15:21:11 2015

Code to compute spike-triggered average.
"""

from __future__ import division
import numpy as np


def compute_sta(stim, rho, num_timesteps):
    """Compute the spike-triggered average from a stimulus and spike-train.
    
    Args:
        stim: stimulus time-series
        rho: spike-train time-series
        num_timesteps: how many timesteps to use in STA
        
    Returns:
        spike-triggered average for num_timesteps timesteps before spike"""
    
    sta = np.zeros((num_timesteps,))

    # This command finds the indices of all of the spikes that occur
    # after 300 ms into the recording.
    spike_times = rho[num_timesteps:].nonzero()[0] + num_timesteps

    # Fill in this value. Note that you should not count spikes that occur
    # before 300 ms into the recording.
    num_spikes = len(spike_times)
    
    # Compute the spike-triggered average of the spikes found.
    # To do this, compute the average of all of the vectors
    # starting 300 ms (exclusive) before a spike and ending at the time of
    # the event (inclusive). Each of these vectors defines a list of
    # samples that is contained within a window of 300 ms before each
    # spike. The average of these vectors should be completed in an
    # element-wise manner.
    #
    # My code follows
    # Collect M (num_spikes) stimulus chunks in a nested list
    s = [stim[i-num_timesteps:i+1] for i in spike_times if i >= num_timesteps]
    
    # Divide the 300 ms windows into num_timesteps bins by creating array of
    # indices to evaluate the stimulus
    # Assumes a 2 ms sampling period
    binstarts = np.linspace(1, 150, num_timesteps, dtype = int)
    
    # Sum the stimulus feature values at each timestep
    for i, time in enumerate(binstarts):
        curSum = 0
        for window in s:
            curSum += window[time]
        sta[i] = curSum
    
    # Compute the average by dividing by the total number of spikes
    sta /= num_spikes
    
    return sta