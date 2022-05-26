import tensorflow as tf
import numpy as np
import os
import time

#Duration of one time step
dt = 0.0002

#Time constant for excitatory neurons
tau_e = 0.020

#Time constant for inhibitory neurons 
tau_i = 0.010

#Correlation time constant for noise
tau_eta = 0.02	

#Noise coefficients
eps1 = (1.0-dt/tau_eta)
eps2 = np.sqrt(2.0*dt/tau_eta)

#Number of patterns
patterns = 5
npops = 2

#Firing rate = k * [relu(membrane potential) ^ alpha]
k = 0.3
alpha = 2