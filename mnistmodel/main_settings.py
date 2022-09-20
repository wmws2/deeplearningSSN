import tensorflow as tf
import numpy as np
import os
import time

#Load target moments 
# targetmean = tf.constant(np.load("targetmean.npy"), dtype=tf.float32)
# targetcov = tf.constant(np.load("targetcov.npy"), dtype=tf.float32)
# targetinput = tf.constant(np.load("targetinput.npy"), dtype=tf.float32)

# targetmean = tf.expand_dims(targetmean, axis=2)
# targetcov = tf.expand_dims(targetcov, axis=3)
# targetinput = tf.expand_dims(targetinput, axis=2)

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

#Firing rate = k * [relu(membrane potential) ^ alpha]
k = 0.3
alpha = 2