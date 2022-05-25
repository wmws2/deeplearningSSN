import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io
from scipy.stats import gamma

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

x = np.expand_dims(np.load('training_images.npy'), axis=1).astype(np.float32) #(50000, 1024)
filter_bank = np.load('filter_banknorm_initial.npy')
filtercount = np.shape(filter_bank)[0]
filtersize = np.shape(filter_bank)[1]
A = filter_bank.reshape(filtercount,filtersize*filtersize).T
CL = np.load('feature_cl.npy')
C = np.tril(CL)@np.tril(CL).T
CL = tf.Variable(CL, dtype=tf.float32)
sigma_x = tf.Variable(np.load('noise_std.npy'), dtype=tf.float32)
alpha = tf.Variable(np.load('gamma_alpha.npy'), dtype=tf.float32)
beta = tf.Variable(np.load('gamma_beta.npy'), dtype=tf.float32)

AT = A.T
ATA = AT@A
ACAT = A@C@AT
Nz = 100
zs = tf.linspace(0.1,1,Nz)

CL_ = tf.linalg.band_part(CL, -1, 0)
C = CL_@tf.transpose(CL_)
tf.print(tf.linalg.inv(C)@C)
ACAT = A@C@AT

idx = tf.random.uniform(shape=[10],minval=0, maxval=49999, dtype=tf.int32)
x_input = tf.gather(x, idx, axis=0)
tf.print(idx)
gammadist = tfd.Gamma(concentration=alpha, rate=beta)
logprior = tf.expand_dims(gammadist.log_prob(zs), axis=0)

mixtureLs = tf.TensorArray(dtype=tf.float32, size=Nz)
for i in tf.range(Nz):
	mixtureLs = mixtureLs.write(i,tf.linalg.cholesky(zs[i]**2 * ACAT + sigma_x**2 * tf.eye(32*32)))
mixtureLs = mixtureLs.stack()

mixture = tfd.MultivariateNormalTriL(scale_tril=mixtureLs)
logprobs = mixture.log_prob(x_input) + logprior

fig,ax = plt.subplots(10,2)
for i in range(10):
	ax[i,0].imshow(x_input[i].numpy().reshape(32,32))
	ax[i,1].plot(zs,tf.transpose(mixture.log_prob(x_input)[i]))
plt.show()

logprobs_baseline = tf.reduce_max(logprobs, axis=1, keepdims=True)
logprobs_sum = tf.math.log(tf.reduce_sum(tf.exp(logprobs-logprobs_baseline),axis=1)) + logprobs_baseline[:,0]
nllh = -tf.reduce_mean(logprobs_sum)



