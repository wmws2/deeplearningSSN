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
C = np.load('feature_covnorm_initial.npy')
CL = tf.Variable(np.linalg.cholesky(C), dtype=tf.float32)
sigma_x = tf.Variable(1.0, dtype=tf.float32)
alpha = tf.Variable(2, dtype=tf.float32)
beta = tf.Variable(0.5, dtype=tf.float32)

AT = A.T
ATA = AT@A
ACAT = A@C@AT
Nz = 100
zs = tf.linspace(0.1,10,Nz)


trainable_variables = [CL, sigma_x, alpha, beta]
optimizer = tf.optimizers.Adam(learning_rate=0.1)
batchsize = 100
@tf.function
def train():

	gradients = [CL*0., sigma_x*0., alpha*0., beta*0.]
	cost = 0.
	CL_ = tf.linalg.band_part(CL, -1, 0)
	C = CL_@tf.transpose(CL_)
	ACAT = A@C@AT
	for j in tf.range(batchsize):
		tf.print(j,end='\r')
		idx = tf.random.uniform(shape=[5],minval=0, maxval=49999, dtype=tf.int32)
		x_input = tf.gather(x, idx, axis=0)

		with tf.GradientTape() as tape:

			gammadist = tfd.Gamma(concentration=alpha, rate=beta)
			logprior = tf.expand_dims(gammadist.log_prob(zs), axis=0)

			CL_ = tf.linalg.band_part(CL, -1, 0)
			C = CL_@tf.transpose(CL_)
			ACAT = A@C@AT

			mixtureLs = tf.TensorArray(dtype=tf.float32, size=Nz)
			for i in tf.range(Nz):
				mixtureLs = mixtureLs.write(i,tf.linalg.cholesky(zs[i]**2 * ACAT + sigma_x**2 * tf.eye(32*32)))
			mixtureLs = mixtureLs.stack()

			mixture = tfd.MultivariateNormalTriL(scale_tril=mixtureLs)
			logprobs = mixture.log_prob(x_input) + logprior
			logprobs_baseline = tf.reduce_max(logprobs, axis=1, keepdims=True)
			logprobs_sum = tf.math.log(tf.reduce_sum(tf.exp(logprobs-logprobs_baseline),axis=1)) + logprobs_baseline[:,0]
			tempcost = -tf.reduce_mean(logprobs_sum)
			tempgradients = tape.gradient(tempcost, trainable_variables)
			cost += tempcost
			gradients = [gradients[j] + gradient for j, gradient in enumerate(tempgradients)]

	gradients = [x/batchsize for x in gradients]
	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return cost/batchsize, tf.reduce_mean(tf.linalg.diag_part(ACAT))*12/(sigma_x**2)

snrs=[]
costs=[]
for i in range(10000):
	
	cost, snr = train()
	print(i, '|', cost.numpy() ,'|', snr.numpy(), sigma_x.numpy(), alpha.numpy(), beta.numpy())
	snrs.append(snr.numpy())
	costs.append(cost.numpy())
	np.save('snrs.npy',snrs)
	np.save('costs.npy',costs)
	np.save('feature_cl.npy',CL.numpy())
	np.save('noise_std.npy',sigma_x.numpy())
	np.save('gamma_alpha.npy',alpha.numpy())
	np.save('gamma_beta.npy',beta.numpy())

