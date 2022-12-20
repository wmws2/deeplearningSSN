import numpy as np
import os
import time
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import multivariate_normal

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
folder = 'rebuttal/'


imgs = np.load('training_images.npy')
x = imgs
A = np.load(folder+'filter_bank_norm.npy').reshape(-1,32*32).T
C = np.load(folder+'feature_C_norm.npy')
sigma_x = np.load(folder+'noise_std.npy')
alpha = np.load(folder+'gamma_alpha.npy')
beta = np.load(folder+'gamma_beta.npy')
mpcovs = np.load(folder+'mpcovs.npy')

AT = A.T
ATA = AT@A
ACAT = A@C@AT

Nz = 500
z_axis = np.linspace(0.01,1.5,Nz)
gammadist = tfd.Normal(loc=alpha, scale=beta)
p_z = tf.expand_dims(gammadist.log_prob(z_axis), axis=0)
mixtureLs = tf.TensorArray(dtype=tf.float32, size=Nz)
for i in tf.range(Nz):
	mixtureLs = mixtureLs.write(i,tf.linalg.cholesky(z_axis[i]**2 * ACAT + sigma_x**2 * tf.eye(32*32)))
mixtureLs = mixtureLs.stack()
mixture = tfd.MultivariateNormalTriL(scale_tril=mixtureLs)

posteriorzs = []
posteriormeans = []
posteriorcovs = []

for s in range(1):

	if s%100==0:
		print(s)

	xm = x[s]
	p_z_given_x = (mixture.log_prob(xm) + p_z).numpy()[0]
	p_z_given_x = np.exp(p_z_given_x - np.max(p_z_given_x))
	p_z_given_x /= np.sum(p_z_given_x)
	plt.plot(z_axis,p_z_given_x,c='g')
	# print(z_axis[np.argmax(p_z_given_x)])

	means = np.array([z_axis[z]/(sigma_x**2) * mpcovs[z]@AT@xm for z in range(Nz)])
	posterior_mean = np.sum(np.expand_dims(p_z_given_x, axis=1) * means, axis=0)

	meanadjusted = means - np.expand_dims(posterior_mean, axis=0)
	meanmeanT = np.expand_dims(meanadjusted, axis=1) * np.expand_dims(meanadjusted, axis=2)
	posterior_cov = np.sum(np.expand_dims(p_z_given_x, axis=(1,2)) * (mpcovs + meanmeanT), axis=0)

	plt.plot(z_axis,np.mean(np.abs(mpcovs), axis=(1,2)),c='r')
	plt.plot(z_axis,np.mean(np.abs(meanmeanT), axis=(1,2)),c='b')
	plt.plot(z_axis,np.mean(np.abs(mpcovs+meanmeanT), axis=(1,2)),c='k')
	plt.show()
	posteriorzs.append(p_z_given_x)
	posteriormeans.append(posterior_mean)
	posteriorcovs.append(posterior_cov)

# np.save(folder+'posteriorzs.npy', np.array(posteriorzs))
# np.save(folder+'posteriormeans.npy', np.array(posteriormeans))
# np.save(folder+'posteriorcovs.npy', np.array(posteriorcovs))
