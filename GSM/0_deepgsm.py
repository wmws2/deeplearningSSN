import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io

import tensorflow as tf
import tensorflow_probability as tfp

imgs = np.load('training_images.npy').astype(np.float32)
imgs = tf.expand_dims(imgs,axis=2)

N = 16
filtercount = 50
filtersize = 1024

tftheta = tf.Variable(tf.random.uniform([filtercount], minval=-np.pi, maxval=np.pi, dtype=tf.float32),name='theta')
tfx = tf.Variable(tf.random.uniform([filtercount], minval=-16, maxval=16, dtype=tf.float32),name='x')
tfy = tf.Variable(tf.random.uniform([filtercount], minval=-16, maxval=16, dtype=tf.float32),name='y')
tfscale = tf.Variable(tf.random.uniform([filtercount], minval=0.2, maxval=1, dtype=tf.float32),name='scale')
trainable_variables = [tftheta,tfx,tfy,tfscale]
optimizer = tf.optimizers.Adam(learning_rate=0.01)

@tf.function
def gabor(N,theta, x_offset, y_offset, scale):

	theta = tf.clip_by_value(theta,-np.pi,np.pi)
	x_offset = tf.clip_by_value(x_offset,-16,16)
	y_offset = tf.clip_by_value(y_offset,-16,16)
	scale = tf.clip_by_value(scale,0.2,1)

	k = 0.5/scale
	sigma = 5*scale
	gamma = 1

	[x, y] = tf.meshgrid(range(-N,N),range(-N,N))
	x = tf.cast(x,tf.float32) - x_offset
	y = tf.cast(y,tf.float32) - y_offset
	x1 = x * tf.cos(theta) + y * tf.sin(theta)
	y1 = -x * tf.sin(theta) + y * tf.cos(theta)
	gauss = tf.exp(-(gamma**2 * x1**2 + y1**2) / (2 * sigma**2))
	sinusoid = tf.cos(k * x1)

	return (gauss*sinusoid)


def train():

	with tf.GradientTape() as tape:

		filterbank = tf.TensorArray(dtype=tf.float32,size=filtercount)
		for i in tf.range(filtercount):
			filterbank = filterbank.write(i,gabor(N,tftheta[i],tfx[i],tfy[i],tfscale[i]))
		filterbank = filterbank.stack()
		filterbank = tf.reshape(filterbank,[filtercount,filtersize])
		left_inverse = tf.linalg.inv(filterbank@tf.transpose(filterbank))@filterbank

		varunex = 0
		for i in tf.range(500):
			if i%1000==0:
				tf.print(i)
			activation = left_inverse@imgs[i]
			signal = tf.reduce_sum(activation*filterbank,axis=0)
			noise = imgs[i,:,0] - signal

			sserr = tf.reduce_sum(noise**2)
			sstot = tf.reduce_sum(imgs[i,:,0]**2)

			varunex +=(sserr/sstot)/500

		gradients = tape.gradient(varunex, trainable_variables)
	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return varunex

for i in range(100000):
	cost = train()
	tf.print(i,cost)
	np.save('./parameters/gaborparameters_' + str(i) + '.npy',np.array([tftheta.numpy(),tfx.numpy(),tfy.numpy(),tfscale.numpy()]))
