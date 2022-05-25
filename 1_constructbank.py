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
params = np.load('./parameters/gaborparameters_100.npy')
tftheta = params[0]
tfx = params[1]
tfy = params[2]
tfscale = params[3]

@tf.function
def gabor(N,theta, x_offset, y_offset, scale):

	theta = tf.clip_by_value(theta,-np.pi,np.pi)
	x_offset = tf.clip_by_value(x_offset,-16,16)
	y_offset = tf.clip_by_value(y_offset,-16,16)
	scale = tf.clip_by_value(scale,0.2,10)

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

filterbank = tf.TensorArray(dtype=tf.float32,size=filtercount)
for i in tf.range(filtercount):
	filterbank = filterbank.write(i,gabor(N,tftheta[i],tfx[i],tfy[i],tfscale[i]))
filterbank = filterbank.stack().numpy()

fig,ax = plt.subplots(5,10)
for i in range(5):
	for j in range(10):
		ax[i,j].imshow(filterbank[i*10+j],vmin=-1,vmax=1,cmap='Greys')
		ax[i,j].axis('off')
		#ax[i,j].set_title(str(tfscale[i*10+j]))
plt.show()

# np.save('filter_bank.npy',filterbank)

# filterbank = tf.reshape(filterbank,[filtercount,filtersize])
# left_inverse = tf.linalg.inv(filterbank@tf.transpose(filterbank))@filterbank

# varunex = 0
# for i in tf.range(50000):
# 	if i%1000==0:
# 		tf.print(i)
# 	activation = left_inverse@imgs[i]
# 	signal = tf.reduce_sum(activation*filterbank,axis=0)
# 	noise = imgs[i,:,0] - signal

# 	sserr = tf.reduce_sum(noise**2)
# 	sstot = tf.reduce_sum(imgs[i,:,0]**2)

# 	varunex +=(sserr/sstot)/50000

# tf.print(varunex)
