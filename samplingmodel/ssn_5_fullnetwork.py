import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from main_settings import *
from function_h import construct_h
from function_Tinv import compute_Tinv

inputh = np.load('./parameters/inputh50000.npy').astype(np.float32)[:,:50]
targetmean = np.load('./parameters/targetmean50000.npy').astype(np.float32)
targetcov = np.load('./parameters/targetcov50000.npy').astype(np.float32)

# h = tf.Variable(np.load('./parameters/simexpand_49_h.npy'), dtype=tf.float32)
# w = tf.Variable(np.load('./parameters/simexpand_49_w.npy'), dtype=tf.float32)
# n = tf.Variable(np.load('./parameters/simexpand_49_n.npy'), dtype=tf.float32)
h = tf.Variable(np.load('./parameters/simfull_640_h.npy'), dtype=tf.float32)
w = tf.Variable(np.load('./parameters/simfull_640_w.npy'), dtype=tf.float32)
n = tf.Variable(np.load('./parameters/simfull_640_n.npy'), dtype=tf.float32)
trainable_variables = [w,h,n]
optimizer = tf.optimizers.Adam(learning_rate=0.0001)


N = 50
tfTinv = compute_Tinv(npops,N)
inputh2 = inputh[:,:N]
targetmean2 = targetmean[:,:N]
targetcov2 = targetcov[:,:N,:N]
mask = np.ones([2*N,2*N])
mask[:,N:] *= -1

@tf.function
def train(subbatchsize):

	idx = tf.random.uniform(shape=[100],minval=0, maxval=39999, dtype=tf.int32)
	inputh3 = tf.gather(inputh2, idx, axis=0)
	targetmean3 = tf.gather(targetmean2, idx, axis=0)
	targetcov3 = tf.gather(targetcov2, idx, axis=0)

	gradients = [w*0.,h*0.,n*0.]
	cost = 0.

	for j in tf.range(5):

		tf.print(j,end='\r')
		tfn = n
		tfh = construct_h(inputh3[j*20:j*20+20], h)
		tfW = tf.abs(w)*mask
		eta = tfn@tf.random.normal([20,2*N,subbatchsize])
		u = tf.zeros([20,2*N,subbatchsize], dtype=tf.float32)

		for t in tf.range(2000):
			du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
			u = tf.clip_by_value(u + du,-30,30)
			eta = eps1*eta + eps2*tfn@tf.random.normal([20,2*N,subbatchsize])

		with tf.GradientTape() as tape:
			tfn = n
			tfh = construct_h(inputh3[j*20:j*20+20], h)
			tfW = tf.abs(w)*mask
			uall = tf.TensorArray(dtype=tf.float32, size=500)
			for t in tf.range(500):
				du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
				u = tf.clip_by_value(u + du,-30,30)
				eta = eps1*eta + eps2*tfn@tf.random.normal([20,2*N,subbatchsize])		
				uall = uall.write(t, u)

			uall = uall.stack() #[time,patterns,neurons,trials]
			umean = tf.reduce_mean(tf.reduce_mean(uall[:,:,:N], axis=3),axis=0) #[patterns,neurons]
			ucov = tf.reduce_mean(tfp.stats.covariance(uall[:,:,:N], sample_axis=3, event_axis=2),axis=0) #[patterns,neurons,neurons]
			ucovbias = tf.square(ucov - targetcov3[j*20:j*20+20])
			uvarbias = tf.linalg.diag_part(ucovbias)
			tempcost = 1e-02 * tf.reduce_mean(tf.square(umean- targetmean3[j*20:j*20+20])) + 2e-00*tf.reduce_mean(uvarbias) + 1e-00*tf.reduce_mean(tf.square(ucov - targetcov3[j*20:j*20+20]))
			cost += tempcost/5
			tempgradients = tape.gradient(tempcost/5, trainable_variables)
			gradients = [gradients[k] + gradient for k, gradient in enumerate(tempgradients)]

	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return cost

for i in range(100000):
	
	cost = train(100)
	tf.print(i,cost)

	if i%5==0:
		np.save('./rebuttalparameters/simfull_' + str(i)+'_w.npy',w.numpy())
		np.save('./rebuttalparameters/simfull_' + str(i)+'_h.npy',h.numpy())
		np.save('./rebuttalparameters/simfull_' + str(i)+'_n.npy',n.numpy())




