import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from main_settings import *
from function_eta import construct_eta
from function_W import construct_W
from function_h import construct_h
from function_Tinv import compute_Tinv

inputh = np.load('./parameters/inputh.npy').astype(np.float32)[:,:50]
targetmean = np.load('./parameters/targetmean.npy').astype(np.float32)
targetcov = np.load('./parameters/targetcov.npy').astype(np.float32)

h = tf.Variable(np.load('./parameters/simexpand_50_h.npy'), dtype=tf.float32)
w = tf.Variable(np.load('./parameters/simexpand_50_w.npy'), dtype=tf.float32)
n = tf.Variable(np.load('./parameters/simexpand_50_n.npy'), dtype=tf.float32)
trainable_variables = [w,h,n]
optimizer = tf.optimizers.Adam(learning_rate=0.01)

N=25

tfTinv = tf.linalg.diag(tf.concat([tf.ones(2*N)*100, tf.ones(2*N)*50],axis=0))
inputh2 = inputh[:,25-N:25+N]
targetmean2 = targetmean[:,25-N:25+N]
targetcov2 = targetcov[:,25-N:25+N,25-N:25+N]

@tf.function
def train(subbatchsize):
	tfn = construct_eta(n)
	tfh = construct_h(inputh2, h)
	tfW = construct_W(w)

	eta = tfn@tf.random.normal([5,4*N,subbatchsize])
	u = tf.zeros([5,4*N,subbatchsize], dtype=tf.float32)

	for t in tf.range(2000):
		du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
		u = tf.clip_by_value(u + du,-30,30)
		eta = eps1*eta + eps2*tfn@tf.random.normal([5,4*N,subbatchsize])

	with tf.GradientTape() as tape:
		tfn = construct_eta(n)
		tfh = construct_h(inputh2, h)
		tfW = construct_W(w)
		uall = tf.TensorArray(dtype=tf.float32, size=500)
		for t in tf.range(500):
			du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
			u = tf.clip_by_value(u + du,-30,30)
			eta = eps1*eta + eps2*tfn@tf.random.normal([5,4*N,subbatchsize])		
			uall = uall.write(t, u)

		uall = uall.stack() #[time,patterns,neurons,trials]
		umean = tf.reduce_mean(tf.reduce_mean(uall[:,:,:2*N], axis=3),axis=0) #[patterns,neurons]
		ucov = tf.reduce_mean(tfp.stats.covariance(uall[:,:,:2*N], sample_axis=3, event_axis=2),axis=0) #[patterns,neurons,neurons]
		ucovbias = tf.square(ucov - targetcov2)
		uvarbias = tf.linalg.diag_part(ucovbias)
		cost = 1e-02 * tf.reduce_mean(tf.square(umean- targetmean2)) + 2e-02*tf.reduce_mean(uvarbias) + 1e-02*tf.reduce_mean(tf.square(ucov - targetcov2))
		gradients = tape.gradient(cost, trainable_variables)

	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return cost

for i in range(10000):
	
	cost = train(500)
	tf.print(i,cost)

	if i%20==0:
		np.save('./parameters/simfull_' + str(i)+'_w.npy',w.numpy())
		np.save('./parameters/simfull_' + str(i)+'_h.npy',h.numpy())
		np.save('./parameters/simfull_' + str(i)+'_n.npy',n.numpy())




