import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import time

from main_settings import *
from function_h import construct_h
from function_Tinv import compute_Tinv

inputh = np.load('./parameters/inith.npy').astype(np.float32) #[100,1]
targetmean = tf.constant(np.load('./parameters/initmean.npy'),dtype=tf.float32) #[100,1]
targetcov = tf.constant(np.load('./parameters/initcov.npy'),dtype=tf.float32) #[100,1,1]
targetvar = np.array([np.diag(targetcov[i]) for i in range(100)])

neurons = 1

tfTinv = compute_Tinv(npops,neurons)

dt = 0.0002
tau_eta = 0.02	
eps1 = (1.0-dt/tau_eta)
eps2 = np.sqrt(2.0*dt/tau_eta)

w = tf.Variable(tf.random.normal([2,2])*0.1, dtype=tf.float32)
h = tf.Variable([tf.random.normal([])*0.1,tf.random.normal([])*0.1,tf.random.normal([])], dtype=tf.float32)

inhvar = tf.constant(targetvar, dtype=tf.float32)
inhmean = tf.constant(targetmean, dtype=tf.float32)

trainable_variables = [w,h]
optimizer = tf.optimizers.Adam(learning_rate=0.01)

mask = np.ones([2,2])
mask[:,1:] *= -1

@tf.function
def trainw(inhmean,inhvar):

	cost = 0.0
	for i in tf.range(50):
		with tf.GradientTape() as tape:
			targetvar2 = tf.concat([targetvar,inhvar],axis=1) #[5,4]
			targetvar2 = tf.cast(tf.expand_dims(targetvar2,axis=2), tf.float32) #[5,4,1]
			targetmean2 = tf.concat([targetmean,inhmean],axis=1) #[5,4]
			targetmean2 = tf.cast(tf.expand_dims(targetmean2,axis=2), tf.float32) #[5,4,1]
			tfh = construct_h(inputh, h)
			tfW = tf.abs(w)*mask

			cost = tf.reduce_mean((targetmean2 - tfh + 0.3*tfW@(targetmean2**2 + targetvar2))**2)
			gradients = tape.gradient(cost, trainable_variables)

		optimizer.apply_gradients(zip(gradients, trainable_variables))

	return cost

@tf.function
def updateinh(subbatchsize):

	tfW = tf.abs(w)*mask
	tfh = construct_h(inputh, h)
	eta = tf.random.normal([100,2,subbatchsize])
	u = tf.zeros([100,2,subbatchsize], dtype=tf.float32)
	for t in tf.range(2000):
		du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
		u = u + du
		eta = eps1*eta + eps2*tf.random.normal([100,2,subbatchsize])

	uall = tf.TensorArray(dtype=tf.float32, size=2000)
	for t in tf.range(2000):
		du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
		u = u + du
		eta = eps1*eta + eps2*tf.random.normal([100,2,subbatchsize])		
		uall = uall.write(t, u)

	uall = uall.stack() #[time,patterns,neurons,trials]

	newinhmean = tf.reduce_mean(tf.reduce_mean(uall[:,:,1:],axis=3),axis=0)#[patterns,neurons]
	newinhvar = tf.reduce_mean(tf.math.reduce_variance(uall[:,:,1:],axis=3),axis=0) #[patterns,neurons,neurons]

	umean = tf.reduce_mean(tf.reduce_mean(uall[:,:,:1], axis=3),axis=0) #[patterns,neurons]
	ucov = tf.reduce_mean(tfp.stats.covariance(uall[:,:,:1], sample_axis=3, event_axis=2),axis=0) #[patterns,neurons,neurons]
	ucovbias = tf.square(ucov - targetcov)
	uvarbias = tf.linalg.diag_part(ucovbias)
	cost = 1e-02 * tf.reduce_mean(tf.square(umean- targetmean)) + 2e-02*tf.reduce_mean(uvarbias) + 1e-02 * tf.reduce_mean(tf.square(ucov - targetcov))

	return newinhmean,newinhvar,cost

iterations = 10
times = 20

costs = np.ones([iterations,times])*np.nan
ws = np.ones([iterations,times,2,2])*np.nan
hs = np.ones([iterations,times,3])*np.nan

for j in range(iterations):

	currentcost = np.nan
	tf.print(j,"Trying...")

	while tf.math.is_nan(currentcost):

		scale = tf.random.uniform([])*2
		inhvar = scale*scale*tf.constant(targetvar, dtype=tf.float32)
		inhmean = scale*tf.constant(targetmean, dtype=tf.float32)
		w.assign(tf.random.normal([2,2])*0.1)
		h.assign([tf.random.normal([])*0.1,tf.random.normal([])*0.1,tf.random.normal([])])

		for var in optimizer.variables():
			var.assign(tf.zeros_like(var))

		inhmean,inhvar,currentcost = updateinh(2000)

	tf.print(j,"Found!")
	costs[j,0] = currentcost.numpy()
	ws[j,0] = tf.identity(w).numpy()
	hs[j,0] = tf.identity(h).numpy()
	tf.print(j,0, currentcost)

	for i in range(1,times):
		trainw(inhmean,inhvar)
		inhmean,inhvar,cost = updateinh(2000)
		tf.print(j,i, cost, currentcost)
		if tf.math.is_nan(cost)==True:
			break
		if cost < currentcost:
		 	currentcost = cost

		costs[j,i] = cost.numpy()
		ws[j,i] = tf.identity(w).numpy()
		hs[j,i] = tf.identity(h).numpy()

	np.save('./parameters/initcosts.npy',costs)
	np.save('./parameters/initws.npy',ws)
	np.save('./parameters/iniths.npy',hs)
