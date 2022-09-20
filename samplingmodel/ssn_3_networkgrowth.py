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

optimizer = tf.optimizers.Adam(learning_rate=0.01)

def train(N,subbatchsize):

	if N==1:
		rank=0
		h = tf.Variable(np.load('./parameters/topiniths.npy')[rank], dtype=tf.float32)
		w = tf.Variable(np.load('./parameters/topinitws.npy')[rank], dtype=tf.float32)
		n = tf.Variable(0.9*np.identity(2) + 0.1*np.ones([2,2]),dtype=tf.float32)

	else:
		h = tf.Variable(np.load('./parameters/simexpand_' + str(N-1) + '_h.npy'), dtype=tf.float32)
		w = tf.Variable(np.load('./parameters/simexpand_' + str(N-1) + '_w.npy'), dtype=tf.float32)
		n = tf.Variable(np.load('./parameters/simexpand_' + str(N-1) + '_n.npy'), dtype=tf.float32)

	trainable_variables = [w,h,n]

	if N<4:
		lr=0.01
	else:
		lr=0.001
	optimizer = tf.optimizers.Adam(learning_rate=lr)

	tfTinv = compute_Tinv(npops,N)
	inputh2 = inputh[:,:N]
	targetmean2 = targetmean[:,:N]
	targetcov2 = targetcov[:,:N,:N]

	mask = np.ones([2*N,2*N])
	mask[:,N:] *= -1


	if N<5:
		multiplier = int(20)
	elif N<10:
		multiplier = int(10)
	elif N<20:
		multiplier = int(5)
	else:
		multiplier = int(4)

	for i in tf.range(50):
		gradients = [w*0.,h*0.,n*0.]
		cost = 0.

		idx = tf.random.uniform(shape=[100],minval=0, maxval=49999, dtype=tf.int32)
		inputh3 = tf.gather(inputh2, idx, axis=0)
		targetmean3 = tf.gather(targetmean2, idx, axis=0)
		targetcov3 = tf.gather(targetcov2, idx, axis=0)

		for j in range(int(20/multiplier)):
			tf.print(j,end='\r')
			tfn = n
			tfh = construct_h(inputh3[j*5*multiplier:j*5*multiplier+5*multiplier], h)
			tfW = tf.abs(w)*mask

			eta = tfn@tf.random.normal([5*multiplier,2*N,subbatchsize])
			u = tf.zeros([5*multiplier,2*N,subbatchsize], dtype=tf.float32)
			for t in tf.range(2000):
				du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
				u = tf.clip_by_value(u + du,-100,100)
				eta = eps1*eta + eps2*tfn@tf.random.normal([5*multiplier,2*N,subbatchsize])

			with tf.GradientTape() as tape:
				tfn = n
				tfh = construct_h(inputh3[j*5*multiplier:j*5*multiplier+5*multiplier], h)
				tfW = tf.abs(w)*mask
				uall = tf.TensorArray(dtype=tf.float32, size=500)
				for t in tf.range(500):
					du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
					u = tf.clip_by_value(u + du,-100,100)
					eta = eps1*eta + eps2*tfn@tf.random.normal([5*multiplier,2*N,subbatchsize])		
					uall = uall.write(t, u)

				uall = uall.stack() #[time,patterns,neurons,trials]
				umean = tf.reduce_mean(tf.reduce_mean(uall[:,:,:N], axis=3),axis=0) #[patterns,neurons]
				ucov = tf.reduce_mean(tfp.stats.covariance(uall[:,:,:N], sample_axis=3, event_axis=2),axis=0) #[patterns,neurons,neurons]
				ucovbias = tf.square(ucov - targetcov3[j*5*multiplier:j*5*multiplier+5*multiplier])
				uvarbias = tf.linalg.diag_part(ucovbias)

				tempcost = 1e-02 * tf.reduce_mean(tf.square(umean- targetmean3[j*5*multiplier:j*5*multiplier+5*multiplier])) + 2e-00*tf.reduce_mean(uvarbias) + 1e-00 * tf.reduce_mean(tf.square(ucov - targetcov3[j*5*multiplier:j*5*multiplier+5*multiplier]))
				cost += tempcost/20*multiplier
				tempgradients = tape.gradient(tempcost/20*multiplier, trainable_variables)
				gradients = [gradients[k] + gradient for k, gradient in enumerate(tempgradients)]

		optimizer.apply_gradients(zip(gradients, trainable_variables))
		tf.print(N,i,cost)

	return cost,w,h,n

for N in range(1,50):
	
	cost,w,h,n = train(N,100)

	if N<50:
		chosen = N-1
		topleft = w[:chosen,:chosen]
		topmid = w[:chosen,chosen:chosen+1]
		topright = w[:chosen,chosen+1:]
		midleft = w[chosen:chosen+1,:chosen]
		midmid = w[chosen:chosen+1,chosen:chosen+1]
		midright = w[chosen:chosen+1,chosen+1:]
		botleft = w[chosen+1:,:chosen]
		botmid = w[chosen+1:,chosen:chosen+1]
		botright = w[chosen+1:,chosen+1:]

		top = tf.concat([topleft,topmid,topmid,topright],axis=1)
		mid = tf.concat([0.5*midleft,0.5*midmid,0.5*midmid,0.5*midright],axis=1)
		bot = tf.concat([botleft,botmid,botmid,botright],axis=1)
		w = tf.concat([top,mid,mid,bot],axis=0)

		chosen = 2*N
		topleft = w[:chosen,:chosen]
		topright = w[:chosen,chosen:]
		botleft = w[chosen:,:chosen]
		botright = w[chosen:,chosen:]

		top = tf.concat([topleft,topright,topright],axis=1)
		bot = tf.concat([0.5*botleft,0.5*botright,0.5*botright],axis=1)
		w = tf.concat([top,bot,bot],axis=0)

		chosen = N-1
		topleft = n[:chosen,:chosen]
		topmid = n[:chosen,chosen:chosen+1]
		topright = n[:chosen,chosen+1:]
		midleft = n[chosen:chosen+1,:chosen]
		midmid = n[chosen:chosen+1,chosen:chosen+1]
		midright = n[chosen:chosen+1,chosen+1:]
		botleft = n[chosen+1:,:chosen]
		botmid = n[chosen+1:,chosen:chosen+1]
		botright = n[chosen+1:,chosen+1:]

		top = tf.concat([topleft,topmid,topmid,topright],axis=1)
		mid = tf.concat([0.5*midleft,0.5*midmid,0.5*midmid,0.5*midright],axis=1)
		bot = tf.concat([botleft,botmid,botmid,botright],axis=1)
		n = tf.concat([top,mid,mid,bot],axis=0)

		chosen = 2*N
		topleft = n[:chosen,:chosen]
		topright = n[:chosen,chosen:]
		botleft = n[chosen:,:chosen]
		botright = n[chosen:,chosen:]

		top = tf.concat([topleft,topright,topright],axis=1)
		bot = tf.concat([0.5*botleft,0.5*botright,0.5*botright],axis=1)
		n = tf.concat([top,bot,bot],axis=0)

	tf.print(tf.shape(w),tf.shape(n))
	np.save('./parameters/simexpand_' + str(N) + '_w.npy',w.numpy())
	np.save('./parameters/simexpand_' + str(N) + '_h.npy',h.numpy())
	np.save('./parameters/simexpand_' + str(N) + '_n.npy',n.numpy())




