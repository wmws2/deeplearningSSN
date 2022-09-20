import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from main_settings import *
from function_h import construct_h
from function_Tinv import compute_Tinv

with np.load('mnist.npz') as f:
    x_train, y_train = f['x_train'].reshape(-1,28*28)/255, f['y_train']
    x_test, y_test = f['x_test'].reshape(-1,28*28)/255, f['y_test']

xenc_train = np.load('encodedin50.npy')[:100,:1]
y_train = y_train[:100]

neurons = 1
tfTinv = compute_Tinv(1,1)
mask = np.ones([2,2])
mask[:,1:] *= -1
subbatchsize = 1000

wdecode = tf.Variable(tf.random.uniform(shape=[1,10], minval=-np.sqrt(6/12), maxval=np.sqrt(6/12)), dtype=tf.float32)
cdecode = tf.Variable(tf.random.uniform(shape=[1,10], minval=-np.sqrt(6/12), maxval=np.sqrt(6/12)), dtype=tf.float32)
lossfunction = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.optimizers.Adam(learning_rate=0.01)

unstable = True
while unstable:

    tf.print('rerolling parameters...')

    w = tf.Variable(tf.random.normal(shape=[2,2], stddev=np.sqrt(1/2)), dtype=tf.float32)
    h = tf.Variable([tf.random.uniform([]),tf.random.uniform([]),tf.random.uniform([])], dtype=tf.float32)
    trainable_variables = [w,h,wdecode,cdecode]

    tfW = tf.abs(w)*mask
    tfh = construct_h(xenc_train, h, 1)
    eta = tf.random.normal([100,2,subbatchsize])
    u = tf.zeros([100,2,subbatchsize], dtype=tf.float32)

    for t in tf.range(2000):
        du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
        u = u + du
        eta = eps1*eta + eps2*tf.random.normal([100,2,subbatchsize])

    uall = tf.TensorArray(dtype=tf.float32, size=500)
    for t in tf.range(500):
        du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
        u = u + du
        eta = eps1*eta + eps2*tf.random.normal([100,2,subbatchsize])		
        uall = uall.write(t, u)

    uall = uall.stack() #[500 100 2 1000]
    unstable = tf.math.reduce_any(tf.math.is_nan(uall))

@tf.function
def train():

    with tf.GradientTape() as tape:

        tfW = tf.abs(w)*mask
        tfh = construct_h(xenc_train, h, 1)
        eta = tf.random.normal([100,2,subbatchsize])
        u = tf.zeros([100,2,subbatchsize], dtype=tf.float32)

        for t in tf.range(2000):
            du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
            u = tf.clip_by_value(u + du,-30,30)
            eta = eps1*eta + eps2*tf.random.normal([100,2,subbatchsize])

        uall = tf.TensorArray(dtype=tf.float32, size=500)
        for t in tf.range(500):
            du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
            u = tf.clip_by_value(u + du,-30,30)
            eta = eps1*eta + eps2*tf.random.normal([100,2,subbatchsize])		
            uall = uall.write(t, u)

        uall = uall.stack() #[500 100 2 1000]
        umean = tf.reduce_mean(uall[:,:,:1], axis=[0,3])
        preds = tf.nn.softmax(umean@wdecode + cdecode, axis=1)
        loss = lossfunction(y_train, preds)
        gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss

for i in range(3000):
    loss = train()

    if i%10 == 0:
        tf.print(loss)
        tosave = [i.numpy() for i in trainable_variables]
        np.save('mnistinit50.npy',tosave)

