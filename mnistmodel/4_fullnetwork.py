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

xenc_train = np.load('encodedin50.npy') #(60000,40)

ninh = 50
nexc = 50
subbatchsize = 100
lossfunction = tf.keras.losses.SparseCategoricalCrossentropy()

params = np.load('./expand/mnistexpand50_97.npy', allow_pickle=True)
w = tf.Variable(params[0], dtype=tf.float32)
h = tf.Variable(params[1], dtype=tf.float32)
n = tf.Variable(params[2], dtype=tf.float32)
wdecode = tf.Variable(params[3], dtype=tf.float32)
cdecode = tf.Variable(params[4], dtype=tf.float32)

tfTinv = compute_Tinv(nexc,ninh)
mask = np.ones([nexc+ninh,nexc+ninh])
mask[:,nexc:] *= -1
trainable_variables = [w,h,n,wdecode,cdecode]
optimizer = tf.optimizers.Adam(learning_rate=0.001)

@tf.function
def train():

    idx = tf.random.uniform(shape=[100],minval=0, maxval=59999, dtype=tf.int32)
    xenc_train_n = tf.gather(xenc_train, idx, axis=0)
    y_train_n = tf.gather(y_train, idx, axis=0)

    tfW = tf.abs(w)*mask
    tfh = construct_h(xenc_train_n, h, ninh)
    tfn = n
    u = tf.zeros([100,nexc+ninh,subbatchsize], dtype=tf.float32)
    eta = tfn@tf.random.normal([100,nexc+ninh,subbatchsize])

    for t in tf.range(2000):
        du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
        u = tf.clip_by_value(u + du,-30,30)
        eta = eps1*eta + eps2*tfn@tf.random.normal([100,nexc+ninh,subbatchsize])

    with tf.GradientTape() as tape:

        tfW = tf.abs(w)*mask
        tfh = construct_h(xenc_train_n, h, ninh)
        tfn = n
        uall = tf.TensorArray(dtype=tf.float32, size=500)

        for t in tf.range(500):
            du = tfTinv@(-u + tfh + tfW@(0.3 * tf.pow(tf.nn.relu(u),2)) + eta)*dt
            u = u + du
            u = tf.clip_by_value(u + du,-30,30)
            eta = eps1*eta + eps2*tfn@tf.random.normal([100,nexc+ninh,subbatchsize])	
            uall = uall.write(t, u)

        uall = uall.stack() #[500 100 nexc+ninh 1000]
        umean = tf.reduce_mean(uall[:,:,:nexc], axis=[0,3])
        preds = tf.nn.softmax(umean@wdecode + cdecode, axis=1)
        loss = lossfunction(y_train_n, preds)
        gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss

for i in range(100000):
    loss = train()
    if i%200 == 0:
        tf.print(i,loss)
        np.save('./expand/mnistfull50_' + str(i) + '.npy',[m.numpy() for m in trainable_variables])