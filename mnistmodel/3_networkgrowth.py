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

def expand(w,chosen):

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

    return w

xenc_train = np.load('encodedin50.npy')[:100] #(60000,40)
y_train = y_train[:100]

ninh = 1
nexc = 1
subbatchsize = 100
lossfunction = tf.keras.losses.SparseCategoricalCrossentropy()

for i in range(98):

    if i==0:
        [w_,h_,wdecode_,cdecode_] = np.load('mnistinit.npy', allow_pickle=True)
        n_ = 0.9*np.identity(2) + 0.1*np.ones([2,2])
        w = tf.Variable(w_, dtype=tf.float32)
        h = tf.Variable(h_, dtype=tf.float32)
        n = tf.Variable(n_, dtype=tf.float32)
        wdecode = tf.Variable(wdecode_, dtype=tf.float32)
        cdecode = tf.Variable(cdecode_, dtype=tf.float32)

    else:
        params = np.load('./expand/mnistexpand50_' + str(i-1) + '.npy', allow_pickle=True)
        w = tf.Variable(params[0], dtype=tf.float32)
        h = tf.Variable(params[1], dtype=tf.float32)
        n = tf.Variable(params[2], dtype=tf.float32)
        wdecode = tf.Variable(params[3], dtype=tf.float32)
        cdecode = tf.Variable(params[4], dtype=tf.float32)

    tfTinv = compute_Tinv(nexc,ninh)
    mask = np.ones([nexc+ninh,nexc+ninh])
    mask[:,nexc:] *= -1
    xenc_train_n = xenc_train[:,:nexc]
    trainable_variables = [w,h,n,wdecode,cdecode]

    if i<5:
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
    else:
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

    for j in tf.range(10):

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
            loss = lossfunction(y_train, preds)
            gradients = tape.gradient(loss, trainable_variables)

        tf.print(i,j,nexc,ninh,loss)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    if nexc < ninh:
        w = expand(w, nexc-1)
        n = expand(n, nexc-1)
        wdecode = tf.concat([wdecode[:nexc-1], 0.5*wdecode[nexc-1:nexc], 0.5*wdecode[nexc-1:nexc]],axis=0)
        nexc += 1
    else:
        w = expand(w, nexc+ninh-1)
        n = expand(n, nexc+ninh-1)
        ninh += 1
    
    trainable_variables = [w,h,n,wdecode,cdecode]
    np.save('./mnistexpand50_' + str(i) + '.npy',[m.numpy() for m in trainable_variables])