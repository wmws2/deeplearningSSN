import numpy as np
import tensorflow as tf

with np.load('mnist.npz') as f:
    x_train, y_train = f['x_train'].reshape(-1,28*28)/255, f['y_train']
    x_test, y_test = f['x_test'].reshape(-1,28*28)/255, f['y_test']

x_train = np.concatenate([x_train,x_test],axis=0)

dims = [784,360,120,50]

enc1 = tf.Variable(tf.random.normal(shape=[dims[0],dims[1]], stddev=np.sqrt(2/dims[1])), dtype=tf.float32)
enc2 = tf.Variable(tf.random.normal(shape=[dims[1],dims[2]], stddev=np.sqrt(2/dims[2])), dtype=tf.float32)
enc3 = tf.Variable(tf.random.uniform(shape=[dims[2],dims[3]], minval=-np.sqrt(6/(dims[2]+dims[3])), maxval=np.sqrt(6/(dims[2]+dims[3]))), dtype=tf.float32)
dec1 = tf.Variable(tf.random.normal(shape=[dims[3],dims[2]], stddev=np.sqrt(2/dims[2])), dtype=tf.float32)
dec2 = tf.Variable(tf.random.normal(shape=[dims[2],dims[1]], stddev=np.sqrt(2/dims[1])), dtype=tf.float32)
dec3 = tf.Variable(tf.random.uniform(shape=[dims[1],dims[0]], minval=-np.sqrt(6/(dims[0]+dims[1])), maxval=np.sqrt(6/(dims[0]+dims[1]))), dtype=tf.float32)

biase1 = tf.Variable(tf.random.normal(shape=[1,dims[1]]), dtype=tf.float32)
biase2 = tf.Variable(tf.random.normal(shape=[1,dims[2]]), dtype=tf.float32)
biase3 = tf.Variable(tf.random.normal(shape=[1,dims[3]]), dtype=tf.float32)
biasd1 = tf.Variable(tf.random.normal(shape=[1,dims[2]]), dtype=tf.float32)
biasd2 = tf.Variable(tf.random.normal(shape=[1,dims[1]]), dtype=tf.float32)
biasd3 = tf.Variable(tf.random.normal(shape=[1,dims[0]]), dtype=tf.float32)

trainable_variables = [enc1, enc2, enc3, dec1, dec2, dec3, biase1, biase2, biase3, biasd1, biasd2, biasd3]
optimizer = tf.optimizers.Adam(learning_rate=0.001)
lossfunction = tf.keras.losses.BinaryCrossentropy()

@tf.function
def train():

    with tf.GradientTape() as tape:

        enclayer1 = tf.nn.relu(x_train@enc1 + biase1)
        enclayer2 = tf.nn.relu(enclayer1@enc2 + biase2)
        enclayer3 = tf.math.sigmoid(enclayer2@enc3 + biase3)

        declayer1 = tf.nn.relu(enclayer3@dec1 + biasd1)
        declayer2 = tf.nn.relu(declayer1@dec2 + biasd2)
        declayer3 = tf.math.sigmoid(declayer2@dec3 + biasd3)

        cost = lossfunction(x_train,declayer3)

        gradients = tape.gradient(cost, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return cost,enclayer3

for i in tf.range(50000):
    cost, encoded = train()
    if i%1000==0:
        tf.print(i,cost)

np.save('variablesin50.npy', [i.numpy() for i in trainable_variables])
np.save('encodedin50.npy', encoded.numpy())