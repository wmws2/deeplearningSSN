import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io
from scipy.ndimage import rotate

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

training_images = np.zeros([50000,1024])

for i in range(5):
	print(i)
	img = unpickle('CIFAR-10/data_batch_' + str(i+1)) #dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
	img = img[b'data']
	R, G, B = img[:,:1024], img[:,1024:2048], img[:,2048:3072]
	training_images[i*10000:(i+1)*10000] = 0.2989 * R + 0.5870 * G + 0.1140 * B

training_images = (training_images - np.mean(training_images, axis=1, keepdims=True))/np.std(training_images, axis=1, keepdims=True)
np.save('training_images.npy',training_images)

img = unpickle('CIFAR-10/test_batch') #dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
img = img[b'data']
R, G, B = img[:,:1024], img[:,1024:2048], img[:,2048:3072]
test_images = 0.2989 * R + 0.5870 * G + 0.1140 * B

test_images = (test_images - np.mean(test_images, axis=1, keepdims=True))/np.std(test_images, axis=1, keepdims=True)
np.save('test_images.npy',test_images)