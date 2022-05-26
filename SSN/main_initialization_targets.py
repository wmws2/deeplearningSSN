import numpy as np
import os
import time

inputh = np.load('./parameters/inputh.npy')[:,:50]
targetmean = np.load('./parameters/targetmean.npy')
targetcov = np.load('./parameters/targetcov.npy')

n=1
inith = inputh[:,:1]
initmean = targetmean[:,:1]
initcov = targetcov[:,:1,:1]

np.save('./parameters/inith.npy',inith)
np.save('./parameters/initmean.npy',initmean)
np.save('./parameters/initcov.npy',initcov)

print(np.shape(inith))