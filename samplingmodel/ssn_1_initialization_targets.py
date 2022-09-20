import numpy as np
import os
import time

inputh = np.load('./parameters/inputh50000.npy')
targetmean = np.load('./parameters/targetmean50000.npy')
targetcov = np.load('./parameters/targetcov50000.npy')

n=1
inith = inputh[:,:1]
initmean = targetmean[:,:1]
initcov = targetcov[:,:1,:1]

np.save('./parameters/inith.npy',inith)
np.save('./parameters/initmean.npy',initmean)
np.save('./parameters/initcov.npy',initcov)

print(np.shape(inith))