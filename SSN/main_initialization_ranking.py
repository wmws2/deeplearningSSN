import numpy as np
import os
import time
import matplotlib.pyplot as plt

costs = np.load('./parameters/initcosts.npy')
hs = np.load('./parameters/iniths.npy')
ws = np.load('./parameters/initws.npy')

print(np.shape(ws))

topws = np.zeros([20,2,2])
tophs = np.zeros([20,3])
topcosts = np.zeros([20])

for i in range(10):
	idx = np.argmin(costs[i])
	topws[i] = ws[i,idx]
	tophs[i] = hs[i,idx]
	topcosts[i] = costs[i,idx]

ranking = np.argsort(topcosts)
print(topcosts[ranking[10:]])
np.save('./parameters/topinitws.npy',topws[ranking[10:]])
np.save('./parameters/topiniths.npy',tophs[ranking[10:]])