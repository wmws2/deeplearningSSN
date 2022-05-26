import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io



#--------------------------------------------------------------------------
means = np.load('./parameters/posteriormeans_50000.npy')
covs = np.load('./parameters/posteriorcovs_50000.npy')

thres = -np.amin(means)
stims = 5

targetmean = np.zeros([stims,50])
targetcov = np.zeros([stims,50,50])

trials = 1

for i in range(stims):
	for n in range(trials):
		print(i,n)
		mean = np.expand_dims(means[i],axis=1)
		L = np.linalg.cholesky(covs[i])
		samples = L@np.random.randn(50,100000)+mean
		samples_thres = samples+thres
		targetmean[i] += np.mean(samples_thres,axis=1)/trials
		targetcov[i] += np.cov(samples_thres)/trials

targetvar = np.array([np.diag(targetcov[i]) for i in range(stims)])
fig,ax = plt.subplots(3)
ax[0].plot(targetmean.T)
ax[1].plot(targetvar.T)
np.save('targetmean50000.npy',means+thres)
np.save('targetcov50000.npy',covs)

filter_bank = np.load('filter_bank_norm.npy')
filtercount = np.shape(filter_bank)[0]
filtersizex = np.shape(filter_bank)[1]
filtersizey = np.shape(filter_bank)[2]
filter_bank_flat = filter_bank.reshape(filtercount,filtersizex*filtersizey)
left_inverse = np.linalg.inv(filter_bank_flat@filter_bank_flat.T)@filter_bank_flat

x = np.load('trainingimages.npy')
inputh = np.zeros([stims,50])
for i in range(stims):
		inputh[i] = left_inverse@np.ndarray.flatten(x[i])

inputh -= np.amin(inputh[:100])
np.save('inputh50000.npy',inputh)




