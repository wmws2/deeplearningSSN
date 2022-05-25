import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io



#--------------------------------------------------------------------------
means = np.load('./parameters/posteriormeans_50000.npy')
covs = np.load('./parameters/posteriorcovs_50000.npy')

thres = -np.amin(means)
print(thres)
means = np.load('./parameters/posteriormeans_rodrigo.npy')
covs = np.load('./parameters/posteriorcovs_rodrigo.npy')

stims = 5

# targetmean = np.zeros([stims,50])
# targetcov = np.zeros([stims,50,50])

# trials = 1

# for i in range(stims):
# 	for n in range(trials):
# 		print(i,n)
# 		mean = np.expand_dims(means[i],axis=1)
# 		L = np.linalg.cholesky(covs[i])
# 		samples = L@np.random.randn(50,100000)+mean

# 		samples_thres = samples+thres
# 		#samples_thres[samples_thres<0] = 0
# 		#samples = scale*(samples_thres**power)

# 		targetmean[i] += np.mean(samples_thres,axis=1)/trials
# 		targetcov[i] += np.cov(samples_thres)/trials

# targetvar = np.array([np.diag(targetcov[i]) for i in range(stims)])
# fig,ax = plt.subplots(3)
# ax[0].plot(targetmean.T)
# ax[1].plot(targetvar.T)
np.save('targetmeanrodrigo.npy',means+thres)
np.save('targetcovrodrigo.npy',covs)

filter_bank = np.load('filter_bank_norm.npy')
filtercount = np.shape(filter_bank)[0]
filtersizex = np.shape(filter_bank)[1]
filtersizey = np.shape(filter_bank)[2]
filter_bank_flat = filter_bank.reshape(filtercount,filtersizex*filtersizey)
left_inverse = np.linalg.inv(filter_bank_flat@filter_bank_flat.T)@filter_bank_flat

x = np.load('rodrigo_images.npy')
inputh = np.zeros([stims,50])
for i in range(stims):
		inputh[i] = left_inverse@np.ndarray.flatten(x[i])

inputh -= np.amin(inputh[:100])
np.save('inputhrodrigo.npy',inputh)



#--------------------------------------------------------------------------

# filter_bank = np.load('filter_bank.npy')
# filtercount = np.shape(filter_bank)[0]
# filtersizex = np.shape(filter_bank)[1]
# filtersizey = np.shape(filter_bank)[2]
# filter_bank_flat = filter_bank.reshape(filtercount,filtersizex*filtersizey)
# left_inverse = np.linalg.inv(filter_bank_flat@filter_bank_flat.T)@filter_bank_flat

# x = np.load('finalstimulus2.npy')
# inputh = np.zeros([5,60])
# for k in range(5):

# 	xm = np.copy(x[k])
# 	xm = np.ndarray.flatten(xm)
# 	inputh[k] = left_inverse@xm
# #inputh[inputh<0] = 0
# #inputh[inputh<0] *= -1
# plt.plot(inputh.T,'r')
# plt.show()
# np.save('finalh2.npy',inputh)

#--------------------------------------------------------------------------

# means = np.load('finalmeans2.npy')
# covs = np.load('finalcovs2.npy')
# targetmean = np.zeros([5,60])
# targetcov = np.zeros([5,60,60])
# trials = 5

# for i in range(5):
# 	for n in range(trials):
# 		print(i,n)
# 		mean = np.expand_dims(means[i],axis=1)
# 		L = np.linalg.cholesky(covs[i])
# 		samples = L@np.random.randn(60,100000)+mean

# 		samples_thres = samples+thres
# 		samples_thres[samples_thres<0] = 0

# 		samples = scale*(samples_thres**power)

# 		targetmean[i] += np.mean(samples,axis=1)/trials
# 		targetcov[i] += np.cov(samples)/trials

# targetvar = np.array([np.diag(targetcov[i]) for i in range(5)])
# fig,ax = plt.subplots(3)
# ax[0].plot(targetmean.T,c='r')
# ax[1].plot(targetvar.T,c='r')
# plt.show()