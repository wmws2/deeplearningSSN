import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io
from scipy.ndimage import rotate

filter_bank = np.load('filter_bank.npy')
filtercount = np.shape(filter_bank)[0]
filtersize = np.shape(filter_bank)[1]
filter_bank_flat = filter_bank.reshape(filtercount,filtersize*filtersize)
left_inverse = np.linalg.inv(filter_bank_flat@filter_bank_flat.T)@filter_bank_flat

#----------------------------------------------------------------------
imgs = np.load('training_images.npy')
fvus = []
for i in range(50000):
	if i%1000 == 0:
		print(i)
		print(np.mean(fvus))
	recon = filter_bank_flat.T@left_inverse@imgs[i]
	error = imgs[i]-recon 

	sserr = np.sum(error**2)
	sstot = np.sum(imgs[i]**2)
	fvus.append(sserr/sstot)

	# fig,ax = plt.subplots(2)
	# ax[0].imshow(imgs[i].reshape(32,32))
	# ax[1].imshow(recons.reshape(32,32))
	# plt.show()

np.save('image_bank_fvu.npy',np.array(fvus))

#----------------------------------------------------------------------
# imgs = np.load('training_images.npy')
# activations = []
# for i in range(50000):
# 	if i%1000 == 0:
# 		print(i)
# 	activations.append(left_inverse@imgs[i])
# np.save('image_bank_activations.npy',np.array(activations))

#----------------------------------------------------------------------
# a = np.load('image_bank_activations.npy').T
# cov = np.cov(a,ddof=0)
# print(np.linalg.cond(cov))
# np.save('feature_cov_initial.npy', cov)