import numpy as np
import os
import time
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import multivariate_normal


CL = np.load('feature_cl_norm.npy')
C = (np.tril(CL)@np.tril(CL).T).astype(np.float64)
# C = np.load('feature_cov_initial.npy')
diag = np.sqrt(np.diag(C))
diag1 = np.expand_dims(diag,axis=0)
diag2 = np.expand_dims(diag,axis=1)

# # C/=(diag1*diag2)


# Full covariance
fig,ax = plt.subplots(1)
im = ax.imshow(C)
ax.axis('off')
fig.colorbar(im)
plt.show()
plt.plot(C)
plt.show()