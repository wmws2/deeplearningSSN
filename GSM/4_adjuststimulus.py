import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.io

# C = np.load('feature_cov_initial.npy')

# filter_bank = np.load('filter_bank.npy')

# scale = np.sqrt(np.diag(C))

# filter_bank *= np.expand_dims(scale,axis=(1,2))
# C /= np.expand_dims(scale,axis=(1))*np.expand_dims(scale,axis=(0))

# np.save('feature_covnorm_initial.npy',C)
# np.save('filter_banknorm_initial.npy',filter_bank)

#------------------------------------------------------------------------------------

CL = np.load('feature_cl.npy')
C = np.tril(CL)@np.tril(CL).T

print(np.linalg.cond(C))
plt.plot(np.linalg.inv(C)@C)
plt.show()

filter_bank = np.load('filter_banknorm_initial.npy')

scale = np.sqrt(np.diag(C))

filter_bank *= np.expand_dims(scale,axis=(1,2))
C /= np.expand_dims(scale,axis=(1))*np.expand_dims(scale,axis=(0))
plt.imshow(C)
plt.show()

print(np.linalg.cond(C))
plt.plot(np.linalg.inv(C)@C)
plt.show()
np.save('feature_C_norm.npy',C)
np.save('filter_bank_norm.npy',filter_bank)


