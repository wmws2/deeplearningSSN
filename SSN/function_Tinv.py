import tensorflow as tf
import numpy as np
import os
import time
from main_settings import *

@tf.function
def compute_Tinv(npops,neurons):

	exc = tf.ones([neurons])*(1/tau_e)
	inh = tf.ones([neurons])*(1/tau_i)

	Tinvcolumn = tf.concat([exc, inh], axis=0)
	Tinvblock = tf.linalg.diag(Tinvcolumn)

	return [Tinvblock]
	