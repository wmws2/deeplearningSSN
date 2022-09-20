import tensorflow as tf
import numpy as np
import os
import time
from main_settings import *

def compute_Tinv(nexc,ninh):

	exc = tf.ones([nexc])*(1/tau_e)
	inh = tf.ones([ninh])*(1/tau_i)

	Tinvcolumn = tf.concat([exc, inh], axis=0)
	Tinvblock = tf.linalg.diag(Tinvcolumn)

	return [Tinvblock]
	