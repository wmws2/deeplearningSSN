import tensorflow as tf
import numpy as np
import os
import time

from main_settings import *

@tf.function
def construct_h(inputh, h):

	h_scale = h[0]
	h_thres = h[1]
	h_power = h[2]

	hpop = h_scale**2 * tf.exp(h_power**2 * tf.math.log(inputh+h_thres**2))
	hpops = tf.tile(hpop, [1,2])
	
	return tf.expand_dims(hpops,axis=2)