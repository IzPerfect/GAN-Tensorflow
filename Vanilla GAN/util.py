import tensorflow as tf
import numpy as np


def make_noise(batch_size, noise_size):
	return np.random.randn(batch_size, noise_size)

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def select_fn(num):
	if num == 0:
		activation_fn = tf.nn.relu
	elif num == 1:
		activation_fn = tf.sigmoid
	else:
		activation_fn = lrelu
		
	return activation_fn