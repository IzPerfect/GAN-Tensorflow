import tensorflow as tf
import numpy as np
from util import *



def G_network(Z, func_num, layer1, layer2, name = 'Generator'):
	with tf.variable_scope(name):
		fn = select_fn(func_num)
		L1 = tf.layers.dense(Z, layer1, activation = fn)
		L2 = tf.layers.dense(L1, layer2, activation = fn)
		L3 = tf.layers.dense(L2, 784, activation = fn)
		
		return L3
		

def D_network(input, func_num, layer1, layer2, keep_prob, is_training, name = 'Discriminator', reuse = None):
	with tf.variable_scope(name) as d_net:
		if reuse:
			d_net.reuse_variables()
			
		fn = select_fn(func_num)
		L1 = tf.layers.dense(input, layer1, activation = fn)
		L1 = tf.layers.dropout(L1, keep_prob, is_training)
		L2 = tf.layers.dense(L1, layer2, activation = fn)
		L2 = tf.layers.dropout(L2, keep_prob, is_training)
		L3 = tf.layers.dense(L2, 1, activation = None)
		L3 = tf.layers.dropout(L3, keep_prob, is_training)
		
		return L3				