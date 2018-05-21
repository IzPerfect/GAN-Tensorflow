# coding: utf-8

import argparse
import os
import tensorflow as tf
from model import *

parser = argparse.ArgumentParser()

parser.add_argument('--test_gan',dest = 'test_gan', default = 'None', help ='Test figure name')
parser.add_argument('--save_num',dest = 'save_num', default = 100, help ='Number of restored model', type = int)
parser.add_argument('--layer1', dest = 'layer1', default = 128, help ='nodes of layer1', type = int)
parser.add_argument('--layer2', dest = 'layer2', default = 128, help ='nodes of layer2', type = int)
parser.add_argument('--noise_size', dest = 'noise_size', default = 256, help ='noise size', type = int)
parser.add_argument('--epoch', dest = 'epoch', default =101, help ='decide epoch', type = int)
parser.add_argument('--batch_size', dest = 'batch_size', default = 100, help = 'decide batch_size', type = int)
parser.add_argument('--learning_rate', dest = 'learning_rate', default = 0.001, help = 'decide batch_size', type = float)
parser.add_argument('--drop_rate', dest = 'drop_rate', default = 0.5, help = 'decide to drop rate', type = float)
parser.add_argument('--actv_fc', dest = 'actv_fc', default = 'relu', help ='select sigmoid, relu, lrelu')

args = parser.parse_args()
# define main
def main(_):
	tfconfig = tf.ConfigProto(allow_soft_placement=True)

	with tf.Session(config=tfconfig) as sess:
		networks = Net(sess, args)
		if args.test_gan != 'None':
			networks.test()
		else:
			networks.train()
		


if __name__ == '__main__':
    tf.app.run()
