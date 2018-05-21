# coding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from module import *

		
class Net(object):
	def __init__(self, sess, args):
		self.sess = sess
		self.test_gan = args.test_gan 
		self.save_num = args.save_num 
		self.layer1 = args.layer1
		self.layer2 = args.layer2
		self.noise_size = args.noise_size
		self.epoch = args.epoch
		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.drop_rate = args.drop_rate
		self.actv_fc = args.actv_fc
		if self.actv_fc == 'relu':
			self.func_num = 0
		elif self.actv_fc == 'lrelu':
			self.func_num = 1
		else:
			self.func_num = 2
		

		
		self.mnist = input_data.read_data_sets('./data/mnist/', one_hot = True)
		self._build_net()
		print('\n***Hyperparmeter setting***\n')
		print('Number of layer 1 : {}'.format(self.layer1))
		print('Number of layer 2 : {}'.format(self.layer2))
		print('Epoch : {}'.format(self.epoch))
		print('batch_size : {}'.format(self.batch_size))
		print('learning_rate : {}'.format(self.learning_rate))
		print('drop_rate : {}'.format(self.drop_rate))

		
		
		print('Network ready!')
		
		
	def _build_net(self):
		self.X = tf.placeholder(tf.float32, [None, 784])
		self.Z = tf.placeholder(tf.float32, [None, self.noise_size])
		self.is_training = tf.placeholder(tf.bool)
		
		self.keep_prob = tf.placeholder(tf.float32)		
		
		
		# make network(Generator and Discriminator)
		self.G = G_network(self.Z, self.func_num, self.layer1,
			self.layer2, 'Generator')
		self.D_real = D_network(self.X, self.func_num, self.layer1,
			self.layer2, self.keep_prob, self.is_training, 'Discriminator')	
		self.D_fake = D_network(self.G, self.func_num, self.layer1,
			self.layer2, self.keep_prob, self.is_training, 'Discriminator', reuse = True)
		
		# define loss function
		self.loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits = self.D_real, labels = tf.ones_like(self.D_real)))
		self.loss_D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits = self.D_fake, labels = tf.zeros_like(self.D_fake)))
		
		self.loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			logits = self.D_fake, labels = tf.ones_like(self.D_fake)))
		
		self.loss_D = self.loss_D_fake + self.loss_D_real
		
		self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		self.disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')
		
		self.d_optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
		self.g_optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
		
		
		self.disc_grad = self.d_optimizer.compute_gradients(self.loss_D, self.disc_vars)
		self.gener_grad = self.g_optimizer.compute_gradients(self.loss_G, self.gen_vars)
		
		self.update_d = self.d_optimizer.apply_gradients(self.disc_grad)
		self.update_g = self.d_optimizer.apply_gradients(self.gener_grad)
		
	
		
	def train(self):
		self.init_op = tf.global_variables_initializer()
		self.sess.run(self.init_op)
		
		
		for step in range(self.epoch):
			self.loss_val_D, self.loss_val_G = 0, 0
			self.total_batch = int(self.mnist.train.num_examples/self.batch_size)
			
			for i in range(self.total_batch):
				self.batch_xs, self.batch_ys = self.mnist.train.next_batch(self.batch_size)
				self.noise_z = make_noise(self.batch_size, self.noise_size)
				
				_, self.loss_val_D = self.sess.run([self.update_d, self.loss_D],
					feed_dict = {self.X:self.batch_xs, self.Z:self.noise_z, self.keep_prob: self.drop_rate, self.is_training : True})
				_, self.loss_val_G = self.sess.run([self.update_g, self.loss_G],
					feed_dict = {self.Z:self.noise_z})
				
					
	
			print('Epoch : {}, D_loss = {:.4f}, G_loss = {:.4f}'.format(step, self.loss_val_D, self.loss_val_G))
			if step % 10 == 0:
				self.test(step, True)
				
				self.save_dir = './gan_weights'
				self.saver = tf.train.Saver()
				
				if not os.path.exists(self.save_dir):
					os.makedirs(self.save_dir)
					
				self.save_path = self.saver.save(self.sess, self.save_dir + '/gan_weights.ckpt-' + str(step))
				print('File saved : ', self.save_path)
			
			
			
	def test(self, n_step=None, train_save = False, fig_num=10):
		
		if not(train_save):
			try:
					self.saver = tf.train.Saver()
					self.init_op = tf.global_variables_initializer()
					self.sess.run(self.init_op)
					
					self.save_path = './gan_weights/gan_weights.ckpt-' + str(self.save_num)
					self.saver.restore(self.sess, self.save_path)
					print('\n***Model restoration complete!***\n')
					
			except:
					print('\n***No model restored***\n')
					exit()
				
		self.noise = make_noise(fig_num, self.noise_size)
		self.images = self.sess.run(self.G, feed_dict = {self.Z:self.noise, self.keep_prob: 1, self.is_training : False})
		
		self.fig = plt.figure()
		
		for i in range(fig_num):
			self.snap = self.fig.add_subplot(1, fig_num, i+1)
			self.snap.set_xticks([])
			self.snap.set_yticks([])
			
		
			plt.imshow(self.images[i].reshape([28, 28]), cmap = 'gray')
		
		if train_save:
			self.save_dir = './result/train'
			
			if not os.path.exists(self.save_dir):
				os.makedirs(self.save_dir)
			
			plt.savefig(self.save_dir + '/train_fig-{}.png'.format(str(n_step)))
			plt.close(self.fig)			
		else:
				
			self.save_dir = './result/test'
		
			if not os.path.exists(self.save_dir):
				os.makedirs(self.save_dir)
		
			plt.savefig(self.save_dir + self.test_gan + '.png')
			plt.close(self.fig)	
				
			