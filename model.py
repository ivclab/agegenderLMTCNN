""" Rude Carnie: Age and Gender Deep Learning with Tensorflow found at
https://github.com/dpressel/rude-carnie
"""
# ==============================================================================
# MIT License
#
# Modifications copyright (c) 2018 Image & Vision Computing Lab, Institute of Information Science, Academia Sinica
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.contrib.layers import *
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
import tensorflow.contrib.slim as slim

TOWER_NAME = 'tower'

def select_model(name):

	if name.startswith('inception'):
		print('selected (fine-tuning) inception model')
		return inception_v3
	elif name == 'levi_hassner_bn':
		print('selected levi hassner batch norm model')
		return levi_hassner_bn
	elif name == 'levi_hassner':
		print('selected levi hassner model')
		return levi_hassner
	elif name == 'mobilenet_multitask':
		print('selected mobilenet multitask model')
		return mobilenet_multitask
	elif name == 'LMTCNN-1-1':
		print('selected dp multitask model')
		return dp_multitask
	elif name == 'LMTCNN-2-1':
		print('selected dp multitask 2 1 model')
		return dp_multitask_2_1

def get_checkpoint(checkpoint_path, requested_step=None, basename='checkpoint'):

	if requested_step is not None:
		model_checkpoint_path = '%s/%s-%s' % (checkpoint_path, basename, requested_step)
		if os.path.exists(model_checkpoint_path) is None:
			print('No checkpoint file found at [%s]' % checkpoint_path)
			exit(-1)
			print(model_checkpoint_path)
		print(model_checkpoint_path)
		return model_checkpoint_path, requested_step

	ckpt = tf.train.get_checkpoint_state(checkpoint_path)
	if ckpt and ckpt.model_checkpoint_path:
		# Restore checkpoint as described in top of this program
		print(ckpt.model_checkpoint_path)
		global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		return ckpt.model_checkpoint_path, global_step
	else:
		print('No checkpoint file found at [%s]' % checkpoint_path)
		exit(-1)

def _activation_summary(x):
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inception_v3(nlabels, images, pkeep, is_training):

	batch_norm_params = {
		"is_training": is_training,
		"trainable": True,
		# Decay for the moving averages.
		"decay": 0.9997,
		# Epsilon to prevent 0s in variance.
		"epsilon": 0.001,
		# Collection containing the moving mean and moving variance.
		"variables_collections": {
			"beta": None,
			"gamma": None,
			"moving_mean": ["moving_vars"],
			"moving_variance": ["moving_vars"],
		}
	}

	weight_decay = 0.00004
	stddev = 0.1
	weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
	with tf.variable_scope("InceptionV3","InceptionV3",[images]) as scope:

		with tf.contrib.slim.arg_scope(
			[tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected],
			weights_regularizer=weights_regularizer,
			trainable=True):

			with tf.contrib.slim.arg_scope(
				[tf.contrib.slim.conv2d],
				weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
				activation_fn=tf.nn.relu,
				normalizer_fn=batch_norm,
				normalizer_params=batch_norm_params):

				net, end_points = inception_v3_base(images, scope=scope)
				with tf.variable_scope("logits"):
					shape = net.get_shape()
					net = avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
					net = tf.nn.dropout(net, pkeep, name='droplast')
					net = flatten(net, scope="flatten")

	with tf.variable_scope('output') as scope:
		weights = tf.Variable(tf.truncated_normal([2048, nlabels], mean=0.0, stddev=0.01), name='weights')
		biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
		output = tf.add(tf.matmul(net, weights), biases, name=scope.name)
		_activation_summary(output)

	return output

def levi_hassner_bn(nlabels, images, pkeep, is_training):

	batch_norm_params = {
		"is_training": is_training,
		"trainable": True,
		# Decay for the moving averages.
		"decay": 0.9997,
		# Epsilon to prevent 0s in variance.
		"epsilon": 0.001,
		# Collection containing the moving mean and moving variance.
		"variables_collections": {
			"beta": None,
			"gamma": None,
			"moving_mean": ["moving_vars"],
			"moving_variance": ["moving_vars"],
		}
	}

	weight_decay = 0.0005
	weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

	with tf.variable_scope("LeviHassnerBN", "LeviHassnerBN", [images]) as scope:

		with tf.contrib.slim.arg_scope(
			[convolution2d, fully_connected],
			weights_regularizer=weights_regularizer,
			biases_initializer=tf.constant_initializer(1.),
			weights_initializer=tf.random_normal_initializer(stddev=0.005),
			trainable=True):

			with tf.contrib.slim.arg_scope(
				[convolution2d],
				weights_initializer=tf.random_normal_initializer(stddev=0.01),
				normalizer_fn=batch_norm,
				normalizer_params=batch_norm_params):

				conv1 = convolution2d(images, 96, [7,7], [4, 4], padding='VALID', biases_initializer=tf.constant_initializer(0.), scope='conv1')
				pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
				conv2 = convolution2d(pool1, 256, [5, 5], [1, 1], padding='SAME', scope='conv2')
				pool2 = max_pool2d(conv2, 3, 2, padding='VALID', scope='pool2')
				conv3 = convolution2d(pool2, 384, [3, 3], [1, 1], padding='SAME', biases_initializer=tf.constant_initializer(0.), scope='conv3')
				pool3 = max_pool2d(conv3, 3, 2, padding='VALID', scope='pool3')
				# can use tf.contrib.layer.flatten
				flat = tf.reshape(pool3, [-1, 384*6*6], name='reshape')
				full1 = fully_connected(flat, 512, scope='full1')
				drop1 = tf.nn.dropout(full1, pkeep, name='drop1')
				full2 = fully_connected(drop1, 512, scope='full2')
				drop2 = tf.nn.dropout(full2, pkeep, name='drop2')

	with tf.variable_scope('output') as scope:
		weights = tf.Variable(tf.random_normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
		biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
		output = tf.add(tf.matmul(drop2, weights), biases, name=scope.name)

	return output

def levi_hassner(nlabels, images, pkeep, is_training):

	weight_decay = 0.0005
	weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

	with tf.variable_scope("LeviHassner", "LeviHassner", [images]) as scope:

		with tf.contrib.slim.arg_scope(
			[convolution2d, fully_connected],
			weights_regularizer=weights_regularizer,
			biases_initializer=tf.constant_initializer(1.),
			weights_initializer=tf.random_normal_initializer(stddev=0.005),
			trainable=True):

			with tf.contrib.slim.arg_scope(
				[convolution2d],
				weights_initializer=tf.random_normal_initializer(stddev=0.01)):

				conv1 = convolution2d(images, 96, [7,7], [4, 4], padding='VALID', biases_initializer=tf.constant_initializer(0.), scope='conv1')
				pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
				norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, beta=0.75, name='norm1')
				conv2 = convolution2d(norm1, 256, [5, 5], [1, 1], padding='SAME', scope='conv2') 
				pool2 = max_pool2d(conv2, 3, 2, padding='VALID', scope='pool2')
				norm2 = tf.nn.local_response_normalization(pool2, 5, alpha=0.0001, beta=0.75, name='norm2')
				conv3 = convolution2d(norm2, 384, [3, 3], [1, 1], biases_initializer=tf.constant_initializer(0.), padding='SAME', scope='conv3')
				pool3 = max_pool2d(conv3, 3, 2, padding='VALID', scope='pool3')
				flat = tf.reshape(pool3, [-1, 384*6*6], name='reshape')
				full1 = fully_connected(flat, 512, scope='full1')
				drop1 = tf.nn.dropout(full1, pkeep, name='drop1')
				full2 = fully_connected(drop1, 512, scope='full2')
				drop2 = tf.nn.dropout(full2, pkeep, name='drop2')

	with tf.variable_scope('output') as scope:
		weights = tf.Variable(tf.random_normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
		biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
		output = tf.add(tf.matmul(drop2, weights), biases, name=scope.name)

	return output

def mobilenet(inputs, width_multiplier=1, scope=None, is_training=True):

	def _depthwise_separable_conv(inputs, num_pwc_filters, width_multiplier,
		sc, downsample=False):
		# helper function to build the depthwise separable convolution layer.
		num_pwc_filters = round(num_pwc_filters*width_multiplier)
		_stride = 2 if downsample else 1
		# skip pointwise by setting num_outputs = None
		depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=_stride,
			depth_multiplier=1, kernel_size=[3,3], scope=sc+'/depthwise_conv')
		bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')

		pointwise_conv = slim.convolution2d(bn, num_pwc_filters, kernel_size=[1, 1], scope=sc+'/pointwise_conv')
		bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
		return bn

	with tf.variable_scope(scope) as sc:
		end_points_collection = sc.name + '_end_points'

		with slim.arg_scope(
			[slim.convolution2d, slim.separable_convolution2d],
			activation_fn=None, outputs_collections=[end_points_collection]):

			with slim.arg_scope([slim.batch_norm],
				is_training=is_training, activation_fn=tf.nn.relu):

				net = slim.convolution2d(inputs, round(32*width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
				net = slim.batch_norm(net, scope='conv_1/batch_norm')
				net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
				net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
				net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
				net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
				net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
				net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')

				net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
				net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
				net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
				net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
				net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')

				net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
				net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
				net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')

			end_points = slim.utils.convert_collection_to_dict(end_points_collection)
			net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
			end_points['squeeze'] = net

			return net, end_points

def mobilenet_multitask(nlabels1, images1, nlabels2, images2, pkeep, is_training):

	batch_norm_params = {
		"is_training": is_training,
		"trainable": True,
		# decay for the moving averages.
		"decay": 0.9997,
		# epsilon to prevent 0s in variance.
		"epsilon": 0.001,
		# collection containing the moving mean and moving variance.
		"variables_collections": {
			"beta": None,
			"gamma": None,
			"moving_mean": ["moving_vars"],
			"moving_variance": ["moving_vars"],
		}
	}

	weight_decay = 0.00004
	stddev = 0.1
	weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

	with tf.variable_scope("MobileNetmultitaskls ", "MobileNetmultitask", [images1]) as scope:

		with slim.arg_scope(
			[slim.convolution2d, slim.separable_convolution2d],
			weights_initializer=slim.initializers.xavier_initializer(),
			biases_initializer=slim.init_ops.zeros_initializer(),
			weights_regularizer=slim.l2_regularizer(weight_decay),
			trainable=True):

			net, end_points = mobilenet(images1, width_multiplier=1, scope=scope, is_training=is_training)

	with tf.variable_scope('ageoutput') as scope:
		ageweights = tf.Variable(tf.truncated_normal([1024, nlabels1], mean=0.0, stddev=0.01), name='ageweights')
		agebiases = tf.Variable(tf.constant(0.0, shape=[nlabels1], dtype=tf.float32), name='agebiases')
		ageoutput = tf.add(tf.matmul(net, ageweights), agebiases, name=scope.name)
		_activation_summary(ageoutput)

	with tf.variable_scope('genderoutput') as scope:
		genderweights = tf.Variable(tf.truncated_normal([1024, nlabels2], mean=0.0, stddev=0.01), name='genderweights')
		genderbiases = tf.Variable(tf.constant(0.0, shape=[nlabels2], dtype=tf.float32), name='genderbiases')
		genderoutput = tf.add(tf.matmul(net, genderweights), genderbiases, name=scope.name)
		_activation_summary(genderoutput)

	return ageoutput, genderoutput

def dp_multitask(nlabels1, images1, nlabels2, images2, pkeep, is_training):

	def _depthwise_separable_conv(inputs, num_pwc_filters, width_multiplier,
		sc, downsample=False):
		# helper function to build the depthwise separable convolution layer.
		num_pwc_filters = round(num_pwc_filters*width_multiplier)
		_stride = 2 if downsample else 1
		# skip pointwise by setting num_outputs = None
		depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=_stride,
			depth_multiplier=1, kernel_size=[3,3], scope=sc+'/depthwise_conv')
		bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')

		pointwise_conv = slim.convolution2d(bn, num_pwc_filters, kernel_size=[1, 1], scope=sc+'/pointwise_conv')
		bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
		return bn

	weight_decay = 0.0005
	weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
	with tf.variable_scope("multitaskdpcnn", "multitaskdpcnn", [images1]) as scope:

		with tf.contrib.slim.arg_scope(
			[convolution2d, fully_connected],
			weights_regularizer=weights_regularizer,
			biases_initializer=tf.constant_initializer(1.),
			weights_initializer=tf.random_normal_initializer(stddev=0.005),
			trainable=True):

			with slim.arg_scope(
				[slim.convolution2d, slim.separable_convolution2d],
				weights_initializer=slim.initializers.xavier_initializer(),
				biases_initializer=slim.init_ops.zeros_initializer(),
				weights_regularizer=slim.l2_regularizer(weight_decay),
				trainable=True):

				conv1 = convolution2d(images1, 96, [7,7], [4, 4], padding='VALID', 
					biases_initializer=tf.constant_initializer(0.), scope='conv1')
				pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
				norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, 
					beta=0.75, name='norm1')
				#convdp1 = _depthwise_separable_conv(norm1, 256, 1.0 , downsample=True,sc='convdp1')
				convdp1 = _depthwise_separable_conv(norm1, 256, 1 , downsample=False,sc='convdp1') 
				convdp2 = _depthwise_separable_conv(convdp1, 384, 1 , downsample=True ,sc='convdp2')
				pool3 = max_pool2d(convdp2, 3, 2, padding='VALID', scope='pool3')
				#flat = tf.reshape(pool3, [-1, 384*3*3], name='reshape')
				flat = tf.reshape(pool3, [-1, 384*6*6], name='reshape')
				full1 = fully_connected(flat, 512, scope='full1')
				drop1 = tf.nn.dropout(full1, pkeep, name='drop1')
				full2 = fully_connected(drop1, 512, scope='full2')
				drop2 = tf.nn.dropout(full2, pkeep, name='drop2')

	with tf.variable_scope('ageoutput') as agescope:
		ageweights = tf.Variable(tf.random_normal([512, nlabels1], mean=0.0, stddev=0.01), name='ageweights', trainable=True)
		agebiases = tf.Variable(tf.constant(0.0, shape=[nlabels1], dtype=tf.float32), name='agebiases', trainable=True)
		ageoutput = tf.add(tf.matmul(drop2, ageweights), agebiases, name=agescope.name)

	with tf.variable_scope('genderoutput') as genderscope:
		genderweights = tf.Variable(tf.random_normal([512, nlabels2], mean=0.0, stddev=0.01), name='genderweights')
		genderbiases = tf.Variable(tf.constant(0.0, shape=[nlabels2], dtype=tf.float32), name='genderbiases')
		genderoutput = tf.add(tf.matmul(drop2, genderweights), genderbiases, name=genderscope.name)

	return ageoutput, genderoutput

def dp_multitask_2_1(nlabels1, images1, nlabels2, images2, pkeep, is_training):

	def _depthwise_separable_conv(inputs, num_pwc_filters, width_multiplier,
		sc, downsample=False):
		# helper function to build the depthwise separable convolution layer.
		num_pwc_filters = round(num_pwc_filters*width_multiplier)
		_stride = 2 if downsample else 1
		# skip pointwise by setting num_outputs = None
		depthwise_conv = slim.separable_convolution2d(inputs, num_outputs=None, stride=_stride,
			depth_multiplier=1, kernel_size=[3,3], scope=sc+'/depthwise_conv')
		bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')

		pointwise_conv = slim.convolution2d(bn, num_pwc_filters, kernel_size=[1, 1], scope=sc+'/pointwise_conv')
		bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
		return bn

	weight_decay = 0.0005
	weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
	with tf.variable_scope("multitaskdpcnn", "multitaskdpcnn", [images1]) as scope:

		with tf.contrib.slim.arg_scope(
			[convolution2d, fully_connected],
			weights_regularizer=weights_regularizer,
			biases_initializer=tf.constant_initializer(1.),
			weights_initializer=tf.random_normal_initializer(stddev=0.005),
			trainable=True):

			with slim.arg_scope(
				[slim.convolution2d, slim.separable_convolution2d],
				weights_initializer=slim.initializers.xavier_initializer(),
				biases_initializer=slim.init_ops.zeros_initializer(),
				weights_regularizer=slim.l2_regularizer(weight_decay),
				trainable=True):

				conv1 = convolution2d(images1, 96, [7,7], [4, 4], padding='VALID', 
					biases_initializer=tf.constant_initializer(0.), scope='conv1')
				pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
				norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, 
					beta=0.75, name='norm1')
				#convdp1 = _depthwise_separable_conv(norm1, 256, 1.0 , downsample=True,sc='convdp1')
				convdp1 = _depthwise_separable_conv(norm1, 256, 2 , downsample=False,sc='convdp1') 
				convdp2 = _depthwise_separable_conv(convdp1, 384, 1 , downsample=True ,sc='convdp2')
				pool3 = max_pool2d(convdp2, 3, 2, padding='VALID', scope='pool3')
				#flat = tf.reshape(pool3, [-1, 384*3*3], name='reshape')
				flat = tf.reshape(pool3, [-1, 384*6*6], name='reshape')
				full1 = fully_connected(flat, 512, scope='full1')
				drop1 = tf.nn.dropout(full1, pkeep, name='drop1')
				full2 = fully_connected(drop1, 512, scope='full2')
				drop2 = tf.nn.dropout(full2, pkeep, name='drop2')

	with tf.variable_scope('ageoutput') as agescope:
		ageweights = tf.Variable(tf.random_normal([512, nlabels1], mean=0.0, stddev=0.01), name='ageweights', trainable=True)
		agebiases = tf.Variable(tf.constant(0.0, shape=[nlabels1], dtype=tf.float32), name='agebiases', trainable=True)
		ageoutput = tf.add(tf.matmul(drop2, ageweights), agebiases, name=agescope.name)

	with tf.variable_scope('genderoutput') as genderscope:
		genderweights = tf.Variable(tf.random_normal([512, nlabels2], mean=0.0, stddev=0.01), name='genderweights')
		genderbiases = tf.Variable(tf.constant(0.0, shape=[nlabels2], dtype=tf.float32), name='genderbiases')
		genderoutput = tf.add(tf.matmul(drop2, genderweights), genderbiases, name=genderscope.name)

	return ageoutput, genderoutput