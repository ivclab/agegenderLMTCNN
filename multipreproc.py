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
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from datetime import datetime
import os
import random
import sys
import threading
import numpy as np 
import tensorflow as tf 
import json

from pdb import set_trace as bp

RESIZE_HEIGHT= 256
RESIZE_WIDTH = 256

tf.app.flags.DEFINE_string('fold_dir','./DataPreparation/FiveFolds/train_val_test_per_fold_agegender',
	'Fold directories')
tf.app.flags.DEFINE_string('data_dir','./adiencedb/aligned',
	'Data directory')
tf.app.flags.DEFINE_string('tf_output_dir','./tfrecord',
	'tfrecord output directory')

tf.app.flags.DEFINE_string('train_list','agegender_train.txt',
	'Training list')
tf.app.flags.DEFINE_string('valid_list','agegender_val.txt',
	'Validation list')
tf.app.flags.DEFINE_string('test_list','agegender_test.txt',
	'Testing list')

tf.app.flags.DEFINE_integer('train_shards',10,
	'number of shards in training tfrecord files.')
tf.app.flags.DEFINE_integer('valid_shards',2,
	'number of shards in validation tfrecord files.')
tf.app.flags.DEFINE_integer('test_shards',2,
	'number of shards in testing tfrecord files.')

tf.app.flags.DEFINE_integer('num_threads',2,
	'number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
	# wrapper for inserting int64 features into example proto.
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
	# wrapper for inserting bytes features into example proto.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_buffer, agelabel, genderlabel, height, width):
	# build an example proto for an example.
	# filename: string, path to an image file, eg. '/path/to/example.jpg'
	# image_buffer: string, JPEG encoding of RGB image
	# agelabel: integer, identifier for the ground truth for the network
	# genderlabel: integer, identifier for the ground truth for the network
	# height: integer, image height in pixels
	# width: integer, image width in pixels
	# returns:
	# example proto

	example = tf.train.Example(features=tf.train.Features(feature={
		'image/ageclass/label': _int64_feature(agelabel),
		'image/genderclass/label': _int64_feature(genderlabel),
		'image/filename': _bytes_feature(str.encode(os.path.basename(filename))),
		'image/encoded': _bytes_feature(image_buffer),
		'image/height': _int64_feature(height),
		'image/width': _int64_feature(width)
		}))

	return example

def _is_png(filename):
	# determine if a file contains a PNG format image.
	# filename: string, path of the image file.
	# returns:
	# boolean indicating if the image is a PNG.
	return '.png' in filename

def _process_image(filename, coder):
	# process a single image file.
	# filename: string, path to an image file eg. '/path/to/example.jpg'
	# coder: instance of ImageCoder to provide tensorflow image coding utils.
	# returns:
	# image_buffer: string, JPEG encoding of RGB image.
	# height: integer, image height in pixels.
	# width: integer, image width in pixels.

	# read the image file.
	with tf.gfile.FastGFile(filename, 'rb') as f:
		image_data = f.read()

	# convert any png to jpeg's for consistency.
	if _is_png(filename):
		print('converting png to jpeg for %s' % filename)
		image_data = coder.png_to_jpeg(image_data)

	# decode the rgb jpeg.
	image = coder.resample_jpeg(image_data)
	return image, RESIZE_HEIGHT, RESIZE_WIDTH

class ImageCoder(object):
	# helper class that provides tensorflow image coding utilities.

	def __init__(self):
		# create a single session to run all image coding calls.
		self._sess = tf.Session()

		# initializes function that converts PNG to JPEG data.
		self._png_data = tf.placeholder(dtype=tf.string)
		image = tf.image.decode_png(self._png_data, channels=3)
		self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

		# initializes function that decodes RGB JPEG data.
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
		cropped = tf.image.resize_images(self._decode_jpeg, [RESIZE_HEIGHT, RESIZE_WIDTH] )
		cropped = tf.cast(cropped, tf.uint8)
		self._recoded = tf.image.encode_jpeg(cropped, format='rgb', quality=100)

	def png_to_jpeg(self, image_data):
		return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

	def resample_jpeg(self, image_data):
		image = self._sess.run(self._recoded, feed_dict={self._decode_jpeg_data: image_data})
		return image

def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
	agelabels, genderlabels, num_shards, subsavefolder):
	# processes and saves list of images as TFRecord in 1 thread.
	# coder: instance of ImageCoder to provide tensorflow image coding utils.
	# thread_index: integer, unique batch to run index is within [0, len(ranges)].
	# ranges: list of pairs of integers specifying ranges of each batches to analyze in parallel.
	# name: string, unique identifier specifying the data set
	# filenames: list of strings; each string is a path to an image file
	# agelabels: list of integer; each integer identifies the ground truth
	# genderlabels: list of integer; each integer identifies the ground truth
	# num_shards: integer number of shards for this data set.

	# each thread produces N shards where N = int(num_shards / num_threads).
	# for instance, if num_shards = 128, and the num_threads =2, then the first
	# thread would produce shards [0, 64).
	num_threads = len(ranges)
	assert not num_shards % num_threads
	num_shards_per_batch = int(num_shards / num_threads)

	shard_ranges = np.linspace(ranges[thread_index][0],
		ranges[thread_index][1], num_shards_per_batch+1).astype(int)
	num_files_in_thread = ranges[thread_index][1]-ranges[thread_index][0]

	counter = 0
	for s in xrange(num_shards_per_batch):
		# generate a sharded version of the file name. e.g. 'train-00002-of-00010'
		shard = thread_index * num_shards_per_batch + s
		output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
		output_file = os.path.join(subsavefolder, output_filename)
		writer = tf.python_io.TFRecordWriter(output_file)

		shard_counter = 0
		files_in_shard = np.arange(shard_ranges[s], shard_ranges[s+1], dtype=int)
		for i in files_in_shard:
			filename = filenames[i]
			agelabel = int(agelabels[i])
			genderlabel = int(genderlabels[i])

			image_buffer, height, width = _process_image(filename, coder)
			print ('***_convert_to_example****')
			print ('filename : %s ' % (filename) )			
			print ('agelabel : %d ' % (agelabel) )			
			print ('genderlabel : %d ' % (genderlabel) )			
			example = _convert_to_example(filename, image_buffer, agelabel,
				genderlabel, height, width)
			writer.write(example.SerializeToString())
			shard_counter += 1
			counter += 1

			if not counter % 1000:
				print('%s [thread %d]: processed %d of %d images in thread batch.' %
					(datetime.now(), thread_index, counter, num_files_in_thread))
				sys.stdout.flush()

		writer.close()
		print('%s [thread %d]: Wrote %d images to %s' %
			(datetime.now(), thread_index, shard_counter, output_file))
		sys.stdout.flush()
		shard_counter=0
	print('%s [thread %d]: wrote %d images to %d shards.' %
		(datetime.now(), thread_index, counter, num_files_in_thread))
	sys.stdout.flush()

def _process_image_files(name, filenames, agelabels, genderlabels, num_shards, subsavefolder):
	# process and save list of images as TFRecord of Example protos.
	# name: string, unique identifier specifying the data set
	# filenames: list of strings; each string is a path to an image file
	# agelabels: list of integer; each integer identifies the ground truth
	# genderlabels: list of integer: each integer identifies the ground truth
	# num_shards: integer number of shards for this data set.
	assert len(filenames) == len(agelabels) == len(genderlabels)

	# break all images into batches with a [ranges[i][0], ranges[i][1]].
	spacing = np.linspace(0, len(filenames), FLAGS.num_threads+1).astype(np.int)
	ranges = []
	threads = []
	for i in xrange(len(spacing)-1):
		ranges.append([spacing[i], spacing[i+1]])

	# launch a thread for each batch.
	print('launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges) )
	sys.stdout.flush()

	# create a mechanism for monitoring when all threads are finished.
	coord = tf.train.Coordinator()

	coder = ImageCoder()

	threads = []
	for thread_index in xrange(len(ranges)):
		args = (coder, thread_index, ranges, name, filenames, agelabels, genderlabels, num_shards, subsavefolder)
		t = threading.Thread(target=_process_image_files_batch, args=args)
		t.start()
		threads.append(t)

	# wait for all the threads to terminate.
	coord.join(threads)
	print('%s: finished writing all %d images in data set.' %
		(datetime.now(), len(filenames)))
	sys.stdout.flush()


def _find_image_files(list_file, data_dir):

	print('Determining list of input files and labels from %s.' % list_file)
	files_labels = [l.strip().split(' ') for l in tf.gfile.FastGFile(list_file, 'r').readlines()]

	agelabels = []
	genderlabels = []
	filenames = []

	# leave label index 0 empty as a background class.
	# label_index = 1

	# construct the list of JPEG files and labels
	for path, agelabel, genderlabel in files_labels:		
		jpeg_file_path = '%s/%s' % (data_dir, path)
		if os.path.exists(jpeg_file_path):
			print ("_find_image_files : jpeg_file_path: %s " % (jpeg_file_path) )			
			print ("_find_image_files : agelabel: %s " % (agelabel) )			
			print ("_find_image_files : genderlabel: %s " % (genderlabel) )			
			filenames.append(jpeg_file_path)
			agelabels.append(agelabel)
			genderlabels.append(genderlabel)

	# age and gender labels
	age_unique_labels = set(agelabels)
	gender_unique_labels = set(genderlabels)

	# shuffle the ordering of all image files in order to guarantee
	# random ordering of the images with respect to label in the 
	# saved TFRecord files. Make the randomization repeatable.
	shuffled_index = list(range(len(filenames)))
	random.seed(12345)
	random.shuffle(shuffled_index)

	filenames = [filenames[i] for i in shuffled_index]
	agelabels = [agelabels[i] for i in shuffled_index]
	genderlabels = [genderlabels[i] for i in shuffled_index]

	print('Found %d JPEG files with %d agelabels and %d genderlabels inside %s.' % 
		(len(filenames), len(age_unique_labels), len(gender_unique_labels), data_dir) )

	return filenames, agelabels, genderlabels

def _process_dataset(name, filename, directory, num_shards, subsavefolder):
	# process a complete data set and save it as a TFRecord.
	# name: string, unique identifier specifying the data set.
	# directory: string, root path to the data set
	# num_shards: integer, number of shards for this data set.
	# labels_file: string, path to the labels file.

	filenames, agelabels, genderlabels = _find_image_files(filename, directory)
	_process_image_files(name, filenames, agelabels, genderlabels, num_shards, subsavefolder)
	age_unique_labels = set(agelabels)
	gender_unique_labels = set(genderlabels)

	return len(agelabels), age_unique_labels, len(genderlabels), gender_unique_labels

def main(unused_argv):
	assert not FLAGS.train_shards % FLAGS.num_threads, (
		'please make the FLAGS.num_threads commersurate with FLAGS.train_shards')
	assert not FLAGS.valid_shards % FLAGS.num_threads, (
		'please make the FLAGS.num_threads commensurate with FLAGS.valid_shards')
	assert not FLAGS.test_shards % FLAGS.num_threads, (
		'please make the FLAGS.num_threads commensurate with FLAGS.test_shards')

	folddirlist = FLAGS.fold_dir.split(os.sep)
	savefolder  = FLAGS.tf_output_dir+os.sep+folddirlist[-1]
	print('saving results to %s ' % (savefolder) )
	if os.path.exists(FLAGS.tf_output_dir) is False:
		print ('Creating %s' % (FLAGS.tf_output_dir) )
		os.makedirs(FLAGS.tf_output_dir)
	if os.path.exists(savefolder) is False:
		print ('Creating %s ' % (savefolder))
		os.makedirs(savefolder)

	folddirlist = os.listdir(FLAGS.fold_dir)
	for folddirname in folddirlist:
		subfolddir = FLAGS.fold_dir+os.sep+folddirname
		subsavefolder = savefolder+os.sep+folddirname
		if os.path.exists(subsavefolder) is False:
			print ('Creating %s ' %(subsavefolder) )
			os.makedirs(subsavefolder)
		# Run it
		agevalid, agevalid_outcomes, gendervalid, gendervalid_outcomes = _process_dataset('validation', '%s/%s' % (subfolddir, FLAGS.valid_list),
			FLAGS.data_dir, FLAGS.valid_shards, subsavefolder)
		agetrain, agetrain_outcomes, gendertrain, gendertrain_outcomes = _process_dataset('train', '%s/%s' % (subfolddir, FLAGS.train_list),
			FLAGS.data_dir,FLAGS.train_shards, subsavefolder)
		agetest, agetest_outcomes, gendertest, gendertest_outcomes = _process_dataset('test', '%s/%s' % (subfolddir, FLAGS.test_list),
			FLAGS.data_dir, FLAGS.test_shards, subsavefolder)

		if len(agevalid_outcomes) != len(agevalid_outcomes | agetrain_outcomes) or len(gendervalid_outcomes) != len(gendervalid_outcomes | gendertrain_outcomes):
			print('Warning: age unattested labels in training data [%s]' 
				% (', '.join(agevalid_outcomes | agetrain_outcomes) - agevalid_outcomes))
			print('Warning: gender unattested labels in training data [%s]' 
				% (', '.join(gendervalid_outcomes | gendertrain_outcomes) - gendervalid_outcomes))
		
		output_file_age = os.path.join(subsavefolder, 'mdage.json')
		output_file_gender = os.path.join(subsavefolder, 'mdgender.json')

		mdage = { 'num_valid_shards': FLAGS.valid_shards,
				'num_train_shards': FLAGS.train_shards,
				'num_test_shards': FLAGS.test_shards,
				'valid_counts': agevalid,
				'train_counts': agetrain,
				'test_counts': agetest,
				'timestamp': str(datetime.now()),
				'nlabels': len(agetrain_outcomes)}
		with open(output_file_age, 'w') as f:
			json.dump(mdage, f)

		mdgender = { 'num_valid_shards': FLAGS.valid_shards,
					'num_train_shards': FLAGS.train_shards,
					'num_test_shards': FLAGS.test_shards,
					'valid_counts': gendervalid,
					'train_counts': gendertrain,
					'test_counts': gendertest,
					'timestamp': str(datetime.now()),
					'nlabels': len(gendertrain_outcomes)}
		with open(output_file_gender, 'w') as f:
			json.dump(mdgender, f)

if __name__ == '__main__':
	tf.app.run()