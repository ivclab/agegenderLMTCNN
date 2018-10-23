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

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
from data import multiinputs, inputs_mod
from model import select_model, get_checkpoint
import os
import json
from utils import *
import csv
import cv2

#convert .pb files
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib

from pdb import set_trace as bp

RESIZE_FINAL = 227
GENDER_LIST = ['M', 'F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128

tf.app.flags.DEFINE_boolean('multitask', True,
	'Whether utilize multitask model')
tf.app.flags.DEFINE_string('model_type', 'LMTCNN-1-1',
	'choose model structure. LMTCNN-1-1 and mobilenet_multitask for multitask. inception, levi_hassner_bn and levi_hassner for singletask ')
tf.app.flags.DEFINE_string('class_type', '',
	'select which single task to train (Age or Gender), only be utilized when multitask=False and choose single task model_type')
tf.app.flags.DEFINE_string('model_dir','./models/train_val_test_per_fold_agegender/test_fold_is_0/LMTCNN-run-30707',
	'trained model directory')
tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
	'Checkpoint basename')
tf.app.flags.DEFINE_string('requested_step', '', 
	'Within the model directory, a requested step to restore e.g., 9000 ')

tf.app.flags.DEFINE_boolean('convertpb', True,
	'Whether want to convert model to become frozen model file')
#tf.app.flags.DEFINE_string('outputlayername', 'ageoutput,genderoutput',
#	'For converting frozen model, entering the output layer name of model structure')
tf.app.flags.DEFINE_string('device_id','/cpu:0',
	'what processing unit to execute inference on')

tf.app.flags.DEFINE_string('filename', '',
	'File (Image) or File list (Text/No header TSV) to process')
tf.app.flags.DEFINE_string('resultfile','',
	'output files (CSV)')
tf.app.flags.DEFINE_boolean('single_look', False,
	'single look at the image or multiple crops')

#tf.app.flags.DEFINE_string('facedetect_model_type','',
#	'what model of face detection to detect the image. eg. MTCNN')

FLAGS = tf.app.flags.FLAGS

def one_of(fname, types):
	return any([fname.endswith('.' + ty) for ty in types])

def resolve_file(fname):
	if os.path.exists(fname): return fname
	for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
		cand = fname + suffix
		if os.path.exists(cand):
			return cand
	return None

def list_images(srcfile):
	with open(srcfile, 'r') as csvfile:
		delim = ',' if srcfile.endswith('.csv') else '\t'
		reader = csv.reader(csvfile, delimiter=delim)
		if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
			print('skipping header')
			_ = next(reader)

		return [row[0] for row in reader]

def classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer):
	try:
		num_batches = math.ceil(len(image_files) / MAX_BATCH_SZ)
		pg = ProgressBar(num_batches)
		for j in range(num_batches):
			start_offset = j * MAX_BATCH_SZ
			end_offset = min((j+1) * MAX_BATCH_SZ, len(image_files))

			batch_image_files = image_files[start_offset:end_offset]
			print(start_offset, end_offset, len(batch_image_files) )
			image_batch = make_multi_image_batch(batch_image_files, coder)
			batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()} )
			batch_sz = batch_results.shape[0]
			for i in range(batch_sz):
				output_i = batch_results[i]
				best_i = np.argmax(output_i)
				best_choice = (label_list[best_i], output_i[best_i])
				print('Guess @ 1 %s, prob = %.2f' % best_choice)
				if writer is not None:
					f = batch_image_files[i]
					writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
			pg.update()
		pg.done()
	except Exception as e:
		print(e)
		print('Failed to run all images')

def classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file, writer):

	#try:
		print('Running file %s ' % image_file)
		image_batch = make_multi_crop_batch(image_file, coder)

		#print('106')

		batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
		output = batch_results[0]
		batch_sz = batch_results.shape[0]

		#print('112')

		for i in range(1, batch_sz):
			output = output + batch_results[i]

		#print('117')

		output /= batch_sz
		best = np.argmax(output)
		best_choice = (label_list[best], output[best])
		print('Guess @ 1 %s, prob = %.2f' % best_choice)

		nlabels = len(label_list)
		if nlabels > 2:
			output[best] = 0
			second_best = np.argmax(output)
			print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

		if writer is not None:
			writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))

	#except Exception as e:
	#	print(e)
	#	print('Failed to run image %s ' % image_file)

def main(argv=None):

	files = []

	#if FLAGS.facedetect_model_type is 'MTCNN':
	#	print('Using face detector %s' %(FLAGS.facedetect_model_type))
	#	img = cv2.imread(FLAGS.filename)
	#	img_resize = cv2.resize(img, (640,480))
	#	img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
	#	faceimagesequence, faceimagelocation, faceimagelandmarks, numberoffaces = MTCNNDetectFace(image)

	# Load Model
	if FLAGS.multitask:
		config = tf.ConfigProto(allow_soft_placement=True)
		with tf.Session(config=config) as sess:
			
			age_nlabels = len(AGE_LIST)
			gender_nlabels = len(GENDER_LIST)

			print('Executing on %s ' % FLAGS.device_id)
			model_fn = select_model(FLAGS.model_type)

			with tf.device(FLAGS.device_id):
				images = tf.placeholder(tf.float32, [None,RESIZE_FINAL,RESIZE_FINAL, 3], name='input')
				agelogits, genderlogits = model_fn(age_nlabels,images, gender_nlabels, images, 1, False)
				init = tf.global_variables_initializer()
				requested_step = FLAGS.requested_step if FLAGS.requested_step else None
				checkpoint_path = '%s' % (FLAGS.model_dir)
				model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
				saver = tf.train.Saver()
				saver.restore(sess, model_checkpoint_path)
				softmax_age_output = tf.nn.softmax(agelogits, name='ageoutput')
				softmax_gender_output = tf.nn.softmax(genderlogits, name='genderoutput')
				print (softmax_age_output)
				print (softmax_gender_output)

				coder = ImageCoder()
				#support a batch mode if no face detection model
				if len(files) == 0:
					if (os.path.isdir(FLAGS.filename)):
						for relpath in os.listdir(FLAGS.filename):
							abspath = os.path.join(FLAGS.filename, relpath)

							if os.path.isfile(abspath) and any([abspath.endswith('.'+ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
								print(abspath)
								files.append(abspath)

					else:
						files.append(FLAGS.filename)
						# if it happens to be a list file, read the list and clobber the files
						if any([FLAGS.filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
							files = list_images(FLAGS.filename)

				writer = None
				output = None
				if FLAGS.resultfile:
					print('Creating output file %s ' % FLAGS.resultfile)
					output = open(FLAGS.resultfile, 'w')
					writer = csv.writer(output)
					writer.writerow(('file', 'label', 'score'))

				image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
				print(image_files)

				if FLAGS.single_look:

					classify_many_single_crop(sess, AGE_LIST, softmax_age_output,
						coder, images, image_files, writer)
					classify_many_single_crop(sess, GENDER_LIST, softmax_gender_output,
						coder, images, image_files, writer)
					
				else:

					for image_file in image_files:
						classify_one_multi_crop(sess, AGE_LIST, softmax_age_output,
							coder, images, image_file, writer)
						classify_one_multi_crop(sess, GENDER_LIST, softmax_gender_output,
							coder, images, image_file, writer)


				if output is not None:
					output.close()


				if FLAGS.convertpb:
					# retrieve the protobuf graph definition
					graph = tf.get_default_graph()
					input_graph_def = graph.as_graph_def()
					output_node_names = 'ageoutput_1,genderoutput_1'
					output_graph_def = graph_util.convert_variables_to_constants(
						sess, #The session is used to retrieve the weights
						input_graph_def, #The graph_def is used to retrieve the nodes
						output_node_names.split(",") #The output node names are used to select the usefull nodes
					)

					# finally we serialize and dump the output graph to the filesystem
					output_pb_file = FLAGS.model_dir+os.sep+FLAGS.model_type+'.pb'
					with tf.gfile.GFile(output_pb_file, "wb") as f:
						f.write(output_graph_def.SerializeToString())
					print("%d ops in the final graph." % len(output_graph_def.node) )

	else:

		config = tf.ConfigProto(allow_soft_placement=True)
		with tf.Session(config=config) as sess:
			
			if FLAGS.class_type == 'Age':
				label_list = AGE_LIST
			elif FLAGS.class_type == 'Gender':
				label_list = GENDER_LIST			
			nlabels = len(label_list)

			print('Executing on %s' % FLAGS.device_id)
			model_fn = select_model(FLAGS.model_type)

			with tf.device(FLAGS.device_id):
				images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3],name='input')
				logits = model_fn(nlabels, images, 1, False)
				init = tf.global_variables_initializer()

				requested_step = FLAGS.requested_step if FLAGS.requested_step else None

				checkpoint_path = '%s' % (FLAGS.model_dir)

				model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)

				saver = tf.train.Saver()
				saver.restore(sess, model_checkpoint_path)

				if FLAGS.class_type == 'Age':
					softmax_output = tf.nn.softmax(logits, name='ageoutput')
				elif FLAGS.class_type == 'Gender':
					softmax_output = tf.nn.softmax(logits, name='genderoutput')

				coder = ImageCoder()

				# Support a batch mode if no face detection model
				if len(files) == 0:

					if (os.path.isdir(FLAGS.filename)):
						for relpath in os.listdir(FLAGS.filename):
							abspath = os.path.join(FLAGS.filename, relpath)

							if os.path.isfile(abspath) and any([abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
								print(abspath)
								files.append(abspath)
					else:
						files.append(FLAGS.filename)
						# If it happens to be a list file, read the list and clobber the files
						if any([FLAGS.filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
							files = list_images(FLAGS.filename)

				writer = None
				output = None
				if FLAGS.resultfile:
					print('Creating output file %s ' % FLAGS.resultfile)
					output = open(FLAGS.resultfile, 'w')
					writer = csv.writer(output)
					writer.writerow(('file', 'label', 'score'))

				image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
				print(image_files)

				if FLAGS.single_look:
					classify_many_single_crop(sess, label_list, softmax_output, coder, images, image_files, writer)
				else:
					for image_file in image_files:
						classify_one_multi_crop(sess, label_list, softmax_output, coder, images, image_file, writer)

				if output is not None:
					output.close()

				if FLAGS.convertpb:
					# retrieve the protobuf graph definition
					graph = tf.get_default_graph()
					input_graph_def = graph.as_graph_def()
					if FLAGS.class_type == 'Age':
						output_node_names = 'ageoutput'
					elif FLAGS.class_type == 'Gender':
						output_node_names = 'genderoutput'

					output_graph_def = graph_util.convert_variables_to_constants(
						sess, #The session is used to retrieve the weights
						input_graph_def, #The graph_def is used to retrieve the nodes
						output_node_names.split(",") #The output node names are used to select the usefull nodes
					)

					# finally we serialize and dump the output graph to the filesystem
					output_pb_file = FLAGS.model_dir+os.sep+FLAGS.model_type+'_'+FLAGS.class_type+'.pb'
					with tf.gfile.GFile(output_pb_file, "wb") as f:
						f.write(output_graph_def.SerializeToString())
					print("%d ops in the final graph." % len(output_graph_def.node) )

if __name__ == '__main__':
	tf.app.run()