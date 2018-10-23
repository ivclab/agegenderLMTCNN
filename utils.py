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

import six.moves
from datetime import datetime
import sys
import math
import time
from data import multiinputs, inputs_mod, standardize_image
import numpy as np
import tensorflow as tf
import cv2
import re

RESIZE_AOI = 256
RESIZE_FINAL = 227


def EyeLocAlignFace(faceimage, righteyex, righteyey, lefteyex, lefteyey):

	deltay = math.fabs(righteyey-lefteyey)
	deltax = math.fabs(righteyex-lefteyex)
	degrees = float((math.atan2(deltay,deltax)*180)/math.pi)

	if (lefteyey > righteyey):
		faceimage = imutils.rotate(faceimage, degrees)
	else:
		faceimage = imutils.rotate(faceimage, -degrees)

	return faceimage

def MTCNNDetectFace(image):

	print ("Initialize Networks ... ")
	g1 = tf.Graph()
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
	with g1.as_default():
		sess_mtcnn = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess_mtcnn.as_default():
			pnet, rnet, onet = align.detect_face.create_mtcnn(sess_mtcnn, None)

	# minimum size of face
	minsize   = 60
	# mtcnn three step's threshold
	threshold = [0.6,0.7,0.7]
	# scale factor
	factor    = 0.709
	margin    = 0

	faceimagesequence  = []
	faceimagelocation  = []
	faceimagelandmarks = []

	bounding_boxes, points = align.detect_face.detect_face(image,minsize,pnet,rnet,onet,threshold,factor)

	numberoffaces = bounding_boxes.shape[0]
	img_size = np.asarray(image.shape)[0:2]
	if numberoffaces > 0:
		det       = bounding_boxes[:,0:4]
		probs     = bounding_boxes[:,4]
		# right eye, left eye, nose, right corner, left corner
		landmarks = points[0:10,:]

	for index in xrange(numberoffaces):
		x1 = np.maximum(det[index, 0]-margin/2, 0).astype(np.int32)
		y1 = np.maximum(det[index, 1]-margin/2, 0).astype(np.int32)
		x2 = np.minimum(det[index, 2]+margin/2, img_size[1]).astype(np.int32)
		y2 = np.minimum(det[index, 3]+margin/2, img_size[0]).astype(np.int32)
		# face image
		faceimage = image[y1:y2, x1:x2]
		faceimage = EyeLocAlignFace(faceimage,landmarks[0, index],landmarks[5, index],landmarks[1, index],landmarks[6, index])
		faceimagesequence.append(faceimage.copy())
		# face location
		location = [x1,y1,x2,y2]
		faceimagelocation.append(location)
		# face landmarks
		# right eye
		rex = int(landmarks[0, index])
		rey = int(landmarks[5, index])
		# left eye
		lex = int(landmarks[1, index])
		ley = int(landmarks[6, index])
		# nose
		nx  = int(landmarks[2, index])
		ny  = int(landmarks[7, index])
		# right corner
		rcx = int(landmarks[3, index])
		rcy = int(landmarks[8, index])
		# left corner
		lcx = int(landmarks[4, index])
		lcy = int(landmarks[9, index])
		location = [rex,rey,lex,ley,nx,ny,rcx,rcy,lcx,lcy]
		faceimagelandmarks.append(location)

	return faceimagesequence, faceimagelocation, faceimagelandmarks, numberoffaces

# Read image files            
class ImageCoder(object):
    
    def __init__(self):
        # Create a single Session to run all image coding calls.
        config = tf.ConfigProto(allow_soft_placement=True)
        self._sess = tf.Session(config=config)
        
        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        self.crop = tf.image.resize_images(self._decode_jpeg, (RESIZE_AOI, RESIZE_AOI))

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})
        
    def decode_jpeg(self, image_data):
        image = self._sess.run(self.crop, #self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

# Modifed from here
# http://stackoverflow.com/questions/3160699/python-progress-bar#3160819
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='='):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def update(self, step=1):
        self.current += step
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        six.print_('\r' + self.fmt % args, end='')

    def done(self):
        self.current = self.total
        self.update(step=0)
        print('')

def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
    filename: string, path of the image file.
    Returns:
    boolean indicating if the image is a PNG.
    """
    return '.png' in filename
        
def make_multi_image_batch(filenames, coder):
    """Process a multi-image batch, each with a single-look
    Args:
    filenames: list of paths
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """

    images = []

    for filename in filenames:
        with tf.gfile.FastGFile(filename, 'rb') as f:
            image_data = f.read()
        # Convert any PNG to JPEG's for consistency.
        if _is_png(filename):
            print('Converting PNG to JPEG for %s' % filename)
            image_data = coder.png_to_jpeg(image_data)
    
        image = coder.decode_jpeg(image_data)

        crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
        image = standardize_image(crop)
        images.append(image)
    image_batch = tf.stack(images)
    return image_batch

def make_multi_crop_batch(filename, coder):
    """Process a single image file.
    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    
    image = coder.decode_jpeg(image_data)

    crops = []
    print('Running multi-cropped image')
    h = image.shape[0]
    w = image.shape[1]
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(standardize_image(crop))
    crops.append(tf.image.flip_left_right(crop))

    corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(standardize_image(cropped))
        flipped = tf.image.flip_left_right(cropped)
        crops.append(standardize_image(flipped))

    image_batch = tf.stack(crops)
    return image_batch