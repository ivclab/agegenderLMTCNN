# Copyright 2015, Gil Levi and Tal Hassner 
#
# The SOFTWARE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use.
# It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally
# be caused through its use.
#
# The purpose of this repository is to assist readers in reproducing our results on age and gender classification for facial images as
# described in the following work:
#
# Gil Levi and Tal Hassner, Age and Gender Classification Using Convolutional Neural Networks, IEEE Workshop on Analysis and Modeling of
# Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015
#
# Project page: http://www.openu.ac.il/home/hassner/projects/cnn_agegender/
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
import os
import random
import sys
import argparse

age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['m','f']

def main(args):

	# creat output dir
	if not os.path.exists(args.outfilesdir):
		os.mkdir(args.outfilesdir)

	for cur_test_fold_ind in range(5):

		# make output dirs
		cur_fold_out_foldername='test_fold_is_{0}'.format(cur_test_fold_ind)
		cur_fold_out_foldername=os.path.join(args.outfilesdir,cur_fold_out_foldername)
		if not os.path.exists(cur_fold_out_foldername):
			os.mkdir(cur_fold_out_foldername)

		# read raw data set
		cur_test_fold_filename = 'fold_{0}_data.txt'.format(cur_test_fold_ind)
		cur_test_fold_filename = os.path.join(args.rawfoldsdir, cur_test_fold_filename)
		with open(cur_test_fold_filename) as f:
			def_lines=f.readlines()

		def_lines.pop(0)
		# for test files 
		full_test_list = []
		for def_line in def_lines:
			
			def_dic={}
			subject_dir   = def_line.split('\t')[0]
			image_subject = def_line.split('\t')[2]
			image_name='landmark_aligned_face.{0}.{1}'.format(image_subject,def_line.split('\t')[1])

			image_age = def_line.split('\t')[3]
			if image_age=='(25 23)':
				 image_age='(25 32)'

			image_gender = def_line.split('\t')[4]

			def_dic['subject_dir']  = subject_dir
			def_dic['image_name']   = image_name
			def_dic['image_subject']= image_subject
			def_dic['image_age']    = image_age
			def_dic['image_gender'] = image_gender

			full_test_list.append(def_dic)

		images_num = len(full_test_list)
		indices=random.sample(set(range(0,images_num)), images_num)

		agegender_test_txt_filename=os.path.join(cur_fold_out_foldername, 'agegender_test.txt')
		if os.path.exists(agegender_test_txt_filename):
			os.remove(agegender_test_txt_filename)

		agegender_test_txt_file = open(agegender_test_txt_filename,'w+')
		for ind in indices:
			subject_dir  = full_test_list[ind]['subject_dir']
			image_name   = full_test_list[ind]['image_name']
			image_age    = full_test_list[ind]['image_age']
			image_gender = full_test_list[ind]['image_gender']
			image_subject= full_test_list[ind]['image_subject']

			if image_age in age_list and image_gender in gender_list:
				image_age_index=age_list.index(image_age)
				image_gender_index=gender_list.index(image_gender)
				s='{0}/{1} {2} {3}\n'.format(subject_dir,image_name,image_age_index,image_gender_index)
				agegender_test_txt_file.write(s)

		agegender_test_txt_file.close()

		# for train, val files
		full_train_list = []
		train_folds_indices=list(set(range(5)) - set([cur_test_fold_ind]))
		for train_fold_ind in train_folds_indices:
			# read raw data
			cur_train_fold_filename='fold_{0}_data.txt'.format(train_fold_ind)
			cur_train_fold_filename=os.path.join(args.rawfoldsdir,cur_train_fold_filename)
			with open(cur_train_fold_filename) as f:
				def_lines = f.readlines()

			def_lines.pop(0)
			for def_line in def_lines:

				def_dic={}
				subject_dir  =def_line.split('\t')[0]
				image_subject=def_line.split('\t')[2]
				image_name='landmark_aligned_face.{0}.{1}'.format(image_subject,def_line.split('\t')[1])

				image_age=def_line.split('\t')[3]
				if image_age == '(25 23)':
					image_age='(25 32)'

				image_gender=def_line.split('\t')[4]

				def_dic['subject_dir']  =subject_dir
				def_dic['image_name']   =image_name
				def_dic['image_subject']=image_subject
				def_dic['image_age']    =image_age
				def_dic['image_gender'] =image_gender

				full_train_list.append(def_dic)

		images_num=len(full_train_list)
		indices=random.sample(set(range(0,images_num)), images_num)

		val_indices=indices[:images_num/10]
		train_indices=indices[(images_num/10) + 1:]
		train_subset_indices=indices[(images_num/10) + 1: 2* (images_num/10)]

		cases=['val','train','train_subset']
		for case,indices in zip(cases,[val_indices,train_indices,train_subset_indices]):

			agegender_txt_filename=os.path.join(cur_fold_out_foldername,'agegender_{0}.txt'.format(case))
			if os.path.exists(agegender_txt_filename):
				os.remove(agegender_txt_filename)

			agegender_txt_file=open(agegender_txt_filename, 'w+')
			for ind in indices:
				subject_dir=full_train_list[ind]['subject_dir']
				image_name=full_train_list[ind]['image_name']
				image_age=full_train_list[ind]['image_age']
				image_gender=full_train_list[ind]['image_gender']
				image_subject=full_train_list[ind]['image_subject']

				if image_age in age_list and image_gender in gender_list:
					image_age_index=age_list.index(image_age)
					image_gender_index=gender_list.index(image_gender)
					s='{0}/{1} {2} {3}\n'.format(subject_dir,image_name,image_age_index,image_gender_index)
					agegender_txt_file.write(s)

			agegender_txt_file.close()


def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--inputdir', type=str, default='./adiencedb/aligned',
		help='directory of adience dataset')
	parser.add_argument('--rawfoldsdir', type=str, default='./DataPreparation/FiveFolds/original_txt_files',
		help='directory of raw folds')
	parser.add_argument('--outfilesdir', type=str, default='./DataPreparation/FiveFolds/train_val_test_per_fold_agegender',
		help='directory stored the output files separate from raw data')

	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))