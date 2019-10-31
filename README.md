# agegenderLMTCNN
Official implementation of [Joint Estimation of Age and Gender from Unconstrained Face Images using Lightweight Multi-task CNN for Mobile Applications](https://arxiv.org/abs/1806.02023)

Created by [Jia-Hong Lee](https://github.com/Jia-HongHenryLee), [Yi-Ming Chan](https://github.com/yimingchan),Ting-Yen Chen, Chu-Song Chen

## Introduction
Automatic age and gender classification based on unconstrained images has become essential techniques on mobile devices. With limited computing power, how to develop a robust system becomes a challenging task. In this paper, we present an efficient convolutional neural network (CNN) called lightweight multi-task CNN for simultaneous age and gender classification. Lightweight multi-task CNN uses depthwise separable convolution to reduce the model size and save the inference time. On the public challenging Adience dataset, the accuracy of age and gender classification is better than baseline multi-task CNN methods.

## Citing Paper
If you find our works useful in your research, please consider citing:

	@inproceedings{lee2018joint,
	title={Joint Estimation of Age and Gender from Unconstrained Face Images using Lightweight Multi-task CNN for Mobile Applications},
	author={Lee, Jia-Hong and Chan, Yi-Ming and Chen, Ting-Yen and Chen, Chu-Song},
	booktitle={2018 IEEE Conference on Multimedia Information Processing and Retrieval (MIPR)},
	pages={162--165},
	year={2018},
	organization={IEEE}
	}

## Prerequisition
- Python 2.7
- Numpy
- OpenCV
- [TensorFlow](https://www.tensorflow.org/install/install_linux) 1.2.0 ~ 1.5.0
```bash
$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0-cp27-none-linux_x86_64.whl
```
## Usage
1. Clone the agegenderLMTCNN repository:
```bash
$ git clone --recursive https://github.com/ivclab/agegenderLMTCNN.git
```
2. Download Adience Dataset:
```bash
$ python download_adiencedb.py
```
3. Split raw data into training set, validation set and testing set per fold for five-fold cross validation.
this project have been generated this txt files in DataPreparation/FiveFolds/train_val_test_per_fold_agegender.
if you want to generate the new one, you can utilize the following command:
```bash
$ python datapreparation.py \
	--inputdir=./adiencedb/aligned \
	--rawfoldsdir=./DataPreparation/FiveFolds/original_txt_files \
	--outfilesdir=./DataPreparation/FiveFolds/train_val_test_per_fold_agegender
```
4. Pre-process raw data to generate tfrecord files of training set, validation set and testing set in tfrecord directory:
```bash
$ python multipreproc.py \
	--fold_dir ./DataPreparation/FiveFolds/train_val_test_per_fold_agegender \
	--data_dir ./adiencedb/aligned \
	--tf_output_dir ./tfrecord \
```
or you can download tfrecord files which have been generated:
```bash
$ python download_tfrecord.py
```
or you can download the tfrecord files on [OneDrive](https://9caff703fcfa4c3fac83-my.sharepoint.com/:u:/g/personal/honghenry_lee_iis_sinica_edu_tw/ES91ijG3cCZCieytYdqczoIBY7JjuePELHhTXPIbxBTo_g?e=ADjDco)

5. Train LMTCNN model or Levi_Hassner model. Trained models will store in models directory:
```bash
# five-fold LMTCNN model for age and gender tasks 
$ ./script/trainfold1_best.sh ~ $ ./script/trainfold5_best.sh 

# five-fold Levi_Hassner model for age task
$ ./script/trainagefold1.sh ~ $ ./script/trainagefold5.sh

# five-fold Levi_Hassner model for gender task
$ ./script/traingenderfold1.sh ~ $ ./script/traingenderfold5.sh
```
or you can download model files which have been generated:
```bash
$ python download_model.py
```
or you can download the model files on [OneDrive]( https://9caff703fcfa4c3fac83-my.sharepoint.com/:u:/g/personal/honghenry_lee_iis_sinica_edu_tw/ESMSGAn0fC5ElnHTMdeMFJMBjDWUbbKUve5nW8kQ2as-9Q?e=KLjYwh)

6. Evalate LMTCNN model or Levi_Hassner models. Result will be store in results directory:
```bash
# five-fold LMTCNN model for age and gender tasks 
$ ./script/evalfold1_best.sh ~ $ ./script/evalfold5_best.sh 

# five-fold Levi_Hassner model for age task
$ ./script/evalagefold1.sh ~ $ ./script/evalagefold5.sh

# five-fold Levi_Hassner model for gender task
$ ./script/evalgenderfold1.sh ~ $ ./script/evalgenderfold5.sh
```

7. Inference aligned facial image and generate frozen model files(.pb file) which model size are illustrated in the paper. The frozen model files(.pb file) are stored in model directory:
```bash
# five-fold LMTCNN model for age and gender tasks 
$ ./script/inference1_best.sh ~ $ ./script/inference5_best.sh 

# five-fold Levi_Hassner model for age task
$ ./script/inferenceage1.sh ~ $ ./script/inferenceage5.sh

# five-fold Levi_Hassner model for gender task
$ ./script/inferencegender1.sh ~ $ ./script/inferencegender5.sh
```

8. Deploying frozen model (.pb file) in Android devices.
please refer the [androidversion](https://github.com/ivclab/agegenderLMTCNN/tree/master/androidversion) directory.

## Reference Resources
[rude-carnie](https://github.com/dpressel/rude-carnie)

Age and Gender Classification using Convolutional Neural Networks(https://www.openu.ac.il/home/hassner/projects/cnn_agegender/)

[AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning)

[MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)


## Contact
Please feel free to leave suggestions or comments to [Jia-Hong Lee](https://github.com/Jia-HongHenryLee)(honghenry.lee@gmail.com), Yi-Ming Chan (yiming@iis.sinica.edu.tw), Ting-Yen Chen (timh20022002@iis.sinica.tw) ,Chu-Song Chen (song@iis.sinica.edu.tw)

