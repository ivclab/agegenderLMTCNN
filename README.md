# agegenderLMTCNN
Official implementation of [Joint Estimation of Age and Gender from Unconstrained Face Images using Lightweight Multi-task CNN for Mobile Applications](https://arxiv.org/abs/1806.02023)

Created by [Jia-Hong Lee](https://github.com/Jia-HongHenryLee), [Yi-Ming Chan](https://github.com/yimingchan),Ting-Yen Chen, Chu-Song Chen

## Introduction
Automatic age and gender classification based on unconstrained images has become essential techniques on mobile devices. With limited computing power, how to develop a robust system becomes a challenging task. In this paper, we present an efficient convolutional neural network (CNN) called lightweight multi-task CNN for simultaneous age and gender classification. Lightweight multi-task CNN uses depthwise separable convolution to reduce the model size and save the inference time. On the public challenging Adience dataset, the accuracy of age and gender classification is better than baseline multi-task CNN methods.

## Citing Paper
If you find our works useful in your research, please consider citing:

	Joint Estimation of Age and Gender from Unconstrained Face Images using Lightweight Multi-task CNN for Mobile Applications
	J.-H. Lee, Y.-M. Chan, T.-Y. Chen, C.-S Chen
	IEEE International Conference on Multimedia Information Processing and Retrieval, MIPR 2018

## Prerequisition
- Python 2.7
- [TensorFlow](https://www.tensorflow.org/install/install_linux) 1.2.0 or higher
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

## Coming Soon ...

## Reference Resources
[rude-carnie](https://github.com/dpressel/rude-carnie)

Age and Gender Classification using Convolutional Neural Networks(https://www.openu.ac.il/home/hassner/projects/cnn_agegender/)

[AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning)


## Contact
Please feel free to leave suggestions or comments to [Jia-Hong Lee](https://github.com/Jia-HongHenryLee), Yi-Ming Chan (yiming@iis.sinica.edu.tw), Ting-Yen Chen (timh20022002@iis.sinica.tw) ,Chu-Song Chen (song@iis.sinica.edu.tw)

