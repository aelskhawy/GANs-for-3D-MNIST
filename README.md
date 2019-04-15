# GANs for 3D MNIST

This codes generates 3D digits (16x16x16) using GANs trained on 3D MNIST data. 

## Training
Training for the Vanilla GAN was perfomed on COLAB. Training for the WSGAN-GP was performed on a Tesla GPU with 12 GB of memory. 
For the Vanilla GAN, the samples uploaded to this repo were cherry-picked. 3D MNIST data is somehow noisy and the vanilla objective (and the wasserstein objective also) in this code was not able to fully approximate the distribution, so the sampling process generates a rubbish sometimes. 

This was just a toy project so i didn't dig deeper to know why this happened and moved on to another dataset where these two objectives were able to give good results and generate good-looking samples. 

## Prerequisites
You need to have python 3.6 and PyTorch 0.4 or 1.0. 

## Data 
The data that has been used is available in the code sub-directory. To learn more about 3D MNIST and how this data was generated please read [here](https://www.kaggle.com/daavoo/3d-mnist) and [here](https://medium.com/shashwats-blog/3d-mnist-b922a3d07334)

## Acknowledgment 
This code is a clone of this [repo](https://github.com/rimchang/3DGAN-Pytorch) with some modifications. 
