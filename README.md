# Pytorch Implementation Examples
This repository holds source codes of **machine learning** and modulized **neural networks** implemented with **[Pytorch](https://pytorch.org/docs/stable/index.html)**.<br/>

Any comments or feedbacks are welcomed, email me at imhgchoi@korea.ac.kr <br/>

## Contents
1. [Gradient Descent](https://github.com/imhgchoi/pytorch_implementations/tree/master/Gradient_Descent) : Not Pytorch -- simple gradient descent with several different conditions.
2. [Logistic Regression](https://github.com/imhgchoi/pytorch_implementations/tree/master/Logistic_Regression) : Not Pytorch
3. [Deep Neural Networks](https://github.com/imhgchoi/pytorch_implementations/tree/master/DNN) : predicting handwritten numbers with MNIST dataset
4. [Convolutional Neural Networks](https://github.com/imhgchoi/pytorch_implementations/tree/master/CNN) : predicting handwritten numbers with MNIST dataset
5. [Recurrent Neural Networks](https://github.com/imhgchoi/pytorch_implementations/tree/master/RNN) : predicting future stock price trend with RNN(LSTM cells)
6. [AutoEncoders](https://github.com/imhgchoi/pytorch_implementations/tree/master/AutoEncoders)
&nbsp;&nbsp;&nbsp; <br/>6.1 [Feed Forward AutoEncoder](https://github.com/imhgchoi/pytorch_implementations/tree/master/AutoEncoders/DNN_AE) : regenerating MNIST images with a feed forward AutoEncoder
&nbsp;&nbsp;&nbsp; <br/>6.2 [Convolutional AutoEncoder](https://github.com/imhgchoi/pytorch_implementations/tree/master/AutoEncoders/CNN_AE) : regenerating MNIST images with a convolutional AutoEncoder
&nbsp;&nbsp;&nbsp; <br/>6.3 [Beta-Variational AutoEncoder](https://github.com/imhgchoi/pytorch_implementations/tree/master/AutoEncoders/beta_VAE) : regenerating MNIST images with a Beta-Variational AutoEncoder <br/>
&nbsp;&nbsp;&nbsp; I found it hard to build a vanilla VAE. So I adopted the Beta-VAE with an incremental Beta to help convergence.
&nbsp;&nbsp;&nbsp; <br/>6.4 [Sparse AutoEncoder](https://github.com/imhgchoi/pytorch_implementations/tree/master/AutoEncoders/sparse_AE) : regenerating MNIST images with a sparse AutoEncoder with 1300 hidden code units.
&nbsp;&nbsp;&nbsp; <br/>6.5 [Denoising AutoEncoder](https://github.com/imhgchoi/pytorch_implementations/tree/master/AutoEncoders/denoising_AE) : regenerating MNIST images that has gaussian noise with a denoising AutoEncoder.
7. [Deep Q Network](https://github.com/imhgchoi/pytorch_implementations/tree/master/DQN)
&nbsp;&nbsp;&nbsp; <br/>7.1 [Feed Forward DQN](https://github.com/imhgchoi/pytorch_implementations/tree/master/DQN/feed_forward) : training Cartpole with an RL feed forward DQN
&nbsp;&nbsp;&nbsp; <br/>7.2 [Convolutional DQN](https://github.com/imhgchoi/pytorch_implementations/tree/master/DQN/CNN_DQN) : training Cartpole with an RL Convolutional DQN. Referenced [here](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#sphx-glr-download-intermediate-reinforcement-q-learning-py), but failed to master the game
#### NOTE : All Neural Network Models are built without train/dev/test splits. Models will be prone to overfitting.
---
  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
