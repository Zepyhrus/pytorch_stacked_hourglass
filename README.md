# Stacked Hourglass Networks in Pytorch

Based on **Stacked Hourglass Networks for Human Pose Estimation.** [Alejandro Newell](http://www-personal.umich.edu/~alnewell/), Kaiyu Yang, and [Jia Deng](https://www.cs.princeton.edu/~jiadeng/). *European Conference on Computer Vision (ECCV)*, 2016. [Github](https://github.com/princeton-vl/pose-hg-train)

PyTorch code also adopted from: **Associative Embedding: End-to-end Learning for Joint Detection and Grouping.**
[Alejandro Newell](http://www-personal.umich.edu/~alnewell/), Zhiao Huang, and [Jia Deng](https://www.cs.princeton.edu/~jiadeng/). *Neural Information Processing Systems (NeurIPS)*, 2017. [Github](https://github.com/princeton-vl/pose-ae-train)

## Getting Started

This repository provides everything necessary to train and evaluate a single-person pose estimation model on MPII. If you plan on training your own model from scratch, we highly recommend using multiple GPUs.

Requirements:

- Python 3 (code has been tested on Python 3.6)
- PyTorch (code tested with 1.0)
- CUDA and cuDNN
- Python packages (not exhaustive): opencv-python, tqdm, cffi, h5py, scipy (tested with 1.1.0)

Structure:
- ```data/```: data loading and data augmentation code
- ```models/```: network architecture definitions
- ```task/```: task-specific functions and training configuration
- ```utils/```: image processing code and miscellaneous helper functions
- ```train.py```: code for model training
- ```test.py```: code for model evaluation

### Compatibility Problem

重新修改代码，使其兼容Scipy>=1.3.0，pytorch>=1.1.0，并且将所有图片的操作都更改为opencv。初步采用2层Hourglass网络。

#### Opencv与Scipy.image的兼容问题

需要明确的一点是，Opencv与Scipy.image包存在比较大的差异，即便是简单如`imread`的函数，在相同的图片下依旧在像素值上有不小的差异。以照片`images/timg.jpeg`这张430\*600\*3的图片为例：
 * 共计774,000个像素；
 * 读取的numpy像素值差绝对值之和为203,566，基本相当于1/3的像素都相差1左右；
 * 像素值差的均方根为0.83，相当于几乎每个像素都差1；

需要注意的是：
 * 采用opencv后，在原先采用scipy.image训练出的模型下，会导致精度略微下降，大约在1%左右；



#### Dataset
Download the full [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de/), and place the images directory in data/MPII/

#### Training and Testing

To train a network, call:

```python train.py -e test_run_001``` (```-e,--exp``` allows you to specify an experiment name)

To continue an experiment where it left off, you can call:

```python train.py -c test_run_001```

All training hyperparameters are defined in ```task/pose.py```, and you can modify ```__config__``` to test different options. It is likely you will have to change the batchsize to accommodate the number of GPUs you have available.

Once a model has been trained, you can evaluate it with:

```python test.py -c test_run_001```

The option "-m n" will automatically stop training after n total iterations (if continuing, would look at total iterations)

#### Pretrained Models

An 8HG pretrained model is available [here](http://www-personal.umich.edu/~cnris/original_8hg/checkpoint.pt). It should yield validation accuracy of 0.901.

A 2HG pretrained model is available [here](http://www-personal.umich.edu/~cnris/original_2hg/checkpoint.pt). It should yield validation accuracy of 0.885.

Models should be formatted as exp/<exp_name>/checkpoint.pt

Note models were trained using batch size of 16 along with Adam optimizer with LR of 1e-3 (instead of RMSProp at 2.5e-4), as they outperformed in validation. Code can easily be modified to use original paper settings. The original paper reported validation accuracy of 0.881, which this code approximately replicated. Above results also were trained for approximately 200k iters, while the original paper trained for less.

#### Training/Validation split

The train/val split is same as that found in authors' [implementation](https://github.com/princeton-vl/pose-hg-train)

#### Note

During training, occasionaly "ConnectionResetError" warning was occasionally displayed between epochs, but did not affect training.  
