Note: This current version of code is largely based on MonoScene, and many thanks to this great work. You can take it as an independent repo. We are working to integrate it with our own code.

## Table of Content
- [Preparing](#preparing)
  - [Installation](#installation)  
  - [Datasets](#datasets)
  - [Pretrained models](#pretrained-models)
- [Running](#running)
  - [Training](#training)
  - [Evaluating](#evaluating)


# Preparing

## Installation

1. Create conda environment:

```
$ conda create -y -n kitti_ssc python=3.8
$ conda activate kitti_ssc
```
2. This code was implemented with python 3.8, pytorch 1.7.1 and CUDA 11.0. Please install [PyTorch](https://pytorch.org/): 

```
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Install torchmetrics:
```
$ pip install torchmetrics==0.6.0
```

4. Install the additional dependencies:

```
$ cd kitti_ssc
$ pip install -r requirements.txt
```

5. Install tbb:
```
$ conda install -c bioconda tbb=2020.2
```

6. Install mmcv, mmsegmentation, mmdet:
```
$ pip install -U openmim
$ mim install mmcv_full==1.4.0
$ mim install mmdet==2.14.0
$ mim install mmsegmentation==0.14.1
```

7. Finally, install kitti_ssc:

```
$ pip install -e ./
```


## Datasets


### SemanticKITTI

1. You need to download

      - The **Semantic Scene Completion dataset v1.1** (SemanticKITTI voxel data (700 MB)) from [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download)
      -  The **KITTI Odometry Benchmark calibration data** (Download odometry data set (calibration files, 1 MB)) and the **RGB images** (Download odometry data set (color, 65 GB)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).


2. Create a folder to store SemanticKITTI preprocess data at `/path/to/kitti/preprocess/folder`.

3. Store paths in environment variables for faster access (**Note: folder 'dataset' is in /path/to/semantic_kitti**):

```
$ export KITTI_PREPROCESS=/path/to/kitti/preprocess/folder
$ export KITTI_ROOT=/path/to/semantic_kitti 
```

4. Preprocess the data to generate labels at a lower scale, which are used to compute the ground truth relation matrices:

```
$ cd kitti_ssc/
$ python kitti_ssc/dataset/semantic_kitti/preprocess.py kitti_root=$KITTI_ROOT kitti_preprocess_root=$KITTI_PREPROCESS
```

## Pretrained models

Download TPVFormer pretrained models [on SemanticKITTI](https://cloud.tsinghua.edu.cn/f/414ef11eefa74be1a350/?dl=1), then put them in the folder `TPVFormer/kitti_ssc/ckpts/tpv10/`.


# Running

## Training

To train TPVFormer with SemanticKITTI, type:

### SemanticKITTI

1. Create folders to store training logs at **/path/to/kitti/logdir**.

2. Store in an environment variable:

```
$ export KITTI_LOG=/path/to/kitti/logdir
```

3. Train TPVFormer using 4 GPUs with batch_size of 4 (1 item per GPU) on Semantic KITTI:

```
$ cd kitti_ssc/
$ python kitti_ssc/scripts/train_kitti_ssc.py \
    dataset=kitti \
    enable_log=true \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS\
    kitti_logdir=$KITTI_LOG \
    n_gpus=4 batch_size=4 context_prior=false \
    model_cfg=../../../configs/tpv10_lr_wd_layer.py
```

## Evaluating 

### SemanticKITTI

To evaluate TPVFormer on SemanticKITTI validation set, type:

```
$ cd kitti_ssc/
$ python kitti_ssc/scripts/eval_kitti_ssc.py \
    dataset=kitti \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS \
    n_gpus=1 batch_size=1 context_prior=false \
    model_cfg=../../../configs/tpv10_lr_wd_layer.py \
    +model_path=../../../ckpts/tpv10/model.ckpt
```
