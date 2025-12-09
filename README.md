<div align="center">

# Spatial Retrieval Augmented Autonomous Driving
## Task: 3D Detection 

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2512.06865/)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://spatialretrievalad.github.io/)

</div>

## 📖 Introduction

This repository contains the implementation of the **3D Detection** task from our paper: **"Spatial Retrieval Augmented Autonomous Driving"**.

We introduce a novel **Spatial Retrieval Paradigm** that retrieves offline geographic images (Satellite/Streetview) based on GPS coordinates to enhance autonomous driving tasks. For Detection, we design a plug-and-play **Spatial Retrieval Adapter** and a **Reliability Estimation Gate** to robustly fuse this external knowledge into BEV representations.

We provides the implementation based on **BEVDet** and **BEVFormer**, finetuned on official checkpoint.

## 🚀 News
- **[2025-12-09]** Code and checkpoints for **3D Detection** (BEVDet & BEVFormer) are released!

## 📊 Model Zoo & Main Results

### BEVDet (ResNet50)

| Method            | Modality | NDS | mAP |   Config   | Download  |
| :---------------- | :------: | :------: | :------: |:--------: | :-------: |
| BEVDet           |    C     |      39.41      |      **30.85**      | - | - |
| **BEVDet + Geo** | C + Geo  |    **39.43**     |     30.69    | [config](BEVDet/configs/bevdet/ggbevdet.py)   | [model](https://pan.baidu.com/s/1WPPsESSX4Nqwnxw34wzzEA?pwd=sdfg) |

> *C: Camera, Geo: Geographic Images.*

### BEVFormer (ResNet101-DCN)

| Method            | Modality | NDS | mAP |   Config   | Download  |
| :---------------- | :------: | :------: | :------: |:--------: | :-------: |
| BEVFormer           |    C     |       51.70       |       41.60       | - | - |
| **BEVFormer + Geo** | C + Geo  |    **51.80**     |     **41.64**    | [config](BEVFormer/projects/configs/bevformer/bevformer_base.py)   | [model](https://pan.baidu.com/s/1V5HpcB44n7Hj6BDn5JtVlw?pwd=sdfg) |

> *C: Camera, Geo: Geographic Images.*

## 📦 Installation

### Please follow the official installation instructions to configure the environment:

- See **BEVDet**: `BEVDet/README.md`
- See **BEVFormer**: `BEVFormer/README.md`


## 📂 Data Preparation

### Step 1: Prepare Base Dataset (Following MMDet3D Workflow)

Please refer to the official dataset configuration instructions to modify the dataset settings.

### Step 2: Generate Geographic Data (nuScenes-Geography-Data)

Configure geographic data tools following the readme in: [SpatialRetrievalAD-Dataset-Devkit](https://github.com/SpatialRetrievalAD/SpatialRetrievalAD-Dataset-Devkit) project, prepare both the nuScenes-Geography dataset and its devkit

After install geographic data tools, configure paths and img settings such as resolution (align with nuscenes input size) in `geoext_gen.py` and run it for streetsat data cache.

Finally, configure paths and run `pkl_merge_bevdet.py` or `pkl_merge_former.py` for merge original mmdet3d pkl and geo pkl.

Optionally, Download from [BEVDet_geo pkl](https://pan.baidu.com/s/1WPPsESSX4Nqwnxw34wzzEA?pwd=sdfg) and [BEVFormer_geo pkl](https://pan.baidu.com/s/1V5HpcB44n7Hj6BDn5JtVlw?pwd=sdfg) for merged pkl.

Finally, define the paths to dataset, the generated .pkl files, and the nuscenes dataset in the BEVDet and BEVFormer config files, and prepare the required official checkpoints (including ResNet-50/101-DCN).

## 🚄 Training & Evaluation

Train BEVDet with 4 GPUs 
```
BEVDet/tools/4card_train.sh
```

Train BEVFormer with 4 GPUs 
```
BEVFormer/tools/train.sh
```


Eval BEVDet with 1 GPU
```
BEVDet/tools/1cardtest.sh
```

Eval BEVFormer with 4 GPU
```
BEVFormer/tools/test.sh
```

## 🖊️ Citation
```
@misc{spad,
      title={Spatial Retrieval Augmented Autonomous Driving}, 
      author={Xiaosong Jia and Chenhe Zhang and Yule Jiang and Songbur Wong and Zhiyuan Zhang and Chen Chen and Shaofeng Zhang and Xuanhe Zhou and Xue Yang and Junchi Yan and Yu-Gang Jiang},
      year={2025},
      eprint={2512.06865},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.06865}, 
}
```
## 🙏 Acknowledgements

This work is based on [BEVDet](https://github.com/HuangJunJie2017/BEVDet) and [BEVFormer](https://github.com/fundamentalvision/BEVFormer).
