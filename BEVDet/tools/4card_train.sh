#!/usr/bin/env bash
source xxx/miniconda3/bin/activate bevdet

cd xxx/BEVDet/tools
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

./dist_train.sh xxx/BEVDet/configs/bevdet/ggbevdet.py 4
