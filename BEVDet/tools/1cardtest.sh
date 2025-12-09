#!/usr/bin/env bash
cd xxx/BEVDet/tools

source xxx/miniconda3/bin/activate bevdet

python test.py xxx/BEVDet/configs/bevdet/ggbevdet.py xxx/BEVDet/tools/work_dirs/open/epoch_4_ema.pth --eval mAP
