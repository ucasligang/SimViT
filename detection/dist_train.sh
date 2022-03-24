#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nohup python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=66667 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} &

#nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port=66667 train.py
# configs/mask_rcnn_capt_micro_fpn_1x_coco.py --launcher pytorch > retinanet_real_capt_micro.out 2>&1 &