#!/usr/bin/env bash
#export NCCL_LL_THRESHOLD=0
#
#CONFIG=$1
#GPUS=$2
#PORT=${PORT:-6666}
#
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    --use_env main.py --config $CONFIG
nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env main_finetune.py --config configs/capt_new/capt_medium.py &


# nohup python -m torch.distributed.launch --nproc_per_node=5 --use_env main_finetune.py --config configs/capt_v2/capt_v2_b0.py &
# nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --config configs/capt_v2_pool/capt_v2_b0.py &


mv nohup.txt checkpoints/capt_micro/
nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --config configs/capt_new/capt_micro.py &