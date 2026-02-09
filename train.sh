#!/usr/bin/env bash
PROJECT_ROOT="."
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"  
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4323 basicsr/train.py -opt options/train.yml