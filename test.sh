#!/usr/bin/env bash
PROJECT_ROOT="."
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"  # 将项目根目录加入PYTHONPATH
python3 basicsr/test.py -opt options/train.yml