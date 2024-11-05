#!/bin/bash

config=config/train_LJSpeech.yaml
list_train=/your/path/LJSpeech_train_training.csv
list_val=/your/path/LJSpeech_val_training.csv

tensorboard_dir=/your/path/ckpt
save_ckpt=/your/path/tensorboard

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train_Radio2Speech.py \
  --config=$config \
  --list_train=$list_train \
  --list_val=$list_val \
  --tensorboard_dir=$tensorboard_dir \
  --save_ckpt=$save_ckpt