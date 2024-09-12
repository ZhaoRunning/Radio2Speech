#!/bin/bash


# LJSpeech (TransUnet)
config=config/eval.yaml
dataset_name=LJSpeech
vocoder_ckpt=/your/path/train_nodev_ljspeech_parallel_wavegan.v1/checkpoint-400000steps.pkl
vocoder_config=/your/path/train_nodev_ljspeech_parallel_wavegan.v1/config.yml
list_val=/your/path/examples/LJSpeech_val.csv
audio_path=/your/path/examples/ljspeech/audio_wave
load_best_model=/your/path/transunet_LJSpeech/net_best.pth
#save_wave_path=

CUDA_VISIBLE_DEVICES=1 python vitunet_vocoder_eval.py \
 --config=$config \
 --vocoder_ckpt=$vocoder_ckpt \
 --vocoder_config=$vocoder_config \
 --dataset_name=$dataset_name \
 --list_val=$list_val \
 --audio_path=$audio_path \
 --load_best_model=$load_best_model


 # TIMIT