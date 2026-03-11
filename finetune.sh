#!/bin/bash
DATASET_NAME="RSTPReid" #"ICFG-PEDES"

CUDA_VISIBLE_DEVICES=6 \
python finetune.py \
    --name finetune \
    --img_aug \
    --batch_size 128 \
    --MLM \
    --dataset_name $DATASET_NAME \
    --loss_names 'sdm+id+mlm' \
    --num_epoch 60 \
    --root_dir ./data \
    --finetune ./checkpoints/best0.pth
