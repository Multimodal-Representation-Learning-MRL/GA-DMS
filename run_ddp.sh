#!/bin/bash
DATASET_NAME="Testing-WebPerson"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


export MASTER_PORT=13462 
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) 
export MASTER_ADDR=localhost 

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Master address: $MASTER_ADDR, Master port: $MASTER_PORT, World size: $WORLD_SIZE"

mlm_loss_weight=0.3

torchrun --nproc_per_node=$WORLD_SIZE --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    train.py \
    --name Pretrain \
    --img_aug \
    --batch_size 128 \
    --MLM \
    --dataset_name $DATASET_NAME \
    --loss_names 'sdm+mlm' \
    --num_epoch 30 \
    --root_dir /mnt/data \
    --pretrain webperson \
    --nam \
    --lr 1e-5 \
    --mlm_loss_weight $mlm_loss_weight \
    