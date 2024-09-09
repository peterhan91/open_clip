#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

cd src
torchrun --nproc_per_node 2 -m open_clip_train.main \
    --dataset-type "lmdb" \
    --train-data "tifs_dataset.lmdb" \
    --warmup 1000 \
    --batch-size 32 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 1 \
    --workers 4 \
    --model "coca_roberta-ViT-B-32" \
    --report-to "wandb" \
    --coca-contrastive-loss-weight 1 \
    --coca-caption-loss-weight 2 \
    --log-every-n-steps 100

