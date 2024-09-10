#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

cd src
python -m open_clip_train.main \
    --dataset-type "lmdb" \
    --train-data "tifs_dataset.lmdb" \
    --warmup 1000 \
    --batch-size 32 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 1 \
    --workers 4 \
    --model "coca_Phikon_pubmedbert" \
    --report-to 'tensorboard' \
    --coca-contrastive-loss-weight 1 \
    --coca-caption-loss-weight 2 \
    --log-every-n-steps 100

