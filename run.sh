#!/bin/bash

cd src
torchrun --nproc_per_node 3 -m open_clip_train.main \
    --dataset-type "lmdb" \
    --train-data "tifs_dataset.lmdb" \
    --warmup 1000 \
    --batch-size 64 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 1 \
    --workers 8 \
    --model "coca_uni_biomedbert" \
    --report-to 'wandb' \
    --coca-contrastive-loss-weight 1 \
    --coca-caption-loss-weight 1 \
    --log-every-n-steps 100 \
    --image-mean 0.485 0.456 0.406 \
    --image-std 0.229 0.224 0.225 \
    --wandb-project-name "open-clip" \

