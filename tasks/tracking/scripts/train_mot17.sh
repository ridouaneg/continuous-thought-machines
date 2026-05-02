#!/usr/bin/env bash
# MOT17 — pedestrian tracking on the MOTChallenge benchmark.
# Download (~5 GB): https://motchallenge.net/data/MOT17.zip
# Expected layout under DATA_ROOT:
#   train/MOT17-{02,04,05,09,10,11,13}-DPM/{img1,gt,seqinfo.ini}
# Validation is a temporal split of training sequences (last val_ratio fraction).
set -e

DATA_ROOT=${DATA_ROOT:-/geovic/ghermi/data/MOT17}

python -m tasks.tracking.train \
    --dataset mot17 \
    --data_root "${DATA_ROOT}" \
    --n_objects 8 \
    --n_frames 8 \
    --img_size 128 \
    --n_bins 16 \
    --stride 4 \
    --val_ratio 0.2 \
    --encoder_type resnet18 \
    --in_channels 3 \
    --d_feat 256 \
    --d_model 512 \
    --d_input 256 \
    --heads 8 \
    --n_synch_out 64 \
    --n_synch_action 64 \
    --synapse_depth 2 \
    --memory_length 16 \
    --memory_hidden_dims 32 \
    --iterations 30 \
    --dropout 0.1 \
    --batch_size 32 \
    --batch_size_test 64 \
    --lr 1e-4 \
    --training_iterations 30001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 2000 \
    --n_test_batches 20 \
    --use_amp \
    --log_dir logs/tracking/mot17 \
    --seed 42 --device 0 \
    "$@"
