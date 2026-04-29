#!/usr/bin/env bash
# DanceTrack — multi-dancer tracking with frequent occlusions and crossings.
# Download: https://github.com/DanceTrack/DanceTrack
# Expected layout under DATA_ROOT:
#   {train,val,test}/dancetrack####/{img1,gt,seqinfo.ini}
set -e

DATA_ROOT=${DATA_ROOT:-/geovic/ghermi/data/DanceTrack}

python -m tasks.tracking.train \
    --dataset dancetrack \
    --data_root "${DATA_ROOT}" \
    --n_objects 8 \
    --n_frames 8 \
    --img_size 128 \
    --n_bins 16 \
    --stride 4 \
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
    --log_dir logs/tracking/dancetrack \
    --seed 42 --device 0
