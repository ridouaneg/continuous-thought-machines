#!/usr/bin/env bash
# HMDB-51 (split 1) — 16 frames per clip, 112x112 crop.
# Expects HMDB-51 videos + split files under ${DATA_ROOT}.
set -e

DATA_ROOT=${DATA_ROOT:-/geovic/ghermi/data/hmdb51}

python -m tasks.video.train \
    --dataset hmdb51 \
    --data_root "${DATA_ROOT}" \
    --fold 1 \
    --n_frames 16 \
    --image_size 112 \
    --d_model 768 \
    --d_input 256 \
    --heads 8 \
    --iterations_per_frame 1 \
    --synapse_depth 4 \
    --n_synch_out 128 \
    --n_synch_action 128 \
    --memory_length 12 \
    --memory_hidden_dims 32 \
    --backbone_type resnet18-2 \
    --pretrained_backbone \
    --freeze_backbone \
    --positional_embedding_type none \
    --batch_size 16 \
    --batch_size_test 16 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --label_smoothing 0.1 \
    --training_iterations 20001 \
    --warmup_steps 2000 \
    --track_every 2000 \
    --save_every 2000 \
    --n_test_batches 30 \
    --dropout 0.1 \
    --log_dir logs/video/hmdb51 \
    --device 0 1 \
    --use_amp \
    --seed 42 \
    "$@"
