#!/usr/bin/env bash
# Kinetics-400 — 32 frames per clip, 112x112 crop, 400 classes.
# Expects the layout produced by the official Kinetics download scripts:
#   ${DATA_ROOT}/kinetics_400_train.csv
#   ${DATA_ROOT}/kinetics_400_val.csv
#   ${DATA_ROOT}/kinetics_400_train/<class>/<youtube_id>_<start:06d>_<end:06d>.mp4
#   ${DATA_ROOT}/kinetics_400_val/<class>/<youtube_id>_<start:06d>_<end:06d>.mp4
# Records whose mp4 is missing are skipped, so this also works with a small
# subset of classes copied to a workstation for a smoke test.
set -e

DATA_ROOT=${DATA_ROOT:-/geovic/ghermi/data/kinetics}

python -m tasks.video.train \
    --dataset kinetics \
    --data_root "${DATA_ROOT}" \
    --n_frames 32 \
    --image_size 112 \
    --d_model 1024 \
    --d_input 256 \
    --heads 8 \
    --iterations_per_frame 1 \
    --synapse_depth 4 \
    --n_synch_out 128 \
    --n_synch_action 128 \
    --memory_length 16 \
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
    --log_dir logs/video/kinetics \
    --device 0 \
    --use_amp \
    --seed 42
