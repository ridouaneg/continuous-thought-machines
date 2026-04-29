#!/usr/bin/env bash
# Countix (RepNet paper) — Kinetics-400 clips annotated with repetition counts.
#
# Before running:
#   1. Download the annotation CSVs from https://sites.google.com/view/repnet
#      and place them as:
#        ${DATA_ROOT}/countix_train.csv
#        ${DATA_ROOT}/countix_val.csv
#   2. Download the Kinetics-400 videos for the IDs in those CSVs (e.g. via
#      youtube-dl or the official Kinetics download scripts) and place them as:
#        ${DATA_ROOT}/videos/<kinetics_id>.mp4
#
# Countix counts go up to ~30, so n_count_buckets=32 covers the full range.
# n_frames=64 gives a Nyquist limit of 32 reps — sufficient for this dataset.
set -e

DATA_ROOT=${DATA_ROOT:-data/repetition/countix}

python -m tasks.repetition.train \
    --dataset countix \
    --data_root "${DATA_ROOT}" \
    --n_frames 64 \
    --image_size 112 \
    --n_count_buckets 32 \
    --d_model 1024 \
    --d_input 256 \
    --heads 8 \
    --iterations_per_frame 1 \
    --synapse_depth 4 \
    --n_synch_out 128 \
    --n_synch_action 128 \
    --memory_length 32 \
    --memory_hidden_dims 32 \
    --backbone_type resnet18-2 \
    --positional_embedding_type none \
    --batch_size 16 \
    --batch_size_test 16 \
    --lr 1e-4 \
    --training_iterations 100001 \
    --warmup_steps 2000 \
    --track_every 2000 \
    --save_every 2000 \
    --n_test_batches 30 \
    --dropout 0.1 \
    --log_dir logs/repetition/countix \
    --device 0 \
    --use_amp \
    --seed 42
