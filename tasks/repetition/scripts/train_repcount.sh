#!/usr/bin/env bash
# RepCount-A (TransRAC paper) — dense repetition-counting benchmark.
#
# Before running:
#   1. Download RepCount-A from https://github.com/SvipRepetitionCounting/TransRAC
#   2. Place annotation CSVs as:
#        ${DATA_ROOT}/annotation/train.csv
#        ${DATA_ROOT}/annotation/valid.csv
#        ${DATA_ROOT}/annotation/test.csv
#      Expected CSV columns: name (video filename), count (integer).
#   3. Place video files as:
#        ${DATA_ROOT}/videos/<video_name>
#
# RepCount-A has counts up to ~50, so n_count_buckets=64 is recommended.
# Alternatively, use 32 buckets and set clips with count > 31 to bucket 31.
#set -e

DATA_ROOT=${DATA_ROOT:-data/repetition/repcount}
DATA_ROOT=""

python -m tasks.repetition.train \
    --dataset repcount \
    --data_root "${DATA_ROOT}" \
    --n_frames 64 \
    --image_size 112 \
    --n_count_buckets 64 \
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
    --batch_size 8 \
    --batch_size_test 8 \
    --lr 1e-4 \
    --training_iterations 100001 \
    --warmup_steps 2000 \
    --track_every 2000 \
    --save_every 2000 \
    --n_test_batches 30 \
    --dropout 0.1 \
    --log_dir logs/repetition/repcount \
    --device 0 \
    --use_amp \
    --seed 42 \
    "$@"
