#!/usr/bin/env bash
# Countix (RepNet paper) — Kinetics-400 clips annotated with repetition counts.
#
# Before running:
#   1. Download the annotation CSVs from https://sites.google.com/view/repnet
#      and place them as:
#        ${DATA_ROOT}/countix_train.csv
#        ${DATA_ROOT}/countix_val.csv
#   2. Provide the Kinetics-400 videos. Two options:
#        a) flat layout — drop them at ${DATA_ROOT}/videos/<kinetics_id>.mp4
#        b) reuse a Kinetics mirror — set KINETICS_ROOT to a tree shaped like
#           ${KINETICS_ROOT}/kinetics_400_<split>/<class>/<id>_<start>_<end>.mp4
#           and the loader will index it by youtube_id.
#
# Countix counts go up to ~30, so n_count_buckets=32 covers the full range.
# n_frames=64 gives a Nyquist limit of 32 reps — sufficient for this dataset.
#set -e

DATA_ROOT="/geovic/ghermi/data/countix/"
KINETICS_ROOT="/geovic/ghermi/data/kinetics/"

python -m tasks.repetition.train \
    --dataset countix \
    --data_root "${DATA_ROOT}" \
    --kinetics_root "${KINETICS_ROOT}" \
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
