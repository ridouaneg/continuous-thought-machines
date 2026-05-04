#!/usr/bin/env bash
# RepCount-A (TransRAC paper) with the CORN-style survival head.
# Identical to train_repcount.sh except --head_type survival and a distinct
# log dir so it doesn't clobber the CE baseline.
#
# Same data prerequisites as train_repcount.sh — see that script's header.
#
# RepCount-A has counts up to ~50; the CE baseline already uses
# n_count_buckets=64 so we mirror that here. Survival's tail-absorbing bin
# at K-1 will catch any clamped samples — bump K higher if the long tail
# matters more for your eval.
#set -e

DATA_ROOT="/geovic/ghermi/data/repcount/"

python -m tasks.repetition.train \
    --dataset repcount \
    --head_type survival \
    --data_root "${DATA_ROOT}" \
    --target_fps 8 \
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
    --log_dir logs/repetition/repcount_survival \
    --device 0 \
    --use_amp \
    --seed 42 \
    "$@"
