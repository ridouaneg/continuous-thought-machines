#!/usr/bin/env bash
# Countix (RepNet paper) with the CORN-style survival head.
# Identical to train_countix.sh except --head_type survival and a distinct
# log dir so it doesn't clobber the CE baseline.
#
# Same data prerequisites as train_countix.sh — see that script's header.
#
# Countix counts go up to ~30, so n_count_buckets=32 covers the full range.
#set -e

DATA_ROOT="/geovic/ghermi/data/countix/"
KINETICS_ROOT="/geovic/ghermi/data/kinetics/"

python -m tasks.repetition.train \
    --dataset countix \
    --head_type survival \
    --data_root "${DATA_ROOT}" \
    --kinetics_root "${KINETICS_ROOT}" \
    --target_fps 8 \
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
    --log_dir logs/repetition/countix_survival \
    --device 0 \
    --use_amp \
    --seed 42
