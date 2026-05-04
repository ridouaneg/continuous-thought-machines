#!/usr/bin/env bash
# UCFRep with the CORN-style survival head.
# Identical to train_ucfrep.sh except --head_type survival and a distinct
# log dir so it doesn't clobber the CE baseline.
#
# Same data prerequisites as train_ucfrep.sh — see that script's header.
#
# UCFRep counts go up to ~53; the CE baseline uses n_count_buckets=32 (so
# the high-count tail clamps to bin 31). We mirror that for an apples-to-
# apples comparison — bump both scripts to 64 if the tail matters.
#set -e

DATA_ROOT="/geovic/ghermi/data/ucfrep/"

python -m tasks.repetition.train \
    --dataset ucfrep \
    --head_type survival \
    --data_root "${DATA_ROOT}" \
    --target_fps 8 \
    --image_size 112 \
    --n_count_buckets 32 \
    --d_model 512 \
    --d_input 256 \
    --heads 8 \
    --iterations_per_frame 1 \
    --synapse_depth 2 \
    --n_synch_out 128 \
    --n_synch_action 128 \
    --memory_length 16 \
    --memory_hidden_dims 32 \
    --backbone_type resnet18-2 \
    --positional_embedding_type none \
    --batch_size 16 \
    --batch_size_test 16 \
    --lr 1e-4 \
    --training_iterations 3001 \
    --warmup_steps 200 \
    --track_every 300 \
    --save_every 500 \
    --n_test_batches 20 \
    --dropout 0.1 \
    --log_dir logs/repetition/ucfrep_survival \
    --device 0 \
    --use_amp \
    --seed 42
