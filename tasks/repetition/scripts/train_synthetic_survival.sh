#!/usr/bin/env bash
# Synthetic oscillating-dots run with the CORN-style survival head.
# Identical to train_synthetic.sh except --head_type survival and a
# distinct log dir so it doesn't clobber the CE baseline.
#set -e

python -m tasks.repetition.train \
    --dataset synthetic \
    --head_type survival \
    --target_fps 8 \
    --clip_duration_s_min 4 \
    --clip_duration_s_max 12 \
    --image_size 64 \
    --max_count 16 \
    --n_count_buckets 32 \
    --d_model 256 \
    --d_input 128 \
    --heads 4 \
    --iterations_per_frame 1 \
    --synapse_depth 2 \
    --n_synch_out 64 \
    --n_synch_action 64 \
    --memory_length 32 \
    --memory_hidden_dims 16 \
    --backbone_type resnet18-1 \
    --positional_embedding_type none \
    --batch_size 16 \
    --batch_size_test 16 \
    --lr 3e-4 \
    --training_iterations 2001 \
    --warmup_steps 200 \
    --track_every 200 \
    --save_every 200 \
    --n_test_batches 10 \
    --log_dir logs/repetition/synthetic_survival \
    --seed 42 --device 0 \
    "$@"
