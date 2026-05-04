#!/usr/bin/env bash
# Synthetic-v2 oscillating-dots with the CORN-style survival head.
# Identical to train_synthetic_v2.sh except --head_type survival and a
# distinct log dir so it doesn't clobber the CE baseline.
#set -e

CUDA_VISIBLE_DEVICES=0 python -m tasks.repetition.train \
    --dataset synthetic-v2 \
    --head_type survival \
    --target_fps 16 \
    --clip_duration_s_min 4 \
    --clip_duration_s_max 12 \
    --image_size 64 \
    --max_count 8 \
    --n_count_buckets 16 \
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
    --no-pretrained_backbone \
    --positional_embedding_type none \
    --batch_size 16 \
    --batch_size_test 16 \
    --lr 1e-3 \
    --training_iterations 10001 \
    --warmup_steps 1000 \
    --track_every 1000 \
    --save_every 1000 \
    --n_test_batches 10 \
    --log_dir logs/repetition/synthetic_v2_survival \
    --device 0 \
    --use_amp \
    --seed 42 \
    "$@"
