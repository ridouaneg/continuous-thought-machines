#!/usr/bin/env bash
# Synthetic oscillating-dots run — CPU-friendly, no downloads.
# A dot bounces N times vertically with a pure cosine trajectory; the CTM
# must learn to count N. Good for verifying the FFT oscillator hypothesis
# before committing GPU time to real datasets.
#set -e

python -m tasks.repetition.train \
    --dataset synthetic \
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
    --log_dir logs/repetition/synthetic \
    --device 0 \
    --use_amp \
    --seed 42 \
    "$@"
