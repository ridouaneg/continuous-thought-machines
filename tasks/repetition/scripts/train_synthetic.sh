#!/usr/bin/env bash
# Synthetic oscillating-dots run — CPU-friendly, no downloads.
# A dot bounces N times vertically with a pure cosine trajectory; the CTM
# must learn to count N. Good for verifying the FFT oscillator hypothesis
# before committing GPU time to real datasets.
#set -e

python -m tasks.repetition.train \
    --dataset synthetic \
    --n_frames 64 \
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
    --training_iterations 5001 \
    --warmup_steps 200 \
    --track_every 500 \
    --save_every 500 \
    --n_test_batches 10 \
    --log_dir logs/repetition/synthetic \
    --seed 42 --device 0 \
    "$@"
