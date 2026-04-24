#!/usr/bin/env bash
# Synthetic "moving shapes" run — CPU-friendly, no downloads.
# Useful for sanity checking the model and populating logs for the
# visualisation notebook.
set -e

python -m tasks.video.train \
    --dataset synthetic \
    --n_frames 16 \
    --image_size 64 \
    --d_model 256 \
    --d_input 128 \
    --heads 4 \
    --iterations_per_frame 1 \
    --synapse_depth 2 \
    --n_synch_out 64 \
    --n_synch_action 64 \
    --memory_length 8 \
    --memory_hidden_dims 16 \
    --backbone_type resnet18-1 \
    --positional_embedding_type none \
    --batch_size 16 \
    --batch_size_test 16 \
    --lr 3e-4 \
    --training_iterations 3001 \
    --warmup_steps 200 \
    --track_every 500 \
    --save_every 500 \
    --n_test_batches 10 \
    --log_dir logs/video/synthetic \
    --seed 42
