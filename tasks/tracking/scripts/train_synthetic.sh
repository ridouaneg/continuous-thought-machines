#!/usr/bin/env bash
# Synthetic bouncing-blobs tracking — CPU-friendly, no downloads.
# N visually-identical Gaussian blobs bounce inside a unit box; the CTM must
# track each blob's identity across frames using only motion cues.
set -e

python -m tasks.tracking.train \
    --dataset synthetic \
    --n_objects 2 \
    --n_frames 8 \
    --img_size 32 \
    --n_bins 16 \
    --blob_sigma_px 1.5 \
    --velocity_scale 0.07 \
    --n_train 50000 \
    --n_test 5000 \
    --encoder_type tiny \
    --in_channels 1 \
    --d_feat 64 \
    --d_model 256 \
    --d_input 128 \
    --heads 4 \
    --n_synch_out 32 \
    --n_synch_action 32 \
    --synapse_depth 1 \
    --memory_length 10 \
    --memory_hidden_dims 16 \
    --iterations 20 \
    --batch_size 64 \
    --batch_size_test 256 \
    --lr 1e-4 \
    --training_iterations 30001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 2000 \
    --n_test_batches 20 \
    --log_dir logs/tracking/synthetic \
    --seed 42 --device 0
