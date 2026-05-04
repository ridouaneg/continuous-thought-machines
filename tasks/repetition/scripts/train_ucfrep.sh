#!/usr/bin/env bash
# UCFRep — UCF-101 videos re-annotated with repetition counts (526 videos:
# 421 train / 105 test).
#
# Before running, prepare annotations + symlink with:
#   python -m tasks.repetition.scripts.prepare_ucfrep \
#       --data_root /geovic/ghermi/data/ucfrep \
#       --ucf101_videos /geovic/geovic/UCF-101/videos
#
# UCFRep counts go up to ~53, so n_count_buckets=64 covers the range.
#set -e

DATA_ROOT="/geovic/ghermi/data/ucfrep/"
#UCF_ROOT="/geovic/geovic/UCF-101/videos/"

python -m tasks.repetition.train \
    --dataset ucfrep \
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
    --warmup_steps 300 \
    --track_every 300 \
    --save_every 500 \
    --n_test_batches 20 \
    --dropout 0.1 \
    --log_dir logs/repetition/ucfrep \
    --device 0 \
    --use_amp \
    --seed 42
