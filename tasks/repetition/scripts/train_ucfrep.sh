#!/usr/bin/env bash
# UCFRep — UCF-101 videos re-annotated with repetition counts.
#
# Before running:
#   1. Download UCF-101 videos and place them as:
#        ${DATA_ROOT}/UCF-101/<ClassName>/<video>.avi
#   2. Download the UCFRep annotation JSON from
#        https://github.com/xiaostuff/UCFRep
#      and place it as:
#        ${DATA_ROOT}/ucfrep_annotations.json
#      Expected JSON: list of {"video_name": "<Class>/<video>.avi",
#                               "count": N, "split": "train|test"}
#
# UCFRep counts go up to ~20, so n_count_buckets=32 is sufficient.
set -e

DATA_ROOT=${DATA_ROOT:-data/repetition/ucfrep}

python -m tasks.repetition.train \
    --dataset ucfrep \
    --data_root "${DATA_ROOT}" \
    --n_frames 64 \
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
    --training_iterations 50001 \
    --warmup_steps 1000 \
    --track_every 1000 \
    --save_every 1000 \
    --n_test_batches 20 \
    --dropout 0.1 \
    --log_dir logs/repetition/ucfrep \
    --device 0 \
    --use_amp \
    --seed 42
