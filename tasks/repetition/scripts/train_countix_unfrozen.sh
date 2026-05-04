#!/usr/bin/env bash
# Countix (RepNet paper) — same setup as train_countix.sh but with the
# ImageNet-pretrained ResNet18-2 backbone *unfrozen*, so the optimiser can
# fine-tune the visual features for repetition-counting. Targets the
# "frozen-features may be the bottleneck" hypothesis from
# logs_jz/repetition/SUMMARY.md.
#
# Before running:
#   1. Place the Countix CSVs at ${DATA_ROOT}/countix_{train,val}.csv
#   2. Provide the Kinetics-400 videos:
#        a) flat ${DATA_ROOT}/videos/<id>.mp4, or
#        b) Kinetics mirror at ${KINETICS_ROOT}/kinetics_400_<split>/...
#set -e

DATA_ROOT="/geovic/ghermi/data/countix/"
KINETICS_ROOT="/geovic/ghermi/data/kinetics/"

python -m tasks.repetition.train \
    --dataset countix \
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
    --no-freeze_backbone \
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
    --log_dir logs/repetition/countix_unfrozen \
    --device 0 \
    --use_amp \
    --seed 42
