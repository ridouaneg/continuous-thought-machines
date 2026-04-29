#!/usr/bin/env bash
# Fine-tune the predictive-coding pretrained CTM on UCF-101.
# Encoder stays frozen; CTM core is initialised from the pretrain checkpoint.
set -e

DATA_ROOT=${DATA_ROOT:-/geovic/geovic/UCF-101}
INIT_FROM=${INIT_FROM:-logs/pretrain/kinetics/checkpoint.pt}

python -m tasks.pretrain.finetune \
    --dataset ucf101 \
    --data_root "${DATA_ROOT}" \
    --fold 1 \
    --n_frames 16 \
    --image_size 112 \
    --d_model 512 \
    --d_input 128 \
    --heads 4 \
    --iterations_per_frame 1 \
    --synapse_depth 4 \
    --n_synch_out 64 \
    --n_synch_action 64 \
    --memory_length 16 \
    --memory_hidden_dims 16 \
    --backbone_type resnet18-2 \
    --positional_embedding_type none \
    --init_from "${INIT_FROM}" \
    --batch_size 16 \
    --batch_size_test 16 \
    --lr 1e-4 \
    --training_iterations 30001 \
    --warmup_steps 1000 \
    --save_every 2000 \
    --track_every 1000 \
    --n_test_batches 30 \
    --dropout 0.1 \
    --log_dir logs/pretrain/finetune_ucf101 \
    --device 0 \
    --use_amp \
    --seed 42
