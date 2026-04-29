#!/usr/bin/env bash
# Predictive-coding pretraining on Kinetics (or any video folder).
# Frozen ImageNet ResNet18 backbone; trains the CTM core only.
#set -e

DATA_ROOT="/geovic/ghermi/data/kinetics/kinetics_400_train/"

python -m tasks.pretrain.pretrain \
    --dataset kinetics \
    --data_root "${DATA_ROOT}" \
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
    --batch_size 16 \
    --lr 1e-4 \
    --training_iterations 10001 \
    --warmup_steps 1000 \
    --save_every 2000 \
    --track_every 500 \
    --gradient_clipping 1.0 \
    --dropout 0.1 \
    --log_dir logs/pretrain/kinetics \
    --device 0 \
    --use_amp \
    --seed 42
