#!/bin/bash
#SBATCH --job-name=ctm_repcount
#SBATCH -A kcn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# RepCount-A (TransRAC paper) — dense repetition-counting benchmark.
set -e

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/continuous_thought_machines/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines
wandb offline

DATA_ROOT="/lustre/fsn1/projects/rech/kcn/ucm72yx/data/repcount/"

python -m tasks.repetition.train \
    --dataset repcount \
    --data_root "${DATA_ROOT}" \
    --n_frames 64 \
    --image_size 112 \
    --n_count_buckets 64 \
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
    --batch_size 8 \
    --batch_size_test 8 \
    --lr 1e-4 \
    --training_iterations 100001 \
    --warmup_steps 2000 \
    --track_every 2000 \
    --save_every 2000 \
    --n_test_batches 30 \
    --dropout 0.1 \
    --log_dir logs/repetition/repcount \
    --device 0 \
    --use_amp \
    --seed 42
