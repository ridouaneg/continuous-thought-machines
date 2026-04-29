#!/bin/bash
#SBATCH --job-name=ctm_dancetrack
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

# DanceTrack — multi-dancer tracking with frequent occlusions and crossings.
set -e

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/continuous_thought_machines/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines
wandb offline

DATA_ROOT="/lustre/fsn1/projects/rech/kcn/ucm72yx/data/DanceTrack/"

python -m tasks.tracking.train \
    --dataset dancetrack \
    --data_root "${DATA_ROOT}" \
    --n_objects 8 \
    --n_frames 8 \
    --img_size 128 \
    --n_bins 16 \
    --stride 4 \
    --encoder_type resnet18 \
    --in_channels 3 \
    --d_feat 256 \
    --d_model 512 \
    --d_input 256 \
    --heads 8 \
    --n_synch_out 64 \
    --n_synch_action 64 \
    --synapse_depth 2 \
    --memory_length 16 \
    --memory_hidden_dims 32 \
    --iterations 30 \
    --dropout 0.1 \
    --batch_size 32 \
    --batch_size_test 64 \
    --lr 1e-4 \
    --training_iterations 30001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 2000 \
    --n_test_batches 20 \
    --use_amp \
    --log_dir logs/tracking/dancetrack \
    --device 0 \
    --seed 42
