#!/bin/bash
#SBATCH --job-name=ctm_video_synth
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

# Synthetic "moving shapes" run — sanity check on H100.
set -e

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines/.venv/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines
wandb offline

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
    --device 0 \
    --seed 42
