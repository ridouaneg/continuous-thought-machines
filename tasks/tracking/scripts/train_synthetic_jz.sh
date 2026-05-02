#!/bin/bash
#SBATCH --job-name=ctm_track_synth
#SBATCH -A oyr@a100
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/ctm/%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/ctm/%j.err

# Synthetic bouncing-blobs tracking — identity tracking from motion cues.
# On-the-fly model: at internal tick t the CTM attends to frame t//ipf and
# predicts that frame's positions. Total iterations = n_frames * ipf.
#set -e

module load arch/a100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines/.venv/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines

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
    --in_channels 1 \
    --backbone_type resnet18-2 \
    --no-pretrained_backbone \
    --no-freeze_backbone \
    --positional_embedding_type learnable-fourier \
    --d_model 256 \
    --d_input 128 \
    --heads 4 \
    --n_synch_out 32 \
    --n_synch_action 32 \
    --synapse_depth 1 \
    --memory_length 10 \
    --memory_hidden_dims 16 \
    --iterations_per_frame 4 \
    --batch_size 64 \
    --batch_size_test 256 \
    --lr 1e-4 \
    --training_iterations 30001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 2000 \
    --n_test_batches 20 \
    --log_dir logs/tracking/synthetic \
    --device 0 \
    --seed 42 \
    "$@"
