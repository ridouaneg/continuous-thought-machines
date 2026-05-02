#!/bin/bash
#SBATCH --job-name=ctm_hmdb51
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

# HMDB-51 (split 1) — 16 frames per clip, 112x112 crop.
#set -e

module load arch/a100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines/.venv/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines

DATA_ROOT="/lustre/fsn1/projects/rech/kcn/ucm72yx/data/hmdb51/"

python -m tasks.video.train \
    --dataset hmdb51 \
    --data_root "${DATA_ROOT}" \
    --fold 1 \
    --n_frames 16 \
    --image_size 112 \
    --d_model 768 \
    --d_input 256 \
    --heads 8 \
    --iterations_per_frame 1 \
    --synapse_depth 4 \
    --n_synch_out 128 \
    --n_synch_action 128 \
    --memory_length 12 \
    --memory_hidden_dims 32 \
    --backbone_type resnet18-2 \
    --pretrained_backbone \
    --freeze_backbone \
    --positional_embedding_type none \
    --batch_size 16 \
    --batch_size_test 16 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --label_smoothing 0.1 \
    --training_iterations 20001 \
    --warmup_steps 2000 \
    --track_every 2000 \
    --save_every 2000 \
    --n_test_batches 30 \
    --dropout 0.1 \
    --log_dir logs/video/hmdb51 \
    --device 0 \
    --use_amp \
    --seed 42 \
    "$@"
