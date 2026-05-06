#!/bin/bash
#SBATCH --job-name=lstm_synth
#SBATCH -A kcn@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/ctm/%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/ctm/%j.err

# LSTM baseline on synthetic-v1 at scale matched to the CIFAR LSTM config:
# d_model=256, d_input=64, heads=16, num_layers=2, dropout=0.0,
# resnet18-1 from scratch, no positional embedding.
# This is the LSTM analogue of train_synthetic_jz.sh — same dataset
# knobs (image_size 64, max_count 8, target_fps 16, duration 4-12s)
# so the comparison vs CTM-RepCount is apples-to-apples on the easy
# case the CTM solves perfectly.

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
module unload cudnn  # venv ships cuDNN 9.19; JZ system cuDNN 9.2 on LD_LIBRARY_PATH crashes nn.LSTM
source /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines/.venv/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines

python -m tasks.repetition.train \
    --model lstm \
    --dataset synthetic \
    --target_fps 16 \
    --clip_duration_s_min 4 \
    --clip_duration_s_max 12 \
    --image_size 64 \
    --max_count 8 \
    --n_count_buckets 16 \
    --d_model 256 \
    --d_input 64 \
    --heads 16 \
    --num_layers 2 \
    --iterations_per_frame 1 \
    --backbone_type resnet18-1 \
    --no-pretrained_backbone \
    --positional_embedding_type none \
    --batch_size 32 \
    --batch_size_test 32 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --training_iterations 10001 \
    --warmup_steps 1000 \
    --use_scheduler \
    --scheduler_type cosine \
    --track_every 1000 \
    --save_every 1000 \
    --n_test_batches 10 \
    --num_workers_train 8 \
    --dropout 0.0 \
    --log_dir logs/repetition/synthetic_lstm \
    --device 0 \
    --use_amp \
    --seed 42
