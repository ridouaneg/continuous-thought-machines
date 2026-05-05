#!/bin/bash
#SBATCH --job-name=ctm_synth_v2_cifar
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

# Synthetic-v2 with the published CIFAR-10 CTM body shape.
# Matches train_countix_cifar_jz.sh body knobs verbatim so a flat curve
# here would be a strong signal that body capacity is *not* the lever
# for repetition counting. Encoder kept at resnet18-1 from-scratch
# (matches the existing synthetic v1/v2 runs).
# Synthetic-v2 train set = 2048 clips (dataset.py:953); at bs 32 that's
# 64 iters/epoch. 50k iters ≈ 780 epochs — long but feasible.

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines/.venv/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines

python -m tasks.repetition.train \
    --dataset synthetic-v2 \
    --target_fps 16 \
    --clip_duration_s_min 4 \
    --clip_duration_s_max 12 \
    --image_size 64 \
    --max_count 8 \
    --n_count_buckets 16 \
    --d_model 256 \
    --d_input 64 \
    --heads 16 \
    --iterations_per_frame 1 \
    --synapse_depth 5 \
    --n_synch_out 256 \
    --n_synch_action 512 \
    --neuron_select_type random-pairing \
    --memory_length 15 \
    --memory_hidden_dims 64 \
    --deep_memory \
    --dropout 0.0 \
    --dropout_nlm 0 \
    --no-do_normalisation \
    --backbone_type resnet18-1 \
    --no-pretrained_backbone \
    --positional_embedding_type none \
    --batch_size 32 \
    --batch_size_test 32 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --training_iterations 50001 \
    --warmup_steps 2000 \
    --use_scheduler \
    --scheduler_type cosine \
    --track_every 2000 \
    --save_every 2000 \
    --n_test_batches 10 \
    --num_workers_train 8 \
    --log_dir logs/repetition/synthetic_v2_cifar \
    --device 0 \
    --use_amp \
    --seed 42
