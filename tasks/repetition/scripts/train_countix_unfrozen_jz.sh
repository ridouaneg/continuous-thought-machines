#!/bin/bash
#SBATCH --job-name=ctm_countix_unfrozen
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

# Countix (RepNet paper) — same setup as train_countix_jz.sh but with the
# ImageNet-pretrained ResNet18-2 backbone *unfrozen*, so the optimiser can
# fine-tune the visual features for repetition-counting. Targets the
# "frozen-features may be the bottleneck" hypothesis from
# logs_jz/repetition/SUMMARY.md. ~12M extra trainable params, so AMP + the
# default 16-batch is what fits cleanly on the A100.
#set -e

module load arch/a100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines/.venv/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines

DATA_ROOT="/lustre/fsn1/projects/rech/kcn/ucm72yx/data/countix/"
KINETICS_ROOT="/lustre/fsn1/projects/rech/kcn/ucm72yx/data/kinetics/"

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
    --training_iterations 10001 \
    --warmup_steps 2000 \
    --track_every 2000 \
    --save_every 2000 \
    --n_test_batches 30 \
    --dropout 0.1 \
    --log_dir logs/repetition/countix_unfrozen \
    --device 0 \
    --use_amp \
    --seed 42
