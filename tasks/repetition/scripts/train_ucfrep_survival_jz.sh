#!/bin/bash
#SBATCH --job-name=ctm_ucfrep_surv
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

# UCFRep with the CORN-style survival head.
# Identical to train_ucfrep_jz.sh except --head_type survival and a
# distinct log dir so it doesn't clobber the CE baseline.
#set -e

module load arch/a100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines/.venv/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines

DATA_ROOT="/lustre/fsn1/projects/rech/kcn/ucm72yx/data/ucfrep/"

python -m tasks.repetition.train \
    --dataset ucfrep \
    --head_type survival \
    --data_root "${DATA_ROOT}" \
    --target_fps 8 \
    --image_size 112 \
    --n_count_buckets 32 \
    --d_model 512 \
    --d_input 256 \
    --heads 8 \
    --iterations_per_frame 1 \
    --synapse_depth 2 \
    --n_synch_out 128 \
    --n_synch_action 128 \
    --memory_length 16 \
    --memory_hidden_dims 32 \
    --backbone_type resnet18-2 \
    --positional_embedding_type none \
    --batch_size 16 \
    --batch_size_test 16 \
    --lr 1e-4 \
    --training_iterations 3001 \
    --warmup_steps 300 \
    --track_every 300 \
    --save_every 500 \
    --n_test_batches 20 \
    --dropout 0.1 \
    --log_dir logs/repetition/ucfrep_survival \
    --device 0 \
    --use_amp \
    --seed 42
