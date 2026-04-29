#!/bin/bash
#SBATCH --job-name=ctm_finetune_ucf101
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

# Fine-tune the predictive-coding pretrained CTM on UCF-101 (split 1).
# Encoder stays frozen; CTM core is initialised from the pretrain checkpoint.
set -e

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/continuous_thought_machines/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines
wandb offline

DATA_ROOT="/lustre/fsn1/projects/rech/kcn/ucm72yx/data/UCF-101/"
INIT_FROM="${INIT_FROM:-logs/pretrain/kinetics/checkpoint.pt}"

python -m tasks.pretrain.finetune \
    --dataset ucf101 \
    --data_root "${DATA_ROOT}" \
    --fold 1 \
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
    --init_from "${INIT_FROM}" \
    --batch_size 16 \
    --batch_size_test 16 \
    --lr 1e-4 \
    --training_iterations 30001 \
    --warmup_steps 1000 \
    --save_every 2000 \
    --track_every 1000 \
    --n_test_batches 30 \
    --dropout 0.1 \
    --log_dir logs/pretrain/finetune_ucf101 \
    --device 0 \
    --use_amp \
    --seed 42
