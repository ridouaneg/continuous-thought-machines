#!/bin/bash
#SBATCH --job-name=ctm_countix_cifar_dinov3_vits16
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

# Encoder lift against train_countix_cifar_rn50_3_jz.sh:
# CIFAR body kept identical, encoder swapped for DINOv3 ViT-S/16
# (the smallest "tiny" DINOv3 ViT, 21M params, embed_dim=384).
# At image_size=112 with patch_size=16 the spatial grid is 7x7
# (49 tokens/frame), so attention is ~4x cheaper than rn50-3's 14x14.
#
# DINOv3 weights are gated. Pre-stage them on a login node:
#   - clone facebookresearch/dinov3 to $REPO_PATH below
#   - download the dinov3_vits16_pretrain_lvd1689m.pth weight file from
#     Meta (or HF) to $WEIGHTS_PATH below.
# The Python loader picks up DINOV3_REPO_PATH / DINOV3_WEIGHTS_PATH
# from the environment (see models/dinov3.py).

module load arch/h100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines/.venv/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines

DATA_ROOT="/lustre/fsn1/projects/rech/kcn/ucm72yx/data/countix/"
KINETICS_ROOT="/lustre/fsn1/projects/rech/kcn/ucm72yx/data/kinetics/"

export DINOV3_REPO_PATH="/lustre/fsn1/projects/rech/kcn/ucm72yx/code/dinov3"
export DINOV3_WEIGHTS_PATH="/lustre/fsn1/projects/rech/kcn/ucm72yx/checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m.pth"

python -m tasks.repetition.train \
    --dataset countix \
    --data_root "${DATA_ROOT}" \
    --kinetics_root "${KINETICS_ROOT}" \
    --target_fps 8 \
    --image_size 112 \
    --n_count_buckets 32 \
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
    --backbone_type dinov3-vits16 \
    --pretrained_backbone \
    --freeze_backbone \
    --positional_embedding_type none \
    --batch_size 32 \
    --batch_size_test 32 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --training_iterations 100001 \
    --warmup_steps 2000 \
    --use_scheduler \
    --scheduler_type cosine \
    --track_every 2000 \
    --save_every 2000 \
    --n_test_batches 30 \
    --num_workers_train 8 \
    --log_dir logs/repetition/countix_cifar_dinov3_vits16 \
    --device 0 \
    --use_amp \
    --seed 42
