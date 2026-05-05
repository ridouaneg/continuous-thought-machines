#!/bin/bash
#SBATCH --job-name=ctm_countix_cifar_rn50_4
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

# Encoder lift against train_countix_cifar_jz.sh:
# CIFAR body kept identical, encoder upgraded to resnet50-4
# (full ImageNet-pretrained ResNet-50, all 4 stages kept).
# get_d_backbone returns 2048 channels (ctm.py:334) — heaviest of the
# encoder sweep. Drop batch_size if OOM.

module load arch/h100
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
    --backbone_type resnet50-4 \
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
    --log_dir logs/repetition/countix_cifar_rn50_4 \
    --device 0 \
    --use_amp \
    --seed 42
