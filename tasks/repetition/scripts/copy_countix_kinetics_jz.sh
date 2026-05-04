#!/bin/bash
#SBATCH --job-name=ctm_copy_countix
#SBATCH -A oyr@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/ctm/%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/ctm/%j.err

# Copy only the Kinetics videos referenced by Countix into the project area.
# IO-bound, so we use the cpu/prepost partition (no GPU needed).
#set -e

module load pytorch-gpu/py3/2.6.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines/.venv/bin/activate
cd /lustre/fsn1/projects/rech/kcn/ucm72yx/code/continuous-thought-machines

python -m tasks.repetition.scripts.copy_countix_kinetics \
    --countix-root /lustre/fsn1/projects/rech/kcn/ucm72yx/data/countix \
    --src          /lustre/fsmisc/dataset/kinetics \
    --dst          /lustre/fsn1/projects/rech/kcn/ucm72yx/data/kinetics \
    --splits train val test \
    --workers 8
