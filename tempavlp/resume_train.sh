#!/bin/bash
#SBATCH --job-name=tempcxr_ddp
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/users/eprakash/tempcxr_%j.out
#SBATCH --error=/users/eprakash/tempcxr_%j.err

# ============================================================
# Modules
# ============================================================
module load slurm
module load nvhpc

# ============================================================
# Conda
# ============================================================
source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

# ============================================================
# Environment variables for PyTorch DDP / NCCL
# ============================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTHONFAULTHANDLER=1

# ============================================================
# Go to project directory
# ============================================================
cd /users/eprakash/temporal/model

# ============================================================
# Launch training (4 GPUs, DDP)
# ============================================================
torchrun \
  --nproc_per_node=4 \
  resume_train.py

