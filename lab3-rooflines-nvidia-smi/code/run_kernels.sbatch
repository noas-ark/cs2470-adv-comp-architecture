#!/bin/bash
#SBATCH -J lab3k
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -o lab3k_%j.log
# Part II / Ex I / Ex II only: sbatch run_kernels.sbatch
cd ~/workspace/ml_workloads_part_3
PY=~/.conda/envs/cs2470_env/bin/python
nvidia-smi -L
$PY profile_kernel.py
$PY plot_roofline.py
