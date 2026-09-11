#!/bin/bash
#SBATCH -J gemma_profile
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 00:45:00
#SBATCH -o gemma_%j.log
set -e
cd ~/workspace/gpu_profiling_part_1
PY=~/.conda/envs/cs2470_env/bin/python
nvidia-smi -L
$PY annotate_gemma2.py
$PY run_profiler.py --model google/gemma-2-2b --out profile_gemma.json
$PY parse_profile.py profile_gemma.json --kernel sgemm --top 10 --json stats_gemma.json | tee parse_gemma.txt
