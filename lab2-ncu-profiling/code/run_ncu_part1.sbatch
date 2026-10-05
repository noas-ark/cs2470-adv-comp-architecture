#!/bin/bash
#SBATCH -J ncu_part1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o ncu_part1_%j.log
# Run as: sbatch --export=ALL,LAUNCH_COUNT=300,OUT=profile_ncu_basic_300 run_ncu_part1.sbatch
set -e
cd ~/workspace/gpu_profiling_part_2
export PATH=/usr/local/cuda-13.0/bin:$PATH
PY=~/.conda/envs/cs2470_env/bin/python
nvidia-smi -L
ncu --version
$PY ~/workspace/gpu_profiling_part_1/annotate_llama.py --restore || true
OUT=${OUT:-profile_ncu_basic}
LC=${LAUNCH_COUNT:+-c $LAUNCH_COUNT}
ncu -o $OUT -f $LC $PY profile_ncu_workload.py
ls -lh $OUT.ncu-rep
