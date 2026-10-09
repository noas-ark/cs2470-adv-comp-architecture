#!/bin/bash
#SBATCH -J lab3
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 01:30:00
#SBATCH -o lab3_%j.log
# Run from ~/workspace/ml_workloads_part_3 as: sbatch run_lab3.sbatch
cd ~/workspace/ml_workloads_part_3
PY=~/.conda/envs/cs2470_env/bin/python
mkdir -p results

echo "===== Part III: nvidia-smi ====="
hostname; nvidia-smi
echo "--- nvidia-smi --lms 1 (2 s sample, header + first lines) ---"
timeout 2 nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -lms 1 | head -15

echo "===== Part II: kernel profiling ====="
$PY profile_kernel.py

echo "===== Part I / Ex I / Ex II: rooflines ====="
$PY plot_roofline.py

echo "===== Part IV: Llama-3.2-1B power ====="
$PY profile_power_workload.py --model meta-llama/Llama-3.2-1B --batch 1
head -5 results/power_Llama-3.2-1B_bs1.csv
$PY plot_power.py results/power_Llama-3.2-1B_bs1.json

echo "===== Ex III / IV: Llama-3.1-8B and Gemma-2-2B, bs 1 and 100 ====="
for M in meta-llama/Llama-3.1-8B google/gemma-2-2b; do
  for B in 1 100; do
    $PY profile_power_workload.py --model $M --batch $B || echo "FAILED $M bs$B"
  done
done
$PY plot_power.py results/power_Llama-3.1-8B_bs1.json --out ex3_llama8b_bs1
$PY plot_power.py results/power_Llama-3.1-8B_bs1.json results/power_Llama-3.1-8B_bs100.json \
                  results/power_gemma-2-2b_bs1.json results/power_gemma-2-2b_bs100.json --out ex4_compare
ls -la results
echo "===== DONE ====="
