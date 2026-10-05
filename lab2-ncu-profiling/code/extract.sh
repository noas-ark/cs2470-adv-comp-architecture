#!/bin/bash
# Dump an Nsight Compute report to text: session info, kernel list, and details/raw pages for kernels 30, 75, 77.
# Needs ncu, which only exists on GPU nodes:
#   sbatch -p gpu --gres=gpu:1 -c 2 -t 00:20:00 -o extract_%j.log extract.sh
cd ~/workspace/gpu_profiling_part_2
export PATH=/usr/local/cuda-13.0/bin:$PATH
which ncu || { ls -d /usr/local/cuda* ; exit 1; }
R=profile_ncu_basic_300.ncu-rep
mkdir -p out
ncu -i $R --page session > out/session.txt 2>&1
ncu -i $R --page raw --csv --metrics gpu__time_duration.sum > out/list.csv 2>&1
for k in 30 75 77; do
  ncu -i $R --launch-skip $k --launch-count 1 --page details --print-details all > out/k$k.txt 2>&1
  ncu -i $R --launch-skip $k --launch-count 1 --page raw > out/k${k}_raw.txt 2>&1
done
wc -l out/*
