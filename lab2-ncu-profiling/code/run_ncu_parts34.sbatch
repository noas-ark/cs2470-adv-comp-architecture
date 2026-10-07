#!/bin/bash
#SBATCH -J ncu_p34
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o ncu_p34_%j.log
# Parts III and IV: NVTX-limited profiling of layer 0's MLP, three ways.
# -c 8 keeps the first 8 kernels inside the range (the first prefill call of layer 0's MLP plus a bit).
cd ~/workspace/gpu_profiling_part_2
export PATH=/usr/local/cuda-13.0/bin:$PATH
PY=~/.conda/envs/cs2470_env/bin/python
mkdir -p out
nvidia-smi -L
NV='--nvtx --nvtx-include Annotation/'
run () {  # run <name> <extra ncu args...>
  name=$1; shift
  t0=$(date +%s)
  ncu -o $name -f $NV -c 8 "$@" $PY profile_ncu_region.py > out/$name.log 2>&1
  echo "$name: exit $? in $(( $(date +%s) - t0 )) s"
  grep "==PROF== Profiling" out/$name.log | head -8
}
run profile_ncu_region                                          # Part III: default set
run profile_ncu_region_full --set full                          # Part IV: full set
run profile_ncu_region_L2 --metrics group:memory__l2_cache_table # Part IV: one metric group
ls -lh profile_ncu_region*.ncu-rep
ncu -i profile_ncu_region.ncu-rep --page raw --csv --metrics gpu__time_duration.sum > out/p3_list.csv 2>&1
ncu -i profile_ncu_region.ncu-rep --launch-count 1 --page details > out/p3_k0.txt 2>&1
ncu -i profile_ncu_region_full.ncu-rep --launch-count 1 --page details > out/p4_full_k0.txt 2>&1
ncu -i profile_ncu_region_L2.ncu-rep --page details --print-details all > out/p4_l2.txt 2>&1
ncu -i profile_ncu_region_L2.ncu-rep --page raw --csv > out/p4_l2_raw.csv 2>&1
echo "sections, basic:"; grep -c "Section:" out/p3_k0.txt; grep "Section:" out/p3_k0.txt
echo "sections, full:";  grep -c "Section:" out/p4_full_k0.txt; grep "Section:" out/p4_full_k0.txt
