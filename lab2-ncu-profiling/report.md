# Lab 2: GPU Profiling Part II (Nsight Compute)

CS2470 Advanced Computer Architecture, Fall 2026

## Setup

- Workload: `meta-llama/Llama-3.2-1B`, fp32, bs 1, prompt `"Hello, how are you?"`, `max_new_tokens=50`
- Script: `code/profile_ncu_workload.py` (1 warmup `generate()` + 1 measured `generate()`)
- HW: **NVIDIA L4** (Ada, CC 8.9, 58 SMs) on node `gpu-dy-gpu-cr-1`, HUIT academic cluster
- Tool: Nsight Compute 2025.3.1 (CUDA 13.0)
- Run: `code/run_ncu_part1.sbatch` → `ncu -o profile_ncu_basic_300 -f -c 300 python profile_ncu_workload.py`
  - `-c 300` caps capture at the first 300 kernel launches instead of Ctrl+C
  - default (basic) section set, 8 passes per kernel
  - Lab 1 annotations removed first (`annotate_llama.py --restore`), so model source is stock
- All 300 captured kernels are from the warmup `generate()`: setup, embedding, and layers 0 to ~4 of prefill
- Text dumps: `code/extract.sh` → `out/session.txt`, `out/list.csv`, `out/k{30,75,77}.txt` (details page), `out/k{30,75,77}_raw.txt` (all raw metrics)
- Kernel IDs are NCU's 0-based IDs; `extract.sh` uses `--launch-skip N --launch-count 1` = kernel N (checked against `list.csv`)

## Exercise I: GPU Architecture

Source: `device__attribute_*` metrics in the raw page (`out/k30_raw.txt`; same on every kernel). Session page only lists SM count, name, CC, total memory.

| # | Question | Answer | Raw value |
|---|---|---|---|
| 1 | GPU clock rate | 2.04 GHz | `clock_rate` = 2,040,000 kHz |
| 2 | Memory clock rate | 6.251 GHz | `memory_clock_rate` = 6,251,000 kHz |
| 3 | L2 cache size | 48 MiB | `l2_cache_size` = 50,331,648 B |
| 4 | Max registers per block | 65,536 | `max_registers_per_block` |
| 5 | Max warps per scheduler | 12 | `max_warps_per_scheduler` (also "GPU Maximum Warps Per Scheduler" in Occupancy) |
| 6 | Memory pools supported | 1 (supported) | `memory_pools_supported` |
| 7 | Warp size | 32 | `warp_size` |
| 8 | Total memory | 22.03 GiB (23.66 GB) | `total_memory` = 23,659,151,360 B |

Notes:

- 2.04 GHz is the max boost clock. Measured SM clock during profiling is only ~800 MHz (Speed of Light "SM Frequency"), since ncu locks clocks to base by default (`--clock-control base`), so durations below are slower than a normal run
- 12 warps/scheduler × 4 schedulers = 48 warps/SM (`max_warps_per_multiprocessor` = 48)
- 192-bit memory bus (`global_memory_bus_width`) = 6 × 32-bit channels, which shows up again in Q5e
- 48 MiB L2 is big but still ~100x smaller than the fp32 weights (~4.9 GB), so weight reads stream from DRAM

## Exercise II: Kernel Analysis

Layer 0 of prefill, as it appears in `list.csv` (for context):

| IDs | Kernels | Model op |
|---|---|---|
| 0 to 13 | fill, scan, elementwise | `generate()` setup: attention mask, position ids, cache position |
| 14 | `indexSelectSmallIndex` | token embedding lookup |
| 20 to 22 | `cos`, `sin`, scale | rotary embedding |
| 24 to 29 | `pow`, `reduce`, `add`, `rsqrt`, `mul`, `mul` | input RMSNorm |
| **30** | `gemmSN_TN_kernel` | `q_proj` |
| 31 to 34 | `ampere_sgemm_32x32` + `splitKreduce` × 2 | `k_proj`, `v_proj` |
| 35 to 64 | elementwise, cat, softmax, small GEMMs | rotary apply, KV cache, SDPA math path |
| 65 | `gemmSN_TN_kernel` | `o_proj` |
| 67 to 72 | RMSNorm kernels | post-attention RMSNorm |
| 73 | `ampere_sgemm_64x32_sliced1x4_tn` | `gate_proj` |
| **75** | `ampere_sgemm_64x32_sliced1x4_tn` | `up_proj` |
| **77** | `ampere_sgemm_64x32_sliced1x4_tn` + 78 `splitKreduce` | `down_proj` |

### Q1. Name and duration of kernel 77

- Name: `ampere_sgemm_64x32_sliced1x4_tn`
- Duration: **327.39 µs**
- What it is: layer 0 MLP `down_proj` in prefill (8192 → 2048)
- Grid (32, 1, 7) × 256 threads: z = 7 is split-K, so kernel 78 (`splitKreduce_kernel`) sums the 7 partial results
- Same kernel as Lab 1 Q4 (prefill MLP SGEMM)
- Method: `out/k77.txt`, GPU Speed of Light → Duration; cross-checked with `list.csv` row 77

### Q2

- Blank in handout (numbering goes from Q1 to Q3)

### Q3. Kernels before the first `indexSelect`

- **14 kernels** (IDs 0 to 13); first `indexSelectSmallIndex` is ID 14 (12.77 µs)
- Only 1 `indexSelect` in the 300 captured: the embedding lookup for the prompt (`nn.Embedding` → `aten::index_select`)
- The 14 before it are `generate()` setup, not the model:
  - `FillFunctor` / elementwise: building attention mask, `unfinished_sequences`, etc.
  - 2 × `DeviceScanInitKernel` + `DeviceScanKernel`: `cumsum` for position ids / cache position
- Method: first row matching `indexSelect` in `out/list.csv`

### Q4. Kernel 75 compute and memory throughput

- Kernel: `ampere_sgemm_64x32_sliced1x4_tn`, layer 0 `up_proj` (2048 → 8192), 343.62 µs, grid (128, 1, 2) × 256
- **Compute (SM) throughput: 38.77%**
- **Memory throughput: 83.50%** (all from DRAM: DRAM throughput also 83.50%)
- Read: memory-bound. Prefill has only 7 tokens (BOS + 6), so the GEMM is a thin 7 × 2048 by 2048 × 8192; it reads 64 MiB of fp32 weights and does little math per byte
- L1/TEX 54.44%, L2 30.84%: weights stream through once, low reuse
- Method: `out/k75.txt`, GPU Speed of Light Throughput

### Q5. Kernel 30

Kernel: `gemmSN_TN_kernel<float, 128, 16, 2, 4, 8, 9, 0, ...>` = layer 0 `q_proj` (2048 → 2048) in prefill, 92.19 µs, grid 256 × block 128

| Sub | Question | Answer | Source |
|---|---|---|---|
| a | L1/TEX cache throughput | **86.52%** | Speed of Light |
| b | L2 cache throughput | **28.68%** | Speed of Light |
| c | Achieved active warps per SM | **15.75** (of 24 theoretical, 48 max) | Occupancy |
| d | Registers per thread | **72** | Launch Statistics |
| e | DRAM cycles active, avg / min / max | **426,930.67 / 424,944 / 428,912** | raw `dram__cycles_active.{avg,min,max}` |

**a/b notes**

- L1 high, L2 low: the small-N GEMM kernel stages tiles in shared memory and re-reads them per output column (LSU pipe at 75.6% = the top compute unit), while each weight goes through L2 once
- DRAM throughput 74.25%, so also near memory-bound like Q4

**c notes**

- Theoretical 24 warps/SM (50%) vs achieved 15.75 (32.8%)
- Gap is mostly the grid: 256 blocks across 58 SMs × 6 blocks/SM = 348 slots → 0.74 waves, so many SMs run under-filled

**e notes**

- sum = 2,561,584 = 6 × avg → averaged over 6 DRAM units (the 6 × 32-bit channels of the 192-bit bus)
- min to max spread < 1% (73.9% to 74.6% of peak): load is spread evenly across channels, no hot partition

**f. Occupancy vs shared memory per block**

- Source: Occupancy → "Impact of Varying Shared Memory Usage Per Block" (`out/k30.txt` has the chart's 200 points, 0.5 KB steps)
- Kernel's current point: 14.85 KB/block → 6 blocks/SM → 24 warps → **50% theoretical**. Shared memory is the binding limit (block limits: smem 6, registers 7, warps 12, SM 24)
- Step function, occupancy only drops as smem/block grows:

| Shared mem per block | Blocks/SM | Theoretical occupancy | Limiter |
|---|---|---|---|
| 0 to ~14 KB | 7 | 58.3% | registers (72 regs × 128 threads) |
| ~14.5 to ~16.5 KB | 6 | 50% | smem (kernel sits here) |
| ~17 to ~20 KB | 5 | 41.7% | smem |
| ~20.5 to ~24.5 KB | 4 | 33.3% | smem |
| ~25 to ~33 KB | 3 | 25% | smem |
| ~33.5 to ~50 KB | 2 | 16.7% | smem |
| ~50.5 to ~99 KB | 1 | 8.3% | smem |
| > ~99 KB | 0 | 0% | can't launch (99 KB opt-in max/block) |

- Shape: flat on the left (capped by registers at 58%), then a staircase down as fewer blocks fit in the 100 KB of shared memory per SM (~1 KB/block reserved by the driver)
- Takeaway: cutting this kernel's smem by ~1 KB would only buy 50% → 58%, since registers become the cap. Achieved occupancy (32.8%) is limited by the 0.74-wave grid anyway, so smem tuning wouldn't help much here

*Method:* `ncu -i ... --page details --print-details all` printed the Occupancy section's graph data as text; I matched each step's x position to blocks/SM = ⌊102.4 KB / (smem + 1 KB)⌋, capped at 7 by registers.
