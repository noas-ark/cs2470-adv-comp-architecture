# Lab 1: GPU Profiling Part I (PyTorch Profiler)

CS2470 Advanced Computer Architecture, Fall 2026

## Setup

- **Workload:** `meta-llama/Llama-3.2-1B` in fp32, batch size 1, prompt `"Hello, how are you?"`, `max_new_tokens=50` (1 prefill pass + 49 decode passes)
- **Hardware:** NVIDIA A10G (Ampere) on the HUIT academic cluster
- **Software:** PyTorch 2.8.0+cu128, transformers 4.57.6
- **Annotations:** `model.generate()` is wrapped in `CS2470Profile_MyCode` (Part III-A). Inside `modeling_llama.py` (Part III-B), `record_function` blocks mark `DecoderLayer`, `RMSNorm`, `LlamaAttention`, `MLP`, `RotaryEmbedding`, `Embedding`, and `LMHead`, plus `QProj`, `KProj`, `VProj`, `ApplyRotary`, `KVCacheUpdate`, `SDPA`, and `OProj` inside attention.

![Full profiled run in Perfetto](images/trace_overview.jpg)
*The full profiled `generate()` call (about 1.84 s). `CS2470Profile_MyCode` spans the run on both the CPU thread (top) and GPU stream 7 (bottom), and each repeated block on the GPU is one forward pass.*

![One decoder layer with annotations](images/decoder_layer.jpg)
*One `CS2470Profile_DecoderLayer` (about 1.8 ms) with `RMSNorm`, `LlamaAttention` (and its `QProj`, `KProj`, `VProj`, `ApplyRotary`, `KVCacheUpdate`, `SDPA`, and `OProj` children), and `MLP` nested beneath it. PyTorch mirrors the annotations onto the GPU stream, where they trail the CPU because kernels execute after they are launched.*

## Exercise I: Understanding Profiled Output

### Q1. How many GPU kernels are executed during one RMSNorm execution?

**6 kernels.** All 1,650 RMSNorm calls in the trace (50 forward passes × 33 norms per pass) launch the same 6:

| # | Code in `LlamaRMSNorm.forward` | aten op | GPU kernel |
|---|---|---|---|
| 1 | `hidden_states.pow(2)` | `aten::pow` | `vectorized_elementwise_kernel<4, pow_tensor_scalar_kernel_impl<float, float>>` |
| 2 | `.mean(-1, keepdim=True)` | `aten::mean` | `reduce_kernel<512, 1, ReduceOp<float, MeanOps<...>>>` |
| 3 | `variance + self.variance_epsilon` | `aten::add` | `vectorized_elementwise_kernel<4, CUDAFunctorOnSelf_add<float>>` |
| 4 | `torch.rsqrt(...)` | `aten::rsqrt` | `vectorized_elementwise_kernel<4, rsqrt_kernel_cuda(...)>` |
| 5 | `hidden_states * torch.rsqrt(...)` | `aten::mul` | `elementwise_kernel<128, 2, BinaryFunctor<float, float, float, MulFunctor<float>>>` |
| 6 | `self.weight * hidden_states` | `aten::mul` | `vectorized_elementwise_kernel<4, BinaryFunctor<float, float, float, MulFunctor<float>>>` |

The two `.to(dtype)` calls in RMSNorm launch nothing because the model already runs in fp32, so casting to the same dtype is a no-op (in bf16 these would add 2 cast kernels). The two multiplies use different kernel templates because #5 broadcasts one scalar per token across the 2048-wide hidden vector, which rules out PyTorch's vectorized path.

![CPU side of one RMSNorm](images/q1_rmsnorm_cpu.jpg)
*CPU side: one `CS2470Profile_RMSNorm` contains 6 aten ops (`pow`, `mean`, `add`, `rsqrt`, `mul`, `mul`), each ending in one `cudaLaunchKernel`.*

![A kernel launch linked to its GPU kernel](images/q1_launch_flow.jpg)
*Selecting the `pow` launch shows the flow arrow to the kernel it started. The kernel begins about 0.5 ms after the launch because the GPU is still finishing the previous layer's MLP.*

![GPU side of the same RMSNorm](images/q1_rmsnorm_gpu.jpg)
*GPU side: the same RMSNorm on stream 7 runs as 6 back-to-back kernels totaling about 13.7 µs.*

*Method:* zoomed into a `CS2470Profile_RMSNorm` slice in Perfetto (6 aten ops, each with one `cudaLaunchKernel` linked to a GPU kernel), then confirmed across all instances by matching launch correlation IDs to kernels in `profile.json`.

### Q2

Left blank in the handout (numbering goes from Q1 to Q3).

### Q3. List the aten functions that occur during the Attention operation.

`LlamaAttention.forward` directly calls 11 aten functions: `aten::linear`, `aten::view`, `aten::transpose`, `aten::unsqueeze`, `aten::mul`, `aten::slice`, `aten::neg`, `aten::cat`, `aten::add`, `aten::scaled_dot_product_attention`, and `aten::reshape`. Including the ops those dispatch to internally, 35 distinct aten functions run in every decode-step attention call. By annotation:

| Step | Called directly | Also runs internally |
|---|---|---|
| `QProj` / `KProj` / `VProj` | `linear`, `view`, `transpose` | `t`, `matmul`, `mm`, `reshape`, `_unsafe_view`, `as_strided` |
| `ApplyRotary` | `unsqueeze`, `mul`, `slice`, `neg`, `cat`, `add` | `as_strided` |
| `KVCacheUpdate` | `cat` | none |
| `SDPA` | `scaled_dot_product_attention`, `transpose` | `_scaled_dot_product_attention_math`, `mul`, `repeat_interleave`, `unsqueeze`, `expand`, `clone`, `empty_like`, `empty`, `copy_`, `flatten`, `view`, `matmul`, `bmm`, `reshape`, `_reshape_alias`, `_unsafe_view`, `_safe_softmax`, `softmax`, `_softmax`, `isneginf`, `all`, `scalar_tensor`, `fill_`, `where`, `to`, `as_strided` |
| (before `OProj`) | `reshape` | `view` |
| `OProj` | `linear` | `t`, `transpose`, `matmul`, `mm`, `reshape`, `view`, `_unsafe_view`, `as_strided` |

The 16 prefill attention calls run 9 additional ops (44 total): `ones`, `tril`, `resize_`, `add_`, and `contiguous` build the causal mask across the prompt tokens, and `_to_copy`, `empty_strided`, `lift_fresh`, and `detach_` create the KV cache on first use. Decode needs neither because it has a single query and an existing cache.

SDPA dispatches to PyTorch's unfused math fallback (`_scaled_dot_product_attention_math`), which is why attention expands into roughly 25 ops. I believe this is because fp32 inputs rule out FlashAttention and grouped-query attention rules out the memory-efficient kernel. `repeat_interleave` is the GQA expansion from 8 KV heads to 32 query heads done by hand.

![One LlamaAttention block in Perfetto](images/q3_attention.jpg)
*One `CS2470Profile_LlamaAttention` (about 1.1 ms of CPU time). `SDPA` holds the deepest stack of aten ops.*

*Method:* I read the ops under each sub-annotation by zooming into one `CS2470Profile_LlamaAttention` slice in Perfetto (screenshot above). Many ops are too thin to read at that zoom, so I confirmed the list with Perfetto's SQL mode (prefix the search bar with `:`). This query takes one decode-step attention slice on the CPU thread and lists every `aten::` op nested beneath it:

```sql
SELECT name, COUNT(*) AS times
FROM descendant_slice((
  SELECT id FROM slice
  WHERE name = 'CS2470Profile_LlamaAttention' AND category = 'user_annotation'
  ORDER BY ts LIMIT 1 OFFSET 20))
WHERE name LIKE 'aten::%'
GROUP BY name ORDER BY MIN(ts)
```

It returns 35 rows, matching the table above. The counts also line up with the code: `aten::linear` runs 4 times (the Q, K, V, and O projections), `aten::scaled_dot_product_attention` once, and `aten::neg` twice (rotary embedding rotates both Q and K). `category = 'user_annotation'` selects the CPU copy of the annotation rather than its GPU-side mirror, and `OFFSET 20` skips the 16 prefill calls (`OFFSET 0` returns the prefill set of 44). Lastly, a script over `profile.json` checked all 800 attention slices: all 784 decode calls produce the identical set of 35, and all 16 prefill calls produce the same 44.

### Q4. What is the start time of the first `ampere_sgemm_64x32_sliced1x4_tn` kernel of the workload's execution?

**00:00:00.006 565 311 in Perfetto, i.e. 6.565 ms after the trace starts** (raw `ts` in `profile.json`: 6168291221092.682 µs). The kernel runs for 322.7 µs inside the first decoder layer's `CS2470Profile_MLP` during prefill. Perfetto reports times relative to the start of the trace (when the profiler began recording), not the start of `CS2470Profile_MyCode`.

This kernel appears 48 times, all during the prefill pass and all inside `CS2470Profile_MLP`: 16 layers × 3 MLP projections (`gate_proj`, `up_proj`, `down_proj`), each taking 297 to 323 µs. It never appears during decode. Prefill pushes every prompt token through the MLP at once, so each projection is a matrix-matrix multiply that cuBLAS maps to this SGEMM kernel. Decode processes one token per step, which turns the same projections into matrix-vector products that run on `gemv` kernels instead.

![First ampere_sgemm_64x32_sliced1x4_tn kernel](images/q4_first_sgemm.jpg)
*The first `ampere_sgemm_64x32_sliced1x4_tn` selected on GPU stream 7, nested under `CS2470Profile_MLP` in the first decoder layer. The details panel shows its start time and duration.*

*Method:* searched the kernel name in Perfetto and stepped to the first of its 48 matches (search results are ordered by time). Confirmed with Perfetto's SQL mode:

```sql
SELECT s.ts - (SELECT start_ts FROM trace_bounds) AS start_ns, s.dur AS dur_ns,
  (SELECT GROUP_CONCAT(a.name, ' > ') FROM ancestor_slice(s.id) a) AS inside
FROM slice s
WHERE s.name = 'ampere_sgemm_64x32_sliced1x4_tn'
ORDER BY s.ts LIMIT 3
```

The first row returns `start_ns = 6565311`, nested under `CS2470Profile_MyCode > CS2470Profile_DecoderLayer > CS2470Profile_MLP`. A second query confirmed that all 48 instances fall before the second forward pass begins on the GPU and all 48 sit inside `CS2470Profile_MLP`.

### Q5. What is the GPU kernel name linked to the 3rd linear layer in Attention execution?

The 3rd linear layer is `v_proj` (attention runs `q_proj`, `k_proj`, `v_proj`, `o_proj` in that order), and the kernel it launches depends on the phase:

| Phase | Calls | Kernel(s) launched by `CS2470Profile_VProj` | Avg duration |
|---|---|---|---|
| Prefill | 16 | `ampere_sgemm_32x32_sliced1x4_tn`, then `cublasLt::splitKreduce_kernel<32, 16, int, float, ...>` | 23.1 µs + 1.6 µs |
| Decode | 784 | `internal::gemvx::kernel<int, int, float, float, float, float, false, true, true, false, 9, false, cublasGemvParamsEx<...>>` | 18.3 µs |

Prefill multiplies every prompt token at once (matrix × matrix), so cuBLAS runs a split-K SGEMM followed by a small kernel that sums the partial results. Decode multiplies a single token (matrix × vector), so it switches to a GEMV kernel. The point here is that the decode kernel matters most for runtime, since it accounts for about 14.4 ms across 784 calls compared to 0.4 ms for all 16 prefill calls. For that reason I would treat the decode `gemvx` kernel as the primary answer.

`v_proj` shares its kernels with `k_proj`, not `q_proj`. With grouped-query attention, K and V each project to 512 values per token (8 heads × 64) while Q and O project to 2048 (32 heads × 64). cuBLAS picks a different kernel for each shape, which is why Q and O run `gemmSN_TN_kernel` in prefill and `gemv2T_kernel_val` in decode instead.

*Method:* followed the `cudaLaunchKernel` flow from `CS2470Profile_VProj` to its GPU kernel in Perfetto for one prefill and one decode call. I then checked all 800 calls with a query that joins each projection's annotation to its kernels through the `flow` table.

<details>
<summary>Perfetto SQL query</summary>

```sql
WITH v AS (
  SELECT id, ts, dur, track_id, name,
         ROW_NUMBER() OVER (PARTITION BY name ORDER BY ts) AS rn
  FROM slice
  WHERE category = 'user_annotation'
    AND name IN ('CS2470Profile_QProj', 'CS2470Profile_KProj',
                 'CS2470Profile_VProj', 'CS2470Profile_OProj'))
SELECT REPLACE(v.name, 'CS2470Profile_', '') AS proj,
       CASE WHEN v.rn <= 16 THEN 'prefill' ELSE 'decode' END AS phase,
       k.name AS kernel, COUNT(*) AS n, CAST(AVG(k.dur) AS INT) AS avg_ns
FROM v
JOIN slice l ON l.track_id = v.track_id AND l.ts >= v.ts
            AND l.ts < v.ts + v.dur AND l.category = 'cuda_runtime'
JOIN flow f ON f.slice_out = l.id
JOIN slice k ON k.id = f.slice_in
GROUP BY proj, phase, kernel
ORDER BY proj, phase DESC
```

`rn <= 16` marks prefill because each projection runs once per layer (16 layers) in the first forward pass.
</details>

### Q6. List all of the annotations that correspond to the `vectorized_elementwise_kernel<4, AUnaryFunctor<float, float, float, MulFunctor<float>>, ...>` kernel.

This kernel multiplies a tensor by a single scalar (`AUnaryFunctor` wraps a binary op with one operand fixed as a constant). It runs 1,700 times under two annotation stacks:

| Annotation stack | aten op | Count | Why |
|---|---|---|---|
| `CS2470Profile_MyCode` > `CS2470Profile_DecoderLayer` > `CS2470Profile_LlamaAttention` > `CS2470Profile_SDPA` | `aten::mul` | 1,600 | 2 per attention call × 800 calls (the math SDPA backend scales Q and K by √scale before the matmul) |
| `CS2470Profile_MyCode` > `CS2470Profile_RotaryEmbedding` | `aten::mul` | 100 | 2 per forward pass × 50 passes (`cos` and `sin` are multiplied by `attention_scaling`) |

Each instance takes about 1.2 to 1.3 µs. The multiplies inside `CS2470Profile_ApplyRotary` do not show up here because `q * cos` and `rotate_half(q) * sin` multiply two tensors, which PyTorch routes to the `BinaryFunctor` kernel instead. The RotaryEmbedding multiplies are also wasted work in this model (`attention_scaling` is 1.0 for Llama 3.2's RoPE, so they multiply by one).

*Method:* searched `AUnaryFunctor` in Perfetto and followed each match's preceding flow back to its `cudaLaunchKernel` to read the labels above it. I then grouped every instance of the kernel by its annotation stack with SQL.

<details>
<summary>Perfetto SQL query</summary>

```sql
SELECT (SELECT GROUP_CONCAT(REPLACE(a.name, 'CS2470Profile_', ''), ' > ')
        FROM ancestor_slice(f.slice_out) a
        WHERE a.name LIKE 'CS2470Profile_%') AS stack,
       (SELECT p.name FROM slice p
        WHERE p.id = (SELECT parent_id FROM slice WHERE id = f.slice_out)) AS op,
       COUNT(*) AS n, CAST(AVG(k.dur) AS INT) AS avg_ns
FROM slice k
JOIN flow f ON f.slice_in = k.id
WHERE k.name LIKE 'void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >%'
GROUP BY stack, op
ORDER BY n DESC
```
</details>

### Q7. How many instances are there of the rsqrt kernel?

**1,650 GPU kernels** (`vectorized_elementwise_kernel<4, rsqrt_kernel_cuda(...)>`), which is one per RMSNorm call (50 forward passes × 33 norms). Every one launches from inside `CS2470Profile_RMSNorm`, and nothing else in the model calls rsqrt.

Searching `rsqrt` in Perfetto returns 3,300 matches, though – each GPU kernel has a matching CPU op (`aten::rsqrt`) that launches it, so a name search double counts. Searching `rsqrt_kernel` matches only the GPU kernel and returns 1,650.

*Method:* compared the Perfetto search counts for `rsqrt` (3,300) and `rsqrt_kernel` (1,650). I then split the matches by category with SQL and checked that every kernel's launch sits under `CS2470Profile_RMSNorm`.

<details>
<summary>Perfetto SQL query</summary>

```sql
SELECT s.category,
       CASE WHEN s.category = 'kernel' THEN 'rsqrt kernel' ELSE s.name END AS what,
       COUNT(*) AS n,
       SUM(CASE WHEN s.category = 'kernel'
                 AND (SELECT COUNT(*) FROM flow f
                      JOIN ancestor_slice(f.slice_out) a
                      WHERE f.slice_in = s.id AND a.name = 'CS2470Profile_RMSNorm') > 0
                THEN 1 ELSE 0 END) AS in_rmsnorm
FROM slice s
WHERE s.name LIKE '%rsqrt%'
GROUP BY s.category, what
```

Returns `cpu_op | aten::rsqrt | 1650 | 0` and `kernel | rsqrt kernel | 1650 | 1650`.
</details>

## Exercise II: Parsing Profiler Data

`code/parse_profile.py` reads the Chrome trace from `prof.export_chrome_trace()` and reports the five statistics the lab asks for. Every number comes from filtering the trace's events by category (`kernel`, `cpu_op`, `user_annotation`) and summing their `dur` field (in microseconds).

| Statistic | How the parser computes it |
|---|---|
| Total CPU time | Wall time of `CS2470Profile_MyCode` (the profiled `generate()` call), plus time inside aten ops as the union of their intervals, since summing nested ops like `aten::linear` > `aten::mm` would double count |
| Total GPU time | Union of all kernel, memcpy, and memset intervals, reported with utilization (GPU busy time / CPU wall time) |
| Specific kernel | Count, total, average, min, and max duration for an exact kernel name (substring match as a fallback) |
| Decoder layers and attention blocks | Count of CPU-side `CS2470Profile_*DecoderLayer` and `CS2470Profile_*Attention` slices. Every label is mirrored onto the GPU stream, so the parser only counts `user_annotation` events to avoid doubling. Forward passes come from `CS2470Profile_LMHead`, which gives layers per pass |
| Top-N kernels | Kernels grouped by name and ranked by total time, with call count and share of GPU kernel time |

```
python code/parse_profile.py profile.json --kernel ampere_sgemm_64x32_sliced1x4_tn --top 10 --json stats.json
```

To run Gemma-2-2b through the same pipeline, `code/run_profiler.py` takes a `--model` flag and `code/annotate_gemma2.py` applies the same `CS2470Profile_` labels to `modeling_gemma2.py` (so the parser runs unchanged). `code/run_gemma.sbatch` runs annotation, profiling, and parsing as one batch job. Both models ran in fp32 with SDPA attention on the same A10G, with the same prompt and 50 new tokens.

### Results

| Statistic | Llama-3.2-1B | Gemma-2-2b |
|---|---|---|
| CPU wall time (`CS2470Profile_MyCode`) | 1,839.99 ms | 3,755.49 ms |
| CPU time inside aten ops | 1,166.52 ms | 2,336.22 ms |
| GPU busy time | 1,041.54 ms | 1,199.99 ms |
| GPU kernels launched | 41,417 | 83,419 |
| GPU utilization (busy / wall) | 56.6% | 32.0% |
| `ampere_sgemm_64x32_sliced1x4_tn` | 48 calls, 14.54 ms total, 302.9 µs avg | 130 calls, 17.70 ms total, 136.1 µs avg |
| Decoder layer executions | 800 (16 layers × 50 passes) | 1,300 (26 layers × 50 passes) |
| Attention block executions | 800 | 1,300 |

**Top 5 GPU kernels, Llama-3.2-1B** (94 distinct kernels)

| Rank | Kernel | Calls | Total | % of GPU kernel time | Where it runs |
|---|---|---|---|---|---|
| 1 | `gemv2T_kernel_val<...>` | 3,136 | 515.49 ms | 49.6% | Decode MLP (2 of 3 projections), `q_proj`, `o_proj` |
| 2 | `internal::gemvx::kernel<...>` | 784 | 202.59 ms | 19.5% | Decode MLP (1 projection) |
| 3 | `internal::gemvx::kernel<...>` | 50 | 201.81 ms | 19.4% | LM head, once per token |
| 4 | `internal::gemvx::kernel<...>` | 1,568 | 28.82 ms | 2.8% | Decode `k_proj` and `v_proj` |
| 5 | `ampere_sgemm_64x32_sliced1x4_tn` | 48 | 14.54 ms | 1.4% | Prefill MLP |

**Top 5 GPU kernels, Gemma-2-2b** (68 distinct kernels)

| Rank | Kernel | Calls | Total | % of GPU kernel time | Where it runs |
|---|---|---|---|---|---|
| 1 | `internal::gemvx::kernel<...>` | 3,822 | 440.30 ms | 36.7% | Decode GEMV, 3 per layer (1,274 decode layer executions) |
| 2 | `gemv2T_kernel_val<...>` | 50 | 230.36 ms | 19.2% | LM head, once per token |
| 3 | `internal::gemvx::kernel<...>` | 1,274 | 215.14 ms | 17.9% | Decode GEMV, 1 per layer |
| 4 | `internal::gemvx::kernel<...>` | 5,096 | 159.18 ms | 13.3% | Decode GEMV, 4 per layer |
| 5 | `elementwise_kernel<128, 2, ...>` | 10,555 | 19.77 ms | 1.6% | Broadcast elementwise ops (e.g. RMSNorm's scaling multiply) |

The biggest difference between the two models is CPU overhead, not GPU work. Gemma-2-2b has about twice the parameters of Llama-3.2-1B (2.6B vs 1.2B), yet its GPU busy time is only 15% higher (1,200 ms vs 1,042 ms). Its wall time doubles (3,755 ms vs 1,840 ms) because it launches twice as many kernels, which comes from 26 layers instead of 16 and 4 RMSNorms per layer instead of 2 (5,250 RMSNorm reductions vs 1,650). Both models spend about 45 µs of CPU time per kernel (1,840 ms / 41,417 and 3,755 ms / 83,419), so wall time tracks the number of kernels rather than the size of the model. The point here is that at batch size 1 in eager PyTorch, both workloads are bound by per-op CPU overhead, and Gemma's deeper, op-heavier layers push GPU utilization from 57% down to 32%.

When the GPU is busy, it is almost entirely running matrix-vector multiplies. GEMV kernels account for 91% of Llama's GPU kernel time and 87% of Gemma's, and the LM head alone takes about 19% in both (201.8 ms and 230.4 ms across 50 calls) because every token multiplies against the full vocabulary (128,256 entries for Llama, 256,000 for Gemma). Attention math itself is small by comparison (about 3 ms of Llama's GEMV time sits inside SDPA).

In closing, I would expect the largest speedup for both models to come from cutting launch count (CUDA graphs, or `torch.compile` to fuse the many small elementwise and norm kernels) rather than from faster matmuls. This holds for batch size 1 in fp32 (larger batches turn these GEMVs into GEMMs and shift the balance back toward the GPU).

*Method:* ran `code/parse_profile.py` on both traces (`--kernel sgemm` for Gemma, which reports both SGEMM variants it uses). The "Where it runs" column for Llama comes from matching each GEMV kernel's `cudaLaunchKernel` to its enclosing `CS2470Profile_` annotation. For Gemma I report calls per decode layer (49 decode passes × 26 layers = 1,274) rather than a projection-level breakdown.
