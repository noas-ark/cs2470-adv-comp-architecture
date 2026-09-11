# Lab 1: GPU Profiling Part I (PyTorch Profiler)

CS2470 Advanced Computer Architecture, Fall 2026

## Setup

- **Workload:** `meta-llama/Llama-3.2-1B` in fp32, batch size 1, prompt `"Hello, how are you?"`, `max_new_tokens=50` (1 prefill pass + 49 decode passes)
- **Hardware:** NVIDIA A10G (Ampere) on the HUIT academic cluster
- **Software:** PyTorch 2.8.0+cu128, transformers 4.57.6
- **Annotations:** `model.generate()` is wrapped in `CS2470Profile_MyCode` (Part III-A). Inside `modeling_llama.py` (Part III-B), `record_function` blocks mark `DecoderLayer`, `RMSNorm`, `LlamaAttention`, `MLP`, `RotaryEmbedding`, `Embedding`, and `LMHead`, plus `QProj`, `KProj`, `VProj`, `ApplyRotary`, `KVCacheUpdate`, `SDPA`, and `OProj` inside attention.

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

*Method:* zoomed into a `CS2470Profile_RMSNorm` slice in Perfetto (6 aten ops, each with one `cudaLaunchKernel` linked to a GPU kernel), then confirmed across all instances by matching launch correlation IDs to kernels in `profile.json`.

### Q2

Left blank in the handout (numbering goes from Q1 to Q3).
