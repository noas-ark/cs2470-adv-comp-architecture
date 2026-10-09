"""Part II + Exercises I/II: time kernels with PyTorch Profiler, write results/kernels.json."""
import json
import os

import torch
import torch.nn.functional as F
from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity

DTYPE = torch.float16
DEVICE = torch.device("cuda")
BYTES = 2  # fp16
ITERS = 20


def gpu_time_s(fn):
    """Mean GPU kernel time per call (seconds), measured with PyTorch Profiler."""
    with torch.inference_mode():
        for _ in range(10):  # warmup
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(ITERS):
                fn()
            torch.cuda.synchronize()
    # Sum only GPU-side kernel events. CPU ops like aten::cudnn_convolution also carry
    # device time, so summing every row double-counts. This matches the profiler's own
    # "Self CUDA time total" line.
    total_us = 0.0
    for e in prof.events():
        if e.device_type == DeviceType.CUDA:
            t = getattr(e, "self_device_time_total", None)
            total_us += t if t is not None else e.self_cuda_time_total
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))
    print(f"Kernel time per call: {total_us / ITERS:.1f} us")
    return total_us / ITERS / 1e6


def gemm(name, M, K, N):
    A = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    B = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
    t = gpu_time_s(lambda: A @ B)
    flops = 2 * M * N * K
    nbytes = BYTES * (M * K + K * N + M * N)
    return record(name, flops, nbytes, t)


def conv(name, N, C, H, W, Kout, R, S):
    x = torch.randn(N, C, H, W, device=DEVICE, dtype=DTYPE)
    w = torch.randn(Kout, C, R, S, device=DEVICE, dtype=DTYPE)
    t = gpu_time_s(lambda: F.conv2d(x, w, padding=R // 2))
    Ho, Wo = H, W  # stride 1, same padding
    flops = 2 * N * Kout * Ho * Wo * C * R * S
    nbytes = BYTES * (N * C * H * W + Kout * C * R * S + N * Kout * Ho * Wo)
    return record(name, flops, nbytes, t)


def record(name, flops, nbytes, t):
    r = {"name": name, "flops": flops, "bytes": nbytes, "time_s": t,
         "arithmetic_intensity": flops / nbytes, "throughput": flops / t}
    print(f"{name}: {flops:.3e} FLOP, {nbytes:.3e} B, {t*1e3:.3f} ms, "
          f"AI={r['arithmetic_intensity']:.2f} FLOP/B, {r['throughput']/1e12:.2f} TFLOP/s\n")
    return r


if __name__ == "__main__":
    print(torch.cuda.get_device_name())
    kernels = [
        gemm("GEMM 4096x4096x4096", 4096, 4096, 4096),
        conv("Conv2d 3x3", 32, 64, 56, 56, 64, 3, 3),  # ResNet-50 conv2_x layer, batch 32
        gemm("GEMM A 8192x8192x4096", 8192, 8192, 4096),
        gemm("GEMM B 1x8192x4096", 1, 8192, 4096),
    ]
    os.makedirs("results", exist_ok=True)
    with open("results/kernels.json", "w") as f:
        json.dump(kernels, f, indent=2)
    print("Wrote results/kernels.json")
