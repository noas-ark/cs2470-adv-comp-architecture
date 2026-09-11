#!/usr/bin/env python3
"""Parse a PyTorch Profiler Chrome trace (profile.json) and report workload statistics.

Reports:
  1. Total CPU time and total GPU time
  2. Time for a specific kernel (exact name or substring)
  3. Number of decoder layers executed
  4. Number of attention blocks executed
  5. Top-N GPU kernels by total time

Usage:
  python parse_profile.py profile.json --kernel ampere_sgemm_64x32_sliced1x4_tn --top 10
  python parse_profile.py profile.json --json stats.json   # also dump results as JSON
"""
import argparse
import json
import re
from collections import defaultdict

PREFIX = "CS2470Profile_"
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def load_events(path):
    """Return the complete ("X") events from a Chrome trace. Times are in microseconds."""
    with open(path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    return [e for e in events if e.get("ph") == "X" and "dur" in e]


def union_us(intervals):
    """Total length of a set of (start, end) intervals, counting any overlap once."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def cpu_annotations(events, pattern):
    """CPU-side record_function slices whose full name matches `pattern` (regex).

    Each record_function label appears twice in the trace: once on the CPU thread
    (cat=user_annotation) and once mirrored onto the GPU stream (cat=gpu_user_annotation).
    Counting only the CPU copy avoids double counting.
    """
    rx = re.compile(pattern)
    return [e for e in events if e.get("cat") == "user_annotation" and rx.fullmatch(e["name"])]


def cpu_gpu_time(events):
    """Task 1: total CPU and GPU time.

    - CPU wall time: duration of CS2470Profile_MyCode (the profiled generate() call).
      Falls back to the span of all CPU events if the annotation is missing.
    - CPU time in aten ops: union of all aten op intervals, so nested ops
      (e.g. aten::linear containing aten::mm) are not double counted.
    - GPU busy time: union of kernel / memcpy / memset intervals on the GPU.
    """
    mycode = cpu_annotations(events, PREFIX + "MyCode")
    if mycode:
        cpu_wall = sum(e["dur"] for e in mycode)
    else:
        cpu = [e for e in events if e.get("cat") in ("cpu_op", "user_annotation")]
        cpu_wall = max(e["ts"] + e["dur"] for e in cpu) - min(e["ts"] for e in cpu)
    cpu_aten = union_us((e["ts"], e["ts"] + e["dur"]) for e in events if e.get("cat") == "cpu_op")
    gpu = [e for e in events if e.get("cat") in GPU_CATS]
    gpu_busy = union_us((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    return {
        "cpu_wall_ms": cpu_wall / 1e3,
        "cpu_in_aten_ms": cpu_aten / 1e3,
        "gpu_busy_ms": gpu_busy / 1e3,
        "gpu_kernel_count": sum(1 for e in gpu if e["cat"] == "kernel"),
        "gpu_utilization": gpu_busy / cpu_wall if cpu_wall else 0.0,
    }


def kernel_stats(events, query):
    """Task 2: count and timing for a kernel. Exact name match wins, otherwise substring."""
    kernels = [e for e in events if e.get("cat") == "kernel"]
    hits = [e for e in kernels if e["name"] == query] or [e for e in kernels if query in e["name"]]
    by_name = defaultdict(list)
    for e in hits:
        by_name[e["name"]].append(e["dur"])
    return {
        name: {
            "count": len(d),
            "total_ms": sum(d) / 1e3,
            "avg_us": sum(d) / len(d),
            "min_us": min(d),
            "max_us": max(d),
        }
        for name, d in by_name.items()
    }


def model_counts(events):
    """Tasks 3 and 4: decoder layer and attention block executions.

    Any label ending in DecoderLayer / Attention counts, so the same parser works for
    CS2470Profile_LlamaAttention, CS2470Profile_Gemma2Attention, etc.
    Forward passes are counted from the LM head (one per generated token).
    """
    layers = cpu_annotations(events, PREFIX + r"\w*DecoderLayer")
    attention = cpu_annotations(events, PREFIX + r"\w*Attention")
    passes = len(cpu_annotations(events, PREFIX + "LMHead")) or len(cpu_annotations(events, PREFIX + "Embedding"))
    return {
        "forward_passes": passes,
        "decoder_layer_executions": len(layers),
        "layers_per_pass": len(layers) / passes if passes else None,
        "attention_executions": len(attention),
    }


def top_kernels(events, n):
    """Task 5: top-N GPU kernels by total time."""
    agg = defaultdict(lambda: [0, 0.0])
    for e in events:
        if e.get("cat") == "kernel":
            agg[e["name"]][0] += 1
            agg[e["name"]][1] += e["dur"]
    total = sum(t for _, t in agg.values()) or 1.0
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])[:n]
    return {
        "distinct_kernels": len(agg),
        "top": [
            {"name": name, "count": c, "total_ms": t / 1e3, "pct_gpu_kernel_time": 100 * t / total}
            for name, (c, t) in rows
        ],
    }


def short(name, width=90):
    name = name[5:] if name.startswith("void ") else name
    return name if len(name) <= width else name[: width - 3] + "..."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="path to profile.json")
    ap.add_argument("--kernel", default="ampere_sgemm_64x32_sliced1x4_tn", help="kernel name or substring")
    ap.add_argument("--top", type=int, default=10, help="number of top kernels to list")
    ap.add_argument("--json", help="optional path to write all results as JSON")
    args = ap.parse_args()

    events = load_events(args.trace)
    t = cpu_gpu_time(events)
    k = kernel_stats(events, args.kernel)
    m = model_counts(events)
    top = top_kernels(events, args.top)

    print(f"== {args.trace} ==\n")
    print("[1] CPU / GPU time")
    print(f"    CPU wall time (CS2470Profile_MyCode) : {t['cpu_wall_ms']:10.2f} ms")
    print(f"    CPU time inside aten ops             : {t['cpu_in_aten_ms']:10.2f} ms")
    print(f"    GPU busy time (kernels + memcpy/set) : {t['gpu_busy_ms']:10.2f} ms  ({t['gpu_kernel_count']:,} kernels)")
    print(f"    GPU utilization (busy / wall)        : {100 * t['gpu_utilization']:9.1f} %\n")

    print(f"[2] Kernel '{args.kernel}'")
    if not k:
        print("    not found in this trace")
    for name, s in k.items():
        print(f"    {short(name)}")
        print(f"      {s['count']} calls | total {s['total_ms']:.3f} ms | avg {s['avg_us']:.1f} us"
              f" | min {s['min_us']:.1f} us | max {s['max_us']:.1f} us")
    print()

    per_pass = f" ({m['layers_per_pass']:g} per forward pass)" if m["layers_per_pass"] else ""
    print(f"[3] Decoder layer executions: {m['decoder_layer_executions']}{per_pass}")
    print(f"[4] Attention block executions: {m['attention_executions']}\n")

    print(f"[5] Top {args.top} GPU kernels by total time (of {top['distinct_kernels']} distinct)")
    print(f"    {'rank':>4}  {'total ms':>9}  {'calls':>6}  {'% GPU':>6}  kernel")
    for i, r in enumerate(top["top"], 1):
        print(f"    {i:>4}  {r['total_ms']:9.2f}  {r['count']:>6}  {r['pct_gpu_kernel_time']:6.1f}  {short(r['name'])}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"trace": args.trace, "time": t, "kernel": k, "counts": m, "top_kernels": top}, f, indent=2)


if __name__ == "__main__":
    main()
