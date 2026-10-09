"""Exercises III/IV: power-over-time plots and energy.

Usage:
    python plot_power.py results/power_Llama-3.1-8B_bs1.json                 # Ex III
    python plot_power.py results/power_Llama-3.1-8B_bs{1,100}.json \
                         results/power_gemma-2-2b_bs{1,100}.json --out ex4    # Ex IV
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

FMT = "%Y/%m/%d %H:%M:%S.%f"


def load(meta_path):
    meta = json.load(open(meta_path))
    df = pd.read_csv(meta["csv"], skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    pcol = [c for c in df.columns if c.startswith("power.draw")][0]
    df["t"] = pd.to_datetime(df["timestamp"].str.strip(), format=FMT)
    df["P"] = pd.to_numeric(df[pcol], errors="coerce")
    df = df.dropna(subset=["P"]).drop_duplicates("t").sort_values("t")
    start, end = pd.to_datetime(meta["start"], format=FMT), pd.to_datetime(meta["end"], format=FMT)
    df["rel"] = (df["t"] - start).dt.total_seconds()

    # Idle = settled power in the last 2 s of the post-workload padding. The pre-window is
    # not idle: the GPU is still in P0 right after the warmup generate().
    tail = df[df["t"] > df["t"].max() - pd.Timedelta(seconds=2)]
    idle = float(tail["P"].median())
    w = df[(df["t"] >= start) & (df["t"] <= end)]
    meta["sample_ms"] = float(w["t"].diff().dt.total_seconds().median() * 1e3)
    dt = w["t"].diff().dt.total_seconds().fillna(0)
    energy = float((w["P"] * dt).sum())               # J, rectangle integration
    meta.update(df=df, idle_w=idle, energy_j=energy,
                energy_per_iter_j=energy / meta["iters"],
                dyn_energy_j=energy - idle * meta["duration_s"],
                avg_w=float(w["P"].mean()), peak_w=float(w["P"].max()),
                label=f"{meta['model'].split('/')[-1]} bs={meta['batch']}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metas", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--deep_idle", type=float, default=None,
                    help="P8 idle power from a plain nvidia-smi with nothing loaded (W)")
    args = ap.parse_args()
    runs = [load(m) for m in args.metas]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]
    for r, c in zip(runs, colors):
        ax.plot(r["df"]["rel"], r["df"]["P"], color=c, linewidth=1.5,
                label=f"{r['label']}: {r['energy_j']:.0f} J, {r['duration_s']:.1f} s")
        ax.axvline(r["duration_s"], color=c, linestyle=":", alpha=0.6)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
    idle = runs[0]["idle_w"]
    ax.axhline(idle, color="black", linestyle="--", label=f"Idle, model loaded ({idle:.1f} W)")
    if args.deep_idle:
        ax.axhline(args.deep_idle, color="black", linestyle=(0, (1, 2)),
                   label=f"Idle, P8 / nothing loaded ({args.deep_idle:.1f} W)")
    ax.axhline(runs[0]["tdp_w"], color="gray", linestyle="--", label=f"TDP ({runs[0]['tdp_w']:.0f} W)")
    ax.set_ylim(0, runs[0]["tdp_w"] * 1.12)
    ax.set_xlabel("Time since workload start (s)", fontsize=14)
    ax.set_ylabel("Power draw (W)", fontsize=14)
    ax.set_title("GPU power during LLM generation (NVIDIA L4)", fontsize=16)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(alpha=0.3)
    out = args.out or os.path.basename(args.metas[0]).replace(".json", "")
    path = f"results/{out}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved {path}\n")

    print(f"{'run':28s} {'time s':>7s} {'avg W':>6s} {'peak W':>6s} {'idle W':>6s} "
          f"{'energy J':>9s} {'J/iter':>7s} {'J above idle':>12s}")
    for r in runs:
        print(f"{r['label']:28s} {r['duration_s']:7.2f} {r['avg_w']:6.1f} {r['peak_w']:6.1f} "
              f"{r['idle_w']:6.1f} {r['energy_j']:9.1f} {r['energy_per_iter_j']:7.1f} {r['dyn_energy_j']:12.1f}")
        per_tok = r["energy_j"] / (r["iters"] * r["batch"] * r["new_tokens"])
        print(f"{'':28s} energy per generated token: {per_tok:.3f} J, median sample interval {r['sample_ms']:.0f} ms")


if __name__ == "__main__":
    main()
