"""Plot SRAM access activity per cycle from ScaleSim trace files (Lab 4, Part I)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
out = sys.argv[2]

def load(name):
    cyc, cnt = [], []
    with open(f"{d}/{name}_SRAM_TRACE.csv") as f:
        for line in f:
            p = line.split(",")
            c = int(float(p[0]))
            n = sum(1 for x in p[1:] if x.strip() and float(x) >= 0)
            cyc.append(c); cnt.append(n)
    return np.array(cyc), np.array(cnt)

series = [("IFMAP", "Input reads", "#2a78d6"),
          ("FILTER", "Filter reads", "#eb6834"),
          ("OFMAP", "Output writes", "#1baf7a")]
data = {k: load(k) for k, _, _ in series}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), gridspec_kw={"width_ratios": [2.2, 1]})
BIN = 200
for r, (k, label, col) in enumerate(series):
    c, n = data[k]
    ax = axes[r, 0]
    m = len(c) // BIN * BIN
    ax.plot(c[:m].reshape(-1, BIN).mean(1), n[:m].reshape(-1, BIN).mean(1), color=col, lw=1.5)
    ax.set_ylabel(f"{label}\n(words/cycle)")
    ax.set_ylim(0, 33)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax = axes[r, 1]
    w = slice(0, 160)
    ax.plot(c[w], n[w], color=col, lw=1.5, drawstyle="steps-post")
    ax.set_ylim(0, 33)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0, 0].set_title(f"Whole run (average of each {BIN} cycles)", loc="left", color=MUTED, fontsize=10)
axes[0, 1].set_title("First 160 cycles (every cycle)", loc="left", color=MUTED, fontsize=10)
axes[2, 0].set_xlabel("Cycle"); axes[2, 1].set_xlabel("Cycle")
fig.suptitle("ScaleSim, AlexNet Conv1 on a 32x32 weight stationary array: SRAM activity per cycle",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=150)
for k, _, _ in series:
    c, n = data[k]
    print(k, "cycles", len(c), "total words", int(n.sum()), "max/cycle", int(n.max()),
          "first active", int(c[n > 0][0]), "last active", int(c[n > 0][-1]))
