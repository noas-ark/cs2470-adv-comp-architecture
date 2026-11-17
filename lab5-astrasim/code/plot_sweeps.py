"""Plot Lab 5 figures from results.csv.

Usage: python plot_sweeps.py   (run from lab5-astrasim/code/)
Writes ../images/exI_npu_scaling.png and ../images/exII_bandwidth.png
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from style import AMBER, CYAN, GRID, INK, MUTED, PINK, VIOLET, header, line_panel, rounded_bars

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
IMG = LAB / "images"
IMG.mkdir(exist_ok=True)

COLORS = {"all_gather": CYAN, "all_reduce": PINK, "all_to_all": VIOLET, "reduce_scatter": AMBER}
NICE = {"all_gather": "All-Gather", "all_reduce": "All-Reduce",
        "all_to_all": "All-to-All", "reduce_scatter": "Reduce-Scatter"}

rows = list(csv.DictReader((LAB / "results.csv").open()))


def fit(pts):
    """Least-squares fit of comm = A/BW + B. Returns (A, B, max_abs_residual)."""
    xs = [1 / b for b, _ in pts]
    ys = [t for _, t in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    A = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    B = my - A * mx
    return A, B, max(abs(A * x + B - y) for x, y in zip(xs, ys))


# ---------- Exercise I: All-Gather on a Ring, NPU count sweep ----------
ex1 = sorted((int(r["npus"]), float(r["comm_us"])) for r in rows if r["group"] == "ex1")
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))

rounded_bars(axs[0], [f"{n} NPUs" for n, _ in ex1], [t for _, t in ex1],
             [CYAN, PINK, VIOLET], "{:,.1f}")
axs[0].set_title("Comm time vs NPU count", fontsize=10.5, loc="left",
                 color=INK, fontweight="bold", pad=10)
axs[0].set_ylabel("Comm time (us)", fontsize=9)

steps = [(n - 1, t) for n, t in ex1]
slope = (steps[-1][1] - steps[0][1]) / (steps[-1][0] - steps[0][0])
intercept = steps[0][1] - slope * steps[0][0]
line_panel(axs[1],
           [("measured", [s for s, _ in steps], [t for _, t in steps], CYAN),
            (f"{slope:.2f} us x (N-1)", [0, 15], [intercept, intercept + slope * 15], MUTED)],
           "ring steps (N - 1)", "Comm time (us)", xticks=[0, 3, 7, 15])
axs[1].lines[1].set_linestyle("--")
axs[1].lines[1].set_marker("")
axs[1].set_title("Same data against ring step count", fontsize=10.5, loc="left",
                 color=INK, fontweight="bold", pad=10)
axs[1].legend(frameon=False, labelcolor=MUTED, fontsize=8.5, loc="upper left")
for n, t in ex1:
    axs[1].annotate(f"{t:,.1f}", (n - 1, t), textcoords="offset points",
                    xytext=(6, -10), fontsize=8, color=INK)

header(fig, "Exercise I: All-Gather scaling on a Ring",
       "1 MiB per rank, 50 GB/s links, 500 ns latency, Ring_4chunks. Max Comm time across NPUs.")
fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.5)
fig.savefig(IMG / "exI_npu_scaling.png", dpi=150)
plt.close(fig)

# ---------- Exercise II: bandwidth sweep, 4 collectives ----------
ex2 = {}
for r in rows:
    if r["group"] == "ex2":
        ex2.setdefault(r["collective"], []).append((float(r["bandwidth_GBs"]), float(r["comm_us"])))
for v in ex2.values():
    v.sort()

order = ["all_to_all", "all_gather", "all_reduce", "reduce_scatter"]
bws = [b for b, _ in ex2["all_gather"]]

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))

line_panel(axs[0], [(NICE[c], [b for b, _ in ex2[c]], [t for _, t in ex2[c]], COLORS[c])
                    for c in order],
           "Link bandwidth (GB/s)", "Comm time (us)", xticks=bws)
for c in order:
    _, B, _ = fit(ex2[c])
    axs[0].axhline(B, color=COLORS[c], lw=0.8, ls=":", alpha=0.55, zorder=1)
axs[0].set_title("Comm time vs bandwidth (dotted = latency floor)", fontsize=10.5,
                 loc="left", color=INK, fontweight="bold", pad=10)
axs[0].legend(frameon=False, labelcolor=MUTED, fontsize=8.5)

line_panel(axs[1], [(NICE[c], [b for b, _ in ex2[c]],
                     [t / ex2[c][0][1] * 100 for _, t in ex2[c]], COLORS[c]) for c in order],
           "Link bandwidth (GB/s)", "Comm time (% of time at 100 GB/s)", xticks=bws)
# All-to-All, All-Reduce and Reduce-Scatter fall almost exactly on top of each other
# here (all shed ~22%), so give them separate dash patterns to keep them readable.
for ln, dash in zip(axs[1].lines, [(None, None), (None, None), (5, 2), (1, 2)]):
    ln.set_dashes(dash) if dash[0] else None
axs[1].set_title("Same curves normalised to each collective's 100 GB/s point",
                 fontsize=10.5, loc="left", color=INK, fontweight="bold", pad=10)
axs[1].set_ylim(0, 105)
axs[1].legend(frameon=False, labelcolor=MUTED, fontsize=8.5)

header(fig, "Exercise II: bandwidth vs collective communication time",
       "8 NPUs, Switch topology, 936.25 ns latency, HGX-H100-validated system, 1 MiB traces. 20 runs.")
fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.5)
fig.savefig(IMG / "exII_bandwidth.png", dpi=150)
plt.close(fig)

print("wrote", IMG / "exI_npu_scaling.png")
print("wrote", IMG / "exII_bandwidth.png")
print()
print("fitted comm = A/BW + B:")
for c in order:
    A, B, res = fit(ex2[c])
    print(f"  {NICE[c]:15} A={A:8.1f}  B={B:7.2f} us  max resid={res:.4f} us")
