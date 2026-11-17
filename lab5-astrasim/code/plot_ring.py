"""Explainer figure: how a ring All-Gather moves data, and why it costs N-1 steps.

Usage: python plot_ring.py   (run from lab5-astrasim/code/)
Writes ../images/ring_allgather_explained.png
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from style import BG, CYAN, GRID, INK, MUTED, PANEL, PINK, VIOLET, AMBER, footer

IMG = Path(__file__).resolve().parent.parent / "images"
IMG.mkdir(exist_ok=True)

N = 4
COLORS = [CYAN, PINK, VIOLET, AMBER]

fig = plt.figure(figsize=(11.5, 4.9))
fig.patch.set_facecolor(BG)
axL = fig.add_axes([0.03, 0.10, 0.30, 0.66])
axR = fig.add_axes([0.40, 0.10, 0.57, 0.66])

# ---- left: the ring itself ----
axL.set_facecolor(PANEL)
axL.set_xlim(-1.5, 1.5); axL.set_ylim(-1.5, 1.5); axL.set_aspect("equal")
axL.set_xticks([]); axL.set_yticks([])
for s in axL.spines.values():
    s.set_color(GRID)
pos = [(math.cos(math.pi / 2 - 2 * math.pi * i / N), math.sin(math.pi / 2 - 2 * math.pi * i / N)) for i in range(N)]
for i, (x, y) in enumerate(pos):
    axL.add_patch(Circle((x, y), 0.30, fc=PANEL, ec=COLORS[i], lw=2.2, zorder=3))
    axL.text(x, y, f"R{i}", ha="center", va="center", color=COLORS[i],
             fontsize=11, fontweight="bold", zorder=4)
for i in range(N):
    x1, y1 = pos[i]; x2, y2 = pos[(i + 1) % N]
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy); ux, uy = dx / d, dy / d
    axL.add_patch(FancyArrowPatch((x1 + ux * 0.34, y1 + uy * 0.34), (x2 - ux * 0.38, y2 - uy * 0.38),
                                  arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=1.5, zorder=2))
axL.text(0, 0, "1 MiB\nper hop", ha="center", va="center", color=MUTED, fontsize=8.5)
axL.set_title("Every rank sends to one neighbour", fontsize=10, loc="left",
              color=INK, fontweight="bold", pad=8)

# ---- right: what each rank holds after each step ----
axR.set_facecolor(PANEL)
axR.set_xlim(-0.6, N + 0.2); axR.set_ylim(-0.9, N - 0.3); axR.invert_yaxis()
axR.set_xticks([]); axR.set_yticks([])
for s in axR.spines.values():
    s.set_color(GRID)

cell, gap = 0.19, 0.012
for step in range(N):                      # columns: start, then N-1 steps
    for rank in range(N):                  # rows: the ranks
        for b in range(N):                 # the N blocks, one slot each
            held = ((rank - b) % N) <= step
            x = step + b * (cell + gap) - 0.38
            y = rank - cell / 2
            axR.add_patch(FancyBboxPatch((x, y), cell, cell,
                                         boxstyle="round,pad=0,rounding_size=0.04",
                                         fc=COLORS[b] if held else "#1b2740",
                                         ec="none", alpha=1.0 if held else 1.0, zorder=3))
    label = "start" if step == 0 else f"after step {step}"
    axR.text(step - 0.38 + (N * (cell + gap) - gap) / 2, -0.72, label,
             ha="center", va="center", color=MUTED if step else INK, fontsize=9,
             fontweight="bold" if step == N - 1 else "normal")
for rank in range(N):
    axR.text(-0.52, rank, f"R{rank}", ha="right", va="center", color=COLORS[rank], fontsize=9.5,
             fontweight="bold")
axR.text(N - 1 - 0.38 + (N * (cell + gap) - gap) / 2, N - 0.55,
         "all ranks hold all 4 blocks", ha="center", va="center", color=INK, fontsize=9)
axR.set_title("A 4-rank ring is complete after 3 steps, not 4", fontsize=10, loc="left",
              color=INK, fontweight="bold", pad=8)

fig.text(0.012, 0.955, "A ring All-Gather needs N-1 steps, and each step costs the same",
         fontsize=15, fontweight="bold", color=INK, va="top")
fig.text(0.012, 0.886,
         "Each rank starts with its own 1 MiB block and passes one block per step. The amount crossing any single link "
         "per step never changes,\nso only the number of steps grows with N. That is why Exercise I is linear in N-1: "
         "3 steps at 4 NPUs, 7 at 8, 15 at 16.",
         fontsize=9.5, color=MUTED, va="top")
footer(fig, "Diagram, not measured output. The measured per-step cost is 21.57 us at 50 GB/s (1 MiB / 50 GB/s = 20.97 us, plus the 500 ns link latency).")
fig.savefig(IMG / "ring_allgather_explained.png", dpi=150, facecolor=BG)
plt.close(fig)
print("wrote", IMG / "ring_allgather_explained.png")
