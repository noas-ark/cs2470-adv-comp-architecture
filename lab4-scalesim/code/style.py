import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, PANEL, INK, MUTED, GRID = "#0b1220", "#131c2e", "#e6edf7", "#8fa0b8", "#243049"
CYAN, PINK, VIOLET = "#38d6f5", "#ff6fa8", "#8b8dfc"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "font.family": "DejaVu Sans", "font.size": 10,
})

def rounded_bars(ax, labels, vals, colors, fmt, width=0.5):
    top = max(vals) * 1.22
    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    r = top * 0.025
    for i, (v, c) in enumerate(zip(vals, colors)):
        ax.add_patch(FancyBboxPatch((i - width / 2, 0), width, v,
                     boxstyle=f"round,pad=0,rounding_size={min(width/2.5, 0.12)}",
                     mutation_aspect=top / (len(vals) + 0.2) * 0.9, fc=c, ec="none", zorder=3))
        ax.text(i, v + r, fmt.format(v), ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK, zorder=4)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=8)
    ax.tick_params(axis="x", length=0, labelsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

def figure(title, subtitle, metrics, labels, series_vals, colors, legend=None):
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.8))
    for ax, (mt, key, fmt) in zip(axs.flat, metrics):
        rounded_bars(ax, labels, [float(v[key]) for v in series_vals], colors, fmt)
        ax.set_title(mt, fontsize=10.5, loc="left", color=INK, fontweight="bold", pad=10)
    fig.text(0.02, 0.965, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.925, subtitle, fontsize=10, color=MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=2.2, w_pad=2)
    return fig
