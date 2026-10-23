import csv
from style import *
rows = {r["name"]: r for r in csv.DictReader(open("results.csv"))}
M = [
    ("Total cycles (incl. prefetch)", "Total Cycles (incl. prefetch)", "{:,.0f}"),
    ("Compute cycles", "Total Cycles", "{:,.0f}"),
    ("Overall utilization (%)", "Overall Util %", "{:.1f}"),
    ("Mapping utilization (%)", "Mapping Efficiency %", "{:.1f}"),
    ("Avg IFMAP SRAM BW (words/cycle)", "Avg IFMAP SRAM BW", "{:.1f}"),
]
sub = "AlexNet Conv1 · local run"
f = figure("scale.cfg vs google.cfg", sub, M + [("Avg OFMAP DRAM BW (words/cycle)", "Avg OFMAP DRAM BW", "{:.1f}")],
           ["scale.cfg", "google.cfg"], [rows["scale"], rows["google"]], [CYAN, PINK])
f.savefig("figures/exI_comparison.png", dpi=170)
M2 = M + [("Avg IFMAP DRAM BW (words/cycle)", "Avg IFMAP DRAM BW", "{:.1f}")]
for fn, title, names, labels in [
    ("exII_array.png", "Sweep: array size", ["arr16", "arr32", "arr64"], ["16x16", "32x32", "64x64"]),
    ("exII_filterbw.png", "Sweep: FilterSRAMBankBandwidth", ["fbw5", "fbw10", "fbw20"], ["5", "10", "20"]),
    ("exII_ifmap.png", "Sweep: IfmapSramSzkB", ["ifmap16", "ifmap64", "ifmap256"], ["16 kB", "64 kB", "256 kB"]),
]:
    f = figure(title, "scale.cfg base · " + sub, M2, labels, [rows[n] for n in names], [VIOLET, CYAN, "#ffd166"])
    f.savefig(f"figures/{fn}", dpi=170)
