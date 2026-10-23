"""Run ScaleSim configs for Lab 4 and collect metrics into results.csv.

Usage: python run_sweeps.py   (run from lab4/, uses simulators_part_1 venv python)
"""
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIM = ROOT / "SCALE-Sim"
PY = ROOT / "simulators_part_1" / "bin" / "python"
TOPO = "topologies/conv_nets/alexnet_part.csv"
RUNS = ROOT / "runs"
CFGS = RUNS / "cfgs"


def make_cfg(base, name, **over):
    """Copy base cfg text, overriding keys. Keys are matched case-sensitively as written in the cfg."""
    text = (SIM / "configs" / base).read_text()
    text = re.sub(r"run_name\s*=.*", f"run_name = {name}", text)
    for k, v in over.items():
        text, n = re.subn(rf"(?m)^{k}\s*:\s*.*$", f"{k}: {v}", text)
        assert n == 1, (k, n)
    p = CFGS / f"{name}.cfg"
    p.write_text(text)
    return p


def build():
    CFGS.mkdir(parents=True, exist_ok=True)
    jobs = []  # (group, name, param, value, cfg_path)
    jobs.append(("exI", "scale", "config", "scale.cfg", SIM / "configs" / "scale.cfg"))
    jobs.append(("exI", "google", "config", "google.cfg", SIM / "configs" / "google.cfg"))
    for n in (16, 32, 64):
        p = make_cfg("scale.cfg", f"arr{n}", ArrayHeight=n, ArrayWidth=n)
        jobs.append(("exII", f"arr{n}", "ArraySize", n, p))
    for bw in (5, 10, 20):
        p = make_cfg("scale.cfg", f"fbw{bw}", FilterSRAMBankBandwidth=bw)
        jobs.append(("exII", f"fbw{bw}", "FilterSRAMBankBandwidth", bw, p))
    for kb in (16, 64, 256):
        p = make_cfg("scale.cfg", f"ifmap{kb}", IfmapSramSzkB=kb)
        jobs.append(("exII", f"ifmap{kb}", "IfmapSramSzkB", kb, p))
    # Extra: filter BW sweep with InterfaceBandwidth=USER, where the knob is actually read
    # In USER mode the BW must be a multiple of FilterSRAMBankNum (10), so 1 and 5 crash
    for bw in (10, 20, 40):
        p = make_cfg("scale.cfg", f"fbw_user{bw}", FilterSRAMBankBandwidth=bw, InterfaceBandwidth="USER")
        jobs.append(("extra", f"fbw_user{bw}", "FilterSRAMBankBandwidth(USER mode)", bw, p))
    return jobs


def run(job):
    group, name, param, value, cfg = job
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*/BANDWIDTH_REPORT.csv")):  # already finished, skip
        return job, 0
    log = out / "stdout.txt"
    with open(log, "w") as f:
        r = subprocess.run([str(PY), "-m", "scalesim.scale", "-c", str(cfg), "-t", TOPO, "-p", str(out)],
                           cwd=SIM, stdout=f, stderr=subprocess.STDOUT)
    return job, r.returncode


def read_csv_row(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr = [h.strip() for h in rows[0] if h.strip()]
    vals = [v.strip() for v in rows[1][: len(hdr)]]
    return dict(zip(hdr, vals))


def collect(job):
    group, name, param, value, cfg = job
    d = next(p for p in (RUNS / name).glob("*") if p.is_dir())  # <log dir>/<run_name>
    comp = read_csv_row(d / "COMPUTE_REPORT.csv")
    bw = read_csv_row(d / "BANDWIDTH_REPORT.csv")
    return {"group": group, "name": name, "param": param, "value": value, **comp, **bw}


if __name__ == "__main__":
    jobs = build()
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(run, jobs))
    bad = [(j[1], rc) for j, rc in res if rc != 0]
    if bad:
        print("FAILED", bad)
        sys.exit(1)
    rows = [collect(j) for j in jobs]
    keys = list(rows[0].keys())
    with open(ROOT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote results.csv with", len(rows), "rows")
