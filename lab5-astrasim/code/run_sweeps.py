"""Run Astra-Sim analytical configs for Lab 5 and collect comm times into results.csv.

Usage: python run_sweeps.py            (run from lab5-astrasim/code/)
       python run_sweeps.py --astra PATH_TO_ASTRA_SIM_CHECKOUT

Astra-Sim reports one Wall time / Comm time per NPU. Both are simulated time in
ticks, and the analytical backend uses 1 tick = 1 ns, so us = ticks / 1000.
Per the lab, we take the MAX across NPUs (not the sum).
"""
import argparse
import csv
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
RUNS = LAB / "runs"
NETS = RUNS / "nets"

EX1_NPUS = [4, 8, 16]
EX2_COLLECTIVES = ["all_gather", "all_reduce", "all_to_all", "reduce_scatter"]
EX2_BWS = [100, 300, 500, 700, 900]
EX2_LATENCY = 936.25  # ns, held fixed per the lab

COMM_RE = re.compile(r"sys\[(\d+)\], Comm time: (\d+)")
WALL_RE = re.compile(r"sys\[(\d+)\], Wall time: (\d+)")


def write_net(path, topology, npus, bandwidth, latency):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"topology: [ {topology} ]\n"
        f"npus_count: [ {npus} ]\n"
        f"bandwidth: [ {bandwidth} ]  # GB/s\n"
        f"latency: [ {latency} ]  # ns\n"
    )
    return path


def run(astra, name, workload, system, network):
    """Run one simulation, save its log, return (max comm ticks, max wall ticks, n_npus)."""
    binary = astra / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
    remote = astra / "examples/remote_memory/analytical/no_memory_expansion.json"
    RUNS.mkdir(parents=True, exist_ok=True)
    log = RUNS / f"{name}.log"
    proc = subprocess.run(
        [str(binary),
         f"--workload-configuration={workload}",
         f"--system-configuration={system}",
         f"--remote-memory-configuration={remote}",
         f"--network-configuration={network}"],
        capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    log.write_text(out)
    comms = [int(m.group(2)) for m in COMM_RE.finditer(out)]
    walls = [int(m.group(2)) for m in WALL_RE.finditer(out)]
    if not comms:
        raise RuntimeError(f"{name}: no Comm time in output, see {log}")
    return max(comms), max(walls), len(comms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--astra", default=str(Path.home() / "workspace/simulators_part_2/astra-sim"))
    args = ap.parse_args()
    astra = Path(args.astra).resolve()
    ex = astra / "examples"

    rows = []

    # --- Part II baseline: the stock Ring_allgather_16npus example, unmodified ---
    comm, wall, n = run(
        astra, "part2_ring_allgather_16npus",
        ex / "workload/microbenchmarks/all_gather/16npus_1MB/all_gather",
        ex / "system/native_collectives/Ring_4chunks.json",
        ex / "network/analytical/Ring_16npus.yml",
    )
    rows.append(dict(group="part2", collective="all_gather", topology="Ring", npus=n,
                     bandwidth_GBs=50.0, latency_ns=500.0,
                     comm_ticks=comm, wall_ticks=wall, comm_us=comm / 1000))

    # --- Exercise I: All-Gather on a Ring, sweep NPU count. BW/latency held fixed ---
    for npus in EX1_NPUS:
        comm, wall, n = run(
            astra, f"ex1_all_gather_{npus}npus",
            ex / f"workload/microbenchmarks/all_gather/{npus}npus_1MB/all_gather",
            ex / "system/native_collectives/Ring_4chunks.json",
            ex / f"network/analytical/Ring_{npus}npus.yml",
        )
        assert n == npus, f"expected {npus} NPUs, log reported {n}"
        rows.append(dict(group="ex1", collective="all_gather", topology="Ring", npus=npus,
                         bandwidth_GBs=50.0, latency_ns=500.0,
                         comm_ticks=comm, wall_ticks=wall, comm_us=comm / 1000))

    # --- Exercise II: 8-NPU Switch, 4 collectives x 5 bandwidths ---
    for coll in EX2_COLLECTIVES:
        for bw in EX2_BWS:
            net = write_net(NETS / f"switch_8npus_{bw}GBs.yml", "Switch", 8, float(bw), EX2_LATENCY)
            comm, wall, n = run(
                astra, f"ex2_{coll}_{bw}GBs",
                ex / f"workload/microbenchmarks/{coll}/8npus_1MB/{coll}",
                ex / "system/native_collectives/HGX-H100-validated.json",
                net,
            )
            assert n == 8, f"expected 8 NPUs, log reported {n}"
            rows.append(dict(group="ex2", collective=coll, topology="Switch", npus=8,
                             bandwidth_GBs=float(bw), latency_ns=EX2_LATENCY,
                             comm_ticks=comm, wall_ticks=wall, comm_us=comm / 1000))

    # --- Extra probes (not required by the lab): isolate the latency floor ---
    # Sweeping bandwidth alone cannot show what the curves flatten onto, so push
    # bandwidth to effectively infinite and separately zero out latency.
    probes = []
    for coll in EX2_COLLECTIVES:
        net = write_net(NETS / "probe_floor.yml", "Switch", 8, 1000000.0, EX2_LATENCY)
        comm, _, _ = run(astra, f"probe_floor_{coll}",
                         ex / f"workload/microbenchmarks/{coll}/8npus_1MB/{coll}",
                         ex / "system/native_collectives/HGX-H100-validated.json", net)
        probes.append(dict(probe="infinite_bw", collective=coll, bandwidth_GBs=1e6,
                           latency_ns=EX2_LATENCY, comm_us=comm / 1000,
                           latency_hops=comm / 1000 * 1000 / EX2_LATENCY))
    for name, bw, lat in [("bw900_lat0", 900.0, 0.0), ("bw100000_lat0", 100000.0, 0.0),
                          ("bw100000_lat936", 100000.0, EX2_LATENCY)]:
        net = write_net(NETS / f"probe_{name}.yml", "Switch", 8, bw, lat)
        comm, _, _ = run(astra, f"probe_{name}",
                         ex / "workload/microbenchmarks/all_gather/8npus_1MB/all_gather",
                         ex / "system/native_collectives/HGX-H100-validated.json", net)
        probes.append(dict(probe=name, collective="all_gather", bandwidth_GBs=bw,
                           latency_ns=lat, comm_us=comm / 1000,
                           latency_hops=(comm / 1000 * 1000 / lat) if lat else float("nan")))
    # Crossover check: All-Gather overtakes All-Reduce as bandwidth grows
    for bw in (190.0, 200.0, 210.0, 250.0, 400.0):
        net = write_net(NETS / f"probe_x{bw:.0f}.yml", "Switch", 8, bw, EX2_LATENCY)
        for coll in ("all_gather", "all_reduce"):
            comm, _, _ = run(astra, f"probe_x{bw:.0f}_{coll}",
                             ex / f"workload/microbenchmarks/{coll}/8npus_1MB/{coll}",
                             ex / "system/native_collectives/HGX-H100-validated.json", net)
            probes.append(dict(probe="crossover", collective=coll, bandwidth_GBs=bw,
                               latency_ns=EX2_LATENCY, comm_us=comm / 1000,
                               latency_hops=float("nan")))
    pout = LAB / "probes.csv"
    with pout.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(probes[0]))
        w.writeheader()
        w.writerows(probes)
    print(f"wrote {pout} ({len(probes)} probe runs)")

    out = LAB / "results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} runs)")
    for r in rows:
        print(f"  {r['group']:6} {r['collective']:14} npus={r['npus']:2} "
              f"bw={r['bandwidth_GBs']:6.1f}  comm={r['comm_us']:10.3f} us")


if __name__ == "__main__":
    main()
