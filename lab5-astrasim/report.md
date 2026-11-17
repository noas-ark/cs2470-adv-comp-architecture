# Lab 5: Astra-Sim

11/16/26 | Noa Schwartz

Lab instructions: [Simulators Part II, Astra-Sim (Google Doc)](https://docs.google.com/document/d/1ax36L7ZgWKiysm1iY2KLCzsGJ7xxu8yGjIKTo4PFKz0/edit?usp=sharing)

Code and data: [github.com/noas-ark/cs2470-adv-comp-architecture/tree/main/lab5-astrasim](https://github.com/noas-ark/cs2470-adv-comp-architecture/tree/main/lab5-astrasim)

## Setup

- **Cluster:** HUIT academic cluster, interactive CPU session on node `general-dy-general-cr-17`, requested with `srun -p general -c 8 --mem=16G --pty bash`. CPU only, no GPU involved, even though the thing being modeled is distributed GPU training
- **Simulator:** [astra-sim/astra-sim](https://github.com/astra-sim/astra-sim) at the revision the lab pins, `4ce9ecfe746a3ee5fe47559a11d91644d641fa7f`, built from source with submodules inside the shared course container (`source ~/169291/astra-sim/activate.sh`, then `astra-sim-shell`)
- **Backend:** analytical, congestion aware (`AstraSim_Analytical_Congestion_Aware`)
- **Inputs:** the stock files shipped in `examples/`. The only thing I edit is a copy of the network YAML, which is what Exercise II sweeps
- **Driver:** `code/run_sweeps.py` runs all 24 simulations and writes `results.csv`. `code/plot_sweeps.py` draws the two figures. Raw per-run logs are in `runs/`
- **Units:** the analytical backend uses 1 tick = 1 ns, so microseconds = ticks / 1000. This is simulated time. Host elapsed time is a separate thing (the whole 16 NPU example finishes in well under a second of real time)
- **Which number I report:** Astra-Sim prints a Wall time and a Comm time per NPU. Per the lab I take the **max across NPUs**, never the sum. In practice every NPU reported the identical value in all 24 runs, so max and min coincide here

Two notes from actually doing this:

**The results do not depend on the machine.** I ran the full sweep twice: once on a separate machine with its own build of the same pinned commit, and once in the course container on the cluster. All 41 runs (24 for the exercises, 17 probes) returned identical tick counts, down to the digit; the cluster side is recorded in `cluster-verification.csv`. The analytical backend has no timing-dependent behaviour, so what comes out is a function of the inputs and the revision, not of the host. Worth knowing before spending a compute allocation on a re-run.

**Note:** inside the container `$HOME` is `/home/nos673`, while the real home is `/shared/home/nos673`. Anything built from `$HOME` fails with "No such file or directory" even though the same path works outside. I used absolute `/shared/home/...` paths instead. The container preserves your working directory, so `pwd` is the quick way to find out where you actually are.

## Part II: Running the simulator

Running `Ring_allgather_16npus.sh` unmodified (All-Gather, 16 NPUs, Ring, 50 GB/s, 500 ns, `Ring_4chunks.json`):

| Metric | Ticks | Microseconds |
| --- | --- | --- |
| Wall time | 323,560 | 323.560 |
| Comm time | 323,560 | 323.560 |
| Exposed communication | 323,560 | 323.560 |

All 16 NPUs report the same three numbers.

**Why are Wall time and Comm time equal?** Because this workload is nothing but communication. The Chakra trace for each rank holds exactly one node, a collective comm op, and no compute node at all (I decoded it, see Part III). Wall time is the total simulated time the NPU was busy and Comm time is the part of it spent in communication, so when there is no compute to run there is nothing for the other category to hold and nothing for the collective to overlap with. The log's third number says the same thing from the other side: all 323,560 cycles are *exposed* communication, meaning none of it was hidden behind compute.

The practical point is that this is the pathological case. In a real training step you would want Comm time well below Wall time, because that gap is the communication you managed to overlap with useful math.

## Part III: How the simulator works

Four input directories, one per concern, all selected by the run script:

| Directory | Format | What it specifies |
| --- | --- | --- |
| `workload/` | binary Chakra execution traces, `.et`, one file per NPU | what each rank actually does |
| `network/` | YAML, for the analytical backend | topology, NPU count, link bandwidth, link latency |
| `system/` | JSON | scheduling policy, chunking, and which algorithm implements each collective |
| `remote_memory/` | JSON | memory expansion model, here `NO_MEMORY_EXPANSION` |

A few things worth noting after poking at them:

**The workload traces are tiny.** Each `.et` is 81 to 85 bytes. They are binary protobuf, so they cannot be edited as text, but they can be read with the Chakra schema that the build generates (`extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py`). Decoding rank 0 of each 8 NPU microbenchmark gives one node each:

| Trace | Node type | `comm_size` |
| --- | --- | --- |
| `all_gather/8npus_1MB` | comm coll | 1,048,576 |
| `all_reduce/8npus_1MB` | comm coll | 1,048,576 |
| `all_to_all/8npus_1MB` | comm coll | 1,048,576 |
| `reduce_scatter/8npus_1MB` | comm coll | 1,048,576 |

So all four say 1 MiB, which is exactly the lab's warning: the same `comm_size` means different amounts of traffic depending on the collective. Exercise II turns out to measure that difference directly.

**The network file is the only knob with a real number in it.** It is four lines, which is why it is the thing to sweep:

```yaml
topology: [ Switch ]
npus_count: [ 8 ]
bandwidth: [ 400.0 ]  # GB/s
latency: [ 936.25 ]  # ns
```

**The system file picks the algorithm, not the collective.** `Ring_4chunks.json` and `HGX-H100-validated.json` both set every `*-implementation` to `ring`. The collective itself comes from the workload trace. The two differ in `active-chunks-per-dimension` (1 vs 2) and `local-mem-bw` (1600 vs 3350). `preferred-dataset-splits: 4` is the 4 chunks in the Ring config's name, and it shows up in the Exercise II results in a way I did not expect (see the floor analysis).

## Exercise I: All-Gather scaling with NPU count

All-Gather on a Ring, per rank input fixed at 1 MiB, bandwidth 50 GB/s, latency 500 ns, `Ring_4chunks.json`. N swept over 4, 8, 16.

| NPUs | Comm time (us) | Ring steps (N-1) | Time per step (us) | Total data gathered |
| --- | --- | --- | --- | --- |
| 4 | 64.744 | 3 | 21.581 | 4 MiB |
| 8 | 151.016 | 7 | 21.574 | 8 MiB |
| 16 | 323.560 | 15 | 21.571 | 16 MiB |

![All-Gather comm time vs NPU count](images/exI_npu_scaling.png)

**The trend is linear in N-1, not in N.** This is cleaner than I expected. The slope between 4 and 8 NPUs is 21.568 us per step and the slope between 8 and 16 is also 21.568 us per step, to three decimals. Fitting a line through all three points gives an intercept of 0.04 us, which is 40 ns, basically zero. So:

```
comm time (us) = 21.568 x (N - 1)
```

The measured ratios say the same thing: 151.016 / 64.744 = 2.3325 against a predicted 7/3 = 2.3333, and 323.560 / 64.744 = 4.9975 against a predicted 15/3 = 5.

**Why.** Ring All-Gather takes N-1 steps. On each step every rank passes one 1 MiB block to its neighbor, so the amount crossing any single link per step is fixed no matter how big the ring is. Only the number of steps grows. Checking the per step cost against first principles: 1 MiB at 50 GB/s is 1,048,576 / 50e9 = 20.97 us, plus the 500 ns link latency gives 21.47 us. Measured is 21.57 us, so roughly 0.1 us per step is chunking and endpoint overhead that the simple model leaves out.

*Method:* the per step figure is just comm time divided by N-1. The slope is the difference between consecutive points divided by the difference in step count, computed in `code/plot_sweeps.py`; both intervals agreeing to three decimals is what makes the straight line claim safe rather than eyeballed.

**The part worth saying out loud:** time grows with N but the data gathered also grows with N, so cost per byte delivered is nearly flat. That is the sense in which ring All-Gather is bandwidth efficient. What it is not is latency efficient, because N-1 sequential steps means the ring gets deeper every time you add a GPU, and each step is a dependency, not something you can parallelize away. A 16 NPU ring spends 323 us moving 1 MiB per rank. That is the argument for hierarchical or tree collectives once N gets large, and it is why real multi node systems do not run one flat ring across every GPU.

## Exercise II: Bandwidth's impact on different collectives

8 NPUs, Switch topology, latency fixed at 936.25 ns, `HGX-H100-validated.json`, 1 MiB traces. Bandwidth swept over 100, 300, 500, 700, 900 GB/s for each of 4 collectives, 20 runs.

Comm time in microseconds:

| Bandwidth (GB/s) | All-Gather | All-Reduce | All-to-All | Reduce-Scatter |
| --- | --- | --- | --- | --- |
| 100 | 97.157 | 70.479 | 139.877 | 35.591 |
| 300 | 49.973 | 58.851 | 116.825 | 29.675 |
| 500 | 40.520 | 56.571 | 112.305 | 28.515 |
| 700 | 36.488 | 55.545 | 110.271 | 27.993 |
| 900 | 34.227 | 55.031 | 109.253 | 27.731 |

![Bandwidth vs comm time, one curve per collective](images/exII_bandwidth.png)

In the right hand panel All-to-All, All-Reduce and Reduce-Scatter sit almost exactly on top of each other, which is the finding, not a plotting mistake. They all shed about 22% while All-Gather sheds 65%.

### Which collective benefits most from added bandwidth

**All-Gather, by a wide margin.** Going from 100 to 900 GB/s:

| Collective | 100 GB/s | 900 GB/s | Saved | Saved % |
| --- | --- | --- | --- | --- |
| All-Gather | 97.157 | 34.227 | 62.93 us | 64.8% |
| All-to-All | 139.877 | 109.253 | 30.62 us | 21.9% |
| All-Reduce | 70.479 | 55.031 | 15.45 us | 21.9% |
| Reduce-Scatter | 35.591 | 27.731 | 7.86 us | 22.1% |

All 20 points fit `comm = A / BW + B` almost perfectly, with a worst case residual of 0.05 us, which is 0.04%. Splitting each curve into its bandwidth sensitive term A and its fixed floor B:

| Collective | A (us x GB/s) | A relative to Reduce-Scatter | Floor B (us) |
| --- | --- | --- | --- |
| All-Gather | 7,079 | 8.00x | 26.37 |
| All-to-All | 3,449 | 3.90x | 105.38 |
| All-Reduce | 1,740 | 1.97x | 53.08 |
| Reduce-Scatter | 885 | 1.00x | 26.74 |

*Method:* A and B are a least squares fit of comm time against 1/BW over all 5 points per collective, not a two point solve, so a bad fit would show up as a large residual. It does not: the worst point in the whole table is off by 0.05 us.

The A column is a data volume ranking, and for three of the four it lands on the textbook number exactly. With N = 8:

- **Reduce-Scatter** takes 1 MiB total and leaves each rank with 1/N of it, so each rank moves (N-1)/N = 0.875 MiB. Call that 1x.
- **All-Reduce** is a Reduce-Scatter followed by an All-Gather over the same buffer, 2(N-1)/N = 1.75 MiB, so **2x**. Measured 1.97x.
- **All-Gather**'s `comm_size` is per rank input, not total, so every rank has to receive the other N-1 ranks' megabyte, 7 MiB. That is **8x** Reduce-Scatter's 0.875 MiB. Measured 8.00x.

All-Gather therefore moves the most bytes of the four while also having the lowest fixed floor, so it is the one with the most to gain from a fatter link. All-to-All lands at 3.90x, close to 4x, which I could not derive cleanly; the ring based All-to-All in this system config does roughly 4 times Reduce-Scatter's work in both bytes and steps, and its floor behaves the same way (see below), so the two are consistent, but I am reporting that as measured rather than derived.

One consequence visible in the left panel: All-Gather starts slower than All-Reduce at 100 GB/s (97 vs 70 us) and ends faster at 900 (34 vs 55). I measured the crossover at 200 GB/s, where both land on 61.75 us. Which collective is "expensive" is not a fixed property, it depends on the link you run it over.

### Why the curves flatten

Because the bandwidth term is hyperbolic and it is sitting on top of a floor that bandwidth cannot touch. A/BW falls off fast early and barely moves later. For All-Gather, each extra 200 GB/s buys:

| Step | Comm time | Change |
| --- | --- | --- |
| 100 -> 300 | 97.157 -> 49.973 | -47.18 us (-48.6%) |
| 300 -> 500 | 49.973 -> 40.520 | -9.45 us (-18.9%) |
| 500 -> 700 | 40.520 -> 36.488 | -4.03 us (-10.0%) |
| 700 -> 900 | 36.488 -> 34.227 | -2.26 us (-6.2%) |

The last 200 GB/s is worth 5% of what the first 200 was worth, for the same hardware cost.

I ran four extra simulations outside the required sweep to check that the floor is real and to find out what it is made of. Setting bandwidth to 1,000,000 GB/s, effectively infinite, so that only the floor is left:

| Collective | Comm time at infinite BW | Fitted floor B | Floor / 936.25 ns |
| --- | --- | --- | --- |
| Reduce-Scatter | 26.746 us | 26.74 | 28.6 |
| All-Gather | 26.368 us | 26.37 | 28.2 |
| All-Reduce | 53.094 us | 53.08 | 56.7 |
| All-to-All | 105.412 us | 105.38 | 112.6 |

Two things fall out. The measured asymptotes match the fitted floors to within 0.03 us, so the fit was describing something real. And dividing by the link latency gives roughly **28, 28, 56, 112**, which is 1 : 1 : 2 : 4. 28 is exactly 7 ring steps x 4 chunks, the `preferred-dataset-splits: 4` from the system config. All-Reduce is 2x because it is a Reduce-Scatter plus an All-Gather, and All-to-All is 4x, the same factor that showed up in its bandwidth term.

To confirm the floor is link latency and not something else like local memory bandwidth, I reran All-Gather at 900 GB/s with latency set to 0:

| Config | Comm time |
| --- | --- |
| 900 GB/s, 936.25 ns latency | 34.227 us |
| 900 GB/s, 0 ns latency | 8.019 us |
| infinite BW, 936.25 ns latency | 26.426 us |
| infinite BW, 0 ns latency | 0.218 us |

Removing latency removes 26.2 us, which is the floor almost exactly, and removing both leaves 0.218 us of residual overhead. So the floor is serialized link latency: 28 hops that have to happen one after another because each ring step depends on the previous one. No amount of bandwidth removes a dependency chain.

## Takeaway

TLDR: both exercises are the same lesson from different directions. Collective communication cost splits into a bytes term that bandwidth fixes and a steps term that it cannot.

Exercise I holds bandwidth constant and grows the step count: time goes up linearly in N-1, exactly. Exercise II holds the step count constant and grows bandwidth: time falls as 1/BW onto a floor set by those steps. The floor is the number of serialized hops times the link latency, and that number comes out of the algorithm and the chunking, not the hardware.

So the design lever depends on which term dominates. For All-Gather at 100 GB/s, 73% of the time is bytes in flight and buying bandwidth is the right move. For All-to-All at 900 GB/s, 96% of the time is the latency floor and more bandwidth is close to wasted money; what helps there is a shallower algorithm or fewer chunks. The HGX baseline's real 400 GB/s sits on the bend for All-Gather: at that point the bytes term is 17.7 us of a 44.1 us total, so 40%.

Caveat on the numbers above the baseline: the lab notes these bandwidth variations are hypothetical and the HGX-H100 validation does not extend to every modified configuration, so the 900 GB/s points are the model's opinion rather than a validated prediction.
