"""Part I + Exercise I + Exercise II: roofline plots.

Usage:
    python plot_roofline.py                     # uses results/kernels.json if present
    python plot_roofline.py --kernels path.json
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Dense FP16 Tensor Core peaks (no sparsity) and DRAM bandwidth, from NVIDIA datasheets.
L4 = {'name': 'NVIDIA L4', 'peak_flops': 121e12, 'peak_memory_bw': 300e9, 'color': 'tab:red'}
A100 = {'name': 'NVIDIA A100 SXM 80GB', 'peak_flops': 312e12, 'peak_memory_bw': 2039e9, 'color': 'tab:blue'}
H100 = {'name': 'NVIDIA H100 SXM', 'peak_flops': 989e12, 'peak_memory_bw': 3350e9, 'color': 'tab:green'}


class RooflinePlotter:
    def __init__(self):
        self.title_size = 20
        self.ax_title_size = 20
        self.xlabel_size = 20
        self.ylabel_size = 20
        self.xtick_size = 20
        self.ytick_size = 20
        self.legend_size = 14

    def plot_roofline(self, platforms, kernel_data=None,
                      results_dir="./results", plot_name="Roofline Model",
                      file_name="roofline", use_log_scale=False):

        os.makedirs(results_dir, exist_ok=True)
        file_name = f"{file_name}.png"
        kernel_data = kernel_data or []

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        max_ai = 0
        max_flops = 0
        min_ai = float('inf')

        # Calculate the intersection point for the roofline
        intersections = []
        for platform in platforms:
            name = platform['name']
            peak_flops = platform['peak_flops']
            peak_memory_bw = platform['peak_memory_bw']
            color = platform.get('color', None)

            # Ridge point: AI where bandwidth * AI == peak compute
            x_intersection = peak_flops / peak_memory_bw
            intersections.append({'name': name, 'x_intersection': x_intersection,
                                  'peak_flops': peak_flops, 'peak_memory_bw': peak_memory_bw,
                                  'color': color})
            print(f"\n{name}:")
            print(f"  Ridge point: {x_intersection:.4f} FLOP/Byte")
            print(f"  Peak FLOPS: {peak_flops:.2e} FLOP/s")
            print(f"  Peak Memory BW: {peak_memory_bw:.2e} Byte/s")

            max_ai = max(max_ai, x_intersection)
            max_flops = max(max_flops, peak_flops)
            min_ai = min(min_ai, x_intersection)

        # Make sure kernels fit on the axes
        for k in kernel_data:
            max_ai = max(max_ai, k['arithmetic_intensity'])
            min_ai = min(min_ai, k['arithmetic_intensity'])

        # Adjust min/max for plotting purposes
        min_x = min_ai / 100 if use_log_scale else 0
        max_x = max_ai * 10

        # Plot the data for the roofline
        for platform_data in intersections:
            name = platform_data['name']
            x_intersection = platform_data['x_intersection']
            peak_flops = platform_data['peak_flops']
            peak_memory_bw = platform_data['peak_memory_bw']
            color = platform_data['color']

            # Memory-bound slope: perf = BW * AI, from min_x up to the ridge
            x_mem = np.geomspace(max(min_x, 1e-3), x_intersection, 200) if use_log_scale \
                else np.linspace(min_x, x_intersection, 200)
            ax.plot(x_mem, peak_memory_bw * x_mem, color=color, linewidth=2.5,
                    label=f"{name} ({peak_flops/1e12:.0f} TFLOP/s, {peak_memory_bw/1e9:.0f} GB/s)")
            # Compute-bound ceiling: flat at peak FLOPS from ridge to max_x
            ax.plot([x_intersection, max_x], [peak_flops, peak_flops], color=color, linewidth=2.5)
            ax.axvline(x_intersection, color=color, linestyle=':', alpha=0.5)

        # Kernels
        markers = ['o', 's', '^', 'D', 'v', 'P', '*']
        for i, k in enumerate(kernel_data):
            ax.scatter(k['arithmetic_intensity'], k['throughput'], s=150,
                       marker=markers[i % len(markers)], color=k.get('color', 'black'),
                       edgecolor='black', zorder=5,
                       label=f"{k['name']} (AI={k['arithmetic_intensity']:.1f}, "
                             f"{k['throughput']/1e12:.1f} TFLOP/s)")

        # Set Axes, Title, Legend
        ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=self.xlabel_size)
        ax.set_ylabel("Throughput (FLOP/s)", fontsize=self.ylabel_size)
        ax.set_title(plot_name, fontsize=self.title_size)
        ax.tick_params(axis='both', labelsize=self.xtick_size)

        if use_log_scale:
            ax.set_yscale("log")
            ax.set_xscale("log")

        ax.legend(loc="lower right", fontsize=self.legend_size)
        ax.grid(True, alpha=0.3)

        output_path = os.path.join(results_dir, file_name)
        print(f"\nSaving plot to {output_path}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()


def load_kernels(path):
    if not os.path.exists(path):
        print(f"No kernel data at {path}; plotting rooflines only. Run profile_kernel.py first.")
        return {}
    with open(path) as f:
        return {k['name']: k for k in json.load(f)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="./results/kernels.json")
    args = ap.parse_args()

    plotter = RooflinePlotter()
    kernels = load_kernels(args.kernels)

    # Part I/II: L4 roofline with the 4096^3 GEMM
    part2 = [dict(kernels[n], color='gold') for n in ['GEMM 4096x4096x4096'] if n in kernels]
    plotter.plot_roofline(platforms=[L4], kernel_data=part2, results_dir="./results",
                          plot_name="L4 Roofline", file_name="roofline_l4_gemm",
                          use_log_scale=True)

    # Exercise I: L4 + A100 + H100, GEMM + Conv (kernels measured on L4)
    ex1 = [dict(kernels[n], color=c) for n, c in
           [('GEMM 4096x4096x4096', 'gold'), ('Conv2d 3x3', 'magenta')] if n in kernels]
    plotter.plot_roofline(platforms=[L4, A100, H100], kernel_data=ex1, results_dir="./results",
                          plot_name="Accelerator Rooflines", file_name="ex1_rooflines",
                          use_log_scale=True)

    # Exercise II: L4 with GEMM A (prefill-like) and GEMM B (decode-like)
    ex2 = [dict(kernels[n], color=c) for n, c in
           [('GEMM A 8192x8192x4096', 'gold'), ('GEMM B 1x8192x4096', 'cyan')] if n in kernels]
    plotter.plot_roofline(platforms=[L4], kernel_data=ex2, results_dir="./results",
                          plot_name="L4: Compute vs Memory Bound GEMMs", file_name="ex2_gemm_ab",
                          use_log_scale=True)

    if ex2:
        ridge = L4['peak_flops'] / L4['peak_memory_bw']
        print(f"\nExercise II (L4 ridge = {ridge:.1f} FLOP/B):")
        for k in ex2:
            bound = "compute-bound" if k['arithmetic_intensity'] > ridge else "memory-bound"
            pct = 100 * k['throughput'] / min(L4['peak_flops'], L4['peak_memory_bw'] * k['arithmetic_intensity'])
            print(f"  {k['name']}: AI={k['arithmetic_intensity']:.2f} -> {bound}, "
                  f"{k['throughput']/1e12:.2f} TFLOP/s ({pct:.0f}% of attainable)")

    print("Finished generating roofline plots!")
