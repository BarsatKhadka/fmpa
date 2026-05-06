"""
Generate side-by-side comparison images: initial.plc vs our placer.
Saves one PNG per benchmark to the images/ directory.

Usage:
    uv run python gen_images.py                         # all 17 benchmarks
    uv run python gen_images.py ibm01 ibm04 ibm16       # specific ones
    PLACE_TIME_BUDGET=300 uv run python gen_images.py   # quick preview (5 min budget)
"""

import os
import sys
import importlib.util

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

IBM_NAMES = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06",
    "ibm07", "ibm08", "ibm09", "ibm10", "ibm11",
    "ibm12", "ibm13", "ibm14", "ibm15", "ibm16",
    "ibm17", "ibm18",
]

ICCAD_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "external", "MacroPlacement", "Testcases", "ICCAD04",
).replace("\\", "/")


def _draw_panel(ax, placement, benchmark, title):
    """Draw one placement panel: hard macros, soft macros, ports."""
    pos = placement.numpy() if isinstance(placement, torch.Tensor) else placement
    sizes = benchmark.macro_sizes.numpy()
    fixed = benchmark.macro_fixed.numpy()
    num_hard = benchmark.num_hard_macros

    ax.set_xlim(0, benchmark.canvas_width)
    ax.set_ylim(0, benchmark.canvas_height)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)

    # Canvas border
    ax.add_patch(Rectangle(
        (0, 0), benchmark.canvas_width, benchmark.canvas_height,
        fill=False, edgecolor="black", linewidth=1.5, zorder=10,
    ))

    # Macros
    for i in range(benchmark.num_macros):
        x, y = pos[i, 0], pos[i, 1]
        w, h = sizes[i, 0], sizes[i, 1]
        is_soft = i >= num_hard
        is_fixed = bool(fixed[i])

        if is_fixed:
            fc, ec, lw, alpha, ls = "red", "darkred", 0.6, 0.55, "solid"
        elif is_soft:
            fc, ec, lw, alpha, ls = "#aec6e8", "#6a8faf", 0.2, 0.30, "dashed"
        else:
            fc, ec, lw, alpha, ls = "#2166ac", "#0a3d6b", 0.6, 0.65, "solid"

        ax.add_patch(Rectangle(
            (x - w / 2, y - h / 2), w, h,
            facecolor=fc, alpha=alpha,
            edgecolor=ec, linewidth=lw, linestyle=ls,
        ))

    # I/O ports
    if benchmark.port_positions.shape[0] > 0:
        pp = benchmark.port_positions.numpy()
        ax.scatter(pp[:, 0], pp[:, 1], s=6, c="limegreen",
                   edgecolors="darkgreen", linewidths=0.3, zorder=5)

    # Stats in corner
    ax.text(
        0.01, 0.99,
        f"{benchmark.num_hard_macros}H + {benchmark.num_soft_macros}S",
        transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    )

    legend_elements = [
        Patch(fc="#2166ac", alpha=0.65, ec="#0a3d6b", label="Hard macros"),
        Patch(fc="#aec6e8", alpha=0.30, ec="#6a8faf", ls="dashed", label="Soft macros"),
        Patch(fc="red",     alpha=0.55, ec="darkred",  label="Fixed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="limegreen",
               markeredgecolor="darkgreen", markersize=5, label="I/O ports"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
              framealpha=0.8).set_zorder(11)


def process(name, placer, out_dir):
    bench_dir = os.path.join(ICCAD_ROOT, name)
    if not os.path.isdir(bench_dir):
        print(f"  [SKIP] directory not found: {bench_dir}", flush=True)
        return

    from macro_place.loader import load_benchmark_from_dir
    benchmark, plc = load_benchmark_from_dir(bench_dir)

    # Save initial positions before running placer (placer reads from benchmark)
    init_pos = benchmark.macro_positions.clone()

    print(f"  [{name}] Running placer ...", flush=True)
    our_pos = placer.place(benchmark)

    # Two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle(name.upper(), fontsize=14, fontweight="bold", y=1.01)

    _draw_panel(axes[0], init_pos, benchmark, "Initial placement (init.plc)")
    _draw_panel(axes[1], our_pos,  benchmark, "Our placer")

    out_path = os.path.join(out_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{name}] Saved -> {out_path}", flush=True)


def main():
    requested = [a for a in sys.argv[1:] if a.startswith("ibm")]
    names = requested if requested else IBM_NAMES

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(out_dir, exist_ok=True)

    # Load our placer once
    spec = importlib.util.spec_from_file_location("soln", "soln.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    placer = mod.Placer()

    print(f"Generating images for: {names}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Time budget: {os.environ.get('PLACE_TIME_BUDGET', '3300')}s per benchmark\n", flush=True)

    for name in names:
        print(f"[{name}]", flush=True)
        try:
            process(name, placer, out_dir)
        except Exception as e:
            import traceback
            print(f"  [{name}] ERROR: {e}", flush=True)
            traceback.print_exc()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
