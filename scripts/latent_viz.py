"""
Visualize latent-space transformations used by `new-soln.py`.

Example:
  uv run python scripts/latent_viz.py --bench ibm06 --out latent_viz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def _resolve_bench(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("ibm"):
        num = int(raw[3:])
        return f"ibm{num:02d}"
    raise ValueError(f"Unknown benchmark name: {raw!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, help="Benchmark name (e.g. ibm06 / ibm6).")
    parser.add_argument("--out", default="latent_viz", help="Output directory.")
    parser.add_argument("--scale", type=float, default=0.22, help="Alpha magnitude as fraction of cell size.")
    parser.add_argument("--max-modes", type=int, default=8, help="Max number of modes to visualize.")
    args = parser.parse_args()

    name = _resolve_bench(args.bench)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from macro_place.loader import load_benchmark
    from macro_place.utils import visualize_placement

    from new_soln import Placer  # type: ignore

    bench, plc = load_benchmark(name)
    placer = Placer()
    placer._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    placer._place_t0 = 0.0
    data = placer._prepare(bench)
    if plc is None:
        plc = placer._try_load_plc(bench)

    base_hard = data.init_hard.copy()
    base_soft = data.init_soft.copy()
    modes = placer._build_latent_modes(base_hard, base_soft, data, extra_hard_seeds=None)
    if not modes:
        raise RuntimeError("No latent modes found.")

    unit = float(max(data.cell_w, data.cell_h))
    alpha_mag = float(args.scale) * unit

    # Base placement
    base = bench.macro_positions.clone()
    base[: data.num_hard] = torch.from_numpy(base_hard).float()
    if data.num_soft > 0:
        base[data.num_hard : data.num_macros] = torch.from_numpy(base_soft).float()
    visualize_placement(base, bench, save_path=str(out_dir / f"{name}_base.png"), plc=plc)

    for i, mode in enumerate(modes[: max(1, int(args.max_modes))]):
        for sign in (-1.0, 1.0):
            alpha = sign * alpha_mag
            hard = base_hard + alpha * mode
            hard = placer._legalize_hard(hard, data)
            placement = bench.macro_positions.clone()
            placement[: data.num_hard] = torch.from_numpy(hard).float()
            if data.num_soft > 0:
                placement[data.num_hard : data.num_macros] = torch.from_numpy(base_soft).float()
            tag = "neg" if sign < 0 else "pos"
            visualize_placement(
                placement,
                bench,
                save_path=str(out_dir / f"{name}_mode{i:02d}_{tag}.png"),
                plc=plc,
            )

    print(f"Wrote images to: {out_dir.resolve()}")


if __name__ == "__main__":
    # Allow running both as `python` and `uv run python` from repo root.
    # Ensure `new-soln.py` can be imported as a module name.
    # (uv runs with repo root on sys.path by default.)
    if os.path.exists("new-soln.py") and not os.path.exists("new_soln.py"):
        # Create a temporary import shim name in-memory via module aliasing.
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("new_soln", "new-soln.py")
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["new_soln"] = mod
    main()

