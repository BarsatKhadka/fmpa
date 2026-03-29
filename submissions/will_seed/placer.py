"""
Will's Seed Attempt v3 — Legalize + SA Refine + Soft Macro FD

Start from the initial placement (which is already good), fix overlaps
minimally, run fast SA to improve wirelength, then optimize soft macros.

Usage:
    uv run evaluate submissions/will_seed/placer.py
    uv run evaluate submissions/will_seed/placer.py --all
"""

import math
import random
import sys
import io
import torch
import numpy as np
from pathlib import Path
from typing import Tuple

from macro_place.benchmark import Benchmark


def _load_plc(name: str):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile"}
    design = ng45.get(name)
    if design:
        base = Path("external/MacroPlacement/Flows/NanGate45") / design / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


def _extract_edges(benchmark, plc) -> Tuple[np.ndarray, np.ndarray]:
    n_hard = benchmark.num_hard_macros
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
    edge_dict = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            ml = sorted(macros)
            w = 1.0 / (len(ml) - 1)
            for i in range(len(ml)):
                for j in range(i + 1, len(ml)):
                    pair = (ml[i], ml[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + w
    if not edge_dict:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float64)
    return (np.array(list(edge_dict.keys()), dtype=np.int32),
            np.array([edge_dict[e] for e in edge_dict], dtype=np.float64))


class WillSeedPlacer:
    def __init__(self, seed: int = 42, sa_iters: int = 5000):
        self.seed = seed
        self.sa_iters = sa_iters

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        np.random.seed(self.seed)

        n_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes.numpy()[:n_hard].astype(np.float64)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        movable = benchmark.get_movable_mask().numpy()[:n_hard]
        movable_idx = np.where(movable)[0]

        if len(movable_idx) == 0:
            return benchmark.macro_positions.clone()

        plc = _load_plc(benchmark.name)
        edges, edge_weights = (_extract_edges(benchmark, plc) if plc
                                else (np.zeros((0, 2), dtype=np.int32), np.zeros(0)))

        # Precompute
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        eye = np.eye(n_hard, dtype=bool)

        # Start from initial placement — legalize minimally
        pos = benchmark.macro_positions.numpy()[:n_hard].copy().astype(np.float64)
        pos = self._legalize(pos, movable, sizes, half_w, half_h, cw, ch, sep_x, sep_y, n_hard)

        # Build neighbor lists
        neighbors = [[] for _ in range(n_hard)]
        for i, j in edges:
            neighbors[i].append(j)
            neighbors[j].append(i)

        # Fast cost and overlap check
        def wl_cost(p):
            if len(edges) == 0:
                return 0.0
            dx = np.abs(p[edges[:, 0], 0] - p[edges[:, 1], 0])
            dy = np.abs(p[edges[:, 0], 1] - p[edges[:, 1], 1])
            return (edge_weights * (dx + dy)).sum()

        def has_overlap(p):
            dx = np.abs(p[:, 0:1] - p[:, 0:1].T)
            dy = np.abs(p[:, 1:2] - p[:, 1:2].T)
            return ((dx < sep_x) & (dy < sep_y) & ~eye).any()

        # SA refinement
        current_cost = wl_cost(pos)
        best_pos = pos.copy()
        best_cost = current_cost

        T_start = max(cw, ch) * 0.15
        T_end = max(cw, ch) * 0.0005

        for step in range(self.sa_iters):
            frac = step / self.sa_iters
            T = T_start * (T_end / T_start) ** frac
            old_pos = pos.copy()

            move = random.random()
            if move < 0.5:
                i = random.choice(movable_idx)
                shift = T * (0.3 + 0.7 * (1 - frac))
                pos[i, 0] = np.clip(pos[i, 0] + random.gauss(0, shift), half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + random.gauss(0, shift), half_h[i], ch - half_h[i])
            elif move < 0.8:
                i = random.choice(movable_idx)
                if neighbors[i] and random.random() < 0.7:
                    cands = [j for j in neighbors[i] if movable[j]]
                    j = random.choice(cands) if cands else random.choice(movable_idx)
                else:
                    j = random.choice(movable_idx)
                if i != j:
                    pi = np.clip(old_pos[j], [half_w[i], half_h[i]], [cw-half_w[i], ch-half_h[i]])
                    pj = np.clip(old_pos[i], [half_w[j], half_h[j]], [cw-half_w[j], ch-half_h[j]])
                    pos[i] = pi; pos[j] = pj
            else:
                i = random.choice(movable_idx)
                if neighbors[i]:
                    j = random.choice(neighbors[i])
                    alpha = random.uniform(0.05, 0.3)
                    pos[i, 0] = np.clip(pos[i, 0]+alpha*(pos[j, 0]-pos[i, 0]), half_w[i], cw-half_w[i])
                    pos[i, 1] = np.clip(pos[i, 1]+alpha*(pos[j, 1]-pos[i, 1]), half_h[i], ch-half_h[i])

            if has_overlap(pos):
                pos = old_pos
                continue

            new_cost = wl_cost(pos)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos = pos.copy()
            else:
                pos = old_pos

        # Set best hard macro positions in plc
        full_pos = benchmark.macro_positions.clone()
        full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)

        if plc is not None:
            for i, macro_idx in enumerate(benchmark.hard_macro_indices):
                plc.modules_w_pins[macro_idx].set_pos(float(best_pos[i, 0]), float(best_pos[i, 1]))

            # Run soft macro FD (suppress verbose output)
            canvas_size = max(cw, ch)
            old_stdout = sys.stdout; sys.stdout = io.StringIO()
            try:
                plc.optimize_stdcells(
                    use_current_loc=False, move_stdcells=True, move_macros=False,
                    log_scale_conns=False, use_sizes=False, io_factor=1.0,
                    num_steps=[15, 15, 15],
                    max_move_distance=[canvas_size/15]*3,
                    attract_factor=[100, 1.0e-3, 1.0e-5],
                    repel_factor=[0, 1.0e6, 1.0e7],
                )
            finally:
                sys.stdout = old_stdout

            # Copy optimized soft macro positions
            for i, idx in enumerate(benchmark.soft_macro_indices):
                x, y = plc.modules_w_pins[idx].get_pos()
                full_pos[n_hard + i, 0] = x
                full_pos[n_hard + i, 1] = y

        return full_pos

    def _legalize(self, pos, movable, sizes, half_w, half_h, cw, ch, sep_x_mat, sep_y_mat, n):
        """Minimal legalization — only move macros that actually overlap."""
        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(n, dtype=bool)
        legal = pos.copy()
        for idx in order:
            if not movable[idx]:
                placed[idx] = True
                continue
            # Check overlap
            if placed.any():
                dx = np.abs(legal[idx, 0] - legal[:, 0])
                dy = np.abs(legal[idx, 1] - legal[:, 1])
                conflicts = (dx < sep_x_mat[idx] + 0.01) & (dy < sep_y_mat[idx] + 0.01) & placed
                conflicts[idx] = False
                if not conflicts.any():
                    placed[idx] = True
                    continue
            # Find closest legal position
            step = max(sizes[idx, 0], sizes[idx, 1]) * 0.3
            best_p = legal[idx].copy(); best_d = float('inf')
            for r in range(1, 120):
                found = False
                for dxm in range(-r, r + 1):
                    for dym in range(-r, r + 1):
                        if abs(dxm) != r and abs(dym) != r:
                            continue
                        cx = np.clip(pos[idx, 0]+dxm*step, half_w[idx], cw-half_w[idx])
                        cy = np.clip(pos[idx, 1]+dym*step, half_h[idx], ch-half_h[idx])
                        if placed.any():
                            dx = np.abs(cx - legal[:, 0])
                            dy = np.abs(cy - legal[:, 1])
                            c = (dx < sep_x_mat[idx]+0.01) & (dy < sep_y_mat[idx]+0.01) & placed
                            c[idx] = False
                            if c.any():
                                continue
                        d = (cx-pos[idx, 0])**2 + (cy-pos[idx, 1])**2
                        if d < best_d:
                            best_d = d; best_p = np.array([cx, cy]); found = True
                if found:
                    break
            legal[idx] = best_p; placed[idx] = True
        return legal
