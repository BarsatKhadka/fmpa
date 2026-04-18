"""
Universal multiworld macro placer.

Override runtime with:
    PLACE_TIME_BUDGET=<seconds> uv run evaluate new-soln.py
"""

import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark

TIME_BUDGET = int(os.environ.get("PLACE_TIME_BUDGET", 180))
SEED = int(os.environ.get("PLACE_SEED", 20260417))

WL_ALPHA = 10.0
BASE_DENSITY_TARGET = 0.88
SPIRAL_POINTS = 20
LEGAL_GAP = 0.03


@dataclass
class PreparedData:
    num_hard: int
    num_macros: int
    num_soft: int
    canvas_w: float
    canvas_h: float
    canvas_area: float
    grid_rows: int
    grid_cols: int
    cell_w: float
    cell_h: float
    init_hard: np.ndarray
    init_soft: np.ndarray
    sizes_hard: np.ndarray
    sizes_soft: np.ndarray
    sizes_all: np.ndarray
    hard_areas: np.ndarray
    movable_hard: np.ndarray
    movable_soft: np.ndarray
    port_pos: np.ndarray
    hpwl_norm: float
    safe_nnp_np: np.ndarray
    nnmask_np: np.ndarray
    safe_nnp_t: torch.Tensor
    nnmask_t: torch.Tensor
    port_t: torch.Tensor
    hard_sizes_t: torch.Tensor
    soft_sizes_t: torch.Tensor
    cell_cx_g: torch.Tensor
    cell_cy_g: torch.Tensor
    soft_occ_owner: np.ndarray
    soft_occ_net: np.ndarray
    hard_occ_owner: np.ndarray
    hard_occ_net: np.ndarray
    hard_adj: np.ndarray
    hard_degree: np.ndarray
    port_pull: np.ndarray
    spectral_xy: np.ndarray
    neighbor_lists: List[np.ndarray]


class Placer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._seed_everything(benchmark.name)

        data = self._prepare(benchmark)
        plc = self._try_load_plc(benchmark)

        candidates: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        base_soft = self._clamp_soft_copy(data.init_soft, data)
        legacy_hard = self._tiny_fix_hard(self._legacy_resolve_hard(data.init_hard, data), data)
        candidates.append(
            (legacy_hard, base_soft.copy(), "legacy-base", self._cheap_score(legacy_hard, base_soft, data))
        )
        pair_hard = self._legalize_hard(data.init_hard, data)
        candidates.append((pair_hard, base_soft.copy(), "pair-base", self._cheap_score(pair_hard, base_soft, data)))
        shelf_hard = self._tiny_fix_hard(self._shelf_legalize_hard(data.init_hard, data), data)
        candidates.append(
            (shelf_hard, base_soft.copy(), "shelf-base", self._cheap_score(shelf_hard, base_soft, data))
        )

        worlds = self._build_worlds(data)
        max_worlds = 4 if data.num_hard <= 350 else 3 if data.num_hard <= 600 else 2
        run_expensive_worlds = TIME_BUDGET >= 300
        if run_expensive_worlds:
            for hard_seed, soft_seed, world_name in worlds[:max_worlds]:
                if self._time_left(t0) < 20:
                    break
                hard = self._legalize_hard(hard_seed, data)
                soft = self._clamp_soft_copy(soft_seed, data)
                candidates.append(
                    (hard.copy(), soft.copy(), f"{world_name}-base", self._cheap_score(hard, soft, data))
                )

                if world_name == "initial":
                    soft = soft.copy()
                else:
                    soft = self._relax_soft(hard, soft_seed, data, sweeps=3, damping=0.78)
                hard, soft = self._refine_world(hard, soft, data, t0)
                hard = self._legalize_hard(hard, data)
                if world_name != "initial":
                    soft = self._relax_soft(hard, soft, data, sweeps=2, damping=0.82)
                score = self._cheap_score(hard, soft, data)
                candidates.append((hard, soft, world_name, score))

        if not candidates:
            hard = self._legalize_hard(data.init_hard, data)
            soft = self._clamp_soft_copy(data.init_soft, data)
            candidates.append((hard, soft, "fallback", self._cheap_score(hard, soft, data)))

        if run_expensive_worlds and len(candidates) >= 2 and self._time_left(t0) >= 15:
            consensus_hard, consensus_soft = self._consensus_candidate(candidates, data)
            consensus_hard = self._legalize_hard(consensus_hard, data)
            consensus_soft = self._relax_soft(
                consensus_hard, consensus_soft, data, sweeps=3, damping=0.80
            )
            consensus_hard, consensus_soft = self._refine_world(
                consensus_hard,
                consensus_soft,
                data,
                t0,
                short_mode=True,
            )
            consensus_hard = self._legalize_hard(consensus_hard, data)
            consensus_soft = self._relax_soft(
                consensus_hard, consensus_soft, data, sweeps=2, damping=0.84
            )
            score = self._cheap_score(consensus_hard, consensus_soft, data)
            candidates.append((consensus_hard, consensus_soft, "consensus", score))

        best_hard, best_soft = self._select_best(candidates, benchmark, plc, data)

        placement = benchmark.macro_positions.clone()
        placement[: data.num_hard] = torch.from_numpy(best_hard).float()
        if data.num_soft > 0:
            placement[data.num_hard : data.num_macros] = torch.from_numpy(best_soft).float()
        return placement

    def _seed_everything(self, benchmark_name: str) -> None:
        seed = (SEED ^ hash(benchmark_name)) & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _prepare(self, benchmark: Benchmark) -> PreparedData:
        num_hard = benchmark.num_hard_macros
        num_macros = benchmark.num_macros
        num_soft = num_macros - num_hard
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        cell_w = canvas_w / benchmark.grid_cols
        cell_h = canvas_h / benchmark.grid_rows

        init_pos = benchmark.macro_positions.cpu().numpy().astype(np.float32)
        sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float32)
        movable = (~benchmark.macro_fixed).cpu().numpy()

        nets_np = [net.cpu().numpy().astype(np.int64) for net in benchmark.net_nodes]
        n_nets = len(nets_np)
        max_nsz = max(len(net) for net in nets_np)
        safe_nnp_np = np.zeros((n_nets, max_nsz), dtype=np.int64)
        nnmask_np = np.zeros((n_nets, max_nsz), dtype=bool)
        for i, nodes in enumerate(nets_np):
            safe_nnp_np[i, : len(nodes)] = nodes
            nnmask_np[i, : len(nodes)] = True

        port_pos = benchmark.port_positions.cpu().numpy().astype(np.float32)
        port_base = num_macros

        hard_adj = np.zeros((num_hard, num_hard), dtype=np.float32)
        port_pull = np.zeros((num_hard, 2), dtype=np.float32)
        port_pull_count = np.zeros(num_hard, dtype=np.float32)
        soft_occ_owner: List[int] = []
        soft_occ_net: List[int] = []
        hard_occ_owner: List[int] = []
        hard_occ_net: List[int] = []

        for net_id, nodes in enumerate(nets_np):
            hard_nodes = nodes[nodes < num_hard]
            soft_nodes = nodes[(nodes >= num_hard) & (nodes < num_macros)] - num_hard
            port_nodes = nodes[nodes >= port_base] - port_base

            for owner in soft_nodes.tolist():
                soft_occ_owner.append(int(owner))
                soft_occ_net.append(net_id)
            for owner in hard_nodes.tolist():
                hard_occ_owner.append(int(owner))
                hard_occ_net.append(net_id)

            if hard_nodes.size > 0 and port_nodes.size > 0:
                avg_port = port_pos[port_nodes].mean(axis=0)
                port_pull[hard_nodes] += avg_port
                port_pull_count[hard_nodes] += 1.0

            if hard_nodes.size > 1:
                weight = 1.0 / float(hard_nodes.size - 1)
                for i, src in enumerate(hard_nodes):
                    dst = hard_nodes[i + 1 :]
                    if dst.size == 0:
                        continue
                    hard_adj[src, dst] += weight
                    hard_adj[dst, src] += weight

        nonzero = port_pull_count > 0
        port_pull[nonzero] /= port_pull_count[nonzero, None]
        port_pull[~nonzero] = np.array([canvas_w / 2, canvas_h / 2], dtype=np.float32)

        hard_degree = hard_adj.sum(axis=1)
        neighbor_lists: List[np.ndarray] = []
        for idx in range(num_hard):
            weights = hard_adj[idx]
            neighbors = np.flatnonzero(weights > 0)
            if neighbors.size > 12:
                top = np.argpartition(weights[neighbors], -12)[-12:]
                neighbors = neighbors[top]
            neighbor_lists.append(neighbors.astype(np.int64))

        spectral_xy = self._spectral_layout(
            hard_adj,
            benchmark.macro_sizes[:num_hard].cpu().numpy().astype(np.float32),
            canvas_w,
            canvas_h,
            port_pull,
        )

        cell_x = torch.arange(benchmark.grid_cols, dtype=torch.float32) * cell_w + cell_w / 2
        cell_y = torch.arange(benchmark.grid_rows, dtype=torch.float32) * cell_h + cell_h / 2
        cell_cx_g = cell_x.unsqueeze(0).expand(benchmark.grid_rows, -1)
        cell_cy_g = cell_y.unsqueeze(1).expand(-1, benchmark.grid_cols)

        return PreparedData(
            num_hard=num_hard,
            num_macros=num_macros,
            num_soft=num_soft,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            canvas_area=canvas_w * canvas_h,
            grid_rows=benchmark.grid_rows,
            grid_cols=benchmark.grid_cols,
            cell_w=cell_w,
            cell_h=cell_h,
            init_hard=init_pos[:num_hard].copy(),
            init_soft=init_pos[num_hard:num_macros].copy(),
            sizes_hard=sizes[:num_hard].copy(),
            sizes_soft=sizes[num_hard:num_macros].copy(),
            sizes_all=sizes.copy(),
            hard_areas=(sizes[:num_hard, 0] * sizes[:num_hard, 1]).astype(np.float32),
            movable_hard=movable[:num_hard].copy(),
            movable_soft=movable[num_hard:num_macros].copy(),
            port_pos=port_pos.copy(),
            hpwl_norm=max(1.0, float(n_nets) * (canvas_w + canvas_h)),
            safe_nnp_np=safe_nnp_np,
            nnmask_np=nnmask_np,
            safe_nnp_t=torch.from_numpy(safe_nnp_np),
            nnmask_t=torch.from_numpy(nnmask_np),
            port_t=torch.from_numpy(port_pos),
            hard_sizes_t=torch.from_numpy(sizes[:num_hard].copy()),
            soft_sizes_t=torch.from_numpy(sizes[num_hard:num_macros].copy()),
            cell_cx_g=cell_cx_g,
            cell_cy_g=cell_cy_g,
            soft_occ_owner=np.asarray(soft_occ_owner, dtype=np.int64),
            soft_occ_net=np.asarray(soft_occ_net, dtype=np.int64),
            hard_occ_owner=np.asarray(hard_occ_owner, dtype=np.int64),
            hard_occ_net=np.asarray(hard_occ_net, dtype=np.int64),
            hard_adj=hard_adj,
            hard_degree=hard_degree,
            port_pull=port_pull,
            spectral_xy=spectral_xy,
            neighbor_lists=neighbor_lists,
        )

    def _spectral_layout(
        self,
        hard_adj: np.ndarray,
        sizes_hard: np.ndarray,
        canvas_w: float,
        canvas_h: float,
        port_pull: np.ndarray,
    ) -> np.ndarray:
        num_hard = hard_adj.shape[0]
        if num_hard == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if num_hard == 1:
            return np.array([[canvas_w / 2, canvas_h / 2]], dtype=np.float32)

        deg = hard_adj.sum(axis=1)
        lap = np.diag(deg + 1e-4) - hard_adj
        _, vecs = np.linalg.eigh(lap.astype(np.float64))

        basis = []
        for col in range(1, min(8, vecs.shape[1])):
            v = vecs[:, col]
            if float(v.std()) > 1e-8:
                basis.append(v.astype(np.float32))
            if len(basis) == 2:
                break
        if len(basis) < 2:
            basis = [
                np.linspace(-1.0, 1.0, num_hard, dtype=np.float32),
                np.random.uniform(-1.0, 1.0, size=num_hard).astype(np.float32),
            ]

        coords = np.stack(basis, axis=1)
        coords[:, 0] += 0.20 * ((port_pull[:, 0] / max(canvas_w, 1e-6)) * 2.0 - 1.0)
        coords[:, 1] += 0.20 * ((port_pull[:, 1] / max(canvas_h, 1e-6)) * 2.0 - 1.0)

        coords = self._normalize_xy(coords)
        pos = np.zeros((num_hard, 2), dtype=np.float32)
        x_span = np.maximum(canvas_w - sizes_hard[:, 0], 1e-3)
        y_span = np.maximum(canvas_h - sizes_hard[:, 1], 1e-3)
        pos[:, 0] = sizes_hard[:, 0] / 2 + coords[:, 0] * x_span
        pos[:, 1] = sizes_hard[:, 1] / 2 + coords[:, 1] * y_span
        return pos

    def _normalize_xy(self, coords: np.ndarray) -> np.ndarray:
        out = coords.copy()
        for axis in range(out.shape[1]):
            lo = float(out[:, axis].min())
            hi = float(out[:, axis].max())
            if hi - lo < 1e-6:
                out[:, axis] = 0.5
            else:
                out[:, axis] = (out[:, axis] - lo) / (hi - lo)
        return out.astype(np.float32)

    def _build_worlds(
        self, data: PreparedData
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        worlds: List[Tuple[np.ndarray, np.ndarray, str]] = []
        worlds.append((data.init_hard.copy(), data.init_soft.copy(), "initial"))

        spectral = data.spectral_xy.copy()
        worlds.append((spectral, data.init_soft.copy(), "spectral"))

        bisect = np.zeros_like(spectral)
        idx = np.arange(data.num_hard, dtype=np.int64)
        self._recursive_bisect(idx, data.spectral_xy, data.hard_areas, bisect, data, axis=0)
        worlds.append((bisect, data.init_soft.copy(), "bisection"))

        hybrid = 0.55 * data.init_hard + 0.45 * spectral
        worlds.append((hybrid.astype(np.float32), data.init_soft.copy(), "hybrid"))
        return worlds

    def _recursive_bisect(
        self,
        indices: np.ndarray,
        spectral_xy: np.ndarray,
        hard_areas: np.ndarray,
        out_pos: np.ndarray,
        data: PreparedData,
        axis: int,
        rect: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        if rect is None:
            rect = (0.0, 0.0, data.canvas_w, data.canvas_h)

        lx, ly, ux, uy = rect
        if indices.size == 0:
            return
        if indices.size == 1:
            idx = int(indices[0])
            w, h = data.sizes_hard[idx]
            x = np.clip((lx + ux) * 0.5, w / 2, data.canvas_w - w / 2)
            y = np.clip((ly + uy) * 0.5, h / 2, data.canvas_h - h / 2)
            out_pos[idx] = np.array([x, y], dtype=np.float32)
            return

        order = indices[np.argsort(spectral_xy[indices, axis], kind="mergesort")]
        cum = np.cumsum(hard_areas[order])
        total = float(cum[-1])
        split_at = int(np.searchsorted(cum, total * 0.5) + 1)
        split_at = max(1, min(split_at, order.size - 1))
        left = order[:split_at]
        right = order[split_at:]
        left_ratio = float(hard_areas[left].sum()) / max(total, 1e-6)
        left_ratio = float(np.clip(left_ratio, 0.20, 0.80))

        if axis == 0:
            mid = lx + (ux - lx) * left_ratio
            self._recursive_bisect(left, spectral_xy, hard_areas, out_pos, data, 1, (lx, ly, mid, uy))
            self._recursive_bisect(right, spectral_xy, hard_areas, out_pos, data, 1, (mid, ly, ux, uy))
        else:
            mid = ly + (uy - ly) * left_ratio
            self._recursive_bisect(left, spectral_xy, hard_areas, out_pos, data, 0, (lx, ly, ux, mid))
            self._recursive_bisect(right, spectral_xy, hard_areas, out_pos, data, 0, (lx, mid, ux, uy))

    def _refine_world(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
        t0: float,
        short_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if short_mode:
            stages = [
                (40, 0.040, 0.55, 120.0, 0.10),
                (30, 0.025, 0.70, 200.0, 0.04),
            ]
        else:
            stages = [
                (70, 0.045, 0.45, 90.0, 0.12),
                (55, 0.030, 0.65, 150.0, 0.06),
                (40, 0.018, 0.85, 250.0, 0.02),
            ]

        anchor = hard.copy()
        current_hard = hard.copy()
        current_soft = soft.copy()

        for steps, lr, density_weight, overlap_weight, anchor_weight in stages:
            if self._time_left(t0) < 12:
                break
            current_soft = self._relax_soft(current_hard, current_soft, data, sweeps=3, damping=0.60)
            current_hard = self._gradient_stage(
                current_hard,
                current_soft,
                anchor,
                data,
                t0,
                steps=steps,
                lr=lr,
                density_weight=density_weight,
                overlap_weight=overlap_weight,
                anchor_weight=anchor_weight,
            )
            current_hard = self._legalize_hard(current_hard, data)

        return current_hard, current_soft

    def _gradient_stage(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        anchor: np.ndarray,
        data: PreparedData,
        t0: float,
        steps: int,
        lr: float,
        density_weight: float,
        overlap_weight: float,
        anchor_weight: float,
    ) -> np.ndarray:
        movable_idx = np.flatnonzero(data.movable_hard).astype(np.int64)
        if movable_idx.size == 0:
            return hard

        hard_t = torch.from_numpy(hard.copy())
        soft_t = torch.from_numpy(soft.copy())
        anchor_t = torch.from_numpy(anchor.copy())
        params = hard_t[movable_idx].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=lr)

        hard_sizes = data.hard_sizes_t
        soft_sizes = data.soft_sizes_t
        safe_nnp_t = data.safe_nnp_t
        nnmask_t = data.nnmask_t

        fixed_hard = hard_t.clone()
        if data.num_soft > 0:
            soft_density = self._density_map(soft_t, soft_sizes, data)
        else:
            soft_density = None

        best_score = float("inf")
        best_params = params.detach().clone()

        for step in range(steps):
            if step % 10 == 0 and self._time_left(t0) < 10:
                break
            optimizer.zero_grad()

            cur_hard = fixed_hard.clone()
            cur_hard[movable_idx] = params
            all_pos = (
                torch.cat([cur_hard, soft_t, data.port_t], dim=0)
                if data.num_soft > 0
                else torch.cat([cur_hard, data.port_t], dim=0)
            )

            xs = all_pos[safe_nnp_t, 0]
            ys = all_pos[safe_nnp_t, 1]
            neg_inf = torch.full_like(xs, -1e9)
            xs_p = torch.where(nnmask_t, xs, neg_inf)
            xs_n = torch.where(nnmask_t, -xs, neg_inf)
            ys_p = torch.where(nnmask_t, ys, neg_inf)
            ys_n = torch.where(nnmask_t, -ys, neg_inf)
            wl = (
                torch.logsumexp(WL_ALPHA * xs_p, dim=1)
                + torch.logsumexp(WL_ALPHA * xs_n, dim=1)
                + torch.logsumexp(WL_ALPHA * ys_p, dim=1)
                + torch.logsumexp(WL_ALPHA * ys_n, dim=1)
            ).sum() / (WL_ALPHA * data.hpwl_norm)

            hard_density = self._density_map(cur_hard, hard_sizes, data)
            density = hard_density if soft_density is None else hard_density + soft_density
            overflow = (density - BASE_DENSITY_TARGET).clamp(min=0.0)
            density_cost = overflow.pow(2).mean() + 0.15 * overflow.max()

            cx = cur_hard[:, 0]
            cy = cur_hard[:, 1]
            sep_x = (hard_sizes[:, 0:1] + hard_sizes[:, 0:1].T) * 0.5
            sep_y = (hard_sizes[:, 1:2] + hard_sizes[:, 1:2].T) * 0.5
            ov_x = (sep_x - (cx[:, None] - cx[None, :]).abs()).clamp(min=0.0)
            ov_y = (sep_y - (cy[:, None] - cy[None, :]).abs()).clamp(min=0.0)
            overlap = ov_x * ov_y
            overlap_cost = (overlap.sum() - overlap.diagonal().sum()) * 0.5 / data.canvas_area

            anchor_delta = (cur_hard - anchor_t).pow(2).mean() / max(data.canvas_area, 1.0)
            loss = wl
            loss = loss + density_weight * density_cost
            loss = loss + overlap_weight * overlap_cost
            loss = loss + anchor_weight * anchor_delta

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                lo_x = torch.from_numpy(data.sizes_hard[movable_idx, 0] * 0.5)
                hi_x = data.canvas_w - lo_x
                lo_y = torch.from_numpy(data.sizes_hard[movable_idx, 1] * 0.5)
                hi_y = data.canvas_h - lo_y
                params[:, 0].clamp_(lo_x, hi_x)
                params[:, 1].clamp_(lo_y, hi_y)

            score = float(loss.detach())
            if score < best_score:
                best_score = score
                best_params = params.detach().clone()

        result = hard.copy()
        result[movable_idx] = best_params.cpu().numpy()
        return result.astype(np.float32)

    def _density_map(
        self,
        pos_t: torch.Tensor,
        sizes_t: torch.Tensor,
        data: PreparedData,
    ) -> torch.Tensor:
        if pos_t.numel() == 0:
            return torch.zeros((data.grid_rows, data.grid_cols), dtype=torch.float32)

        sigma_x = sizes_t[:, 0].view(-1, 1, 1) * 0.5 + data.cell_w * 0.5
        sigma_y = sizes_t[:, 1].view(-1, 1, 1) * 0.5 + data.cell_h * 0.5
        dx = (pos_t[:, 0].view(-1, 1, 1) - data.cell_cx_g.unsqueeze(0)) / sigma_x
        dy = (pos_t[:, 1].view(-1, 1, 1) - data.cell_cy_g.unsqueeze(0)) / sigma_y
        kernel = (1.0 - dx.abs()).clamp(min=0.0) * (1.0 - dy.abs()).clamp(min=0.0)
        area_scale = (sizes_t[:, 0] * sizes_t[:, 1]).view(-1, 1, 1) / max(
            data.cell_w * data.cell_h, 1e-6
        )
        kernel_sum = kernel.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return (kernel / kernel_sum * area_scale).sum(dim=0)

    def _relax_soft(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
        sweeps: int,
        damping: float,
    ) -> np.ndarray:
        if data.num_soft == 0 or data.soft_occ_owner.size == 0:
            return soft.copy()

        cur = self._clamp_soft_copy(soft, data)
        size_lo_x = data.sizes_soft[:, 0] * 0.5
        size_hi_x = data.canvas_w - size_lo_x
        size_lo_y = data.sizes_soft[:, 1] * 0.5
        size_hi_y = data.canvas_h - size_lo_y
        max_step = max(data.canvas_w, data.canvas_h) * 0.04

        for _ in range(sweeps):
            all_pos = self._all_pos_np(hard, cur, data)
            centers = self._net_box_centers_np(all_pos, data)
            target_x = np.bincount(
                data.soft_occ_owner,
                weights=centers[data.soft_occ_net, 0],
                minlength=data.num_soft,
            ).astype(np.float32)
            target_y = np.bincount(
                data.soft_occ_owner,
                weights=centers[data.soft_occ_net, 1],
                minlength=data.num_soft,
            ).astype(np.float32)
            counts = np.bincount(data.soft_occ_owner, minlength=data.num_soft).astype(np.float32)
            mask = counts > 0
            target = cur.copy()
            target[mask, 0] = target_x[mask] / counts[mask]
            target[mask, 1] = target_y[mask] / counts[mask]

            target = damping * cur + (1.0 - damping) * target
            delta = target - cur
            norm = np.linalg.norm(delta, axis=1)
            step_mask = norm > max_step
            if np.any(step_mask):
                delta[step_mask] *= (max_step / norm[step_mask])[:, None]
                target = cur + delta

            movable = data.movable_soft
            cur[movable] = target[movable]
            cur[:, 0] = np.clip(cur[:, 0], size_lo_x, size_hi_x)
            cur[:, 1] = np.clip(cur[:, 1], size_lo_y, size_hi_y)

        return cur.astype(np.float32)

    def _clamp_soft_copy(self, soft: np.ndarray, data: PreparedData) -> np.ndarray:
        if data.num_soft == 0:
            return soft.copy().astype(np.float32)
        out = soft.copy().astype(np.float32)
        out[:, 0] = np.clip(
            out[:, 0],
            data.sizes_soft[:, 0] * 0.5,
            data.canvas_w - data.sizes_soft[:, 0] * 0.5,
        )
        out[:, 1] = np.clip(
            out[:, 1],
            data.sizes_soft[:, 1] * 0.5,
            data.canvas_h - data.sizes_soft[:, 1] * 0.5,
        )
        return out

    def _all_pos_np(self, hard: np.ndarray, soft: np.ndarray, data: PreparedData) -> np.ndarray:
        if data.num_soft > 0:
            return np.vstack([hard, soft, data.port_pos])
        return np.vstack([hard, data.port_pos])

    def _net_box_centers_np(self, all_pos: np.ndarray, data: PreparedData) -> np.ndarray:
        xs = all_pos[data.safe_nnp_np, 0]
        ys = all_pos[data.safe_nnp_np, 1]
        inf = 1e15
        x_lo = np.where(data.nnmask_np, xs, inf).min(axis=1)
        x_hi = np.where(data.nnmask_np, xs, -inf).max(axis=1)
        y_lo = np.where(data.nnmask_np, ys, inf).min(axis=1)
        y_hi = np.where(data.nnmask_np, ys, -inf).max(axis=1)
        centers = np.stack([(x_lo + x_hi) * 0.5, (y_lo + y_hi) * 0.5], axis=1)
        return centers.astype(np.float32)

    def _legacy_resolve_hard(self, hard: np.ndarray, data: PreparedData, max_rounds: int = 20) -> np.ndarray:
        pos = hard.copy().astype(np.float64)
        for _ in range(max_rounds):
            pos = self._legacy_pair_resolve(pos, data, max_iter=1000)
            total = self._exact_hard_overlap_area(pos, data)
            if total < 1e-10:
                break
            jitter_x = np.random.uniform(-data.sizes_hard[:, 0] * 0.1, data.sizes_hard[:, 0] * 0.1)
            jitter_y = np.random.uniform(-data.sizes_hard[:, 1] * 0.1, data.sizes_hard[:, 1] * 0.1)
            movable = data.movable_hard
            pos[movable, 0] += jitter_x[movable]
            pos[movable, 1] += jitter_y[movable]
            self._clamp_hard(pos, data)
        return pos.astype(np.float32)

    def _legacy_pair_resolve(
        self,
        pos: np.ndarray,
        data: PreparedData,
        max_iter: int = 500,
    ) -> np.ndarray:
        p = pos.copy().astype(np.float64)
        for _ in range(max_iter):
            moved = False
            for i in range(data.num_hard):
                for j in range(i + 1, data.num_hard):
                    xi, yi = p[i]
                    xj, yj = p[j]
                    ox = (data.sizes_hard[i, 0] + data.sizes_hard[j, 0]) * 0.5 - abs(xi - xj)
                    oy = (data.sizes_hard[i, 1] + data.sizes_hard[j, 1]) * 0.5 - abs(yi - yj)
                    if ox <= 0 or oy <= 0:
                        continue
                    if ox < oy:
                        push = (ox + 0.01) * 0.5
                        dx_i = -push if xi < xj else push
                        dx_j = -dx_i
                        if data.movable_hard[i]:
                            p[i, 0] = np.clip(
                                xi + dx_i,
                                data.sizes_hard[i, 0] * 0.5,
                                data.canvas_w - data.sizes_hard[i, 0] * 0.5,
                            )
                        if data.movable_hard[j]:
                            p[j, 0] = np.clip(
                                xj + dx_j,
                                data.sizes_hard[j, 0] * 0.5,
                                data.canvas_w - data.sizes_hard[j, 0] * 0.5,
                            )
                    else:
                        push = (oy + 0.01) * 0.5
                        dy_i = -push if yi < yj else push
                        dy_j = -dy_i
                        if data.movable_hard[i]:
                            p[i, 1] = np.clip(
                                yi + dy_i,
                                data.sizes_hard[i, 1] * 0.5,
                                data.canvas_h - data.sizes_hard[i, 1] * 0.5,
                            )
                        if data.movable_hard[j]:
                            p[j, 1] = np.clip(
                                yj + dy_j,
                                data.sizes_hard[j, 1] * 0.5,
                                data.canvas_h - data.sizes_hard[j, 1] * 0.5,
                            )
                    moved = True
            if not moved:
                break
        return p.astype(np.float64)

    def _shelf_legalize_hard(self, hard: np.ndarray, data: PreparedData) -> np.ndarray:
        pos = hard.copy().astype(np.float32)
        sizes = data.sizes_hard
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) * 0.5
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) * 0.5
        order = np.argsort(-(sizes[:, 0] * sizes[:, 1]))
        placed = np.zeros(data.num_hard, dtype=bool)
        legal = pos.copy()

        for idx in order:
            if not data.movable_hard[idx]:
                placed[idx] = True
                continue

            if placed.any():
                dx = np.abs(legal[idx, 0] - legal[:, 0])
                dy = np.abs(legal[idx, 1] - legal[:, 1])
                coll = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
                coll[idx] = False
                if not coll.any():
                    placed[idx] = True
                    continue

            step = max(float(sizes[idx, 0]), float(sizes[idx, 1])) * 0.25
            best_p = legal[idx].copy()
            best_d = float("inf")
            for radius in range(1, 150):
                found = False
                for dxm in range(-radius, radius + 1):
                    for dym in range(-radius, radius + 1):
                        if abs(dxm) != radius and abs(dym) != radius:
                            continue
                        cx = np.clip(
                            pos[idx, 0] + dxm * step,
                            sizes[idx, 0] * 0.5,
                            data.canvas_w - sizes[idx, 0] * 0.5,
                        )
                        cy = np.clip(
                            pos[idx, 1] + dym * step,
                            sizes[idx, 1] * 0.5,
                            data.canvas_h - sizes[idx, 1] * 0.5,
                        )
                        if placed.any():
                            dx = np.abs(cx - legal[:, 0])
                            dy = np.abs(cy - legal[:, 1])
                            coll = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
                            coll[idx] = False
                            if coll.any():
                                continue
                        dist = float((cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2)
                        if dist < best_d:
                            best_d = dist
                            best_p = np.array([cx, cy], dtype=np.float32)
                            found = True
                if found:
                    break
            legal[idx] = best_p
            placed[idx] = True

        return legal.astype(np.float32)

    def _exact_hard_overlap_area(self, pos: np.ndarray, data: PreparedData) -> float:
        total = 0.0
        for i in range(data.num_hard):
            dx = np.abs(pos[i, 0] - pos[:, 0])
            dy = np.abs(pos[i, 1] - pos[:, 1])
            ox = np.maximum(0.0, (data.sizes_hard[i, 0] + data.sizes_hard[:, 0]) * 0.5 - dx)
            oy = np.maximum(0.0, (data.sizes_hard[i, 1] + data.sizes_hard[:, 1]) * 0.5 - dy)
            area = ox * oy
            area[i] = 0.0
            total += float(area.sum())
        return total * 0.5

    def _tiny_fix_hard(self, hard: np.ndarray, data: PreparedData, rounds: int = 200) -> np.ndarray:
        p = hard.copy().astype(np.float64)
        for _ in range(rounds):
            moved = False
            p[:, 0] = np.clip(
                p[:, 0],
                data.sizes_hard[:, 0] * 0.5,
                data.canvas_w - data.sizes_hard[:, 0] * 0.5,
            )
            p[:, 1] = np.clip(
                p[:, 1],
                data.sizes_hard[:, 1] * 0.5,
                data.canvas_h - data.sizes_hard[:, 1] * 0.5,
            )
            for i in range(data.num_hard):
                wi, hi = data.sizes_hard[i]
                lix, uix = p[i, 0] - wi / 2, p[i, 0] + wi / 2
                liy, uiy = p[i, 1] - hi / 2, p[i, 1] + hi / 2
                for j in range(i + 1, data.num_hard):
                    wj, hj = data.sizes_hard[j]
                    ljx, ujx = p[j, 0] - wj / 2, p[j, 0] + wj / 2
                    ljy, ujy = p[j, 1] - hj / 2, p[j, 1] + hj / 2
                    if lix >= ujx or uix <= ljx or liy >= ujy or uiy <= ljy:
                        continue
                    moved = True
                    ox = min(uix, ujx) - max(lix, ljx)
                    oy = min(uiy, ujy) - max(liy, ljy)
                    if ox < oy:
                        push = (ox + 1e-4) * 0.5
                        if p[i, 0] < p[j, 0]:
                            p[i, 0] -= push
                            p[j, 0] += push
                        else:
                            p[i, 0] += push
                            p[j, 0] -= push
                    else:
                        push = (oy + 1e-4) * 0.5
                        if p[i, 1] < p[j, 1]:
                            p[i, 1] -= push
                            p[j, 1] += push
                        else:
                            p[i, 1] += push
                            p[j, 1] -= push
            if not moved:
                break
        self._clamp_hard(p, data)
        return p.astype(np.float32)

    def _legalize_hard(self, hard: np.ndarray, data: PreparedData) -> np.ndarray:
        pos = hard.copy().astype(np.float32)
        self._clamp_hard(pos, data)

        for _ in range(18):
            moved = False
            for i in range(data.num_hard):
                for j in range(i + 1, data.num_hard):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    ox = (
                        (data.sizes_hard[i, 0] + data.sizes_hard[j, 0]) * 0.5
                        + LEGAL_GAP
                        - abs(dx)
                    )
                    oy = (
                        (data.sizes_hard[i, 1] + data.sizes_hard[j, 1]) * 0.5
                        + LEGAL_GAP
                        - abs(dy)
                    )
                    if ox <= 0 or oy <= 0:
                        continue
                    if ox < oy:
                        push = (ox + LEGAL_GAP) * 0.5
                        shift_i = -push if dx >= 0 else push
                        shift_j = -shift_i
                        if data.movable_hard[i]:
                            pos[i, 0] += shift_i
                        if data.movable_hard[j]:
                            pos[j, 0] += shift_j
                    else:
                        push = (oy + LEGAL_GAP) * 0.5
                        shift_i = -push if dy >= 0 else push
                        shift_j = -shift_i
                        if data.movable_hard[i]:
                            pos[i, 1] += shift_i
                        if data.movable_hard[j]:
                            pos[j, 1] += shift_j
                    moved = True
            self._clamp_hard(pos, data)
            if not moved:
                break

        total_overlap, local_overlap = self._hard_overlap_stats(pos, data)
        if total_overlap <= 1e-8:
            return pos

        order = np.argsort(-(local_overlap + 0.05 * data.hard_areas + 0.01 * data.hard_degree))
        target = hard.copy()
        for idx in order:
            if local_overlap[idx] <= 1e-8 or not data.movable_hard[idx]:
                continue
            pos[idx] = self._nearest_legal_point(int(idx), target[idx], pos, data)

        self._clamp_hard(pos, data)
        return pos

    def _clamp_hard(self, pos: np.ndarray, data: PreparedData) -> None:
        pos[:, 0] = np.clip(
            pos[:, 0],
            data.sizes_hard[:, 0] * 0.5,
            data.canvas_w - data.sizes_hard[:, 0] * 0.5,
        )
        pos[:, 1] = np.clip(
            pos[:, 1],
            data.sizes_hard[:, 1] * 0.5,
            data.canvas_h - data.sizes_hard[:, 1] * 0.5,
        )

    def _hard_overlap_stats(self, pos: np.ndarray, data: PreparedData) -> Tuple[float, np.ndarray]:
        local = np.zeros(data.num_hard, dtype=np.float32)
        total = 0.0
        for i in range(data.num_hard):
            dx = np.abs(pos[i, 0] - pos[:, 0])
            dy = np.abs(pos[i, 1] - pos[:, 1])
            ox = np.maximum(
                0.0,
                (data.sizes_hard[i, 0] + data.sizes_hard[:, 0]) * 0.5 + LEGAL_GAP - dx,
            )
            oy = np.maximum(
                0.0,
                (data.sizes_hard[i, 1] + data.sizes_hard[:, 1]) * 0.5 + LEGAL_GAP - dy,
            )
            area = ox * oy
            area[i] = 0.0
            local[i] = float(area.sum())
            total += float(area.sum())
        return total * 0.5, local

    def _nearest_legal_point(
        self,
        idx: int,
        target_xy: np.ndarray,
        pos: np.ndarray,
        data: PreparedData,
    ) -> np.ndarray:
        best = pos[idx].copy()
        best_score = float("inf")
        step = max(data.cell_w, data.cell_h, float(max(data.sizes_hard[idx]) * 0.25))

        candidates: List[np.ndarray] = [target_xy.astype(np.float32), pos[idx].copy()]
        candidates.append(data.port_pull[idx].astype(np.float32))

        for radius in range(1, 42):
            r = step * radius
            for p in range(SPIRAL_POINTS):
                theta = (2.0 * math.pi * p) / SPIRAL_POINTS
                cand = np.array(
                    [target_xy[0] + r * math.cos(theta), target_xy[1] + r * math.sin(theta)],
                    dtype=np.float32,
                )
                candidates.append(cand)

        for cand in candidates:
            cand = cand.copy()
            cand[0] = np.clip(
                cand[0],
                data.sizes_hard[idx, 0] * 0.5,
                data.canvas_w - data.sizes_hard[idx, 0] * 0.5,
            )
            cand[1] = np.clip(
                cand[1],
                data.sizes_hard[idx, 1] * 0.5,
                data.canvas_h - data.sizes_hard[idx, 1] * 0.5,
            )
            if self._hard_overlaps_any(idx, cand, pos, data):
                continue
            neigh = data.neighbor_lists[idx]
            if neigh.size > 0:
                local_wl = np.abs(cand[0] - pos[neigh, 0]).sum() + np.abs(cand[1] - pos[neigh, 1]).sum()
            else:
                local_wl = 0.0
            score = float(np.square(cand - target_xy).sum()) + 0.04 * float(local_wl)
            if score < best_score:
                best_score = score
                best = cand

        return best.astype(np.float32)

    def _hard_overlaps_any(
        self,
        idx: int,
        cand: np.ndarray,
        pos: np.ndarray,
        data: PreparedData,
    ) -> bool:
        dx = np.abs(cand[0] - pos[:, 0])
        dy = np.abs(cand[1] - pos[:, 1])
        ox = (data.sizes_hard[idx, 0] + data.sizes_hard[:, 0]) * 0.5 + LEGAL_GAP - dx
        oy = (data.sizes_hard[idx, 1] + data.sizes_hard[:, 1]) * 0.5 + LEGAL_GAP - dy
        overlap = (ox > 1e-6) & (oy > 1e-6)
        overlap[idx] = False
        return bool(overlap.any())

    def _cheap_score(self, hard: np.ndarray, soft: np.ndarray, data: PreparedData) -> float:
        all_pos = self._all_pos_np(hard, soft, data)
        xs = all_pos[data.safe_nnp_np, 0]
        ys = all_pos[data.safe_nnp_np, 1]
        inf = 1e15
        wl = (
            np.where(data.nnmask_np, xs, -inf).max(axis=1)
            - np.where(data.nnmask_np, xs, inf).min(axis=1)
            + np.where(data.nnmask_np, ys, -inf).max(axis=1)
            - np.where(data.nnmask_np, ys, inf).min(axis=1)
        ).sum() / data.hpwl_norm
        density = self._density_grid_np(np.vstack([hard, soft]), data)
        topk = self._top_k_mean(density, 0.10)
        overlap, _ = self._hard_overlap_stats(hard, data)
        return float(wl + 0.50 * topk + 250.0 * overlap / data.canvas_area)

    def _density_grid_np(self, macros: np.ndarray, data: PreparedData) -> np.ndarray:
        grid = np.zeros((data.grid_rows, data.grid_cols), dtype=np.float32)
        for idx in range(macros.shape[0]):
            w, h = data.sizes_all[idx]
            x, y = macros[idx]
            lx = x - w * 0.5
            ux = x + w * 0.5
            ly = y - h * 0.5
            uy = y + h * 0.5
            c_lo = max(0, int(lx / data.cell_w))
            c_hi = min(data.grid_cols - 1, int(ux / data.cell_w))
            r_lo = max(0, int(ly / data.cell_h))
            r_hi = min(data.grid_rows - 1, int(uy / data.cell_h))
            for row in range(r_lo, r_hi + 1):
                y0 = row * data.cell_h
                y1 = y0 + data.cell_h
                oy = max(0.0, min(uy, y1) - max(ly, y0))
                if oy <= 0:
                    continue
                for col in range(c_lo, c_hi + 1):
                    x0 = col * data.cell_w
                    x1 = x0 + data.cell_w
                    ox = max(0.0, min(ux, x1) - max(lx, x0))
                    if ox > 0:
                        grid[row, col] += (ox * oy) / max(data.cell_w * data.cell_h, 1e-6)
        return grid

    def _top_k_mean(self, grid: np.ndarray, frac: float) -> float:
        flat = grid.reshape(-1)
        top_k = max(1, int(len(flat) * frac))
        return float(np.partition(flat, -top_k)[-top_k:].mean())

    def _consensus_candidate(
        self,
        candidates: Sequence[Tuple[np.ndarray, np.ndarray, str, float]],
        data: PreparedData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.array([c[3] for c in candidates], dtype=np.float32)
        scores -= scores.min()
        weights = np.exp(-6.0 * scores / max(scores.std() + 1e-4, 1e-3))
        weights /= weights.sum()
        hard = np.zeros_like(candidates[0][0])
        soft = np.zeros_like(candidates[0][1])
        for w, (hard_i, soft_i, _, _) in zip(weights, candidates):
            hard += w * hard_i
            soft += w * soft_i
        return hard.astype(np.float32), soft.astype(np.float32)

    def _select_best(
        self,
        candidates: Sequence[Tuple[np.ndarray, np.ndarray, str, float]],
        benchmark: Benchmark,
        plc,
        data: PreparedData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from macro_place.utils import validate_placement

        if plc is None:
            legal = []
            for cand in candidates:
                placement = benchmark.macro_positions.clone()
                placement[: data.num_hard] = torch.from_numpy(cand[0]).float()
                if data.num_soft > 0:
                    placement[data.num_hard : data.num_macros] = torch.from_numpy(cand[1]).float()
                ok, _ = validate_placement(placement, benchmark)
                if ok:
                    legal.append(cand)
            pool = legal if legal else list(candidates)
            best = min(pool, key=lambda x: x[3])
            return best[0], best[1]

        from macro_place.objective import compute_proxy_cost

        best_cost = float("inf")
        best_pair = (candidates[0][0], candidates[0][1])
        for hard, soft, _, _ in candidates:
            placement = benchmark.macro_positions.clone()
            placement[: data.num_hard] = torch.from_numpy(hard).float()
            if data.num_soft > 0:
                placement[data.num_hard : data.num_macros] = torch.from_numpy(soft).float()
            ok, _ = validate_placement(placement, benchmark)
            if not ok:
                continue
            result = compute_proxy_cost(placement, benchmark, plc)
            if float(result["proxy_cost"]) < best_cost:
                best_cost = float(result["proxy_cost"])
                best_pair = (hard.copy(), soft.copy())
        return best_pair

    def _time_left(self, t0: float) -> float:
        return TIME_BUDGET - (time.time() - t0)

    def _try_load_plc(self, benchmark: Benchmark):
        try:
            from macro_place._plc import PlacementCost

            for path in [
                f"external/MacroPlacement/Testcases/ICCAD04/{benchmark.name}",
                f"external/MacroPlacement/Flows/NanGate45/{benchmark.name}/netlist/output_CT_Grouping",
            ]:
                netlist = f"{path}/netlist.pb.txt"
                if os.path.exists(netlist):
                    plc = PlacementCost(netlist.replace("\\", "/"))
                    plc_path = f"{path}/initial.plc"
                    if os.path.exists(plc_path):
                        plc.restore_placement(plc_path, ifInital=True, ifReadComment=True)
                    return plc
        except Exception:
            return None
        return None
