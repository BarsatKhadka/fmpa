"""
Universal multiworld macro placer.

Override runtime with:
    PLACE_TIME_BUDGET=<seconds> uv run evaluate new-soln.py
"""

import math
import os
import random
import time
import zlib
from itertools import permutations
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
CONG_ALPHA = 8.0


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
    net_weights_t: torch.Tensor
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
    hard_net_lists: List[np.ndarray]
    fast_cong_engine: Optional[dict] = None


class Placer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._seed_everything(benchmark.name)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = self._prepare(benchmark)
        plc = self._try_load_plc(benchmark)
        data.fast_cong_engine = self._build_fast_cong_engine(benchmark, plc)

        candidates: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        base_soft = self._clamp_soft_copy(data.init_soft, data)
        legacy_hard = self._tiny_fix_hard(self._legacy_resolve_hard(data.init_hard, data), data)
        legacy_valid = self._placement_is_valid(legacy_hard, base_soft, benchmark, data)
        latent_seed_hard = legacy_hard.copy()
        latent_seed_soft = base_soft.copy()
        candidates.append(
            (legacy_hard, base_soft.copy(), "legacy-base", self._cheap_score(legacy_hard, base_soft, data))
        )
        pair_hard = self._legalize_hard(data.init_hard, data)
        shelf_hard = self._tiny_fix_hard(self._shelf_legalize_hard(data.init_hard, data), data)
        pair_candidate = (pair_hard, base_soft.copy(), "pair-base", self._cheap_score(pair_hard, base_soft, data))
        shelf_candidate = (
            shelf_hard,
            base_soft.copy(),
            "shelf-base",
            self._cheap_score(shelf_hard, base_soft, data),
        )
        extra_hard_seeds: List[np.ndarray] = [pair_hard, shelf_hard]
        if legacy_valid:
            candidates.append(pair_candidate if pair_candidate[3] <= shelf_candidate[3] else shelf_candidate)
        else:
            candidates.append(pair_candidate)
            candidates.append(shelf_candidate)

        large_global_first = data.num_hard >= 520

        # GPU multi-start analytical placement: explores distinct basins cheaply.
        # This is the main quality lever once CUDA is available.
        if self._device.type == "cuda" and self._time_left(t0) >= 30:
            ms = self._gpu_multistart_analytical(
                data,
                t0,
                hard_seeds=[
                    legacy_hard,
                    self._legalize_hard(data.spectral_xy, data),
                    self._legalize_hard(self._quadrant_permute_world(data.init_hard, data), data),
                ],
                soft_seed=base_soft.copy(),
            )
            for hard, soft, name, score in ms:
                candidates.append((hard, soft, name, score))

        if plc is not None and large_global_first and self._time_left(t0) >= 26:
            struct_hard, struct_soft, struct_name = self._partition_floorplan_search(
                latent_seed_hard,
                latent_seed_soft,
                benchmark,
                plc,
                data,
                t0,
            )
            if struct_hard is not None:
                candidates.append(
                    (
                        struct_hard,
                        struct_soft,
                        struct_name,
                        self._cheap_score(struct_hard, struct_soft, data),
                    )
                )
                if large_global_first:
                    extra_hard_seeds.append(struct_hard.copy())

        if plc is not None and self._time_left(t0) >= 25 and large_global_first:
            if legacy_valid:
                warm_t0 = time.time()
                warm_hard, warm_soft, warm_accept = self._incremental_oracle_refine(
                    legacy_hard,
                    base_soft.copy(),
                    benchmark,
                    plc,
                    data,
                    warm_t0,
                    mode="light",
                )
                if warm_accept > 0:
                    latent_seed_hard = warm_hard.copy()
                    latent_seed_soft = warm_soft.copy()
                    candidates.append(
                        (
                            warm_hard,
                            warm_soft,
                            "legacy-base-incre-light",
                            self._cheap_score(warm_hard, warm_soft, data),
                        )
                    )
            stage_t0 = time.time()
            cem_hard, cem_soft, cem_name = self._latent_basis_search(
                latent_seed_hard,
                latent_seed_soft,
                benchmark,
                plc,
                data,
                stage_t0,
                extra_hard_seeds=extra_hard_seeds,
            )
            if cem_hard is not None:
                candidates.append(
                    (
                        cem_hard,
                        cem_soft,
                        cem_name,
                        self._cheap_score(cem_hard, cem_soft, data),
                    )
                )
                latent_seed_hard = cem_hard.copy()
                latent_seed_soft = cem_soft.copy()
                if self._time_left(stage_t0) >= 18:
                    local_t0 = time.time()
                    cem_ref_hard, cem_ref_soft, cem_accept = self._incremental_oracle_refine(
                        cem_hard,
                        cem_soft,
                        benchmark,
                        plc,
                        data,
                        local_t0,
                        mode="strong",
                    )
                    if cem_accept > 0:
                        cem_swap_hard, cem_swap_soft, cem_swap_accept = self._connected_swap_refine(
                            cem_ref_hard,
                            cem_ref_soft,
                            benchmark,
                            plc,
                            data,
                            local_t0,
                        )
                        if cem_swap_accept > 0:
                            cem_ref_hard = cem_swap_hard
                            cem_ref_soft = cem_swap_soft
                        candidates.append(
                            (
                                cem_ref_hard,
                                cem_ref_soft,
                                f"{cem_name}-local",
                                self._cheap_score(cem_ref_hard, cem_ref_soft, data),
                            )
                        )
                        if plc is not None and data.num_hard <= 520 and self._time_left(local_t0) >= 14:
                            exact_t0 = time.time()
                            oracle_hard, oracle_soft, oracle_accept = self._incremental_oracle_refine(
                                cem_ref_hard,
                                cem_ref_soft,
                                benchmark,
                                plc,
                                data,
                                exact_t0,
                                mode="strong",
                                force_exact=True,
                            )
                            if oracle_accept > 0:
                                oracle_swap_hard, oracle_swap_soft, oracle_swap_accept = self._connected_swap_refine(
                                    oracle_hard,
                                    oracle_soft,
                                    benchmark,
                                    plc,
                                    data,
                                    exact_t0,
                                    force_exact=True,
                                )
                                if oracle_swap_accept > 0:
                                    oracle_hard = oracle_swap_hard
                                    oracle_soft = oracle_swap_soft
                                candidates.append(
                                    (
                                        oracle_hard,
                                        oracle_soft,
                                        f"{cem_name}-oracle",
                                        self._cheap_score(oracle_hard, oracle_soft, data),
                                    )
                                )
                        if self._time_left(local_t0) >= 18:
                            cluster_t0 = time.time()
                            cluster_hard, cluster_soft, cluster_accept = self._cluster_shift_refine(
                                cem_ref_hard,
                                cem_ref_soft,
                                benchmark,
                                plc,
                                data,
                                cluster_t0,
                            )
                            if cluster_accept > 0:
                                candidates.append(
                                    (
                                        cluster_hard,
                                        cluster_soft,
                                        f"{cem_name}-cluster",
                                        self._cheap_score(cluster_hard, cluster_soft, data),
                                    )
                                )

        if plc is not None and self._time_left(t0) >= 18 and legacy_valid and not large_global_first:
            local_t0 = time.time()
            refined_hard, refined_soft, accepted = self._incremental_oracle_refine(
                legacy_hard,
                base_soft.copy(),
                benchmark,
                plc,
                data,
                local_t0,
                mode="light",
            )
            if accepted > 0:
                swap_hard, swap_soft, swap_accepted = self._connected_swap_refine(
                    refined_hard,
                    refined_soft,
                    benchmark,
                    plc,
                    data,
                    local_t0,
                )
                if swap_accepted > 0:
                    refined_hard = swap_hard
                    refined_soft = swap_soft
                candidates.append(
                    (
                        refined_hard,
                        refined_soft,
                        "legacy-base-incre-light",
                        self._cheap_score(refined_hard, refined_soft, data),
                    )
                )
                latent_seed_hard = refined_hard.copy()
                latent_seed_soft = refined_soft.copy()

        if plc is not None and self._time_left(t0) >= 25 and not large_global_first:
            stage_t0 = time.time()
            cem_hard, cem_soft, cem_name = self._latent_basis_search(
                latent_seed_hard,
                latent_seed_soft,
                benchmark,
                plc,
                data,
                stage_t0,
                extra_hard_seeds=extra_hard_seeds,
            )
            if cem_hard is not None:
                candidates.append(
                    (
                        cem_hard,
                        cem_soft,
                        cem_name,
                        self._cheap_score(cem_hard, cem_soft, data),
                    )
                )
                if self._time_left(stage_t0) >= 18:
                    local_t0 = time.time()
                    cem_ref_hard, cem_ref_soft, cem_accept = self._incremental_oracle_refine(
                        cem_hard,
                        cem_soft,
                        benchmark,
                        plc,
                        data,
                        local_t0,
                        mode="strong",
                    )
                    if cem_accept > 0:
                        cem_swap_hard, cem_swap_soft, cem_swap_accept = self._connected_swap_refine(
                            cem_ref_hard,
                            cem_ref_soft,
                            benchmark,
                            plc,
                            data,
                            local_t0,
                        )
                        if cem_swap_accept > 0:
                            cem_ref_hard = cem_swap_hard
                            cem_ref_soft = cem_swap_soft
                        candidates.append(
                            (
                                cem_ref_hard,
                                cem_ref_soft,
                                f"{cem_name}-local",
                                self._cheap_score(cem_ref_hard, cem_ref_soft, data),
                            )
                        )
                        if plc is not None and data.num_hard <= 520 and self._time_left(local_t0) >= 14:
                            exact_t0 = time.time()
                            oracle_hard, oracle_soft, oracle_accept = self._incremental_oracle_refine(
                                cem_ref_hard,
                                cem_ref_soft,
                                benchmark,
                                plc,
                                data,
                                exact_t0,
                                mode="strong",
                                force_exact=True,
                            )
                            if oracle_accept > 0:
                                oracle_swap_hard, oracle_swap_soft, oracle_swap_accept = self._connected_swap_refine(
                                    oracle_hard,
                                    oracle_soft,
                                    benchmark,
                                    plc,
                                    data,
                                    exact_t0,
                                    force_exact=True,
                                )
                                if oracle_swap_accept > 0:
                                    oracle_hard = oracle_swap_hard
                                    oracle_soft = oracle_swap_soft
                                candidates.append(
                                    (
                                        oracle_hard,
                                        oracle_soft,
                                        f"{cem_name}-oracle",
                                        self._cheap_score(oracle_hard, oracle_soft, data),
                                    )
                                )
                        if self._time_left(local_t0) >= 18:
                            cluster_t0 = time.time()
                            cluster_hard, cluster_soft, cluster_accept = self._cluster_shift_refine(
                                cem_ref_hard,
                                cem_ref_soft,
                                benchmark,
                                plc,
                                data,
                                cluster_t0,
                            )
                            if cluster_accept > 0:
                                candidates.append(
                                    (
                                        cluster_hard,
                                        cluster_soft,
                                        f"{cem_name}-cluster",
                                        self._cheap_score(cluster_hard, cluster_soft, data),
                                    )
                        )

        worlds = self._build_worlds(data)
        max_worlds = 5 if data.num_hard <= 350 else 4 if data.num_hard <= 600 else 2
        run_expensive_worlds = plc is not None and data.num_hard <= 600 and self._time_left(t0) >= (
            36 if data.num_hard <= 350 else 44 if data.num_hard <= 650 else 52
        )
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
                if self._time_left(t0) >= 14:
                    world_t0 = time.time()
                    oracle_hard, oracle_soft, oracle_accept = self._incremental_oracle_refine(
                        hard,
                        soft,
                        benchmark,
                        plc,
                        data,
                        world_t0,
                        mode="light",
                    )
                    if oracle_accept > 0:
                        swap_hard, swap_soft, swap_accept = self._connected_swap_refine(
                            oracle_hard,
                            oracle_soft,
                            benchmark,
                            plc,
                            data,
                            world_t0,
                        )
                        if swap_accept > 0:
                            oracle_hard = swap_hard
                            oracle_soft = swap_soft
                        candidates.append(
                            (
                                oracle_hard,
                                oracle_soft,
                                f"{world_name}-oracle",
                                self._cheap_score(oracle_hard, oracle_soft, data),
                            )
                        )

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
            if plc is not None and self._time_left(t0) >= 14:
                consensus_t0 = time.time()
                oracle_hard, oracle_soft, oracle_accept = self._incremental_oracle_refine(
                    consensus_hard,
                    consensus_soft,
                    benchmark,
                    plc,
                    data,
                    consensus_t0,
                    mode="light",
                )
                if oracle_accept > 0:
                    candidates.append(
                        (
                            oracle_hard,
                            oracle_soft,
                            "consensus-oracle",
                            self._cheap_score(oracle_hard, oracle_soft, data),
                        )
                    )
                    if data.num_hard <= 520 and self._time_left(consensus_t0) >= 10:
                        swap_hard, swap_soft, swap_accept = self._connected_swap_refine(
                            oracle_hard,
                            oracle_soft,
                            benchmark,
                            plc,
                            data,
                            consensus_t0,
                            force_exact=True,
                        )
                        if swap_accept > 0:
                            candidates.append(
                                (
                                    swap_hard,
                                    swap_soft,
                                    "consensus-oracle-swap",
                                    self._cheap_score(swap_hard, swap_soft, data),
                                )
                            )

        best_hard, best_soft = self._select_best(candidates, benchmark, plc, data)

        placement = benchmark.macro_positions.clone()
        placement[: data.num_hard] = torch.from_numpy(best_hard).float()
        if data.num_soft > 0:
            placement[data.num_hard : data.num_macros] = torch.from_numpy(best_soft).float()
        return placement

    def _gpu_multistart_analytical(
        self,
        data: PreparedData,
        t0: float,
        hard_seeds: Sequence[np.ndarray],
        soft_seed: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, str, float]]:
        # Runs a short continuation schedule on GPU from multiple hard-macro seeds.
        out: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        if self._device.type != "cuda" or self._time_left(t0) < 20:
            return out

        if data.num_hard <= 220:
            schedule = [
                (90, 0.045, 0.40, 0.18, 0.65),
                (70, 0.030, 0.62, 0.10, 0.35),
            ]
        elif data.num_hard <= 450:
            schedule = [
                (70, 0.040, 0.45, 0.16, 0.55),
                (55, 0.026, 0.68, 0.09, 0.28),
            ]
        else:
            # Large designs: avoid the O(N^2) overlap term by using overlap_weight=0.
            schedule = [
                (50, 0.034, 0.55, 0.00, 0.45),
                (40, 0.022, 0.78, 0.00, 0.22),
            ]

        for k, seed in enumerate(hard_seeds[:4]):
            if self._time_left(t0) < 14:
                break
            hard = seed.copy().astype(np.float32)
            soft = self._clamp_soft_copy(soft_seed, data)
            hard = self._clamp_hard_copy(hard, data)
            anchor = hard.copy()

            for steps, lr, den_w, ov_w, snap_noise in schedule:
                if self._time_left(t0) < 12:
                    break
                hard = self._gradient_stage(
                    hard,
                    soft,
                    anchor,
                    data,
                    t0,
                    steps=steps,
                    lr=lr,
                    density_weight=den_w,
                    overlap_weight=ov_w,
                    anchor_weight=0.06,
                    snap_noise=snap_noise,
                )
                # Always legalize between stages so the next stage sees a plausible state.
                hard = self._legalize_hard(hard, data)
                if data.num_soft > 0:
                    soft = self._relax_soft(hard, soft, data, sweeps=1, damping=0.88)

            score = self._cheap_score(hard, soft, data)
            out.append((hard.copy(), soft.copy(), f"gpu-ms-{k}", score))

        out.sort(key=lambda x: x[3])
        return out[:3]

    def _clamp_hard_copy(self, hard: np.ndarray, data: PreparedData) -> np.ndarray:
        out = hard.copy().astype(np.float32)
        self._clamp_hard(out, data)
        return out

    def _seed_everything(self, benchmark_name: str) -> None:
        seed = (SEED ^ self._stable_seed(benchmark_name)) & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _stable_seed(self, text: str) -> int:
        return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF

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
        net_weights_t = benchmark.net_weights.detach().to(dtype=torch.float32).cpu()
        port_base = num_macros

        hard_adj = np.zeros((num_hard, num_hard), dtype=np.float32)
        port_pull = np.zeros((num_hard, 2), dtype=np.float32)
        port_pull_count = np.zeros(num_hard, dtype=np.float32)
        soft_occ_owner: List[int] = []
        soft_occ_net: List[int] = []
        hard_occ_owner: List[int] = []
        hard_occ_net: List[int] = []
        hard_net_lists: List[List[int]] = [[] for _ in range(num_hard)]

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
                hard_net_lists[int(owner)].append(net_id)

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
            net_weights_t=net_weights_t,
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
            hard_net_lists=[np.array(nets, dtype=np.int64) for nets in hard_net_lists],
            fast_cong_engine=None,
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
        quadrant = self._quadrant_permute_world(data.init_hard, data)
        worlds.append((quadrant, data.init_soft.copy(), "quadrant"))
        return worlds

    def _quadrant_permute_world(self, hard: np.ndarray, data: PreparedData) -> np.ndarray:
        permuted = hard.copy().astype(np.float32)
        if data.num_hard == 0:
            return permuted

        cx = data.canvas_w * 0.5
        cy = data.canvas_h * 0.5
        quad_bounds = [
            (0.0, 0.0, cx, cy),
            (cx, 0.0, data.canvas_w, cy),
            (0.0, cy, cx, data.canvas_h),
            (cx, cy, data.canvas_w, data.canvas_h),
        ]
        quads: List[List[int]] = [[], [], [], []]
        for idx in range(data.num_hard):
            if not data.movable_hard[idx]:
                continue
            x, y = hard[idx]
            q = (1 if x > cx else 0) + (2 if y > cy else 0)
            quads[q].append(idx)

        order = [2, 0, 3, 1]
        for src_q, dst_q in enumerate(order):
            src_lx, src_ly, src_ux, src_uy = quad_bounds[src_q]
            dst_lx, dst_ly, dst_ux, dst_uy = quad_bounds[dst_q]
            src_w = max(src_ux - src_lx, 1e-6)
            src_h = max(src_uy - src_ly, 1e-6)
            for idx in quads[src_q]:
                w, h = data.sizes_hard[idx]
                rel_x = float(np.clip((hard[idx, 0] - src_lx) / src_w, 0.05, 0.95))
                rel_y = float(np.clip((hard[idx, 1] - src_ly) / src_h, 0.05, 0.95))
                dst_x = dst_lx + rel_x * (dst_ux - dst_lx)
                dst_y = dst_ly + rel_y * (dst_uy - dst_ly)
                permuted[idx, 0] = np.clip(dst_x, w * 0.5, data.canvas_w - w * 0.5)
                permuted[idx, 1] = np.clip(dst_y, h * 0.5, data.canvas_h - h * 0.5)
        return permuted.astype(np.float32)

    def _partition_floorplan_search(
        self,
        base_hard: np.ndarray,
        base_soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        if plc is None or data.num_hard < 120 or self._time_left(t0) < 22:
            return None, None, ""

        n_clusters = 4
        labels = self._kmeans_labels(data.spectral_xy, n_clusters, iters=16)
        cluster_ids = np.unique(labels)
        if cluster_ids.size < 3:
            return None, None, ""

        rects = self._region_boxes(int(cluster_ids.size), data)
        if len(rects) != int(cluster_ids.size):
            return None, None, ""

        base_exact = self._proxy_components_if_valid(base_hard, base_soft, benchmark, plc, data)
        base_proxy = base_exact["proxy"] if base_exact is not None else float("inf")
        base_cheap_comps = self._cheap_components(base_hard, base_soft, data)
        calib = self._init_surrogate_calib(base_cheap_comps, base_exact)
        base_cheap = self._surrogate_score_from_components(base_cheap_comps, calib)

        if data.num_hard <= 260:
            exact_budget = 8
            jitter_trials = 2
        elif data.num_hard <= 520:
            exact_budget = 6
            jitter_trials = 1
        else:
            exact_budget = 0
            jitter_trials = 1

        if data.num_hard > 520:
            perm_list = [
                (0, 1, 2, 3),
                (1, 0, 3, 2),
                (2, 3, 0, 1),
                (3, 2, 1, 0),
                (0, 2, 1, 3),
                (2, 0, 3, 1),
                (1, 3, 0, 2),
                (3, 1, 2, 0),
            ][: len(rects) * 2]
            transforms = [(False, False, False)]
            fill_scales = [0.92, 1.00]
        else:
            perm_list = list(permutations(range(len(rects))))
            transforms = [
                (False, False, False),
                (True, False, False),
                (False, True, False),
                (True, True, False),
            ]
            fill_scales = [0.84]
        rng = np.random.default_rng((SEED ^ self._stable_seed(f"{benchmark.name}-partition")) & 0xFFFFFFFF)

        proposals: List[Tuple[float, np.ndarray]] = []
        for perm in perm_list:
            if self._time_left(t0) < 14:
                break
            for fill_scale in fill_scales:
                for flip_x, flip_y, swap_xy in transforms:
                    cand_hard = self._apply_partition_assignment(
                        base_hard,
                        labels,
                        cluster_ids,
                        rects,
                        perm,
                        data,
                        flip_x=flip_x,
                        flip_y=flip_y,
                        swap_xy=swap_xy,
                        fill_scale=fill_scale,
                    )
                    if data.num_hard > 520:
                        self._clamp_hard(cand_hard, data)
                    else:
                        cand_hard = self._legacy_resolve_hard(cand_hard, data, max_rounds=8)
                        cand_hard = self._tiny_fix_hard(cand_hard, data)
                    if jitter_trials > 0:
                        cheap = self._robust_latent_score(
                            cand_hard,
                            base_soft,
                            data,
                            rng,
                            calib,
                            jitter_trials=jitter_trials,
                        )
                    elif data.num_hard > 520:
                        cheap = self._coarse_partition_score(cand_hard, base_soft, data)
                    else:
                        cheap = self._surrogate_score_from_components(
                            self._cheap_components(cand_hard, base_soft, data),
                            calib,
                        )
                    proposals.append((cheap, cand_hard.astype(np.float32)))

        if not proposals:
            return None, None, ""

        proposals.sort(key=lambda item: item[0])
        best_cheap = proposals[0][0]
        best_cheap_hard = proposals[0][1].copy().astype(np.float32)
        if data.num_hard > 520:
            cheap_soft = base_soft.copy().astype(np.float32)
            if data.num_soft > 0:
                cheap_soft = self._relax_soft(best_cheap_hard, cheap_soft, data, sweeps=1, damping=0.92)
            return best_cheap_hard.astype(np.float32), cheap_soft.astype(np.float32), "partition-floorplan-cheap"
        best_hard = None
        best_soft = None
        best_exact = base_proxy
        for _, cand_hard in proposals[:exact_budget]:
            if self._time_left(t0) < 10:
                break
            cand_soft = base_soft.copy().astype(np.float32)
            if data.num_soft > 0:
                cand_soft = self._relax_soft(cand_hard, cand_soft, data, sweeps=1, damping=0.90)
            comps = self._proxy_components_if_valid(cand_hard, cand_soft, benchmark, plc, data)
            if comps is None:
                continue
            if comps["proxy"] + 1e-6 < best_exact:
                best_exact = comps["proxy"]
                best_hard = cand_hard.copy()
                best_soft = cand_soft.copy()

        if best_hard is not None:
            return best_hard.astype(np.float32), best_soft.astype(np.float32), "partition-floorplan"
        if best_cheap + 1e-4 < base_cheap:
            cheap_soft = base_soft.copy().astype(np.float32)
            if data.num_soft > 0:
                cheap_soft = self._relax_soft(best_cheap_hard, cheap_soft, data, sweeps=1, damping=0.92)
            return best_cheap_hard.astype(np.float32), cheap_soft.astype(np.float32), "partition-floorplan-cheap"
        return None, None, ""

    def _region_boxes(
        self,
        n_regions: int,
        data: PreparedData,
    ) -> List[Tuple[float, float, float, float]]:
        if n_regions <= 0:
            return []
        if n_regions == 1:
            return [(0.0, 0.0, data.canvas_w, data.canvas_h)]
        if n_regions == 2:
            mid = data.canvas_w * 0.5
            return [(0.0, 0.0, mid, data.canvas_h), (mid, 0.0, data.canvas_w, data.canvas_h)]
        if n_regions == 3:
            step = data.canvas_w / 3.0
            return [
                (0.0, 0.0, step, data.canvas_h),
                (step, 0.0, 2.0 * step, data.canvas_h),
                (2.0 * step, 0.0, data.canvas_w, data.canvas_h),
            ]
        if n_regions == 4:
            cx = data.canvas_w * 0.5
            cy = data.canvas_h * 0.5
            return [
                (0.0, 0.0, cx, cy),
                (cx, 0.0, data.canvas_w, cy),
                (0.0, cy, cx, data.canvas_h),
                (cx, cy, data.canvas_w, data.canvas_h),
            ]
        cols = int(math.ceil(math.sqrt(n_regions)))
        rows = int(math.ceil(n_regions / cols))
        rects: List[Tuple[float, float, float, float]] = []
        for row in range(rows):
            for col in range(cols):
                if len(rects) >= n_regions:
                    break
                lx = data.canvas_w * col / cols
                ux = data.canvas_w * (col + 1) / cols
                ly = data.canvas_h * row / rows
                uy = data.canvas_h * (row + 1) / rows
                rects.append((lx, ly, ux, uy))
        return rects

    def _apply_partition_assignment(
        self,
        base_hard: np.ndarray,
        labels: np.ndarray,
        cluster_ids: np.ndarray,
        rects: Sequence[Tuple[float, float, float, float]],
        perm: Sequence[int],
        data: PreparedData,
        flip_x: bool,
        flip_y: bool,
        swap_xy: bool,
        fill_scale: float,
    ) -> np.ndarray:
        cand = base_hard.copy().astype(np.float32)
        for src_idx, cluster_id in enumerate(cluster_ids):
            members = np.flatnonzero(labels == int(cluster_id))
            if members.size == 0:
                continue
            movable_members = members[data.movable_hard[members]]
            if movable_members.size == 0:
                continue

            pts = base_hard[movable_members].astype(np.float32)
            center = pts.mean(axis=0)
            span = np.maximum(pts.max(axis=0) - pts.min(axis=0), np.array([data.cell_w, data.cell_h], dtype=np.float32))
            rel = pts - center[None, :]
            if swap_xy:
                rel = rel[:, ::-1]
            if flip_x:
                rel[:, 0] *= -1.0
            if flip_y:
                rel[:, 1] *= -1.0

            dst_lx, dst_ly, dst_ux, dst_uy = rects[int(perm[src_idx])]
            dst_center = np.array([(dst_lx + dst_ux) * 0.5, (dst_ly + dst_uy) * 0.5], dtype=np.float32)
            inner_w = max((dst_ux - dst_lx) * fill_scale, data.cell_w)
            inner_h = max((dst_uy - dst_ly) * fill_scale, data.cell_h)
            scale = float(np.clip(min(inner_w / span[0], inner_h / span[1]), 0.45, 1.85))
            mapped = dst_center[None, :] + rel * scale

            widths = data.sizes_hard[movable_members, 0] * 0.5
            heights = data.sizes_hard[movable_members, 1] * 0.5
            cand[movable_members, 0] = np.clip(mapped[:, 0], widths, data.canvas_w - widths)
            cand[movable_members, 1] = np.clip(mapped[:, 1], heights, data.canvas_h - heights)
        return cand.astype(np.float32)

    def _coarse_partition_score(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
    ) -> float:
        all_pos = self._all_pos_np(hard, soft, data)
        xs = all_pos[data.safe_nnp_np, 0]
        ys = all_pos[data.safe_nnp_np, 1]
        mask = data.nnmask_np
        big = 1e9
        xs_min = np.where(mask, xs, big).min(axis=1)
        xs_max = np.where(mask, xs, -big).max(axis=1)
        ys_min = np.where(mask, ys, big).min(axis=1)
        ys_max = np.where(mask, ys, -big).max(axis=1)
        wl = float(((xs_max - xs_min) + (ys_max - ys_min)).sum() / max(data.hpwl_norm, 1e-6))
        overlap, _ = self._hard_overlap_stats(hard, data)
        disp = float(np.square(hard - data.init_hard).mean() / max(data.canvas_area, 1e-6))
        return float(wl + 120.0 * overlap / max(data.canvas_area, 1e-6) + 0.08 * disp)

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
                (40, 0.040, 0.55, 120.0, 0.10, 0.40),
                (30, 0.025, 0.70, 200.0, 0.04, 0.18),
            ]
        else:
            stages = [
                (70, 0.045, 0.45, 90.0, 0.12, 0.55),
                (55, 0.030, 0.65, 150.0, 0.06, 0.28),
                (40, 0.018, 0.85, 250.0, 0.02, 0.10),
            ]

        anchor = hard.copy()
        current_hard = hard.copy()
        current_soft = soft.copy()

        for steps, lr, density_weight, overlap_weight, anchor_weight, snap_noise in stages:
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
                snap_noise=snap_noise,
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
        snap_noise: float,
    ) -> np.ndarray:
        movable_idx = np.flatnonzero(data.movable_hard).astype(np.int64)
        if movable_idx.size == 0:
            return hard

        device = getattr(self, "_device", torch.device("cpu"))
        hard_t = torch.from_numpy(hard.copy()).to(device)
        soft_t = torch.from_numpy(soft.copy()).to(device)
        anchor_t = torch.from_numpy(anchor.copy()).to(device)
        params = hard_t[movable_idx].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=lr)

        hard_sizes = data.hard_sizes_t.to(device)
        soft_sizes = data.soft_sizes_t.to(device)
        safe_nnp_t = data.safe_nnp_t.to(device)
        nnmask_t = data.nnmask_t.to(device)
        port_t = data.port_t.to(device)

        fixed_hard = hard_t.clone()
        if data.num_soft > 0:
            soft_density = self._density_map(soft_t, soft_sizes, data, device=device)
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
                torch.cat([cur_hard, soft_t, port_t], dim=0)
                if data.num_soft > 0
                else torch.cat([cur_hard, port_t], dim=0)
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

            hard_density = self._density_map(cur_hard, hard_sizes, data, device=device)
            density = hard_density if soft_density is None else hard_density + soft_density
            overflow = (density - BASE_DENSITY_TARGET).clamp(min=0.0)
            density_cost = overflow.pow(2).mean() + 0.15 * overflow.max()

            # Differentiable congestion surrogate (RUDY-like) on GPU.
            # Enabled in later part of the stage to avoid fighting early global spreading.
            cong_cost = xs.new_tensor(0.0)
            if step >= int(steps * 0.55):
                cong_map = self._rudy_congestion_map(
                    all_pos,
                    data,
                    device=device,
                    alpha=CONG_ALPHA,
                )
                cong_cost = self._abu_logsumexp(cong_map, frac=0.05)

            cx = cur_hard[:, 0]
            cy = cur_hard[:, 1]
            sep_x = (hard_sizes[:, 0:1] + hard_sizes[:, 0:1].T) * 0.5
            sep_y = (hard_sizes[:, 1:2] + hard_sizes[:, 1:2].T) * 0.5
            ov_x = (sep_x - (cx[:, None] - cx[None, :]).abs()).clamp(min=0.0)
            ov_y = (sep_y - (cy[:, None] - cy[None, :]).abs()).clamp(min=0.0)
            overlap = ov_x * ov_y
            if data.num_hard > 520:
                overlap_cost = xs.new_tensor(0.0)
            else:
                overlap_cost = (overlap.sum() - overlap.diagonal().sum()) * 0.5 / data.canvas_area

            anchor_delta = (cur_hard - anchor_t).pow(2).mean() / max(data.canvas_area, 1.0)
            loss = wl
            loss = loss + density_weight * density_cost
            loss = loss + overlap_weight * overlap_cost
            loss = loss + 0.55 * cong_cost
            loss = loss + anchor_weight * anchor_delta

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                lo_x = torch.from_numpy(data.sizes_hard[movable_idx, 0] * 0.5).to(device)
                hi_x = data.canvas_w - lo_x
                lo_y = torch.from_numpy(data.sizes_hard[movable_idx, 1] * 0.5).to(device)
                hi_y = data.canvas_h - lo_y
                params[:, 0].clamp_(lo_x, hi_x)
                params[:, 1].clamp_(lo_y, hi_y)
                if snap_noise > 1e-6:
                    frac = 1.0 - (step / max(steps - 1, 1))
                    noise_x = torch.empty_like(params[:, 0]).uniform_(-0.5 * data.cell_w, 0.5 * data.cell_w)
                    noise_y = torch.empty_like(params[:, 1]).uniform_(-0.5 * data.cell_h, 0.5 * data.cell_h)
                    params[:, 0].add_(noise_x * snap_noise * frac)
                    params[:, 1].add_(noise_y * snap_noise * frac)
                    params[:, 0].clamp_(lo_x, hi_x)
                    params[:, 1].clamp_(lo_y, hi_y)

            score = float(loss.detach())
            if score < best_score:
                best_score = score
                best_params = params.detach().clone()

        result = hard.copy()
        result[movable_idx] = best_params.detach().cpu().numpy()
        return result.astype(np.float32)

    def _density_map(
        self,
        pos_t: torch.Tensor,
        sizes_t: torch.Tensor,
        data: PreparedData,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if pos_t.numel() == 0:
            return torch.zeros((data.grid_rows, data.grid_cols), dtype=torch.float32, device=pos_t.device)
        if device is None:
            device = pos_t.device

        cell_cx_g = data.cell_cx_g.to(device)
        cell_cy_g = data.cell_cy_g.to(device)
        sigma_x = sizes_t[:, 0].view(-1, 1, 1) * 0.5 + data.cell_w * 0.5
        sigma_y = sizes_t[:, 1].view(-1, 1, 1) * 0.5 + data.cell_h * 0.5
        dx = (pos_t[:, 0].view(-1, 1, 1) - cell_cx_g.unsqueeze(0)) / sigma_x
        dy = (pos_t[:, 1].view(-1, 1, 1) - cell_cy_g.unsqueeze(0)) / sigma_y
        kernel = (1.0 - dx.abs()).clamp(min=0.0) * (1.0 - dy.abs()).clamp(min=0.0)
        area_scale = (sizes_t[:, 0] * sizes_t[:, 1]).view(-1, 1, 1) / max(
            data.cell_w * data.cell_h, 1e-6
        )
        kernel_sum = kernel.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return (kernel / kernel_sum * area_scale).sum(dim=0)

    def _abu_logsumexp(self, grid: torch.Tensor, frac: float = 0.05) -> torch.Tensor:
        flat = grid.reshape(-1)
        k = max(1, int(flat.numel() * float(frac)))
        topk, _ = torch.topk(flat, k)
        # Smooth top-k mean: logsumexp is more stable than hard mean for optimization.
        return torch.logsumexp(topk, dim=0) - math.log(float(k))

    def _rudy_congestion_map(
        self,
        all_pos: torch.Tensor,
        data: PreparedData,
        device: torch.device,
        alpha: float,
    ) -> torch.Tensor:
        # Smooth bbox via logsumexp max/min, then RUDY = (dx+dy)/(dx*dy+eps).
        xs = all_pos[data.safe_nnp_t.to(device), 0]
        ys = all_pos[data.safe_nnp_t.to(device), 1]
        mask = data.nnmask_t.to(device)
        neg_inf = torch.full_like(xs, -1e9)
        xs_p = torch.where(mask, xs, neg_inf)
        xs_n = torch.where(mask, -xs, neg_inf)
        ys_p = torch.where(mask, ys, neg_inf)
        ys_n = torch.where(mask, -ys, neg_inf)
        x_max = torch.logsumexp(alpha * xs_p, dim=1) / alpha
        x_min = -torch.logsumexp(alpha * xs_n, dim=1) / alpha
        y_max = torch.logsumexp(alpha * ys_p, dim=1) / alpha
        y_min = -torch.logsumexp(alpha * ys_n, dim=1) / alpha
        span_x = (x_max - x_min).clamp(min=data.cell_w * 0.25)
        span_y = (y_max - y_min).clamp(min=data.cell_h * 0.25)
        rudy = (span_x + span_y) / (span_x * span_y + 1e-6)
        w = data.net_weights_t.to(device)
        if w.numel() == rudy.numel():
            rudy = rudy * w

        cx = 0.5 * (x_max + x_min)
        cy = 0.5 * (y_max + y_min)
        sigma_x = 0.5 * span_x + data.cell_w
        sigma_y = 0.5 * span_y + data.cell_h

        cell_x = (data.cell_cx_g[0]).to(device)
        cell_y = (data.cell_cy_g[:, 0]).to(device)
        dx = (cx[:, None] - cell_x[None, :]) / sigma_x[:, None]
        dy = (cy[:, None] - cell_y[None, :]) / sigma_y[:, None]
        kx = (1.0 - dx.abs()).clamp(min=0.0, max=1.0)
        ky = (1.0 - dy.abs()).clamp(min=0.0, max=1.0)
        # Separable accumulation without materializing (nets x rows x cols).
        # (rows, cols) = sum_n rudy[n] * ky[n, row] * kx[n, col]
        grid = torch.einsum("n,nr,nc->rc", rudy, ky, kx)
        # Normalize to keep scale roughly stable across benchmarks.
        return grid / (grid.mean() + 1e-6)

    def _hash_resolve_hard(self, hard: np.ndarray, data: PreparedData, sweeps: int = 4) -> np.ndarray:
        # Large-N overlap reduction with spatial hashing (avoids O(N^2) loops).
        pos = hard.copy().astype(np.float32)
        self._clamp_hard(pos, data)

        if data.num_hard <= 1:
            return pos

        bin_w = float(max(np.median(data.sizes_hard[:, 0]), data.cell_w, 1.0))
        bin_h = float(max(np.median(data.sizes_hard[:, 1]), data.cell_h, 1.0))
        inv_w = 1.0 / max(bin_w, 1e-6)
        inv_h = 1.0 / max(bin_h, 1e-6)

        for _ in range(max(1, sweeps)):
            bins: dict[tuple[int, int], list[int]] = {}
            for i in range(data.num_hard):
                bx = int(pos[i, 0] * inv_w)
                by = int(pos[i, 1] * inv_h)
                bins.setdefault((bx, by), []).append(i)

            moved_any = False
            for i in range(data.num_hard):
                if not data.movable_hard[i]:
                    continue
                bx = int(pos[i, 0] * inv_w)
                by = int(pos[i, 1] * inv_h)
                push_x = 0.0
                push_y = 0.0
                for dxbin in (-1, 0, 1):
                    for dybin in (-1, 0, 1):
                        cell = (bx + dxbin, by + dybin)
                        if cell not in bins:
                            continue
                        for j in bins[cell]:
                            if j == i:
                                continue
                            dx = float(pos[i, 0] - pos[j, 0])
                            dy = float(pos[i, 1] - pos[j, 1])
                            ox = float(
                                (data.sizes_hard[i, 0] + data.sizes_hard[j, 0]) * 0.5
                                + LEGAL_GAP
                                - abs(dx)
                            )
                            oy = float(
                                (data.sizes_hard[i, 1] + data.sizes_hard[j, 1]) * 0.5
                                + LEGAL_GAP
                                - abs(dy)
                            )
                            if ox <= 0.0 or oy <= 0.0:
                                continue
                            if ox < oy:
                                push = 0.5 * ox
                                push_x += -push if dx >= 0 else push
                            else:
                                push = 0.5 * oy
                                push_y += -push if dy >= 0 else push

                step = float(max(data.cell_w, data.cell_h, 1e-3))
                push_x = float(np.clip(push_x, -2.0 * step, 2.0 * step))
                push_y = float(np.clip(push_y, -2.0 * step, 2.0 * step))
                if abs(push_x) > 1e-6 or abs(push_y) > 1e-6:
                    pos[i, 0] += push_x
                    pos[i, 1] += push_y
                    moved_any = True

            self._clamp_hard(pos, data)
            if not moved_any:
                break

        return pos.astype(np.float32)

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

    def _placement_tensor(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        data: PreparedData,
    ) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        placement[: data.num_hard] = torch.from_numpy(hard).float()
        if data.num_soft > 0:
            placement[data.num_hard : data.num_macros] = torch.from_numpy(soft).float()
        return placement

    def _placement_is_valid(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        data: PreparedData,
    ) -> bool:
        from macro_place.utils import validate_placement

        placement = self._placement_tensor(hard, soft, benchmark, data)
        ok, _ = validate_placement(placement, benchmark)
        return bool(ok)

    def _proxy_cost_if_valid(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
    ) -> float:
        if plc is None:
            return float("inf")

        from macro_place.objective import compute_proxy_cost
        from macro_place.utils import validate_placement

        placement = self._placement_tensor(hard, soft, benchmark, data)
        ok, _ = validate_placement(placement, benchmark)
        if not ok:
            return float("inf")
        result = compute_proxy_cost(placement, benchmark, plc)
        return float(result["proxy_cost"])

    def _proxy_components_if_valid(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
    ) -> Optional[dict]:
        if plc is None:
            return None

        from macro_place.objective import compute_proxy_cost
        from macro_place.utils import validate_placement

        placement = self._placement_tensor(hard, soft, benchmark, data)
        ok, _ = validate_placement(placement, benchmark)
        if not ok:
            return None
        result = compute_proxy_cost(placement, benchmark, plc)
        return {
            "proxy": float(result["proxy_cost"]),
            "wl": float(result["wirelength_cost"]),
            "den": float(result["density_cost"]),
            "cong": float(result["congestion_cost"]),
        }

    def _incremental_oracle_refine(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
        mode: str,
        force_exact: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        if plc is None or self._time_left(t0) < 12:
            return hard.copy(), soft.copy(), 0

        if mode == "strong":
            config = {
                "sweeps": 2,
                "side_budget": 3,
                "cheap_trials": 5,
                "exact_trials": 4,
                "density_weight": 0.15,
                "boundary_weight": 0.28,
                "disp_weight": 0.025,
                "inner_alpha": 0.65,
                "edge_inset": 0.10 * max(data.cell_w, data.cell_h),
                "accept_margin": 1e-5,
            }
        else:
            config = {
                "sweeps": 1,
                "side_budget": 2,
                "cheap_trials": 4,
                "exact_trials": 3,
                "density_weight": 0.10,
                "boundary_weight": 0.20,
                "disp_weight": 0.035,
                "inner_alpha": 0.45,
                "edge_inset": 0.14 * max(data.cell_w, data.cell_h),
                "accept_margin": 1e-5,
            }

        if data.num_hard <= 220:
            macro_budget = 10 if mode == "strong" else 8
        elif data.num_hard <= 450:
            macro_budget = 8 if mode == "strong" else 6
        else:
            macro_budget = 6 if mode == "strong" else 4

        use_exact = force_exact and plc is not None and mode == "strong" and data.num_hard <= 520
        cur_hard = hard.copy().astype(np.float32)
        cur_soft = soft.copy().astype(np.float32)
        cur_cheap = self._cheap_score(cur_hard, cur_soft, data)
        cur_exact = self._proxy_cost_if_valid(cur_hard, cur_soft, benchmark, plc, data) if use_exact else float("inf")

        accepted = 0
        for _ in range(config["sweeps"]):
            if self._time_left(t0) < 11:
                break
            poor_scores = self._diagnose_poor_macros(cur_hard, cur_soft, data)
            order = np.argsort(-poor_scores)
            sweep_accepted = 0
            sweep_misses = 0
            for idx in order[:macro_budget]:
                if poor_scores[idx] <= 1e-8 or self._time_left(t0) < 10:
                    break
                proposals = self._macro_reposition_proposals(
                    int(idx),
                    cur_hard,
                    cur_soft,
                    data,
                    config,
                )
                if not proposals:
                    continue
                best_hard = None
                best_soft = None
                best_cheap = cur_cheap
                best_exact = cur_exact
                ranked: List[Tuple[float, np.ndarray, np.ndarray]] = []
                for cand_hard, _ in proposals[: max(config["cheap_trials"], config["exact_trials"])]:
                    cand_cheap = self._cheap_score(cand_hard, cur_soft, data)
                    ranked.append((cand_cheap, cand_hard.astype(np.float32), cur_soft.copy().astype(np.float32)))
                    if not use_exact and cand_cheap + config["accept_margin"] < best_cheap:
                        best_cheap = cand_cheap
                        best_hard = cand_hard
                        best_soft = cur_soft.copy().astype(np.float32)

                ranked.sort(key=lambda item: item[0])
                for cand_cheap, cand_hard, cand_soft in ranked[: config["exact_trials"]]:
                    if not use_exact:
                        break
                    if self._time_left(t0) < 9:
                        break
                    if data.num_soft > 0:
                        relaxed_soft = self._relax_soft(
                            cand_hard,
                            cand_soft,
                            data,
                            sweeps=1,
                            damping=0.88 if mode == "strong" else 0.92,
                        )
                        relaxed_cheap = self._cheap_score(cand_hard, relaxed_soft, data)
                        if relaxed_cheap + 1e-6 < cand_cheap:
                            cand_soft = relaxed_soft.astype(np.float32)
                            cand_cheap = relaxed_cheap

                    cand_exact = self._proxy_cost_if_valid(cand_hard, cand_soft, benchmark, plc, data)
                    if not math.isfinite(cand_exact):
                        continue
                    if math.isfinite(best_exact):
                        better = cand_exact + config["accept_margin"] < best_exact
                    else:
                        better = best_hard is None or cand_exact + config["accept_margin"] < float("inf")
                    if better or (
                        math.isfinite(cand_exact)
                        and math.isfinite(best_exact)
                        and abs(cand_exact - best_exact) <= config["accept_margin"]
                        and cand_cheap + 1e-6 < best_cheap
                    ):
                        best_exact = cand_exact
                        best_cheap = cand_cheap
                        best_hard = cand_hard
                        best_soft = cand_soft

                if best_hard is not None and best_soft is not None:
                    cur_hard = best_hard
                    cur_soft = best_soft
                    cur_cheap = best_cheap
                    cur_exact = best_exact
                    accepted += 1
                    sweep_accepted += 1
                    sweep_misses = 0
                else:
                    sweep_misses += 1
                    if accepted == 0 and sweep_misses >= 2:
                        break
            if sweep_accepted == 0:
                break

        if accepted > 0 and data.num_soft > 0 and self._time_left(t0) >= 9:
            relaxed_soft = self._relax_soft(
                cur_hard,
                cur_soft,
                data,
                sweeps=2 if mode == "strong" else 1,
                damping=0.86 if mode == "strong" else 0.90,
            )
            relaxed_cheap = self._cheap_score(cur_hard, relaxed_soft, data)
            relaxed_exact = self._proxy_cost_if_valid(cur_hard, relaxed_soft, benchmark, plc, data) if use_exact else float("inf")
            if use_exact and math.isfinite(cur_exact):
                better = math.isfinite(relaxed_exact) and relaxed_exact + 1e-6 < cur_exact
            else:
                better = relaxed_cheap + 1e-6 < cur_cheap
            if better or (
                use_exact
                and math.isfinite(relaxed_exact)
                and math.isfinite(cur_exact)
                and abs(relaxed_exact - cur_exact) <= 1e-6
                and relaxed_cheap + 1e-6 < cur_cheap
            ):
                cur_soft = relaxed_soft
                cur_exact = relaxed_exact
                cur_cheap = relaxed_cheap

        return cur_hard.astype(np.float32), cur_soft.astype(np.float32), accepted

    def _diagnose_poor_macros(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
    ) -> np.ndarray:
        scores = np.zeros(data.num_hard, dtype=np.float32)
        if data.num_hard <= 1:
            return scores

        density_grid = self._density_grid_np(np.vstack([hard, soft]), data)
        all_pos = self._all_pos_np(hard, soft, data)
        net_centers = self._net_box_centers_np(all_pos, data)
        mean_area = float(max(data.hard_areas.mean(), 1e-6))
        boundary_band = 2.5 * max(data.cell_w, data.cell_h)
        max_span = max(data.canvas_w, data.canvas_h, 1e-6)

        for idx in range(data.num_hard):
            if not data.movable_hard[idx]:
                continue

            boundary_clear = self._boundary_clearance(int(idx), hard[idx], data)
            deltas = hard - hard[idx]
            dist2 = np.square(deltas[:, 0]) + np.square(deltas[:, 1])
            dist2[idx] = np.inf
            k = min(4, data.num_hard - 1)
            if k <= 0:
                continue
            nearest = np.argpartition(dist2, k - 1)[:k]

            tol_x = 0.20 * (data.sizes_hard[idx, 0] + data.sizes_hard[nearest, 0])
            tol_y = 0.20 * (data.sizes_hard[idx, 1] + data.sizes_hard[nearest, 1])
            west = bool(np.any(hard[nearest, 0] < hard[idx, 0] - tol_x))
            east = bool(np.any(hard[nearest, 0] > hard[idx, 0] + tol_x))
            south = bool(np.any(hard[nearest, 1] < hard[idx, 1] - tol_y))
            north = bool(np.any(hard[nearest, 1] > hard[idx, 1] + tol_y))
            if boundary_clear <= boundary_band or (west and east and south and north):
                continue

            density_val = self._macro_density_sample(int(idx), hard[idx], density_grid, data)
            net_target = self._macro_net_target(int(idx), net_centers, data)
            net_pull = float(np.linalg.norm(net_target - hard[idx]) / max_span)
            area_scale = math.sqrt(float(data.hard_areas[idx] / mean_area))
            degree_scale = 1.0 + 0.18 * math.log1p(float(data.hard_net_lists[idx].size))
            boundary_scale = boundary_clear / max_span
            scores[idx] = float(
                (boundary_scale * (0.65 + 0.35 * density_val) + 0.20 * net_pull)
                * area_scale
                * degree_scale
            )

        return scores

    def _macro_net_target(
        self,
        idx: int,
        net_centers: np.ndarray,
        data: PreparedData,
    ) -> np.ndarray:
        nets = data.hard_net_lists[idx]
        if nets.size == 0:
            return data.port_pull[idx].astype(np.float32)
        target = net_centers[nets].mean(axis=0)
        target = 0.80 * target + 0.20 * data.port_pull[idx]
        return target.astype(np.float32)

    def _boundary_clearance(self, idx: int, xy: np.ndarray, data: PreparedData) -> float:
        left = float(xy[0] - data.sizes_hard[idx, 0] * 0.5)
        right = float(data.canvas_w - (xy[0] + data.sizes_hard[idx, 0] * 0.5))
        bottom = float(xy[1] - data.sizes_hard[idx, 1] * 0.5)
        top = float(data.canvas_h - (xy[1] + data.sizes_hard[idx, 1] * 0.5))
        return min(left, right, bottom, top)

    def _boundary_target(
        self,
        idx: int,
        side: str,
        ref_xy: np.ndarray,
        data: PreparedData,
        inset: float,
    ) -> np.ndarray:
        half_w = data.sizes_hard[idx, 0] * 0.5
        half_h = data.sizes_hard[idx, 1] * 0.5
        if side == "left":
            return np.array(
                [
                    half_w + inset,
                    np.clip(ref_xy[1], half_h, data.canvas_h - half_h),
                ],
                dtype=np.float32,
            )
        if side == "right":
            return np.array(
                [
                    data.canvas_w - half_w - inset,
                    np.clip(ref_xy[1], half_h, data.canvas_h - half_h),
                ],
                dtype=np.float32,
            )
        if side == "bottom":
            return np.array(
                [
                    np.clip(ref_xy[0], half_w, data.canvas_w - half_w),
                    half_h + inset,
                ],
                dtype=np.float32,
            )
        return np.array(
            [
                np.clip(ref_xy[0], half_w, data.canvas_w - half_w),
                data.canvas_h - half_h - inset,
            ],
            dtype=np.float32,
        )

    def _macro_density_sample(
        self,
        idx: int,
        xy: np.ndarray,
        density_grid: np.ndarray,
        data: PreparedData,
    ) -> float:
        w, h = data.sizes_hard[idx]
        lx = xy[0] - w * 0.5
        ux = xy[0] + w * 0.5
        ly = xy[1] - h * 0.5
        uy = xy[1] + h * 0.5
        c_lo = max(0, int(lx / data.cell_w))
        c_hi = min(data.grid_cols - 1, int(ux / data.cell_w))
        r_lo = max(0, int(ly / data.cell_h))
        r_hi = min(data.grid_rows - 1, int(uy / data.cell_h))
        total = 0.0
        total_w = 0.0
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
                if ox <= 0:
                    continue
                frac = (ox * oy) / max(data.cell_w * data.cell_h, 1e-6)
                total += float(density_grid[row, col]) * frac
                total_w += frac
        return total / max(total_w, 1e-6)

    def _macro_hpwl_delta(
        self,
        idx: int,
        cand: np.ndarray,
        all_pos: np.ndarray,
        data: PreparedData,
    ) -> float:
        nets = data.hard_net_lists[idx]
        if nets.size == 0:
            return 0.0

        delta = 0.0
        for net_id in nets:
            nodes = data.safe_nnp_np[net_id][data.nnmask_np[net_id]]
            xs = all_pos[nodes, 0].copy()
            ys = all_pos[nodes, 1].copy()
            old_span = float(xs.max() - xs.min() + ys.max() - ys.min())
            loc = np.where(nodes == idx)[0]
            if loc.size == 0:
                continue
            xs[loc] = cand[0]
            ys[loc] = cand[1]
            new_span = float(xs.max() - xs.min() + ys.max() - ys.min())
            delta += new_span - old_span

        return delta / max(data.hpwl_norm, 1e-6)

    def _macro_reposition_proposals(
        self,
        idx: int,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
        config,
    ) -> List[Tuple[np.ndarray, float]]:
        density_grid = self._density_grid_np(np.vstack([hard, soft]), data)
        all_pos = self._all_pos_np(hard, soft, data)
        net_centers = self._net_box_centers_np(all_pos, data)
        current = hard[idx].copy()
        net_target = self._macro_net_target(idx, net_centers, data)
        before_density = self._macro_density_sample(idx, current, density_grid, data)
        before_boundary = self._boundary_clearance(idx, current, data)
        max_span = max(data.canvas_w, data.canvas_h, 1e-6)

        side_gap = {
            "left": current[0] - data.sizes_hard[idx, 0] * 0.5,
            "right": data.canvas_w - (current[0] + data.sizes_hard[idx, 0] * 0.5),
            "bottom": current[1] - data.sizes_hard[idx, 1] * 0.5,
            "top": data.canvas_h - (current[1] + data.sizes_hard[idx, 1] * 0.5),
        }
        nearest_side = min(side_gap, key=side_gap.get)
        center_x_side = "left" if current[0] <= data.canvas_w * 0.5 else "right"
        center_y_side = "bottom" if current[1] <= data.canvas_h * 0.5 else "top"
        target_x_side = "left" if net_target[0] <= current[0] else "right"
        target_y_side = "bottom" if net_target[1] <= current[1] else "top"

        side_order: List[str] = []
        for side in [nearest_side, center_x_side, center_y_side, target_x_side, target_y_side]:
            if side not in side_order:
                side_order.append(side)

        ref_points = [current, net_target, data.port_pull[idx]]
        neigh = data.neighbor_lists[idx]
        if neigh.size > 0:
            ref_points.append(np.median(hard[neigh], axis=0).astype(np.float32))

        seen = set()
        proposals: List[Tuple[np.ndarray, float]] = []
        for side in side_order[: config["side_budget"]]:
            for ref in ref_points:
                for target in [
                    self._boundary_target(idx, side, ref, data, config["edge_inset"]),
                    current + config["inner_alpha"] * (self._boundary_target(idx, side, ref, data, config["edge_inset"]) - current),
                ]:
                    cand = self._nearest_legal_point(idx, target, hard, data)
                    key = (round(float(cand[0]), 4), round(float(cand[1]), 4))
                    if key in seen:
                        continue
                    seen.add(key)
                    if float(np.linalg.norm(cand - current)) <= 1e-4:
                        continue
                    density_delta = self._macro_density_sample(idx, cand, density_grid, data) - before_density
                    boundary_delta = (
                        self._boundary_clearance(idx, cand, data) - before_boundary
                    ) / max_span
                    disp = float(np.linalg.norm(cand - current) / max_span)
                    delta_wl = self._macro_hpwl_delta(idx, cand, all_pos, data)
                    local_score = (
                        delta_wl
                        + config["density_weight"] * density_delta
                        + config["boundary_weight"] * boundary_delta
                        + config["disp_weight"] * disp
                    )
                    cand_hard = hard.copy()
                    cand_hard[idx] = cand
                    proposals.append((cand_hard.astype(np.float32), float(local_score)))

        proposals.sort(key=lambda item: item[1])
        return proposals[: max(4, config["cheap_trials"] + 1)]

    def _connected_swap_refine(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
        force_exact: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        cur_hard = hard.copy().astype(np.float32)
        cur_soft = soft.copy().astype(np.float32)
        cur_cheap = self._cheap_score(cur_hard, cur_soft, data)
        use_exact = force_exact and plc is not None and data.num_hard <= 520
        cur_exact = self._proxy_cost_if_valid(cur_hard, cur_soft, benchmark, plc, data) if use_exact else float("inf")
        poor_scores = self._diagnose_poor_macros(cur_hard, cur_soft, data)

        if data.num_hard <= 220:
            seed_budget = 8
            partner_budget = 6
        elif data.num_hard <= 450:
            seed_budget = 6
            partner_budget = 5
        else:
            seed_budget = 5
            partner_budget = 4

        accepted = 0
        order = np.argsort(-poor_scores)
        for idx in order[:seed_budget]:
            if poor_scores[idx] <= 1e-8 or self._time_left(t0) < 8:
                break
            if not data.movable_hard[idx]:
                continue

            partners = []
            for j in data.neighbor_lists[idx]:
                j = int(j)
                if j == idx or not data.movable_hard[j]:
                    continue
                size_ratio = max(
                    float(data.sizes_hard[idx, 0] / max(data.sizes_hard[j, 0], 1e-6)),
                    float(data.sizes_hard[j, 0] / max(data.sizes_hard[idx, 0], 1e-6)),
                    float(data.sizes_hard[idx, 1] / max(data.sizes_hard[j, 1], 1e-6)),
                    float(data.sizes_hard[j, 1] / max(data.sizes_hard[idx, 1], 1e-6)),
                )
                if size_ratio > 3.0:
                    continue
                area_gap = abs(float(data.hard_areas[idx] - data.hard_areas[j])) / max(
                    float(max(data.hard_areas[idx], data.hard_areas[j])),
                    1e-6,
                )
                partners.append((area_gap, j))

            if not partners:
                continue
            partners.sort(key=lambda item: item[0])

            best_hard = None
            best_soft = None
            best_cheap = cur_cheap
            best_exact = cur_exact
            for _, j in partners[:partner_budget]:
                cand_hard = cur_hard.copy()
                cand_hard[idx] = cur_hard[j]
                cand_hard[j] = cur_hard[idx]
                if self._hard_overlaps_any(idx, cand_hard[idx], cand_hard, data):
                    continue
                if self._hard_overlaps_any(j, cand_hard[j], cand_hard, data):
                    continue
                cand_soft = cur_soft.copy().astype(np.float32)
                cand_cheap = self._cheap_score(cand_hard, cand_soft, data)
                if data.num_soft > 0 and self._time_left(t0) >= 7:
                    relaxed_soft = self._relax_soft(cand_hard, cand_soft, data, sweeps=1, damping=0.92)
                    relaxed_cheap = self._cheap_score(cand_hard, relaxed_soft, data)
                    if relaxed_cheap + 1e-6 < cand_cheap:
                        cand_soft = relaxed_soft.astype(np.float32)
                        cand_cheap = relaxed_cheap
                if use_exact:
                    cand_exact = self._proxy_cost_if_valid(cand_hard, cand_soft, benchmark, plc, data)
                    if not math.isfinite(cand_exact):
                        continue
                    better = cand_exact + 1e-6 < best_exact if math.isfinite(best_exact) else True
                    if better or (
                        math.isfinite(best_exact)
                        and abs(cand_exact - best_exact) <= 1e-6
                        and cand_cheap + 1e-6 < best_cheap
                    ):
                        best_exact = cand_exact
                        best_cheap = cand_cheap
                        best_hard = cand_hard
                        best_soft = cand_soft
                elif cand_cheap + 1e-5 < best_cheap:
                    best_cheap = cand_cheap
                    best_hard = cand_hard
                    best_soft = cand_soft

            if best_hard is not None and best_soft is not None:
                cur_hard = best_hard
                cur_soft = best_soft
                cur_cheap = best_cheap
                if use_exact:
                    cur_exact = best_exact
                accepted += 1

        if accepted > 0 and data.num_soft > 0 and self._time_left(t0) >= 7:
            relaxed_soft = self._relax_soft(cur_hard, cur_soft, data, sweeps=1, damping=0.90)
            relaxed_cheap = self._cheap_score(cur_hard, relaxed_soft, data)
            relaxed_exact = self._proxy_cost_if_valid(cur_hard, relaxed_soft, benchmark, plc, data) if use_exact else float("inf")
            better = False
            if use_exact and math.isfinite(cur_exact):
                better = math.isfinite(relaxed_exact) and relaxed_exact + 1e-6 < cur_exact
            elif relaxed_cheap + 1e-6 < cur_cheap:
                better = True
            if better or (
                use_exact
                and math.isfinite(relaxed_exact)
                and math.isfinite(cur_exact)
                and abs(relaxed_exact - cur_exact) <= 1e-6
                and relaxed_cheap + 1e-6 < cur_cheap
            ):
                cur_soft = relaxed_soft
                cur_cheap = relaxed_cheap
                if use_exact:
                    cur_exact = relaxed_exact

        return cur_hard.astype(np.float32), cur_soft.astype(np.float32), accepted

    def _latent_basis_search(
        self,
        base_hard: np.ndarray,
        base_soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
        extra_hard_seeds: Optional[Sequence[np.ndarray]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        if self._time_left(t0) < 20:
            return None, None, ""

        modes = self._build_latent_modes(base_hard, base_soft, data, extra_hard_seeds=extra_hard_seeds)
        if not modes:
            return None, None, ""

        if data.num_hard <= 220:
            generations = 4
            samples = 8
            exact_elites = 4
            jitter_trials = 2
        elif data.num_hard <= 450:
            generations = 3
            samples = 6
            exact_elites = 3
            jitter_trials = 1
        else:
            generations = 2
            samples = 4
            exact_elites = 2
            jitter_trials = 1

        rng = np.random.default_rng((SEED ^ ((self._stable_seed(benchmark.name) << 1) & 0xFFFFFFFF)) & 0xFFFFFFFF)
        mode_scale = 0.08 * max(data.cell_w, data.cell_h)
        dim = len(modes)
        mu = np.zeros(dim, dtype=np.float32)
        sigma = np.full(dim, mode_scale, dtype=np.float32)

        base_cheap_comps = self._cheap_components(base_hard, base_soft, data)
        base_exact_comps = self._proxy_components_if_valid(base_hard, base_soft, benchmark, plc, data)
        calib = self._init_surrogate_calib(base_cheap_comps, base_exact_comps)
        base_cheap = self._surrogate_score_from_components(base_cheap_comps, calib)
        best_cheap = base_cheap
        best_cheap_hard = base_hard.copy().astype(np.float32)
        best_cheap_soft = base_soft.copy().astype(np.float32)
        best_hard = base_hard.copy().astype(np.float32)
        best_soft = base_soft.copy().astype(np.float32)
        base_exact = base_exact_comps["proxy"] if base_exact_comps is not None else float("inf")
        best_exact = base_exact
        best_alpha = mu.copy()

        for _ in range(generations):
            if self._time_left(t0) < 12:
                break

            pool = []
            for _sample in range(samples):
                alpha = rng.normal(mu, sigma).astype(np.float32)
                cand_hard = self._apply_latent_alpha(base_hard, modes, alpha, data)
                cheap_comps = self._cheap_components(cand_hard, base_soft, data)
                robust = self._robust_latent_score(
                    cand_hard,
                    base_soft,
                    data,
                    rng,
                    calib,
                    jitter_trials=jitter_trials,
                )
                if robust + 1e-6 < best_cheap:
                    best_cheap = robust
                    best_cheap_hard = cand_hard.copy()
                pool.append((robust, alpha, cand_hard, cheap_comps))

            pool.sort(key=lambda item: item[0])
            elite_count = max(1, samples // 2)
            elite = pool[:elite_count]
            elite_alphas = np.stack([item[1] for item in elite], axis=0)
            mu = elite_alphas.mean(axis=0).astype(np.float32)
            sigma = np.maximum(
                elite_alphas.std(axis=0).astype(np.float32),
                0.02 * max(data.cell_w, data.cell_h),
            )

            for robust, alpha, cand_hard, cheap_comps in pool[:exact_elites]:
                exact_comps = self._proxy_components_if_valid(cand_hard, base_soft, benchmark, plc, data)
                calib = self._update_surrogate_calib(calib, cheap_comps, exact_comps)
                exact = exact_comps["proxy"] if exact_comps is not None else float("inf")
                if exact + 1e-6 < best_exact:
                    best_exact = exact
                    best_hard = cand_hard.copy()
                    best_soft = base_soft.copy()
                    best_alpha = alpha.copy()
                    best_cheap = min(best_cheap, robust)
            mu = 0.55 * mu + 0.45 * best_alpha

        sweep_hard, sweep_soft, sweep_alpha, sweep_score = self._latent_mode_sweep(
            base_hard,
            base_soft,
            modes,
            benchmark,
            plc,
            data,
            t0,
            calib,
        )
        if sweep_hard is not None and sweep_score + 1e-6 < best_exact:
            best_hard = sweep_hard
            best_soft = sweep_soft
            best_alpha = sweep_alpha
            best_exact = sweep_score

        exact_hard, exact_soft, exact_alpha, exact_score = self._exact_latent_pattern_search(
            base_hard,
            base_soft,
            modes,
            best_alpha,
            benchmark,
            plc,
            data,
            t0,
        )
        if exact_hard is not None and exact_score + 1e-6 < best_exact:
            best_hard = exact_hard
            best_soft = exact_soft
            best_alpha = exact_alpha
            best_exact = exact_score

        if best_exact + 1e-6 < base_exact:
            return best_hard.astype(np.float32), best_soft.astype(np.float32), "legacy-base-latent-cem"
        if best_cheap + 1e-4 < base_cheap:
            return (
                best_cheap_hard.astype(np.float32),
                best_cheap_soft.astype(np.float32),
                "legacy-base-latent-cem-cheap",
            )
        return None, None, ""

    def _build_latent_modes(
        self,
        base_hard: np.ndarray,
        base_soft: np.ndarray,
        data: PreparedData,
        extra_hard_seeds: Optional[Sequence[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        modes: List[np.ndarray] = []
        movable = data.movable_hard.astype(np.float32)[:, None]
        center = np.array([data.canvas_w * 0.5, data.canvas_h * 0.5], dtype=np.float32)

        for delta in [
            data.init_hard - base_hard,
            data.spectral_xy - base_hard,
            data.port_pull - base_hard,
        ]:
            mode = self._normalize_mode(delta * movable)
            if mode is not None:
                modes.append(mode)

        if extra_hard_seeds is not None:
            for seed in extra_hard_seeds:
                if seed is None or seed.shape != base_hard.shape:
                    continue
                mode = self._normalize_mode((seed - base_hard) * movable)
                if mode is not None:
                    modes.append(mode)

        bisect = np.zeros_like(base_hard, dtype=np.float32)
        idx = np.arange(data.num_hard, dtype=np.int64)
        self._recursive_bisect(idx, data.spectral_xy, data.hard_areas, bisect, data, axis=0)
        mode = self._normalize_mode((bisect - base_hard) * movable)
        if mode is not None:
            modes.append(mode)

        radial = (base_hard - center[None, :]) * movable
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        radial = radial / np.maximum(radial_norm, 1e-6)
        mode = self._normalize_mode(radial)
        if mode is not None:
            modes.append(mode)

        density_grid = self._density_grid_np(np.vstack([base_hard, base_soft]), data)
        density_over = np.maximum(density_grid - BASE_DENSITY_TARGET, 0.0)
        mode = self._hotspot_escape_mode(density_over, base_hard, data)
        if mode is not None:
            modes.append(mode)

        all_pos = self._all_pos_np(base_hard, base_soft, data)
        cong_grid = self._fast_cong_grid_np(all_pos, data)
        if cong_grid is not None:
            cong_over = np.maximum(cong_grid - 1.0, 0.0)
            mode = self._hotspot_escape_mode(cong_over, base_hard, data)
            if mode is not None:
                modes.append(mode)

        if data.num_hard >= 4:
            deg = data.hard_adj.sum(axis=1)
            lap = np.diag(deg + 1e-4) - data.hard_adj
            _, vecs = np.linalg.eigh(lap.astype(np.float64))
            added = 0
            for col in range(1, min(8, vecs.shape[1])):
                eig = vecs[:, col].astype(np.float32)
                if float(np.std(eig)) < 1e-6:
                    continue
                eig = eig / max(float(np.std(eig)), 1e-6)
                for axis in range(2):
                    directional = np.zeros_like(base_hard, dtype=np.float32)
                    directional[:, axis] = eig
                    mode = self._normalize_mode(directional * movable)
                    if mode is not None:
                        modes.append(mode)
                added += 1
                if added >= 2:
                    break

        return modes[:8]

    def _hotspot_escape_mode(
        self,
        grid: np.ndarray,
        hard: np.ndarray,
        data: PreparedData,
    ) -> Optional[np.ndarray]:
        if grid.size == 0 or float(grid.max()) <= 1e-6:
            return None

        grad_r, grad_c = np.gradient(grid.astype(np.float32))
        mode = np.zeros_like(hard, dtype=np.float32)
        for idx in range(data.num_hard):
            if not data.movable_hard[idx]:
                continue
            col = int(np.clip(hard[idx, 0] / max(data.cell_w, 1e-6), 0, data.grid_cols - 1))
            row = int(np.clip(hard[idx, 1] / max(data.cell_h, 1e-6), 0, data.grid_rows - 1))
            mode[idx, 0] = -grad_c[row, col]
            mode[idx, 1] = -grad_r[row, col]
        return self._normalize_mode(mode)

    def _exact_latent_pattern_search(
        self,
        base_hard: np.ndarray,
        base_soft: np.ndarray,
        modes: Sequence[np.ndarray],
        start_alpha: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, float]:
        if plc is None or self._time_left(t0) < 12 or len(modes) == 0:
            return None, None, start_alpha.copy(), float("inf")

        if data.num_hard <= 220:
            max_rounds = 3
            init_step = 0.10 * max(data.cell_w, data.cell_h)
        elif data.num_hard <= 450:
            max_rounds = 2
            init_step = 0.08 * max(data.cell_w, data.cell_h)
        else:
            max_rounds = 1
            init_step = 0.06 * max(data.cell_w, data.cell_h)

        cache: dict[Tuple[int, ...], Tuple[float, np.ndarray]] = {}

        def eval_alpha(alpha: np.ndarray) -> Tuple[float, np.ndarray]:
            key = tuple(int(round(float(a) * 1000.0)) for a in alpha)
            if key in cache:
                return cache[key]
            cand_hard = self._apply_latent_alpha(base_hard, modes, alpha, data)
            score = self._proxy_cost_if_valid(cand_hard, base_soft, benchmark, plc, data)
            cache[key] = (score, cand_hard)
            return score, cand_hard

        best_alpha = start_alpha.copy().astype(np.float32)
        best_score, best_hard = eval_alpha(best_alpha)
        best_soft = base_soft.copy().astype(np.float32)
        step = np.full(len(modes), init_step, dtype=np.float32)

        for _ in range(max_rounds):
            if self._time_left(t0) < 10:
                break
            improved = False
            for dim in range(len(modes)):
                if self._time_left(t0) < 10:
                    break
                candidates: List[Tuple[float, np.ndarray, np.ndarray]] = []
                for sign in (-1.0, 1.0):
                    alpha = best_alpha.copy()
                    alpha[dim] += sign * step[dim]
                    score, hard = eval_alpha(alpha)
                    candidates.append((score, alpha, hard))
                score, alpha, hard = min(candidates, key=lambda item: item[0])
                if score + 1e-6 < best_score:
                    best_score = score
                    best_alpha = alpha
                    best_hard = hard
                    improved = True
            if not improved:
                step *= 0.5
                if float(step.max()) < 0.02 * max(data.cell_w, data.cell_h):
                    break

        if math.isfinite(best_score):
            return best_hard.astype(np.float32), best_soft.astype(np.float32), best_alpha.astype(np.float32), float(best_score)
        return None, None, start_alpha.copy(), float("inf")

    def _latent_mode_sweep(
        self,
        base_hard: np.ndarray,
        base_soft: np.ndarray,
        modes: Sequence[np.ndarray],
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
        calib: Optional[dict],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, float]:
        if plc is None or self._time_left(t0) < 12 or len(modes) == 0:
            return None, None, np.zeros(len(modes), dtype=np.float32), float("inf")

        if data.num_hard <= 220:
            scales = [0.08, 0.16, 0.28]
            exact_budget = 8
        elif data.num_hard <= 450:
            scales = [0.08, 0.18, 0.32]
            exact_budget = 6
        else:
            scales = [0.10, 0.22, 0.38]
            exact_budget = 4

        unit = max(data.cell_w, data.cell_h)
        proposals: List[Tuple[float, np.ndarray, np.ndarray]] = []
        zero = np.zeros(len(modes), dtype=np.float32)
        for dim in range(len(modes)):
            if self._time_left(t0) < 10:
                break
            for scale in scales:
                step = scale * unit
                for sign in (-1.0, 1.0):
                    alpha = zero.copy()
                    alpha[dim] = sign * step
                    cand_hard = self._apply_latent_alpha(base_hard, modes, alpha, data)
                    cheap = self._robust_latent_score(
                        cand_hard,
                        base_soft,
                        data,
                        np.random.default_rng(self._stable_seed(f"{benchmark.name}-{dim}-{scale}-{sign}")),
                        calib,
                        jitter_trials=1 if data.num_hard > 300 else 2,
                    )
                    proposals.append((cheap, alpha, cand_hard))

        proposals.sort(key=lambda item: item[0])
        best_score = float("inf")
        best_hard = None
        best_soft = None
        best_alpha = zero.copy()
        for _, alpha, cand_hard in proposals[:exact_budget]:
            if self._time_left(t0) < 10:
                break
            score = self._proxy_cost_if_valid(cand_hard, base_soft, benchmark, plc, data)
            if score + 1e-6 < best_score:
                best_score = score
                best_hard = cand_hard.astype(np.float32)
                best_soft = base_soft.copy().astype(np.float32)
                best_alpha = alpha.astype(np.float32)

        if best_hard is not None:
            return best_hard, best_soft, best_alpha, float(best_score)
        return None, None, zero.copy(), float("inf")

    def _cluster_shift_refine(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        if plc is None or data.num_hard < 260 or self._time_left(t0) < 12:
            return hard.copy(), soft.copy(), 0

        if data.num_hard >= 700:
            n_clusters = 5
            rounds = 2
        elif data.num_hard >= 450:
            n_clusters = 4
            rounds = 2
        else:
            n_clusters = 3
            rounds = 1

        labels = self._kmeans_labels(data.spectral_xy, n_clusters)
        cur_hard = hard.copy().astype(np.float32)
        cur_soft = soft.copy().astype(np.float32)
        base = self._proxy_components_if_valid(cur_hard, cur_soft, benchmark, plc, data)
        if base is None:
            return hard.copy(), soft.copy(), 0
        best_score = base["proxy"]
        accepted = 0

        density_grid = self._density_grid_np(np.vstack([cur_hard, cur_soft]), data)
        density_over = np.maximum(density_grid - BASE_DENSITY_TARGET, 0.0)
        hotspot_mode = self._hotspot_escape_mode(density_over, cur_hard, data)
        cluster_dirs: List[np.ndarray] = []
        if hotspot_mode is not None:
            cluster_dirs.append(hotspot_mode)
        cong_grid = self._fast_cong_grid_np(self._all_pos_np(cur_hard, cur_soft, data), data)
        if cong_grid is not None:
            cong_mode = self._hotspot_escape_mode(np.maximum(cong_grid - 1.0, 0.0), cur_hard, data)
            if cong_mode is not None:
                cluster_dirs.append(cong_mode)

        center = np.array([data.canvas_w * 0.5, data.canvas_h * 0.5], dtype=np.float32)
        step_scales = [1.5 * max(data.cell_w, data.cell_h), 0.75 * max(data.cell_w, data.cell_h)]
        for _ in range(rounds):
            if self._time_left(t0) < 10:
                break
            improved = False
            for cluster_id in range(n_clusters):
                members = np.flatnonzero(labels == cluster_id)
                if members.size == 0:
                    continue
                centroid = cur_hard[members].mean(axis=0)
                away = centroid - center
                norm = float(np.linalg.norm(away))
                if norm < 1e-6:
                    away = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    away = away / norm

                dir_list: List[np.ndarray] = [
                    away.astype(np.float32),
                    np.array([1.0, 0.0], dtype=np.float32),
                    np.array([-1.0, 0.0], dtype=np.float32),
                    np.array([0.0, 1.0], dtype=np.float32),
                    np.array([0.0, -1.0], dtype=np.float32),
                ]
                for mode in cluster_dirs:
                    v = mode[members].mean(axis=0)
                    vn = float(np.linalg.norm(v))
                    if vn > 1e-6:
                        dir_list.append((v / vn).astype(np.float32))

                local_best = None
                local_score = best_score
                for direction in dir_list:
                    for step in step_scales:
                        if self._time_left(t0) < 10:
                            break
                        cand_hard = cur_hard.copy()
                        cand_hard[members] += direction[None, :] * step
                        self._clamp_hard(cand_hard, data)
                        cand_hard = self._legacy_resolve_hard(cand_hard, data, max_rounds=6)
                        cand_hard = self._tiny_fix_hard(cand_hard, data)
                        cand = self._proxy_components_if_valid(cand_hard, cur_soft, benchmark, plc, data)
                        if cand is None:
                            continue
                        if cand["proxy"] + 1e-6 < local_score:
                            local_score = cand["proxy"]
                            local_best = cand_hard
                if local_best is not None:
                    cur_hard = local_best
                    best_score = local_score
                    accepted += 1
                    improved = True
            if not improved:
                break

        return cur_hard.astype(np.float32), cur_soft.astype(np.float32), accepted

    def _kmeans_labels(self, points: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
        n = points.shape[0]
        if n == 0 or k <= 1:
            return np.zeros(n, dtype=np.int64)
        k = min(k, n)
        order = np.argsort(points[:, 0] + 0.31 * points[:, 1], kind="mergesort")
        seeds = np.linspace(0, n - 1, k, dtype=np.int64)
        centers = points[order[seeds]].astype(np.float32).copy()
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(iters):
            dist2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(dist2, axis=1).astype(np.int64)
            new_centers = centers.copy()
            for idx in range(k):
                mask = labels == idx
                if np.any(mask):
                    new_centers[idx] = points[mask].mean(axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        return labels

    def _normalize_mode(self, mode: np.ndarray) -> Optional[np.ndarray]:
        rms = math.sqrt(float(np.square(mode).mean()))
        if rms < 1e-6:
            return None
        return (mode / rms).astype(np.float32)

    def _apply_latent_alpha(
        self,
        base_hard: np.ndarray,
        modes: Sequence[np.ndarray],
        alpha: np.ndarray,
        data: PreparedData,
    ) -> np.ndarray:
        cand = base_hard.copy().astype(np.float32)
        for a, mode in zip(alpha, modes):
            cand += float(a) * mode
        self._clamp_hard(cand, data)
        if data.num_hard > 520:
            cand = self._hash_resolve_hard(cand, data, sweeps=5)
        else:
            cand = self._legacy_resolve_hard(cand, data, max_rounds=8)
            cand = self._tiny_fix_hard(cand, data)
        return cand.astype(np.float32)

    def _robust_latent_score(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
        rng: np.random.Generator,
        calib: Optional[dict],
        jitter_trials: int,
    ) -> float:
        score = self._surrogate_score_from_components(self._cheap_components(hard, soft, data), calib)
        if jitter_trials <= 0:
            return score

        total = score
        movable = data.movable_hard
        for _ in range(jitter_trials):
            noisy = hard.copy().astype(np.float32)
            noise = rng.uniform(-0.5, 0.5, size=noisy.shape).astype(np.float32)
            noise[:, 0] *= data.cell_w
            noise[:, 1] *= data.cell_h
            noise[~movable] = 0.0
            noisy += noise
            self._clamp_hard(noisy, data)
            if data.num_hard > 520:
                noisy = self._hash_resolve_hard(noisy, data, sweeps=2)
            else:
                noisy = self._legacy_resolve_hard(noisy, data, max_rounds=4)
                noisy = self._tiny_fix_hard(noisy, data)
            total += self._surrogate_score_from_components(self._cheap_components(noisy, soft, data), calib)
        return total / float(jitter_trials + 1)

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

    def _build_fast_cong_engine(self, benchmark: Benchmark, plc) -> Optional[dict]:
        if plc is None or benchmark.num_nets == 0:
            return None
        try:
            hpm = float(plc.hroutes_per_micron)
            vpm = float(plc.vroutes_per_micron)
            halloc = float(plc.hrouting_alloc)
            valloc = float(plc.vrouting_alloc)
            smooth = int(plc.smooth_range)
        except Exception:
            return None

        src_list: List[int] = []
        snk_list: List[int] = []
        w_list: List[float] = []
        net_weights = benchmark.net_weights.cpu().numpy()
        for ni, nodes_t in enumerate(benchmark.net_nodes):
            nodes = nodes_t.cpu().numpy().astype(np.int64)
            if len(nodes) < 2:
                continue
            src = int(nodes[0])
            weight = float(net_weights[ni]) if ni < len(net_weights) else 1.0
            for dst in nodes[1:]:
                src_list.append(src)
                snk_list.append(int(dst))
                w_list.append(weight)

        if not src_list:
            return None

        return {
            "hpm": hpm,
            "vpm": vpm,
            "halloc": halloc,
            "valloc": valloc,
            "smooth": smooth,
            "src": np.asarray(src_list, dtype=np.int32),
            "snk": np.asarray(snk_list, dtype=np.int32),
            "w": np.asarray(w_list, dtype=np.float32),
        }

    def _fast_cong_grid_np(self, all_pos: np.ndarray, data: PreparedData) -> Optional[np.ndarray]:
        eng = data.fast_cong_engine
        if eng is None:
            return None

        gw = data.canvas_w / data.grid_cols
        gh = data.canvas_h / data.grid_rows
        h_cap = gh * float(eng["hpm"])
        v_cap = gw * float(eng["vpm"])
        if h_cap < 1e-9 or v_cap < 1e-9:
            return None

        px = np.clip((all_pos[:, 0] / gw).astype(np.int32), 0, data.grid_cols - 1)
        py = np.clip((all_pos[:, 1] / gh).astype(np.int32), 0, data.grid_rows - 1)

        src = eng["src"]
        snk = eng["snk"]
        w = eng["w"]
        sr = py[src]
        sc = px[src]
        kr = py[snk]
        kc = px[snk]
        col_lo = np.minimum(sc, kc)
        col_hi = np.maximum(sc, kc)
        row_lo = np.minimum(sr, kr)
        row_hi = np.maximum(sr, kr)

        h_diff = np.zeros((data.grid_rows, data.grid_cols + 1), dtype=np.float32)
        np.add.at(h_diff, (sr, col_lo), w)
        np.add.at(h_diff, (sr, col_hi), -w)
        h_grid = h_diff[:, :-1].cumsum(axis=1)

        v_diff = np.zeros((data.grid_rows + 1, data.grid_cols), dtype=np.float32)
        np.add.at(v_diff, (row_lo, kc), w)
        np.add.at(v_diff, (row_hi, kc), -w)
        v_grid = v_diff[:-1, :].cumsum(axis=0)

        halloc = float(eng["halloc"])
        valloc = float(eng["valloc"])
        for i in range(data.num_hard):
            xi, yi = all_pos[i]
            wi, hi = data.sizes_hard[i]
            lx = xi - wi * 0.5
            ux = xi + wi * 0.5
            ly = yi - hi * 0.5
            uy = yi + hi * 0.5
            c0 = max(0, int(lx / gw))
            c1 = min(data.grid_cols - 1, int(ux / gw))
            r0 = max(0, int(ly / gh))
            r1 = min(data.grid_rows - 1, int(uy / gh))
            for row in range(r0, r1 + 1):
                y0 = row * gh
                y1 = y0 + gh
                oy = max(0.0, min(uy, y1) - max(ly, y0))
                if oy <= 0:
                    continue
                for col in range(c0, c1 + 1):
                    x0 = col * gw
                    x1 = x0 + gw
                    ox = max(0.0, min(ux, x1) - max(lx, x0))
                    if ox <= 0:
                        continue
                    v_grid[row, col] += ox * valloc
                    h_grid[row, col] += oy * halloc

        h_grid /= h_cap
        v_grid /= v_cap
        return (h_grid + v_grid).astype(np.float32)

    def _fast_cong_np(self, all_pos: np.ndarray, data: PreparedData) -> float:
        combined_grid = self._fast_cong_grid_np(all_pos, data)
        if combined_grid is None:
            return 0.0
        combined = combined_grid.reshape(-1)
        top_k = max(1, int(combined.size * 0.05))
        return float(np.partition(combined, -top_k)[-top_k:].mean())

    def _cheap_components(self, hard: np.ndarray, soft: np.ndarray, data: PreparedData) -> dict:
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
        cong = 0.0
        if data.fast_cong_engine is not None and len(data.fast_cong_engine["src"]) <= 250_000:
            cong = self._fast_cong_np(all_pos, data)
        return {
            "wl": float(wl),
            "den": float(topk),
            "cong": float(cong),
            "overlap": float(overlap / max(data.canvas_area, 1e-6)),
        }

    def _surrogate_score_from_components(self, comps: dict, calib: Optional[dict] = None) -> float:
        if calib is None:
            wl_scale = 1.0
            den_scale = 0.5
            cong_scale = 0.35
        else:
            wl_scale = float(calib["wl"])
            den_scale = float(calib["den"])
            cong_scale = float(calib["cong"])
        return float(
            wl_scale * comps["wl"]
            + den_scale * comps["den"]
            + cong_scale * comps["cong"]
            + 250.0 * comps["overlap"]
        )

    def _init_surrogate_calib(self, cheap: dict, exact: Optional[dict]) -> dict:
        calib = {"wl": 1.0, "den": 0.5, "cong": 0.35}
        if exact is None:
            return calib
        eps = 1e-5
        calib["wl"] = float(np.clip(exact["wl"] / max(cheap["wl"], eps), 0.2, 6.0))
        calib["den"] = float(np.clip(0.5 * exact["den"] / max(cheap["den"], eps), 0.05, 8.0))
        if cheap["cong"] > eps:
            calib["cong"] = float(np.clip(0.5 * exact["cong"] / max(cheap["cong"], eps), 0.05, 8.0))
        return calib

    def _update_surrogate_calib(self, calib: dict, cheap: dict, exact: Optional[dict], eta: float = 0.30) -> dict:
        if exact is None:
            return calib

        eps = 1e-5
        targets = {
            "wl": exact["wl"] / max(cheap["wl"], eps),
            "den": 0.5 * exact["den"] / max(cheap["den"], eps),
            "cong": 0.5 * exact["cong"] / max(cheap["cong"], eps) if cheap["cong"] > eps else calib["cong"],
        }
        bounds = {"wl": (0.15, 8.0), "den": (0.03, 12.0), "cong": (0.03, 12.0)}
        updated = dict(calib)
        for key in ["wl", "den", "cong"]:
            ratio = float(np.clip(targets[key] / max(calib[key], eps), 0.25, 4.0))
            updated[key] = float(np.clip(calib[key] * math.exp(eta * math.log(ratio)), *bounds[key]))
        return updated

    def _cheap_score(self, hard: np.ndarray, soft: np.ndarray, data: PreparedData) -> float:
        return self._surrogate_score_from_components(self._cheap_components(hard, soft, data))

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
        return max(0.0, TIME_BUDGET - (time.time() - t0))

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
