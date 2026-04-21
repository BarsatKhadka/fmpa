"""
Macro placer: Oracle SA + net-centroid-guided moves.

Key ideas:
1. Cheap SA warm-up: fast WL+density optimization to escape the worst positions
2. Oracle SA: use exact PlacementCost as the SA cost function with temperature
   - Temperature enables escaping initial.plc's local minimum basin
   - Net-centroid-guided moves: force-directed steps as SA proposals (smarter than random)
   - With ~200 evals in 300s, SA can explore meaningfully with exact congestion signal
3. initial.plc always in candidate set — guaranteed fallback

Override: PLACE_TIME_BUDGET=<s> uv run evaluate soln.py
"""

import math
import os
import random
import time

import numpy as np
import torch

from macro_place.benchmark import Benchmark

OVERLAP_WEIGHT  = 200.0
N_ISLANDS       = 2           # fewer islands; most budget goes to oracle SA
REPAIR_STEPS    = 200
TIME_BUDGET     = int(os.environ.get("PLACE_TIME_BUDGET", 3300))

# Cheap SA params (warm-up phase)
T_SA_START  = 0.04
T_SA_END    = 0.001
MAX_STEP    = 0.4
SWAP_PROB   = 0.15

# Oracle SA params
T_ORC_START = 0.08   # proxy_cost units; at this T, exp(-0.02/0.08)=78% accept for +0.02 worse
T_ORC_END   = 0.002  # nearly greedy at end

# Fast cong SA params (Phase 1b) — higher T to accept congestion-improving moves
T_CONG_START = 0.14  # 3.5× higher than Phase 1a to allow cong-for-WL tradeoffs
T_CONG_END   = 0.003


class Placer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()

        num_hard  = benchmark.num_hard_macros
        canvas_w  = float(benchmark.canvas_width)
        canvas_h  = float(benchmark.canvas_height)
        grid_rows = benchmark.grid_rows
        grid_cols = benchmark.grid_cols
        sizes_np  = benchmark.macro_sizes.numpy().astype(np.float64)
        port_pos  = benchmark.port_positions.numpy().astype(np.float64)
        fixed_np  = benchmark.macro_fixed.numpy()

        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_idx  = torch.where(movable_mask)[0].tolist()
        soft_movable_idx = torch.where(
            benchmark.get_movable_mask() & benchmark.get_soft_macro_mask()
        )[0].tolist()  # movable soft macros for density-aware placement

        nets_np = [n.numpy() for n in benchmark.net_nodes]
        macro_to_nets: dict[int, list[int]] = {}
        for ni, nodes in enumerate(nets_np):
            for node in nodes:
                if int(node) < num_hard:
                    macro_to_nets.setdefault(int(node), []).append(ni)

        hpwl_norm = max(1.0, len(nets_np) * (canvas_w + canvas_h))

        n_nets  = len(nets_np)
        max_nsz = max(len(n) for n in nets_np)
        nnp     = np.full((n_nets, max_nsz), -1, dtype=np.int64)
        for i, n in enumerate(nets_np):
            nnp[i, :len(n)] = n
        nnmask   = (nnp >= 0)
        safe_nnp = np.maximum(nnp, 0)

        # ── Gradient-friendly net matrix: filter huge fan-out nets ───────────
        # High-fanout nets (clocks, resets) can have 1000s of pins, causing
        # n_nets×max_nsz to exceed memory limits and skip gradient entirely.
        # Filtering nets with >64 pins barely affects WL gradient quality
        # (per-pin force is ~1/fanout → negligible for large nets).
        MAX_GRAD_FANOUT = 64
        if max_nsz > MAX_GRAD_FANOUT:
            grad_nets_np  = [n for n in nets_np if len(n) <= MAX_GRAD_FANOUT]
            grad_n_nets   = len(grad_nets_np)
            grad_max_nsz  = max((len(n) for n in grad_nets_np), default=1)
            grad_nnp      = np.full((grad_n_nets, grad_max_nsz), -1, dtype=np.int64)
            for i, n in enumerate(grad_nets_np):
                grad_nnp[i, :len(n)] = n
            grad_nnmask  = (grad_nnp >= 0)
            grad_safe_nnp = np.maximum(grad_nnp, 0)
            grad_hpwl_norm = max(1.0, grad_n_nets * (canvas_w + canvas_h))
        else:
            grad_safe_nnp  = safe_nnp
            grad_nnmask    = nnmask
            grad_hpwl_norm = hpwl_norm

        pos_init = benchmark.macro_positions.numpy().astype(np.float64).copy()
        pos_init = self._resolve(pos_init, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)

        plc = self._try_load_plc(benchmark)

        net_weights_np = benchmark.net_weights.numpy() if hasattr(benchmark, 'net_weights') else np.ones(len(nets_np))
        fast_eng = self._build_fast_engine(nets_np, net_weights_np, plc) if plc is not None else None

        def cheap_cost(p):
            ap   = self._all_pos(p, port_pos)
            wl   = self._hpwl_vec(ap, safe_nnp, nnmask) / hpwl_norm
            dens = self._density_grid(p, num_hard, sizes_np, grid_rows, grid_cols, canvas_w, canvas_h)
            den  = self._top_k_mean(dens, 0.10)
            ov   = self._total_overlap(p, num_hard, sizes_np)
            if fast_eng is not None:
                cong = self._fast_cong(ap, num_hard, sizes_np, fast_eng,
                                       grid_rows, grid_cols, canvas_w, canvas_h)
            else:
                cong = 0.0
            return wl + 0.5 * den + 0.5 * cong + OVERLAP_WEIGHT * ov

        # ── Phase 0: CEM + Evolutionary gradient placement (30% of budget) ──────
        # CEM: sample K candidates in spectral θ-space (low-dim structured transforms),
        # evaluate with cheap WL+density cost, keep top-N_POP as gradient seeds.
        # Structured brute force in compressed space: batch explore → gradient-refine elites.
        grad_end_total = t0 + (TIME_BUDGET - 30) * 0.30
        _gpu = torch.cuda.is_available()
        N_POP   = 4 if _gpu else 2   # GPU: 4 diverse starts; CPU: 2
        N_GENS  = 3    # number of evolutionary generations

        def cem_cost(p):
            """Fast WL+density for CEM candidate screening (no congestion —
            congestion is evaluated via cheap_cost after gradient refinement)."""
            ap   = self._all_pos(p, port_pos)
            wl   = self._hpwl_vec(ap, safe_nnp, nnmask) / hpwl_norm
            dens = self._density_grid(p, num_hard, sizes_np, grid_rows, grid_cols, canvas_w, canvas_h)
            den  = self._top_k_mean(dens, 0.10)
            ov   = self._total_overlap(p, num_hard, sizes_np)
            return wl + 0.5 * den + OVERLAP_WEIGHT * ov

        # Budget: at most 15% of gradient window, or 90s max (fast screening)
        cem_budget = min(90.0, max(8.0, (grad_end_total - time.time()) * 0.15))
        population = self._spectral_cem(
            pos_init, movable_idx, sizes_np, nets_np, net_weights_np,
            num_hard, canvas_w, canvas_h, fixed_np,
            n_pop=N_POP, cem_budget=cem_budget, cheap_cost_fn=cem_cost,
        )
        # Pad with perturbed copies if CEM returned fewer than N_POP
        while len(population) < N_POP:
            pb = self._perturb(pos_init, movable_idx, sizes_np, canvas_w, canvas_h, 0.12)
            pb = self._resolve(pb, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
            population.append(pb)

        best_cheap_cost = float('inf')
        best_cheap_pos  = pos_init.copy()

        time_per_member = max(5.0, (grad_end_total - time.time()) / (N_POP * N_GENS))

        for gen in range(N_GENS):
            if time.time() >= grad_end_total - 5:
                break
            # ── Gradient step: evolve each member ──
            evolved = []
            for p_mem in population:
                slot_end = min(time.time() + time_per_member, grad_end_total)
                if time.time() >= slot_end - 2:
                    evolved.append(p_mem)
                    continue
                pg = self._gradient_phase(
                    p_mem, movable_idx, sizes_np, port_pos, canvas_w, canvas_h,
                    num_hard, grad_safe_nnp, grad_nnmask, grid_rows, grid_cols, grad_hpwl_norm, slot_end,
                    fast_eng=fast_eng,
                )
                pg = self._resolve(pg, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                evolved.append(pg)

            # ── Selection: rank by cheap_cost ──
            evo_costs = [cheap_cost(p) for p in evolved]
            for c, p in zip(evo_costs, evolved):
                if c < best_cheap_cost:
                    best_cheap_cost = c
                    best_cheap_pos  = p.copy()

            # ── Crossover: spatial block crossover between top parents ──
            rank_order = sorted(range(len(evolved)), key=lambda i: evo_costs[i])
            if gen < N_GENS - 1 and len(rank_order) >= 2 and time.time() < grad_end_total - 10:
                pa = evolved[rank_order[0]]
                pb = evolved[rank_order[1]]
                # Maintain N_POP: top-2 parents + (N_POP-2) crossover children
                new_pop = [pa, pb]
                for _ in range(N_POP - 2):
                    child = self._spatial_crossover(pa, pb, movable_idx, canvas_w, canvas_h)
                    child = self._resolve(child, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                    new_pop.append(child)
                population = new_pop
            else:
                population = evolved

        pos_grad = best_cheap_pos.copy()

        # ── Phase 0b: Force-directed soft macro placement (CG solve) ─────────
        # With hard macros fixed at gradient result, optimally place soft macros
        # to minimize quadratic WL (Step C of the A-F pipeline).
        # Runs in seconds; significantly improves WL contributed by soft macros.
        if time.time() < grad_end_total + 30 and soft_movable_idx:
            pos_grad_soft = self._force_directed_soft(
                pos_grad, num_hard, nets_np, net_weights_np,
                port_pos, canvas_w, canvas_h, sizes_np, fixed_np)
            c_soft = cheap_cost(pos_grad_soft)
            if c_soft < best_cheap_cost:
                best_cheap_cost = c_soft
                best_cheap_pos  = pos_grad_soft.copy()
                pos_grad = pos_grad_soft.copy()

        # ── Phase 1: Cheap SA warm-up (30%–37% of budget) ────────────────────
        # Fast WL+density SA from gradient result; diversify with perturbed start.
        sa_end = t0 + (TIME_BUDGET - 30) * 0.37

        islands = []
        for i in range(N_ISLANDS):
            p = self._perturb(pos_grad if i == 0 else pos_init, movable_idx, sizes_np, canvas_w, canvas_h,
                              0.0 if i == 0 else 0.15)
            p = self._resolve(p, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
            islands.append(p)

        costs = [cheap_cost(p) for p in islands]
        if float(min(costs)) < best_cheap_cost:
            best_cheap_pos  = islands[int(np.argmin(costs))].copy()
            best_cheap_cost = float(min(costs))

        while time.time() < sa_end:
            elapsed_frac = min(1.0, (time.time() - t0) / max(1.0, (TIME_BUDGET - 30) * 0.22))
            T = T_SA_START * (T_SA_END / T_SA_START) ** elapsed_frac

            for i in range(N_ISLANDS):
                if time.time() >= sa_end: break
                islands[i] = self._run_sa(
                    islands[i], nets_np, macro_to_nets, movable_idx,
                    sizes_np, port_pos, canvas_w, canvas_h,
                    grid_rows, grid_cols, num_hard, T, 500, hpwl_norm,
                    safe_nnp, nnmask,
                )
                islands[i] = self._resolve(islands[i], num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                c = cheap_cost(islands[i])
                costs[i] = c
                if c < best_cheap_cost:
                    best_cheap_cost = c; best_cheap_pos = islands[i].copy()

        # ── Phase 1b: Fast congestion-aware SA (37%→45% of budget) ───────────
        # T_CONG_START=0.14 lets it trade WL/density for congestion improvement.
        # Phase 1b cong SA: no n_pairs limit for numpy-based cheap_cost (no memory issue).
        # But cap at 1M pairs where fast_cong takes >100ms/call (too slow for SA).
        cong_sa_end = t0 + (TIME_BUDGET - 30) * 0.50
        n_pairs_eng = len(fast_eng['src']) if fast_eng is not None else 0
        if fast_eng is not None and n_pairs_eng < 1_000_000 and time.time() < cong_sa_end:
            p1b = self._run_fast_sa_cong(
                best_cheap_pos.copy(), movable_idx, sizes_np, port_pos,
                canvas_w, canvas_h, num_hard, fixed_np, cheap_cost, cong_sa_end,
                fast_eng=fast_eng, grid_rows=grid_rows, grid_cols=grid_cols,
            )
            c1b = cheap_cost(p1b)
            if c1b < best_cheap_cost:
                best_cheap_cost = c1b
                best_cheap_pos  = p1b.copy()

        # ── Phase 2: Oracle SA + surrogate calibration (30%–90% of budget) ───────
        # Outer loop: each iteration = oracle probe → calibrate surrogate → gradient → SA.
        # Inner oracle SA: temperature-driven exploration with exact proxy.
        hard_deadline  = t0 + TIME_BUDGET - 10
        oracle_end     = min(hard_deadline - 5, t0 + (TIME_BUDGET - 30) * 0.90)
        best_pos       = best_cheap_pos.copy()
        oracle_is_fast = True   # updated after first oracle call

        if plc is not None:
            # Evaluate pos_init as the reference baseline.
            init_r    = self._resolve_fully(pos_init, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
            t_oc0 = time.time()
            init_cost, init_comps = self._true_cost(init_r, benchmark, plc, return_components=True)
            oracle_call_secs = time.time() - t_oc0
            # If oracle calls are very slow (ibm17: 552s each), skip oracle SA entirely.
            # Use gradient result directly — two oracle calls would burn >30% of budget.
            oracle_is_fast = (oracle_call_secs < 60.0)

            # Best of gradient result vs pos_init as primary start.
            starts = [init_r]
            starts_cost = [init_cost]
            if oracle_is_fast:
                start_r_cheap = self._resolve_fully(best_cheap_pos, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                sc_cheap, cheap_comps = self._true_cost(start_r_cheap, benchmark, plc, return_components=True)
            else:
                start_r_cheap = best_cheap_pos.copy()
                sc_cheap, cheap_comps = float('inf'), None
            if sc_cheap < init_cost:
                starts[0] = start_r_cheap
                starts_cost[0] = sc_cheap
                ref_comps = cheap_comps
            else:
                ref_comps = init_comps

            # Surrogate calibration: compare oracle components with surrogate estimates.
            # If oracle density > surrogate, gradient phase under-penalized density → boost λ_den.
            # If oracle congestion > 1.0 (over capacity), boost λ_cong. (Arora-Hazan-Kale 2012)
            ref_pos = starts[0]
            surr_den  = self._top_k_mean(
                self._density_grid(ref_pos, num_hard, sizes_np, grid_rows, grid_cols, canvas_w, canvas_h), 0.10)
            surr_cong = self._fast_cong(self._all_pos(ref_pos, port_pos), num_hard, sizes_np, fast_eng,
                                        grid_rows, grid_cols, canvas_w, canvas_h) if fast_eng is not None else 1.0
            true_den  = (ref_comps or {}).get('density_cost',    surr_den) if ref_comps else surr_den
            true_cong = (ref_comps or {}).get('congestion_cost',  surr_cong) if ref_comps else surr_cong

            eta = 0.25
            calib_lam_den   = float(np.clip(0.40 * math.exp(eta * (true_den  - surr_den)),  0.05, 1.50))
            calib_lam_cong  = float(np.clip(0.25 * math.exp(eta * (true_cong - surr_cong)), 0.01, 0.80))
            calib_gamma_scale = 1.0  # keep gamma default for second pass

            # Second gradient pass with calibrated parameters (warm-start from best_cheap_pos).
            # Runs for BOTH fast and slow oracle — slow oracle benefits most since no SA follows.
            # For slow oracle: use more of the budget (up to 70% vs 55% for fast oracle).
            if oracle_is_fast:
                grad2_budget = t0 + (TIME_BUDGET - 30) * 0.55
            else:
                grad2_budget = t0 + (TIME_BUDGET - 30) * 0.80  # more time for slow-oracle
            if time.time() < grad2_budget - 20:
                pos_grad2 = self._gradient_phase(
                    best_cheap_pos, movable_idx, sizes_np, port_pos, canvas_w, canvas_h,
                    num_hard, grad_safe_nnp, grad_nnmask, grid_rows, grid_cols, grad_hpwl_norm, grad2_budget,
                    fast_eng=fast_eng,
                    lam_den_override=calib_lam_den,
                    lam_cong_override=calib_lam_cong,
                    gamma_scale=calib_gamma_scale,
                )
                pos_grad2 = self._resolve(pos_grad2, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                c_grad2 = cheap_cost(pos_grad2)
                if c_grad2 < best_cheap_cost:
                    best_cheap_pos  = pos_grad2.copy()
                    best_cheap_cost = c_grad2
                if oracle_is_fast:
                    start_r_cheap2 = self._resolve_fully(best_cheap_pos, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                    sc_cheap2, _ = self._true_cost(start_r_cheap2, benchmark, plc, return_components=True)
                    if sc_cheap2 < starts_cost[0]:
                        starts[0] = start_r_cheap2
                        starts_cost[0] = sc_cheap2
                    # Always add calibrated gradient as extra oracle SA start — even if
                    # surrogate cost is worse, it may be in a different gradient basin
                    # that oracle SA can escape into.
                    if sc_cheap2 < float('inf'):
                        starts.append(start_r_cheap2)
                        starts_cost.append(sc_cheap2)
                else:
                    # Slow oracle: update best_pos from calibrated gradient result
                    best_pos = self._resolve_fully(best_cheap_pos, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)

            # Build diverse start pool for oracle SA.
            # Pool strategy: warm start + gradient result + perturbed + spectral + random.
            # More diversity = higher chance of escaping the dominant basin.
            if oracle_is_fast:
                for noise in [0.20, 0.40, 0.60, 0.80]:
                    p_pert = self._perturb(pos_init, movable_idx, sizes_np, canvas_w, canvas_h, noise)
                    p_pert = self._resolve(p_pert, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                    starts.append(p_pert)
                    starts_cost.append(float('inf'))
                # Add spectral-start candidates as diverse seeds
                for _ in range(3):
                    p_spec = self._spectral_start(pos_init, movable_idx, sizes_np, nets_np, net_weights_np,
                                                  num_hard, canvas_w, canvas_h, fixed_np, time.time() + 5.0)
                    if p_spec is not None:
                        p_spec = self._resolve(p_spec, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                        starts.append(p_spec)
                        starts_cost.append(float('inf'))
                # Add quadrant-shuffle starts
                for _ in range(2):
                    p_quad = self._random_start(pos_init, movable_idx, sizes_np, canvas_w, canvas_h, fixed_np)
                    p_quad = self._resolve(p_quad, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                    starts.append(p_quad)
                    starts_cost.append(float('inf'))
                # Fresh uniform-random gradient starts: macros placed randomly on canvas,
                # then gradient-optimized. These explore basins entirely outside initial.plc
                # and are the key to escaping topological local minima (DREAMPlace insight:
                # global gradient placement from scratch finds fundamentally different solutions).
                _fresh_grad_budget = min(90.0, max(20.0, (oracle_end - time.time()) * 0.06))
                for _ in range(2):
                    if time.time() >= oracle_end - _fresh_grad_budget - 10:
                        break
                    _p_unif = pos_init.copy()
                    for _idx in movable_idx:
                        _w, _h = sizes_np[_idx, 0], sizes_np[_idx, 1]
                        _p_unif[_idx, 0] = float(np.random.uniform(_w / 2, canvas_w - _w / 2))
                        _p_unif[_idx, 1] = float(np.random.uniform(_h / 2, canvas_h - _h / 2))
                    _p_unif = self._resolve(_p_unif, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                    _grad_end = min(time.time() + _fresh_grad_budget, oracle_end - 20.0)
                    if time.time() < _grad_end - 5:
                        _p_unif_g = self._gradient_phase(
                            _p_unif, movable_idx, sizes_np, port_pos, canvas_w, canvas_h,
                            num_hard, grad_safe_nnp, grad_nnmask, grid_rows, grid_cols,
                            grad_hpwl_norm, _grad_end, fast_eng=fast_eng,
                        )
                        _p_unif_g = self._resolve(_p_unif_g, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                        starts.append(_p_unif_g)
                        starts_cost.append(float('inf'))

            best_global_cost = min(starts_cost)
            best_pos = starts[int(np.argmin(starts_cost))].copy()

            if oracle_is_fast:
                # Fixed-slot approach: each start gets at most slot_dur seconds.
                # This prevents long-converged starts from hogging budget — with
                # 3300s, 3 starts × 900s each is wasteful since SA converges ~300s.
                # Better: 10+ starts × 300s = full budget used, more diversity.
                total_oracle_time = max(1.0, oracle_end - time.time())
                slot_dur = min(max(180.0, total_oracle_time / max(len(starts), 1)),
                               max(180.0, total_oracle_time / 10.0))
                # Sort starts: known-cost first (best warm start), unknowns last
                known   = [(c, p) for c, p in zip(starts_cost, starts) if c < float('inf')]
                unknown = [(float('inf'), p) for c, p in zip(starts_cost, starts) if c == float('inf')]
                sorted_starts = sorted(known, key=lambda x: x[0]) + unknown

                for si, (s_cost_init, s_pos) in enumerate(sorted_starts):
                    if time.time() >= oracle_end - 5:
                        break
                    slot_end = min(time.time() + slot_dur, oracle_end)
                    result_s = self._plc_oracle_sa(
                        s_pos, movable_idx, sizes_np, port_pos, canvas_w, canvas_h,
                        num_hard, fixed_np, benchmark, plc, slot_end, s_cost_init,
                        macro_to_nets, nets_np,
                        fast_eng=fast_eng, grid_rows=grid_rows, grid_cols=grid_cols,
                    )
                    result_sr = self._resolve_fully(result_s, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                    cost_s = self._true_cost(result_sr, benchmark, plc)
                    if cost_s < best_global_cost:
                        best_global_cost = cost_s; best_pos = result_sr.copy()
                    # If budget remains after exhausting starts, add fresh perturbed restarts
                    if time.time() < oracle_end - slot_dur - 30 and si == len(sorted_starts) - 1:
                        p_extra = self._perturb(best_pos, movable_idx, sizes_np, canvas_w, canvas_h, 0.35)
                        p_extra = self._resolve(p_extra, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                        sorted_starts.append((float('inf'), p_extra))
            else:
                # Oracle too slow for SA: run congestion SA with remaining budget.
                # Phase 1b already ran at 37-50% of budget; use remaining time here.
                slow_sa_deadline = hard_deadline - 15
                best_pos = best_cheap_pos.copy()
                if (fast_eng is not None and time.time() < slow_sa_deadline - 30):
                    best_pos = self._run_fast_sa_cong(
                        best_pos, movable_idx, sizes_np, port_pos,
                        canvas_w, canvas_h, num_hard, fixed_np, cheap_cost,
                        slow_sa_deadline,
                        fast_eng=fast_eng, grid_rows=grid_rows, grid_cols=grid_cols,
                    )
                    c_slow = cheap_cost(best_pos)
                    if c_slow < best_cheap_cost:
                        best_cheap_cost = c_slow
                        best_cheap_pos  = best_pos.copy()
                best_pos = self._resolve_fully(best_pos, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)

        # ── Phase 3: Final selection — oracle result vs pos_init ──────────────
        # Always resolve_fully best_pos first so phase 3 comparison is valid.
        best_pos = self._resolve_fully(best_pos, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
        # Only run oracle comparison in Phase 3 if oracle is fast enough (skip for ibm17-scale).
        _oracle_fast_p3 = locals().get('oracle_is_fast', True) and plc is not None
        if _oracle_fast_p3 and time.time() < hard_deadline - 3:
            best_final_cost = self._true_cost(best_pos, benchmark, plc) if plc else float('inf')
            if time.time() < hard_deadline - 2:
                init_rr = self._resolve_fully(pos_init, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                tc_init = self._true_cost(init_rr, benchmark, plc)
                if tc_init < best_final_cost:
                    best_pos = init_rr.copy()
        result = benchmark.macro_positions.clone()
        result[:] = torch.from_numpy(best_pos).float()
        # Float32 quantization can reintroduce tiny overlaps (error ~1e-4 for large coords).
        # Re-resolve in float32 space to guarantee VALID output.
        best_pos32 = result.numpy().astype(np.float64)
        best_pos32 = self._resolve(best_pos32, num_hard, sizes_np, canvas_w, canvas_h, fixed_np, max_iter=200, min_iter=10)
        result[:] = torch.from_numpy(best_pos32).float()
        return result

    # ── Oracle SA with temperature ────────────────────────────────────────────

    def _plc_oracle_sa(self, pos, movable_idx, sizes, port_pos, canvas_w, canvas_h,
                       num_hard, fixed, benchmark, plc, deadline, init_cost,
                       macro_to_nets, nets_np,
                       fast_eng=None, grid_rows=1, grid_cols=1):
        """
        SA using exact PlacementCost as the objective function.

        Move types (probability-weighted):
        - 25% swap: exchange positions of two macros
        - 25% net-centroid: move macro toward weighted centroid of its net endpoints
        - 20% small Gaussian: fine-tune individual macro position
        - 15% large jump: escape basin (20% of canvas)
        - 10% congestion hotspot escape: move macro away from worst congestion cell
        - 05% cluster shuffle: permute positions of 3-5 connected macros
        """
        best_pos  = pos.copy()
        cur_pos   = pos.copy()
        best_cost = init_cost
        cur_cost  = init_cost
        n_mov     = len(movable_idx)
        step_small = (canvas_w + canvas_h) * 0.03
        step_large = (canvas_w + canvas_h) * 0.18

        # Congestion hotspot tracking (refresh every 40 oracle steps)
        hotspot_cell = None
        hotspot_refresh_steps = 40

        # Orientation tracking — key state alongside positions
        _ORIENTS = ['N', 'FN', 'S', 'FS', 'E', 'FE', 'W', 'FW']
        has_orient = (plc is not None and hasattr(benchmark, 'hard_macro_indices'))
        if has_orient:
            cur_orient  = {benchmark.hard_macro_indices[i]: plc.get_macro_orientation(benchmark.hard_macro_indices[i])
                           for i in movable_idx}
            best_orient = dict(cur_orient)
        else:
            cur_orient = best_orient = {}

        step        = 0
        t_run_start = time.time()

        # Cached combined position array (hard macros + ports), updated incrementally
        cur_all = self._all_pos(cur_pos, port_pos)

        while time.time() < deadline - 1.5:
            elapsed     = time.time() - t_run_start
            time_budget = max(1.0, deadline - 1.5 - t_run_start)
            frac        = min(1.0, elapsed / time_budget)
            # Geometric cooling based on time fraction (not step count)
            T = T_ORC_START * (T_ORC_END / T_ORC_START) ** frac

            # Periodically refresh congestion hotspot using fast surrogate
            if fast_eng is not None and step % hotspot_refresh_steps == 0:
                cg = self._fast_cong_grid(self._all_pos(cur_pos, port_pos), num_hard, sizes,
                                          fast_eng, grid_rows, grid_cols, canvas_w, canvas_h)
                r_hot, c_hot = np.unravel_index(cg.argmax(), cg.shape)
                gw = canvas_w / grid_cols; gh = canvas_h / grid_rows
                hotspot_cell = ((c_hot + 0.5) * gw, (r_hot + 0.5) * gh)

            cand = cur_pos.copy()
            # Temperature-adaptive move distribution:
            # High T (frac<0.3): 35% swap + 20% large + 15% net-centroid + 15% small + 10% hotspot + 5% orient
            # Low T  (frac>0.7): 15% swap + 5% large + 35% net-centroid + 30% small + 10% hotspot + 5% orient
            # Interpolate linearly between high-T and low-T distributions.
            _p_swap    = 0.35 - 0.20 * frac   # 0.35 → 0.15
            _p_centroid = 0.15 + 0.20 * frac  # 0.15 → 0.35
            _p_small   = 0.15 + 0.15 * frac   # 0.15 → 0.30
            _p_large   = 0.20 - 0.15 * frac   # 0.20 → 0.05
            # hotspot=0.10, orient=0.05 held constant; renormalise implicitly via boundaries
            _b1 = _p_swap
            _b2 = _b1 + _p_centroid
            _b3 = _b2 + _p_small
            _b4 = _b3 + _p_large
            _b5 = _b4 + 0.10   # hotspot
            # orient occupies [_b5, 1.0]

            move_type = random.random()

            if move_type < _b1:
                # Swap two random movable macros
                idx  = movable_idx[random.randrange(n_mov)]
                idx2 = movable_idx[random.randrange(n_mov)]
                if idx == idx2: continue
                w,  h  = sizes[idx,  0], sizes[idx,  1]
                w2, h2 = sizes[idx2, 0], sizes[idx2, 1]
                p1, p2 = cur_pos[idx].copy(), cur_pos[idx2].copy()
                # Bounds check: can swapped positions fit?
                if (p2[0]<w/2 or p2[0]>canvas_w-w/2 or p2[1]<h/2 or p2[1]>canvas_h-h/2 or
                    p1[0]<w2/2 or p1[0]>canvas_w-w2/2 or p1[1]<h2/2 or p1[1]>canvas_h-h2/2):
                    continue
                cand[idx] = p2; cand[idx2] = p1

            elif move_type < _b2:
                # Net-centroid-guided move: move macro toward centroid of its net peers.
                # Force-directed step: naturally minimizes wirelength.
                idx = movable_idx[random.randrange(n_mov)]
                w, h = sizes[idx, 0], sizes[idx, 1]
                nets_i = macro_to_nets.get(idx, [])
                if not nets_i:
                    # Fallback to Gaussian if isolated
                    cand[idx, 0] = float(np.clip(
                        cur_pos[idx,0] + np.random.normal(0, step_small), w/2, canvas_w-w/2))
                    cand[idx, 1] = float(np.clip(
                        cur_pos[idx,1] + np.random.normal(0, step_small), h/2, canvas_h-h/2))
                else:
                    # Weighted centroid: weight by 1/net_size (smaller nets → stronger pull)
                    cx_sum = 0.0; cy_sum = 0.0; wt_sum = 0.0
                    for ni in nets_i:
                        nodes = nets_np[ni]
                        # Include all nodes except idx itself
                        others = [int(n) for n in nodes if int(n) != idx]
                        if not others: continue
                        wt = 1.0 / max(1, len(nodes))
                        for n in others:
                            cx_sum += wt * cur_all[n, 0]
                            cy_sum += wt * cur_all[n, 1]
                            wt_sum += wt
                    if wt_sum < 1e-9:
                        cand[idx, 0] = float(np.clip(
                            cur_pos[idx,0] + np.random.normal(0, step_small), w/2, canvas_w-w/2))
                        cand[idx, 1] = float(np.clip(
                            cur_pos[idx,1] + np.random.normal(0, step_small), h/2, canvas_h-h/2))
                    else:
                        target_x = cx_sum / wt_sum
                        target_y = cy_sum / wt_sum
                        # Move a random fraction [0.3, 0.9] toward centroid, plus small noise
                        frac_move = np.random.uniform(0.3, 0.9)
                        nx = cur_pos[idx,0] + frac_move * (target_x - cur_pos[idx,0])
                        ny = cur_pos[idx,1] + frac_move * (target_y - cur_pos[idx,1])
                        # Add small noise to avoid degenerate solutions
                        nx += np.random.normal(0, step_small * 0.3)
                        ny += np.random.normal(0, step_small * 0.3)
                        cand[idx, 0] = float(np.clip(nx, w/2, canvas_w-w/2))
                        cand[idx, 1] = float(np.clip(ny, h/2, canvas_h-h/2))

            elif move_type < _b3:
                # Small Gaussian: fine-tune
                idx = movable_idx[random.randrange(n_mov)]
                w, h = sizes[idx, 0], sizes[idx, 1]
                cand[idx, 0] = float(np.clip(
                    cur_pos[idx,0] + np.random.normal(0, step_small), w/2, canvas_w-w/2))
                cand[idx, 1] = float(np.clip(
                    cur_pos[idx,1] + np.random.normal(0, step_small), h/2, canvas_h-h/2))

            elif move_type < _b4:
                # Large jump: escape basin
                idx = movable_idx[random.randrange(n_mov)]
                w, h = sizes[idx, 0], sizes[idx, 1]
                cand[idx, 0] = float(np.clip(
                    cur_pos[idx,0] + np.random.normal(0, step_large), w/2, canvas_w-w/2))
                cand[idx, 1] = float(np.clip(
                    cur_pos[idx,1] + np.random.normal(0, step_large), h/2, canvas_h-h/2))

            elif move_type < _b5 and has_orient:
                # Orientation flip: try a random orientation for one macro.
                # Pin offsets change → WL and routing patterns shift without moving the footprint.
                idx = movable_idx[random.randrange(n_mov)]
                plc_idx = benchmark.hard_macro_indices[idx]
                old_orient = cur_orient[plc_idx]
                new_orient = random.choice([o for o in _ORIENTS if o != old_orient])
                plc.update_macro_orientation(plc_idx, new_orient)
                cost = self._true_cost(cur_pos, benchmark, plc)
                step += 1
                delta = cost - cur_cost
                if delta < 0 or (T > 1e-9 and random.random() < math.exp(max(-30.0, -delta / T))):
                    cur_orient[plc_idx] = new_orient
                    cur_cost = cost
                    if cost < best_cost:
                        best_cost = cost; best_pos = cur_pos.copy()
                        best_orient = dict(cur_orient)
                else:
                    plc.update_macro_orientation(plc_idx, old_orient)
                continue

            else:
                # Cluster shuffle: permute positions of 3-5 related macros
                k = random.randint(3, min(5, n_mov))
                cluster = random.sample(movable_idx, k)
                positions = [cur_pos[i].copy() for i in cluster]
                random.shuffle(positions)
                valid = True
                for i, cidx in enumerate(cluster):
                    w, h = sizes[cidx, 0], sizes[cidx, 1]
                    p = positions[i]
                    if p[0]<w/2 or p[0]>canvas_w-w/2 or p[1]<h/2 or p[1]>canvas_h-h/2:
                        valid = False; break
                    cand[cidx] = p
                if not valid: continue

            # Fast local resolve: only check moved macros against all others.
            # O(k×n) instead of O(n²); k=1-5, n=num_hard. ~100x faster than full resolve.
            if move_type < _b1:
                moved_set = [idx, idx2]
            elif move_type < _b4:
                moved_set = [idx]
            else:
                moved_set = cluster
            cand = self._local_resolve(cand, moved_set, num_hard, sizes, canvas_w, canvas_h, fixed)
            cost = self._true_cost(cand, benchmark, plc)
            step += 1

            delta = cost - cur_cost
            if delta < 0 or (T > 1e-9 and random.random() < math.exp(max(-30.0, -delta / T))):
                cur_pos  = cand.copy()
                cur_cost = cost
                # Update cached combined position array for centroid moves
                cur_all[:len(cur_pos)] = cur_pos
                if cost < best_cost:
                    best_cost = cost; best_pos = cand.copy()
                    best_orient = dict(cur_orient)

        # Restore plc to best-found orientations before returning
        if has_orient:
            for plc_idx, orient in best_orient.items():
                plc.update_macro_orientation(plc_idx, orient)

        return best_pos

    # ── SA (cheap warm-up) ─────────────────────────────────────────────────────

    def _run_sa(self, pos, nets_np, macro_to_nets, movable_idx,
                sizes, port_pos, canvas_w, canvas_h,
                grid_rows, grid_cols, num_hard, T, n_steps, hpwl_norm,
                safe_nnp, nnmask):
        p = pos.copy(); ap = self._all_pos(p, port_pos)
        cell_w=canvas_w/grid_cols; cell_h=canvas_h/grid_rows; cell_a=cell_w*cell_h
        lx=p[:num_hard,0]-sizes[:num_hard,0]/2; rx=p[:num_hard,0]+sizes[:num_hard,0]/2
        ly=p[:num_hard,1]-sizes[:num_hard,1]/2; ry=p[:num_hard,1]+sizes[:num_hard,1]/2
        all_x=ap[safe_nnp,0]; all_y=ap[safe_nnp,1]
        INF=1e15
        mx=np.where(nnmask,all_x,-INF); mn=np.where(nnmask,all_x,INF)
        ryn=np.where(nnmask,all_y,-INF); lyn=np.where(nnmask,all_y,INF)
        net_hpwl=(mx.max(1)-mn.min(1))+(ryn.max(1)-lyn.min(1))
        dens=self._density_grid(p,num_hard,sizes,grid_rows,grid_cols,canvas_w,canvas_h)
        cur_density=self._top_k_mean(dens,0.10)
        cur_wl_norm=float(net_hpwl.sum())/hpwl_norm
        cur_cost=cur_wl_norm+0.5*cur_density
        best_cost=cur_cost; best_p=p.copy()
        n_mov=len(movable_idx); step_scale=max(0.02,MAX_STEP*(T/T_SA_START))

        for _ in range(n_steps):
            is_swap=(random.random()<SWAP_PROB)
            if is_swap:
                idx=movable_idx[random.randrange(n_mov)]; idx2=movable_idx[random.randrange(n_mov)]
                if idx==idx2: continue
                nx,ny=p[idx2,0],p[idx2,1]; nx2,ny2=p[idx,0],p[idx,1]
                w,h=sizes[idx,0],sizes[idx,1]; w2,h2=sizes[idx2,0],sizes[idx2,1]
                if (nx<w/2 or nx>canvas_w-w/2 or ny<h/2 or ny>canvas_h-h/2 or
                    nx2<w2/2 or nx2>canvas_w-w2/2 or ny2<h2/2 or ny2>canvas_h-h2/2): continue
                ox_c,oy_c=p[idx,0],p[idx,1]; ox_c2,oy_c2=p[idx2,0],p[idx2,1]
                nets_aff=list(set(macro_to_nets.get(idx,[]))|set(macro_to_nets.get(idx2,[])))
                new_hpwls={}; delta_wl=0.
                for ni in nets_aff:
                    nodes=nets_np[ni]; xs_n=ap[nodes,0].copy(); ys_n=ap[nodes,1].copy()
                    for k,node in enumerate(nodes):
                        if node==idx: xs_n[k]=nx; ys_n[k]=ny
                        elif node==idx2: xs_n[k]=nx2; ys_n[k]=ny2
                    nh=(xs_n.max()-xs_n.min())+(ys_n.max()-ys_n.min())
                    new_hpwls[ni]=nh; delta_wl+=nh-net_hpwl[ni]
                self._add_macro_to_dens(dens,ox_c,oy_c,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,-1)
                self._add_macro_to_dens(dens,ox_c2,oy_c2,w2,h2,cell_w,cell_h,grid_rows,grid_cols,cell_a,-1)
                self._add_macro_to_dens(dens,nx,ny,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,+1)
                self._add_macro_to_dens(dens,nx2,ny2,w2,h2,cell_w,cell_h,grid_rows,grid_cols,cell_a,+1)
                new_density=self._top_k_mean(dens,0.10)
                self._add_macro_to_dens(dens,nx,ny,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,-1)
                self._add_macro_to_dens(dens,nx2,ny2,w2,h2,cell_w,cell_h,grid_rows,grid_cols,cell_a,-1)
                self._add_macro_to_dens(dens,ox_c,oy_c,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,+1)
                self._add_macro_to_dens(dens,ox_c2,oy_c2,w2,h2,cell_w,cell_h,grid_rows,grid_cols,cell_a,+1)
                ov_old=0.; ov_new=0.
                for ii,(xxi,yyi,wwi,hhi) in [(idx,(ox_c,oy_c,w,h)),(idx2,(ox_c2,oy_c2,w2,h2))]:
                    lxi=xxi-wwi/2;rxi=xxi+wwi/2;lyi=yyi-hhi/2;ryi=yyi+hhi/2
                    oxa=np.maximum(0.,np.minimum(rxi,rx)-np.maximum(lxi,lx))
                    oya=np.maximum(0.,np.minimum(ryi,ry)-np.maximum(lyi,ly))
                    ov_old+=(oxa*oya).sum()-oxa[ii]*oya[ii]
                for ii,(xxi,yyi,wwi,hhi) in [(idx,(nx,ny,w,h)),(idx2,(nx2,ny2,w2,h2))]:
                    lxi=xxi-wwi/2;rxi=xxi+wwi/2;lyi=yyi-hhi/2;ryi=yyi+hhi/2
                    oxa=np.maximum(0.,np.minimum(rxi,rx)-np.maximum(lxi,lx))
                    oya=np.maximum(0.,np.minimum(ryi,ry)-np.maximum(lyi,ly))
                    ov_new+=(oxa*oya).sum()-oxa[ii]*oya[ii]
                delta_cost=(delta_wl/hpwl_norm+0.5*(new_density-cur_density)+OVERLAP_WEIGHT*(ov_new-ov_old))
                accept=(delta_cost<0 or (T>1e-12 and random.random()<math.exp(max(-30.,-delta_cost/T))))
                if accept:
                    p[idx,0]=nx;p[idx,1]=ny;ap[idx,0]=nx;ap[idx,1]=ny
                    p[idx2,0]=nx2;p[idx2,1]=ny2;ap[idx2,0]=nx2;ap[idx2,1]=ny2
                    lx[idx]=nx-w/2;rx[idx]=nx+w/2;ly[idx]=ny-h/2;ry[idx]=ny+h/2
                    lx[idx2]=nx2-w2/2;rx[idx2]=nx2+w2/2;ly[idx2]=ny2-h2/2;ry[idx2]=ny2+h2/2
                    for ni in nets_aff: net_hpwl[ni]=new_hpwls[ni]
                    self._add_macro_to_dens(dens,ox_c,oy_c,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,-1)
                    self._add_macro_to_dens(dens,ox_c2,oy_c2,w2,h2,cell_w,cell_h,grid_rows,grid_cols,cell_a,-1)
                    self._add_macro_to_dens(dens,nx,ny,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,+1)
                    self._add_macro_to_dens(dens,nx2,ny2,w2,h2,cell_w,cell_h,grid_rows,grid_cols,cell_a,+1)
                    cur_density=new_density; cur_wl_norm+=delta_wl/hpwl_norm
                    cur_cost=cur_wl_norm+0.5*cur_density
                    if cur_cost<best_cost: best_cost=cur_cost; best_p=p.copy()
            else:
                idx=movable_idx[random.randrange(n_mov)]
                w,h=sizes[idx,0],sizes[idx,1]; ox_c,oy_c=p[idx,0],p[idx,1]
                nx=float(np.clip(ox_c+np.random.normal(0,step_scale),w/2,canvas_w-w/2))
                ny=float(np.clip(oy_c+np.random.normal(0,step_scale),h/2,canvas_h-h/2))
                nets_i=macro_to_nets.get(idx,[])
                new_hpwls=[]; delta_wl=0.
                for ni in nets_i:
                    nodes=nets_np[ni]; mask=(nodes==idx)
                    xs_n=np.where(mask,nx,ap[nodes,0]); ys_n=np.where(mask,ny,ap[nodes,1])
                    nh=(xs_n.max()-xs_n.min())+(ys_n.max()-ys_n.min())
                    new_hpwls.append(nh); delta_wl+=nh-net_hpwl[ni]
                self._add_macro_to_dens(dens,ox_c,oy_c,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,-1)
                self._add_macro_to_dens(dens,nx,ny,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,+1)
                new_density=self._top_k_mean(dens,0.10)
                self._add_macro_to_dens(dens,nx,ny,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,-1)
                self._add_macro_to_dens(dens,ox_c,oy_c,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,+1)
                lx_o=ox_c-w/2;rx_o=ox_c+w/2;ly_o=oy_c-h/2;ry_o=oy_c+h/2
                oa=np.maximum(0.,np.minimum(rx_o,rx)-np.maximum(lx_o,lx))
                ob2=np.maximum(0.,np.minimum(ry_o,ry)-np.maximum(ly_o,ly))
                ov_o=(oa*ob2).sum()-oa[idx]*ob2[idx]
                lx_n=nx-w/2;rx_n=nx+w/2;ly_n=ny-h/2;ry_n=ny+h/2
                oa2=np.maximum(0.,np.minimum(rx_n,rx)-np.maximum(lx_n,lx))
                ob3=np.maximum(0.,np.minimum(ry_n,ry)-np.maximum(ly_n,ly))
                ov_n=(oa2*ob3).sum()-oa2[idx]*ob3[idx]
                delta_cost=(delta_wl/hpwl_norm+0.5*(new_density-cur_density)+OVERLAP_WEIGHT*(ov_n-ov_o))
                accept=(delta_cost<0 or (T>1e-12 and random.random()<math.exp(max(-30.,-delta_cost/T))))
                if accept:
                    p[idx,0]=nx;p[idx,1]=ny;ap[idx,0]=nx;ap[idx,1]=ny
                    lx[idx]=lx_n;rx[idx]=rx_n;ly[idx]=ly_n;ry[idx]=ry_n
                    for ni,nh in zip(nets_i,new_hpwls): net_hpwl[ni]=nh
                    self._add_macro_to_dens(dens,ox_c,oy_c,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,-1)
                    self._add_macro_to_dens(dens,nx,ny,w,h,cell_w,cell_h,grid_rows,grid_cols,cell_a,+1)
                    cur_density=new_density; cur_wl_norm+=delta_wl/hpwl_norm
                    cur_cost=cur_wl_norm+0.5*cur_density
                    if cur_cost<best_cost: best_cost=cur_cost; best_p=p.copy()
        return best_p

    # ── Gradient-based placement (ePlace-style smooth WL + density) ──────────

    def _gradient_phase(self, pos, movable_idx, sizes_np, port_pos,
                        canvas_w, canvas_h, num_hard,
                        safe_nnp, nnmask, grid_rows, grid_cols, hpwl_norm, deadline,
                        fast_eng=None, lam_den_override=None, lam_cong_override=None,
                        gamma_scale=1.0):
        """Phase 0/2b: Adam on smooth WL + soft density + soft congestion.
        Override params allow surrogate calibration from oracle feedback."""
        n_mov = len(movable_idx)
        if n_mov == 0 or time.time() >= deadline:
            return pos.copy()

        n_nets, max_nsz = safe_nnp.shape
        # Skip if net matrix too large (memory / speed guard — relaxed for GPU)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mem_limit = 80_000_000 if dev.type == 'cuda' else 10_000_000
        if n_nets * max_nsz > mem_limit:
            return pos.copy()

        mov_arr = np.array(movable_idx, dtype=np.int64)
        mov_t   = torch.from_numpy(mov_arr).to(dev)
        n_ports = port_pos.shape[0]

        # Full reference positions: ALL macros (hard+soft) + ports, no grad on fixed
        if n_ports > 0:
            pos_full_np = np.vstack([pos, port_pos]).astype(np.float32)
        else:
            pos_full_np = pos.astype(np.float32)
        pos_ref = torch.from_numpy(pos_full_np).to(dev)   # no requires_grad

        # Trainable: movable positions only
        x_param = torch.nn.Parameter(
            torch.from_numpy(pos_full_np[mov_arr].copy()).to(dev)
        )
        hw_bnd = torch.from_numpy(sizes_np[mov_arr, 0].astype(np.float32)).to(dev) / 2
        hh_bnd = torch.from_numpy(sizes_np[mov_arr, 1].astype(np.float32)).to(dev) / 2
        cw_t   = torch.tensor(float(canvas_w), device=dev)
        ch_t   = torch.tensor(float(canvas_h), device=dev)

        # Net tensors
        safe_t = torch.from_numpy(safe_nnp.astype(np.int64)).to(dev)   # (n_nets, max_nsz)
        mask_t = torch.from_numpy(nnmask).to(dev)                      # (n_nets, max_nsz) bool

        # Density tensors — hard macros only for the primary gradient phase.
        # Soft macros are handled by a separate gradient pass (_gradient_phase_soft).
        n_den_macros = num_hard  # density kernel spans hard macros only here
        sizes_den_t = torch.from_numpy(sizes_np[:n_den_macros].astype(np.float32)).to(dev)
        cell_w  = float(canvas_w) / grid_cols
        cell_h  = float(canvas_h) / grid_rows
        cx_t    = torch.arange(grid_cols, dtype=torch.float32, device=dev) * cell_w + cell_w / 2
        cy_t    = torch.arange(grid_rows, dtype=torch.float32, device=dev) * cell_h + cell_h / 2
        hw_den  = sizes_den_t[:, 0] / 2 + cell_w / 2  # (n_den_macros,)
        hh_den  = sizes_den_t[:, 1] / 2 + cell_h / 2

        canvas_diag  = float(canvas_w + canvas_h)
        gamma_start  = canvas_diag * 0.04 * gamma_scale   # coarse smooth at start
        gamma_end    = canvas_diag * 0.004 * gamma_scale  # sharp at end
        target_den   = 0.55
        lam_den_start = (lam_den_override * 0.5) if lam_den_override else 0.10
        lam_den_end   = lam_den_override if lam_den_override else 0.60
        lam_cong_max  = lam_cong_override if lam_cong_override else 0.30
        lr            = canvas_diag * 0.002
        t_start       = time.time()
        t_total       = max(1.0, deadline - t_start)

        # ePlace-style density overflow tracking: if density stays high after
        # the normal gradient run, we do a density-overflow pass with lam_den
        # ramped much higher (up to 4×) to force macros to spread out.
        # This mimics ePlace's Lagrangian multiplier update for density.
        overflow_boost = 1.0   # multiplier on lam_den_end; updated after each pass
        _best_pos_grad = pos.copy()
        _best_pos_cost = float('inf')

        # Precompute congestion tensors if fast_eng available.
        # GPU: no n_pairs limit (matmul is O(p×r + p×c) not O(p×r×c)).
        # CPU: limit to 200K pairs (memory guard for einsum).
        cong_enabled = False
        macro_block_enabled = False
        cong_use_matmul = False
        if fast_eng is not None:
            n_pairs = len(fast_eng['src'])
            pairs_limit = 2_000_000 if dev.type == 'cuda' else 200_000
            if n_pairs < pairs_limit:
                src_ct = torch.from_numpy(fast_eng['src'].astype(np.int64)).to(dev)
                snk_ct = torch.from_numpy(fast_eng['snk'].astype(np.int64)).to(dev)
                w_ct   = torch.from_numpy(fast_eng['w'].astype(np.float32)).to(dev)
                hpm_t  = float(fast_eng['hpm'])
                vpm_t  = float(fast_eng['vpm'])
                halloc = float(fast_eng.get('halloc', 0.5))
                valloc = float(fast_eng.get('valloc', 0.5))
                gw_t   = float(canvas_w) / grid_cols
                gh_t   = float(canvas_h) / grid_rows
                h_cap_t = gh_t * hpm_t
                v_cap_t = gw_t * vpm_t
                r_idx_t = torch.arange(grid_rows, dtype=torch.float32, device=dev)
                c_idx_t = torch.arange(grid_cols, dtype=torch.float32, device=dev)
                cong_enabled = (h_cap_t > 1e-9 and v_cap_t > 1e-9 and t_total > 20.0)
                # For large n_pairs on GPU, use (w*row_w_H).T @ col_H matmul
                # instead of 3-way einsum to avoid (p,r,c) intermediate tensor.
                cong_use_matmul = (dev.type == 'cuda' and n_pairs >= 50_000)

        # Adam with cosine annealing: starts at lr_max, decays to lr_min.
        optimizer = torch.optim.Adam([x_param], lr=lr, betas=(0.9, 0.999))
        lr_min = lr * 0.02

        # DREAMPlace-style per-macro preconditioner: p_i = 1/sqrt(degree_i).
        # Macros touching many nets receive smaller gradient steps — they have
        # many conflicting net forces that would otherwise cause oscillation.
        # Computed from safe_nnp: count appearances of each movable macro.
        _mov_to_k = {idx: k for k, idx in enumerate(movable_idx)}
        _deg = np.zeros(n_mov, dtype=np.float32)
        for _v in safe_nnp.ravel():
            _k = _mov_to_k.get(int(_v))
            if _k is not None:
                _deg[_k] += 1.0
        _deg = np.maximum(_deg, 1.0)
        _pc = 1.0 / np.sqrt(_deg)
        _pc /= max(float(_pc.max()), 1e-9)
        precond_t = torch.from_numpy(_pc).to(dev).unsqueeze(1)  # (n_mov, 1)

        # Pre-compute density overflow constants (used in loop and after)
        top_k_den_h = max(1, int(grid_rows * grid_cols * 0.10))
        tau_den_h   = 0.3

        step = 0
        t_first = None

        while time.time() < deadline:
            t_s  = time.time()
            frac = min(1.0, (t_s - t_start) / t_total)
            # Cosine annealing LR: high early for exploration, near-zero late for refinement
            cur_lr = lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * frac))
            optimizer.param_groups[0]['lr'] = cur_lr
            # Anneal: gamma decreases (sharper WL), lam_den increases (stronger spread)
            g       = gamma_start * (gamma_end / gamma_start) ** frac
            lam_den = lam_den_start + (lam_den_end - lam_den_start) * frac

            # Assemble full position tensor; gradients flow through x_param only
            pos_cur = pos_ref.index_put((mov_t,), x_param)  # (n_total, 2)

            # ── Smooth HPWL (weighted-average / logsumexp model) ──
            BIG   = 1e10
            node_x = pos_cur[safe_t, 0]   # (n_nets, max_nsz)
            node_y = pos_cur[safe_t, 1]
            nx_max = torch.where(mask_t, node_x, torch.full_like(node_x, -BIG))
            nx_min = torch.where(mask_t, node_x, torch.full_like(node_x,  BIG))
            ny_max = torch.where(mask_t, node_y, torch.full_like(node_y, -BIG))
            ny_min = torch.where(mask_t, node_y, torch.full_like(node_y,  BIG))
            wl_x   = (torch.logsumexp(nx_max / g, 1) + torch.logsumexp(-nx_min / g, 1)) * g
            wl_y   = (torch.logsumexp(ny_max / g, 1) + torch.logsumexp(-ny_min / g, 1)) * g
            wl_loss = (wl_x + wl_y).sum() / hpwl_norm

            # ── Soft density (tent-kernel bell) — hard macros only ──
            mx  = pos_cur[:n_den_macros, 0]
            my  = pos_cur[:n_den_macros, 1]
            dx  = mx.unsqueeze(1) - cx_t.unsqueeze(0)   # (n_den_macros, grid_cols)
            dy  = my.unsqueeze(1) - cy_t.unsqueeze(0)   # (n_den_macros, grid_rows)
            kx  = torch.clamp(1.0 - dx.abs() / hw_den.unsqueeze(1), min=0.0)
            ky  = torch.clamp(1.0 - dy.abs() / hh_den.unsqueeze(1), min=0.0)
            density  = torch.einsum('ic,ir->rc', kx, ky) / (cell_w * cell_h)
            dens_flat = density.view(-1)
            den_loss = torch.logsumexp(dens_flat / tau_den_h, 0) * tau_den_h / top_k_den_h

            # ── Differentiable congestion (H-then-V 2-pin model) ──────────────
            # Enabled only after 40% of gradient time, to let WL+density settle first.
            cong_loss = torch.tensor(0.0, device=dev)
            lam_cong  = 0.0
            if cong_enabled and frac > 0.60:
                lam_cong = lam_cong_max * min(1.0, (frac - 0.60) / 0.30)
                # Src/snk positions
                xs = pos_cur[src_ct, 0]; ys = pos_cur[src_ct, 1]
                xt = pos_cur[snk_ct, 0]; yt = pos_cur[snk_ct, 1]
                # H routing: at source row (bilinear), cols [min(xs,xt), max(xs,xt)]
                r_s  = ys / gh_t
                row_w_H = torch.clamp(1.0 - (r_s.unsqueeze(1) - r_idx_t).abs(), min=0.0)
                c_lo = torch.minimum(xs, xt) / gw_t
                c_hi = torch.maximum(xs, xt) / gw_t
                al = 6.0  # sharpness in grid-cell units
                col_H = torch.sigmoid(al * (c_idx_t - c_lo.unsqueeze(1))) * \
                        torch.sigmoid(al * (c_hi.unsqueeze(1) - c_idx_t))
                # Use matmul for large n_pairs on GPU (avoids (p,r,c) intermediate)
                if cong_use_matmul:
                    H_cong = ((w_ct.unsqueeze(1) * row_w_H).T @ col_H) / h_cap_t
                else:
                    H_cong = torch.einsum('p,pr,pc->rc', w_ct, row_w_H, col_H) / h_cap_t
                # V routing: at sink col (bilinear), rows [min(ys,yt), max(ys,yt)]
                c_t  = xt / gw_t
                col_w_V = torch.clamp(1.0 - (c_t.unsqueeze(1) - c_idx_t).abs(), min=0.0)
                r_lo = torch.minimum(ys, yt) / gh_t
                r_hi = torch.maximum(ys, yt) / gh_t
                row_V = torch.sigmoid(al * (r_idx_t - r_lo.unsqueeze(1))) * \
                        torch.sigmoid(al * (r_hi.unsqueeze(1) - r_idx_t))
                if cong_use_matmul:
                    V_cong = ((w_ct.unsqueeze(1) * row_V).T @ col_w_V) / v_cap_t
                else:
                    V_cong = torch.einsum('p,pr,pc->rc', w_ct, row_V, col_w_V) / v_cap_t
                # ── Macro routing blockage: hard macros block routing channels ──
                # Oracle formula per cell (r,c): V_macro += x_ov(macro,col_c) × valloc
                # applied for all rows r that the macro overlaps.
                # Differentiable: use row_cov (fraction of row covered) as soft indicator.
                if macro_block_enabled:
                    mx_hard = pos_cur[:num_hard, 0]
                    my_hard = pos_cur[:num_hard, 1]
                    lx_mac = (mx_hard - hw_macro).unsqueeze(1)  # (num_hard, 1)
                    rx_mac = (mx_hard + hw_macro).unsqueeze(1)
                    col_lo = cell_x_edges[:-1].unsqueeze(0)
                    col_hi = cell_x_edges[1:].unsqueeze(0)
                    x_ov = torch.clamp(torch.minimum(rx_mac, col_hi) -
                                       torch.maximum(lx_mac, col_lo), min=0.0)  # (num_hard, grid_cols)
                    ly_mac = (my_hard - hh_macro).unsqueeze(1)
                    ry_mac = (my_hard + hh_macro).unsqueeze(1)
                    row_lo = cell_y_edges[:-1].unsqueeze(0)
                    row_hi = cell_y_edges[1:].unsqueeze(0)
                    y_ov = torch.clamp(torch.minimum(ry_mac, row_hi) -
                                       torch.maximum(ly_mac, row_lo), min=0.0)  # (num_hard, grid_rows)
                    # row_cov: fraction of each row covered by macro (soft binary indicator)
                    row_cov = y_ov / gw_t  # normalize to [0,1] approx
                    # V_macro[r,c] = sum_i x_ov[i,c] × row_cov[i,r] × valloc / v_cap
                    V_macro = torch.einsum('ic,ir->rc', x_ov, row_cov) * (valloc / v_cap_t)
                    # H_macro[r,c] = sum_i col_cov[i,c] × y_ov[i,r] × halloc / h_cap
                    col_cov = x_ov / gh_t
                    H_macro = torch.einsum('ic,ir->rc', col_cov, y_ov) * (halloc / h_cap_t)
                    H_cong = H_cong + H_macro
                    V_cong = V_cong + V_macro

                cong_combined = H_cong + V_cong
                excess_cong = torch.clamp(cong_combined - 1.0, min=0.0)
                # Soft ABU (top-5%) via logsumexp: matches oracle's abu(xx, 0.05) better
                # than plain mean(). Concentrates gradient on worst-congested cells.
                n_cells = excess_cong.numel()
                top_k = max(1, int(n_cells * 0.05))
                tau_cong = 0.5  # temperature for soft-max approximation
                excess_flat = excess_cong.view(-1)
                # logsumexp over all cells (differentiable smooth-max approximating top-5%)
                cong_loss = (torch.logsumexp(excess_flat / tau_cong, dim=0) * tau_cong) / top_k

            loss = wl_loss + lam_den * den_loss + lam_cong * cong_loss
            optimizer.zero_grad()
            loss.backward()
            # Apply preconditioner: scale movable-macro gradients before Adam step.
            # This implements degree-weighted gradient scaling (DREAMPlace §3.2).
            with torch.no_grad():
                x_param.grad.data.mul_(precond_t)
            optimizer.step()

            # Snap-noise: perturb within ±quarter gcell to build robustness
            # to grid snapping (Flaxman-Kalai-McMahan stochastic smoothing)
            with torch.no_grad():
                x_param.data[:, 0] += torch.empty_like(x_param.data[:, 0]).uniform_(-cell_w/4, cell_w/4)
                x_param.data[:, 1] += torch.empty_like(x_param.data[:, 1]).uniform_(-cell_h/4, cell_h/4)

            # Clamp to canvas
            with torch.no_grad():
                x_param.data[:, 0] = torch.max(
                    torch.min(x_param.data[:, 0], cw_t - hw_bnd), hw_bnd)
                x_param.data[:, 1] = torch.max(
                    torch.min(x_param.data[:, 1], ch_t - hh_bnd), hh_bnd)

            step += 1

            # Bail if first step is too slow (large benchmark)
            if t_first is None:
                t_first = time.time() - t_s
                if t_first > 10.0:
                    break

        # ── Density overflow pass (ePlace-style Lagrangian update) ───────────
        # After main gradient, measure density overflow. If >15% of cells overflow
        # target density, run a second pass with lam_den boosted 3× to force spreading.
        # Guard: snapshot x_param before; revert if surrogate WL+density worsens.
        if time.time() < deadline - 15:
            with torch.no_grad():
                pos_ov = pos_ref.index_put((mov_t,), x_param)
                mx_ov = pos_ov[:n_den_macros, 0]
                my_ov = pos_ov[:n_den_macros, 1]
                dx_ov = mx_ov.unsqueeze(1) - cx_t.unsqueeze(0)
                dy_ov = my_ov.unsqueeze(1) - cy_t.unsqueeze(0)
                kx_ov = torch.clamp(1.0 - dx_ov.abs() / hw_den.unsqueeze(1), min=0.0)
                ky_ov = torch.clamp(1.0 - dy_ov.abs() / hh_den.unsqueeze(1), min=0.0)
                dens_ov = torch.einsum('ic,ir->rc', kx_ov, ky_ov) / (cell_w * cell_h)
                overflow_frac = float((dens_ov > target_den).float().mean())
                # Surrogate cost before overflow pass (WL + density, no overlap)
                node_x_bef = pos_ov[safe_t, 0]
                node_y_bef = pos_ov[safe_t, 1]
                nx_max_bef = torch.where(mask_t, node_x_bef, torch.full_like(node_x_bef, -1e10))
                nx_min_bef = torch.where(mask_t, node_x_bef, torch.full_like(node_x_bef,  1e10))
                ny_max_bef = torch.where(mask_t, node_y_bef, torch.full_like(node_y_bef, -1e10))
                ny_min_bef = torch.where(mask_t, node_y_bef, torch.full_like(node_y_bef,  1e10))
                wl_bef = ((torch.logsumexp(nx_max_bef / gamma_end, 1) +
                           torch.logsumexp(-nx_min_bef / gamma_end, 1) +
                           torch.logsumexp(ny_max_bef / gamma_end, 1) +
                           torch.logsumexp(-ny_min_bef / gamma_end, 1)) * gamma_end).sum() / hpwl_norm
                den_bef = torch.logsumexp(dens_ov.view(-1) / tau_den_h, 0) * tau_den_h / top_k_den_h
                cost_before = float(wl_bef + lam_den_end * den_bef)
                x_param_snapshot = x_param.data.clone()

            if overflow_frac > 0.15:
                # High density overflow → boost lam_den 3× to force spreading
                lam_den_boost = lam_den_end * 3.0
                optimizer2 = torch.optim.Adam([x_param], lr=lr * 0.4, betas=(0.9, 0.999))
                t_ov_start = time.time()
                t_ov_total = max(1.0, deadline - t_ov_start - 3)
                while time.time() < deadline - 3:
                    t_ov_s = time.time()
                    frac_ov = min(1.0, (t_ov_s - t_ov_start) / t_ov_total)
                    cur_lr2 = (lr * 0.4) * (1.0 - frac_ov) + (lr * 0.01) * frac_ov
                    optimizer2.param_groups[0]['lr'] = cur_lr2
                    g_ov = gamma_end

                    pos_cur2 = pos_ref.index_put((mov_t,), x_param)
                    node_x2 = pos_cur2[safe_t, 0]
                    node_y2 = pos_cur2[safe_t, 1]
                    nx_max2 = torch.where(mask_t, node_x2, torch.full_like(node_x2, -1e10))
                    nx_min2 = torch.where(mask_t, node_x2, torch.full_like(node_x2,  1e10))
                    ny_max2 = torch.where(mask_t, node_y2, torch.full_like(node_y2, -1e10))
                    ny_min2 = torch.where(mask_t, node_y2, torch.full_like(node_y2,  1e10))
                    wl_x2   = (torch.logsumexp(nx_max2 / g_ov, 1) + torch.logsumexp(-nx_min2 / g_ov, 1)) * g_ov
                    wl_y2   = (torch.logsumexp(ny_max2 / g_ov, 1) + torch.logsumexp(-ny_min2 / g_ov, 1)) * g_ov
                    wl_loss2 = (wl_x2 + wl_y2).sum() / hpwl_norm

                    mx2 = pos_cur2[:n_den_macros, 0]
                    my2 = pos_cur2[:n_den_macros, 1]
                    dx2 = mx2.unsqueeze(1) - cx_t.unsqueeze(0)
                    dy2 = my2.unsqueeze(1) - cy_t.unsqueeze(0)
                    kx2 = torch.clamp(1.0 - dx2.abs() / hw_den.unsqueeze(1), min=0.0)
                    ky2 = torch.clamp(1.0 - dy2.abs() / hh_den.unsqueeze(1), min=0.0)
                    dens2 = torch.einsum('ic,ir->rc', kx2, ky2) / (cell_w * cell_h)
                    dens2_flat = dens2.view(-1)
                    den_loss2 = torch.logsumexp(dens2_flat / tau_den_h, 0) * tau_den_h / top_k_den_h

                    loss2 = wl_loss2 + lam_den_boost * den_loss2
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()
                    with torch.no_grad():
                        x_param.data[:, 0] = torch.max(
                            torch.min(x_param.data[:, 0], cw_t - hw_bnd), hw_bnd)
                        x_param.data[:, 1] = torch.max(
                            torch.min(x_param.data[:, 1], ch_t - hh_bnd), hh_bnd)

                # Guard: revert if overflow pass worsened surrogate cost
                with torch.no_grad():
                    pos_af = pos_ref.index_put((mov_t,), x_param)
                    node_x_af = pos_af[safe_t, 0]
                    node_y_af = pos_af[safe_t, 1]
                    nx_max_af = torch.where(mask_t, node_x_af, torch.full_like(node_x_af, -1e10))
                    nx_min_af = torch.where(mask_t, node_x_af, torch.full_like(node_x_af,  1e10))
                    ny_max_af = torch.where(mask_t, node_y_af, torch.full_like(node_y_af, -1e10))
                    ny_min_af = torch.where(mask_t, node_y_af, torch.full_like(node_y_af,  1e10))
                    wl_af = ((torch.logsumexp(nx_max_af / gamma_end, 1) +
                              torch.logsumexp(-nx_min_af / gamma_end, 1) +
                              torch.logsumexp(ny_max_af / gamma_end, 1) +
                              torch.logsumexp(-ny_min_af / gamma_end, 1)) * gamma_end).sum() / hpwl_norm
                    mx_af = pos_af[:n_den_macros, 0]; my_af = pos_af[:n_den_macros, 1]
                    dx_af = mx_af.unsqueeze(1) - cx_t.unsqueeze(0)
                    dy_af = my_af.unsqueeze(1) - cy_t.unsqueeze(0)
                    kx_af = torch.clamp(1.0 - dx_af.abs() / hw_den.unsqueeze(1), min=0.0)
                    ky_af = torch.clamp(1.0 - dy_af.abs() / hh_den.unsqueeze(1), min=0.0)
                    dens_af = torch.einsum('ic,ir->rc', kx_af, ky_af) / (cell_w * cell_h)
                    den_af = torch.logsumexp(dens_af.view(-1) / tau_den_h, 0) * tau_den_h / top_k_den_h
                    cost_after = float(wl_af + lam_den_end * den_af)
                    if cost_after > cost_before:
                        x_param.data = x_param_snapshot  # revert

        result = pos.copy()
        with torch.no_grad():
            result[mov_arr] = x_param.data.cpu().numpy().astype(np.float64)
        return result

    # ── Soft macro gradient: WL + density-all (Step C2) ────────────────────────

    def _gradient_phase_soft(self, pos, soft_movable_idx, sizes_np, port_pos,
                              canvas_w, canvas_h, num_hard,
                              safe_nnp, nnmask, grid_rows, grid_cols, hpwl_norm, deadline):
        """Optimize soft macro positions with WL + density(all macros) gradient.

        Hard macros are fixed at their current positions. Soft macros spread
        away from dense areas (reducing oracle ABU density) while maintaining WL.
        Uses ALL macros in density tensor so gradient aligns with oracle metric.
        """
        n_soft_mov = len(soft_movable_idx)
        if n_soft_mov == 0 or time.time() >= deadline:
            return pos.copy()

        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_nets, max_nsz = safe_nnp.shape
        mem_limit = 80_000_000 if dev.type == 'cuda' else 10_000_000
        if n_nets * max_nsz > mem_limit:
            return pos.copy()

        mov_arr = np.array(soft_movable_idx, dtype=np.int64)
        mov_t   = torch.from_numpy(mov_arr).to(dev)
        n_ports = port_pos.shape[0]
        n_total_macros = pos.shape[0]  # hard + soft

        if n_ports > 0:
            pos_full_np = np.vstack([pos, port_pos]).astype(np.float32)
        else:
            pos_full_np = pos.astype(np.float32)
        pos_ref = torch.from_numpy(pos_full_np).to(dev)

        x_param = torch.nn.Parameter(
            torch.from_numpy(pos_full_np[mov_arr].copy()).to(dev)
        )
        hw_bnd = torch.from_numpy(sizes_np[mov_arr, 0].astype(np.float32)).to(dev) / 2
        hh_bnd = torch.from_numpy(sizes_np[mov_arr, 1].astype(np.float32)).to(dev) / 2
        cw_t   = torch.tensor(float(canvas_w), device=dev)
        ch_t   = torch.tensor(float(canvas_h), device=dev)

        safe_t = torch.from_numpy(safe_nnp.astype(np.int64)).to(dev)
        mask_t = torch.from_numpy(nnmask).to(dev)

        # Density over ALL macros: soft gradient aligns with oracle ABU metric
        sizes_all_t = torch.from_numpy(sizes_np[:n_total_macros].astype(np.float32)).to(dev)
        cell_w = float(canvas_w) / grid_cols
        cell_h = float(canvas_h) / grid_rows
        cx_t_d = torch.arange(grid_cols, dtype=torch.float32, device=dev) * cell_w + cell_w / 2
        cy_t_d = torch.arange(grid_rows, dtype=torch.float32, device=dev) * cell_h + cell_h / 2
        hw_den = sizes_all_t[:, 0] / 2 + cell_w / 2
        hh_den = sizes_all_t[:, 1] / 2 + cell_h / 2
        # Use ABU-style density loss (logsumexp top-10%) to match oracle metric.
        n_cells = grid_rows * grid_cols
        top_k_den = max(1, int(n_cells * 0.10))
        tau_den = 0.3  # temperature for soft-max density approximation

        canvas_diag = float(canvas_w + canvas_h)
        gamma = canvas_diag * 0.005  # fixed fine-grained gamma (soft macros are small)
        lr = canvas_diag * 0.001
        lam_den = 5.0  # very strong density penalty: overcome WL clustering pull
        optimizer = torch.optim.Adam([x_param], lr=lr, betas=(0.9, 0.999))
        t_start = time.time()
        t_total = max(1.0, deadline - t_start)
        step = 0
        t_first = None

        while time.time() < deadline:
            t_s = time.time()
            frac = min(1.0, (t_s - t_start) / t_total)
            # Cosine annealing
            lr_min = lr * 0.02
            cur_lr = lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * frac))
            optimizer.param_groups[0]['lr'] = cur_lr

            pos_cur = pos_ref.index_put((mov_t,), x_param)

            BIG = 1e10
            node_x = pos_cur[safe_t, 0]
            node_y = pos_cur[safe_t, 1]
            nx_max = torch.where(mask_t, node_x, torch.full_like(node_x, -BIG))
            nx_min = torch.where(mask_t, node_x, torch.full_like(node_x,  BIG))
            ny_max = torch.where(mask_t, node_y, torch.full_like(node_y, -BIG))
            ny_min = torch.where(mask_t, node_y, torch.full_like(node_y,  BIG))
            wl_x = (torch.logsumexp(nx_max / gamma, 1) + torch.logsumexp(-nx_min / gamma, 1)) * gamma
            wl_y = (torch.logsumexp(ny_max / gamma, 1) + torch.logsumexp(-ny_min / gamma, 1)) * gamma
            wl_loss = (wl_x + wl_y).sum() / hpwl_norm

            # Density over ALL macros — gradient flows through soft positions only
            mx  = pos_cur[:n_total_macros, 0]
            my  = pos_cur[:n_total_macros, 1]
            dx  = mx.unsqueeze(1) - cx_t_d.unsqueeze(0)
            dy  = my.unsqueeze(1) - cy_t_d.unsqueeze(0)
            kx  = torch.clamp(1.0 - dx.abs() / hw_den.unsqueeze(1), min=0.0)
            ky  = torch.clamp(1.0 - dy.abs() / hh_den.unsqueeze(1), min=0.0)
            density  = torch.einsum('ic,ir->rc', kx, ky) / (cell_w * cell_h)
            # ABU top-10% density via logsumexp — directly matches oracle density metric
            dens_flat = density.view(-1)
            den_loss = torch.logsumexp(dens_flat / tau_den, dim=0) * tau_den / top_k_den

            loss = wl_loss + lam_den * den_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_param.data[:, 0] = torch.max(
                    torch.min(x_param.data[:, 0], cw_t - hw_bnd), hw_bnd)
                x_param.data[:, 1] = torch.max(
                    torch.min(x_param.data[:, 1], ch_t - hh_bnd), hh_bnd)

            step += 1
            if t_first is None:
                t_first = time.time() - t_s
                if t_first > 10.0:
                    break

        result = pos.copy()
        with torch.no_grad():
            result[mov_arr] = x_param.data.cpu().numpy().astype(np.float64)
        return result

    # ── Force-directed soft macro placement (Step C) ─────────────────────────

    def _force_directed_soft(self, pos, num_hard, nets_np, net_weights_np,
                              port_pos, canvas_w, canvas_h, sizes_np, fixed_np):
        """Optimize soft macro positions via CG solve on the netlist Laplacian.

        Step C from the A-F pipeline: with hard macros fixed, find soft macro
        positions that minimize quadratic WL. Solves a sparse linear system.
        Uses star (not clique) net model: each node connects to net centroid
        with weight w — O(n_pins) per net vs O(n²) for clique.
        """
        try:
            from scipy.sparse import lil_matrix
            from scipy.sparse.linalg import cg as sp_cg
        except ImportError:
            return pos.copy()

        t0_fd = time.time()
        n_total = pos.shape[0]
        n_soft = n_total - num_hard
        if n_soft == 0:
            return pos.copy()

        soft_movable = [i for i in range(num_hard, n_total) if not fixed_np[i]]
        if not soft_movable:
            return pos.copy()

        soft_to_row = {macro_idx: row for row, macro_idx in enumerate(soft_movable)}
        n_s = len(soft_movable)

        def get_xy(idx):
            if idx < n_total:
                return pos[idx, 0], pos[idx, 1]
            p_idx = idx - n_total
            if 0 <= p_idx < len(port_pos):
                return float(port_pos[p_idx, 0]), float(port_pos[p_idx, 1])
            return 0.0, 0.0

        # Star net model: each node connects to fixed-anchor centroid with weight w.
        # For a net with k soft + m fixed nodes, soft nodes are pulled toward
        # the fixed centroid with weight w, and soft-soft pairs attract each other.
        L_ss_x = lil_matrix((n_s, n_s), dtype=np.float64)
        L_ss_y = lil_matrix((n_s, n_s), dtype=np.float64)
        rhs_x  = np.zeros(n_s)
        rhs_y  = np.zeros(n_s)

        for ni, nodes in enumerate(nets_np):
            n_nodes = len(nodes)
            if n_nodes < 2:
                continue
            w = float(net_weights_np[ni]) if ni < len(net_weights_np) else 1.0
            # Separate soft-movable vs fixed pins
            soft_pins = [int(v) for v in nodes if soft_to_row.get(int(v)) is not None]
            fixed_pins = [int(v) for v in nodes if soft_to_row.get(int(v)) is None]
            if not soft_pins:
                continue
            edge_w = w / max(1, n_nodes - 1)
            # Fixed centroid contribution
            if fixed_pins:
                fx = np.mean([get_xy(v)[0] for v in fixed_pins])
                fy = np.mean([get_xy(v)[1] for v in fixed_pins])
                anchor_w = edge_w * len(fixed_pins)
                for sp in soft_pins:
                    r = soft_to_row[sp]
                    L_ss_x[r, r] += anchor_w
                    L_ss_y[r, r] += anchor_w
                    rhs_x[r] += anchor_w * fx
                    rhs_y[r] += anchor_w * fy
            # Soft-soft: star model from first soft pin to others — O(n) not O(n²)
            if len(soft_pins) > 1:
                ra = soft_to_row[soft_pins[0]]
                for bi in range(1, len(soft_pins)):
                    rb = soft_to_row[soft_pins[bi]]
                    L_ss_x[ra, ra] += edge_w; L_ss_x[rb, rb] += edge_w
                    L_ss_x[ra, rb] -= edge_w; L_ss_x[rb, ra] -= edge_w
                    L_ss_y[ra, ra] += edge_w; L_ss_y[rb, rb] += edge_w
                    L_ss_y[ra, rb] -= edge_w; L_ss_y[rb, ra] -= edge_w

        eps = 1e-6
        for i in range(n_s):
            if L_ss_x[i, i] < eps: L_ss_x[i, i] = eps
            if L_ss_y[i, i] < eps: L_ss_y[i, i] = eps

        L_x = L_ss_x.tocsr()
        L_y = L_ss_y.tocsr()
        x0_x = np.array([pos[i, 0] for i in soft_movable])
        x0_y = np.array([pos[i, 1] for i in soft_movable])

        sol_x, _ = sp_cg(L_x, rhs_x, x0=x0_x, maxiter=200, atol=1e-5)
        sol_y, _ = sp_cg(L_y, rhs_y, x0=x0_y, maxiter=200, atol=1e-5)

        result = pos.copy()
        for row_idx, macro_idx in enumerate(soft_movable):
            w_i, h_i = sizes_np[macro_idx, 0], sizes_np[macro_idx, 1]
            result[macro_idx, 0] = float(np.clip(sol_x[row_idx], w_i/2, canvas_w - w_i/2))
            result[macro_idx, 1] = float(np.clip(sol_y[row_idx], h_i/2, canvas_h - h_i/2))
        return result

    # ── Fast congestion engine ────────────────────────────────────────────────

    def _build_fast_engine(self, nets_np, net_weights_np, plc):
        """Precompute 2-pin star decomposition for vectorised congestion."""
        try:
            hpm    = float(plc.hroutes_per_micron)
            vpm    = float(plc.vroutes_per_micron)
            halloc = float(plc.hrouting_alloc)
            valloc = float(plc.vrouting_alloc)
            smooth = int(plc.smooth_range)
        except Exception:
            return None

        src_list = []; snk_list = []; w_list = []
        for ni, nodes in enumerate(nets_np):
            if len(nodes) < 2:
                continue
            w  = float(net_weights_np[ni]) if ni < len(net_weights_np) else 1.0
            s  = int(nodes[0])
            for k in range(1, len(nodes)):
                src_list.append(s)
                snk_list.append(int(nodes[k]))
                w_list.append(w)

        return {
            'hpm': hpm, 'vpm': vpm, 'halloc': halloc, 'valloc': valloc,
            'smooth': smooth,
            'src': np.array(src_list, dtype=np.int32),
            'snk': np.array(snk_list, dtype=np.int32),
            'w':   np.array(w_list,   dtype=np.float64),
        }

    def _fast_cong(self, all_pos, num_hard, sizes, eng, grid_rows, grid_cols, canvas_w, canvas_h):
        """Vectorised routing congestion using diff-array + cumsum (~10ms)."""
        gw = canvas_w / grid_cols
        gh = canvas_h / grid_rows
        # Match oracle: grid_h_routes = grid_height * hroutes_per_micron
        #               grid_v_routes = grid_width  * vroutes_per_micron
        h_cap = gh * eng['hpm']
        v_cap = gw * eng['vpm']
        if h_cap < 1e-9 or v_cap < 1e-9:
            return 0.0

        px = np.clip((all_pos[:, 0] / gw).astype(np.int32), 0, grid_cols - 1)
        py = np.clip((all_pos[:, 1] / gh).astype(np.int32), 0, grid_rows - 1)

        src = eng['src']; snk = eng['snk']; w = eng['w']
        sr = py[src]; sc = px[src]
        kr = py[snk]; kc = px[snk]
        col_lo = np.minimum(sc, kc)
        col_hi = np.maximum(sc, kc)
        row_lo = np.minimum(sr, kr)
        row_hi = np.maximum(sr, kr)

        # H routing: at source row, cols [col_lo, col_hi)
        H_diff = np.zeros((grid_rows, grid_cols + 1))
        np.add.at(H_diff, (sr, col_lo), w)
        np.add.at(H_diff, (sr, col_hi), -w)
        H = H_diff[:, :-1].cumsum(axis=1)

        # V routing: at sink col, rows [row_lo, row_hi)
        V_diff = np.zeros((grid_rows + 1, grid_cols))
        np.add.at(V_diff, (row_lo, kc), w)
        np.add.at(V_diff, (row_hi, kc), -w)
        V = V_diff[:-1, :].cumsum(axis=0)

        # Macro routing congestion (reduces available capacity)
        halloc = eng['halloc']; valloc = eng['valloc']
        for i in range(num_hard):
            xi = all_pos[i, 0]; yi = all_pos[i, 1]
            wi = sizes[i, 0];   hi = sizes[i, 1]
            lxm = xi - wi/2; rxm = xi + wi/2
            lym = yi - hi/2; rym = yi + hi/2
            c0 = max(0, int(lxm / gw)); c1 = min(grid_cols-1, int(rxm / gw))
            r0 = max(0, int(lym / gh)); r1 = min(grid_rows-1, int(rym / gh))
            for r in range(r0, r1+1):
                for c in range(c0, c1+1):
                    ox = max(0., min(rxm, (c+1)*gw) - max(lxm, c*gw))
                    oy = max(0., min(rym, (r+1)*gh) - max(lym, r*gh))
                    V[r, c] += ox * valloc
                    H[r, c] += oy * halloc

        H /= h_cap
        V /= v_cap

        smooth = eng['smooth']
        if smooth > 0:
            try:
                from scipy.ndimage import uniform_filter1d
                V = uniform_filter1d(V, size=2*smooth+1, axis=1, mode='reflect')
                H = uniform_filter1d(H, size=2*smooth+1, axis=0, mode='reflect')
            except ImportError:
                pass

        combined = (H + V).ravel()
        top_k = max(1, int(len(combined) * 0.05))
        return float(np.partition(combined, -top_k)[-top_k:].mean())

    def _fast_cong_grid(self, all_pos, num_hard, sizes, eng, grid_rows, grid_cols, canvas_w, canvas_h):
        """Returns the full (grid_rows × grid_cols) normalized congestion grid."""
        gw = canvas_w / grid_cols
        gh = canvas_h / grid_rows
        h_cap = gh * eng['hpm']
        v_cap = gw * eng['vpm']
        if h_cap < 1e-9 or v_cap < 1e-9:
            return np.zeros((grid_rows, grid_cols))
        px = np.clip((all_pos[:, 0] / gw).astype(np.int32), 0, grid_cols - 1)
        py = np.clip((all_pos[:, 1] / gh).astype(np.int32), 0, grid_rows - 1)
        src = eng['src']; snk = eng['snk']; w = eng['w']
        sr = py[src]; sc = px[src]; kr = py[snk]; kc = px[snk]
        col_lo = np.minimum(sc, kc); col_hi = np.maximum(sc, kc)
        row_lo = np.minimum(sr, kr); row_hi = np.maximum(sr, kr)
        H_diff = np.zeros((grid_rows, grid_cols + 1))
        np.add.at(H_diff, (sr, col_lo), w); np.add.at(H_diff, (sr, col_hi), -w)
        H = H_diff[:, :-1].cumsum(axis=1)
        V_diff = np.zeros((grid_rows + 1, grid_cols))
        np.add.at(V_diff, (row_lo, kc), w); np.add.at(V_diff, (row_hi, kc), -w)
        V = V_diff[:-1, :].cumsum(axis=0)
        halloc = eng['halloc']; valloc = eng['valloc']
        for i in range(num_hard):
            xi = all_pos[i, 0]; yi = all_pos[i, 1]
            wi = sizes[i, 0]; hi = sizes[i, 1]
            lxm = xi - wi/2; rxm = xi + wi/2; lym = yi - hi/2; rym = yi + hi/2
            c0 = max(0, int(lxm / gw)); c1 = min(grid_cols-1, int(rxm / gw))
            r0 = max(0, int(lym / gh)); r1 = min(grid_rows-1, int(rym / gh))
            for r in range(r0, r1+1):
                for c in range(c0, c1+1):
                    ox = max(0., min(rxm, (c+1)*gw) - max(lxm, c*gw))
                    oy = max(0., min(rym, (r+1)*gh) - max(lym, r*gh))
                    V[r, c] += ox * valloc; H[r, c] += oy * halloc
        H /= h_cap; V /= v_cap
        return H + V

    def _run_fast_sa_cong(self, pos, movable_idx, sizes, port_pos, canvas_w, canvas_h,
                          num_hard, fixed_np, fast_cost_fn, deadline, fast_eng=None,
                          grid_rows=1, grid_cols=1):
        """Non-incremental SA using full proxy cost (WL+density+cong) for Phase 1b.
        Every 40 steps a congestion-hotspot escape move is attempted: find the most
        congested grid cell, move the nearest movable macro away from it."""
        cur_pos   = pos.copy()
        cur_cost  = fast_cost_fn(cur_pos)
        best_pos  = cur_pos.copy()
        best_cost = cur_cost
        n_mov     = len(movable_idx)
        t_start   = time.time()
        step_scale = (canvas_w + canvas_h) * 0.06
        step      = 0
        hotspot_cell = None  # (cx, cy) of last congestion hotspot

        while time.time() < deadline:
            elapsed = time.time() - t_start
            frac    = min(1.0, elapsed / max(1.0, deadline - t_start))
            T = T_CONG_START * (T_CONG_END / T_CONG_START) ** frac

            # Periodically refresh congestion hotspot
            if fast_eng is not None and step % 40 == 0:
                ap = self._all_pos(cur_pos, port_pos)
                cg = self._fast_cong_grid(ap, num_hard, sizes, fast_eng,
                                          grid_rows, grid_cols, canvas_w, canvas_h)
                r_hot, c_hot = np.unravel_index(np.argmax(cg), cg.shape)
                gw = canvas_w / grid_cols; gh = canvas_h / grid_rows
                hotspot_cell = ((c_hot + 0.5) * gw, (r_hot + 0.5) * gh)

            step += 1
            cand = cur_pos.copy()
            move_r = random.random()

            # 15% congestion hotspot escape: move nearest macro away from hotspot
            if move_r < 0.15 and hotspot_cell is not None:
                hx, hy = hotspot_cell
                dists = [abs(cur_pos[idx, 0] - hx) + abs(cur_pos[idx, 1] - hy)
                         for idx in movable_idx]
                idx = movable_idx[int(np.argmin(dists))]
                w, h = sizes[idx, 0], sizes[idx, 1]
                dx = cur_pos[idx, 0] - hx; dy = cur_pos[idx, 1] - hy
                push = (canvas_w + canvas_h) * 0.08
                mag  = max(1e-6, abs(dx) + abs(dy))
                cand[idx, 0] = float(np.clip(cur_pos[idx, 0] + push * dx / mag + np.random.normal(0, step_scale * 0.3), w/2, canvas_w-w/2))
                cand[idx, 1] = float(np.clip(cur_pos[idx, 1] + push * dy / mag + np.random.normal(0, step_scale * 0.3), h/2, canvas_h-h/2))
            elif move_r < 0.35:
                idx  = movable_idx[random.randrange(n_mov)]
                idx2 = movable_idx[random.randrange(n_mov)]
                if idx == idx2: continue
                w,  h  = sizes[idx,  0], sizes[idx,  1]
                w2, h2 = sizes[idx2, 0], sizes[idx2, 1]
                p1, p2 = cur_pos[idx].copy(), cur_pos[idx2].copy()
                if (p2[0] < w/2 or p2[0] > canvas_w-w/2 or p2[1] < h/2 or p2[1] > canvas_h-h/2 or
                    p1[0] < w2/2 or p1[0] > canvas_w-w2/2 or p1[1] < h2/2 or p1[1] > canvas_h-h2/2):
                    continue
                cand[idx] = p2; cand[idx2] = p1
            else:
                idx = movable_idx[random.randrange(n_mov)]
                w, h = sizes[idx, 0], sizes[idx, 1]
                scale = step_scale * max(0.1, 1.0 - frac * 0.8)
                cand[idx, 0] = float(np.clip(cur_pos[idx, 0] + np.random.normal(0, scale), w/2, canvas_w-w/2))
                cand[idx, 1] = float(np.clip(cur_pos[idx, 1] + np.random.normal(0, scale), h/2, canvas_h-h/2))

            cost  = fast_cost_fn(cand)
            delta = cost - cur_cost
            if delta < 0 or (T > 1e-9 and random.random() < math.exp(max(-30.0, -delta / T))):
                cur_pos  = cand
                cur_cost = cost
                if cost < best_cost:
                    best_cost = cost
                    best_pos  = cand.copy()

        best_pos = self._resolve(best_pos, num_hard, sizes, canvas_w, canvas_h, fixed_np, max_iter=500)
        return best_pos

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _local_resolve(self, pos, moved_indices, num_hard, sizes, canvas_w, canvas_h, fixed, max_iter=120):
        """Resolve overlaps for `moved_indices` macros vs all others.
        O(k×n) per pass. Falls back to full resolve if residual overlaps remain."""
        p = pos.copy(); GAP = 0.01
        for _ in range(max_iter):
            moved = False
            for i in moved_indices:
                xi, yi = p[i, 0], p[i, 1]; wi, hi = sizes[i, 0], sizes[i, 1]
                for j in range(num_hard):
                    if j == i: continue
                    xj, yj = p[j, 0], p[j, 1]; wj, hj = sizes[j, 0], sizes[j, 1]
                    ox = (wi+wj)/2 - abs(xi-xj); oy = (hi+hj)/2 - abs(yi-yj)
                    if ox <= 0 or oy <= 0: continue
                    if ox < oy:
                        push = (ox+GAP)/2; dx_i = -push if xi < xj else +push
                        if not fixed[i]:
                            p[i,0] = np.clip(xi + dx_i, wi/2, canvas_w-wi/2)
                    else:
                        push = (oy+GAP)/2; dy_i = -push if yi < yj else +push
                        if not fixed[i]:
                            p[i,1] = np.clip(yi + dy_i, hi/2, canvas_h-hi/2)
                    xi, yi = p[i, 0], p[i, 1]
                    moved = True
            if not moved: break
        # Safety: if moved macros still overlap anything, do a quick full resolve pass
        has_residual = False
        for i in moved_indices:
            xi, yi = p[i, 0], p[i, 1]; wi, hi = sizes[i, 0], sizes[i, 1]
            for j in range(num_hard):
                if j == i: continue
                xj, yj = p[j, 0], p[j, 1]; wj, hj = sizes[j, 0], sizes[j, 1]
                if (wi+wj)/2 - abs(xi-xj) > 0 and (hi+hj)/2 - abs(yi-yj) > 0:
                    has_residual = True; break
            if has_residual: break
        if has_residual:
            p = self._resolve(p, num_hard, sizes, canvas_w, canvas_h, fixed, max_iter=100)
        return p

    def _resolve(self, pos, num_hard, sizes, canvas_w, canvas_h, fixed, max_iter=500, min_iter=0):
        # Cap iterations: target <0.5s per call. n_pairs * max_iter * 0.2µs < 500ms
        n_pairs = max(1, num_hard * (num_hard - 1) // 2)
        time_cap = max(3, int(250_000 // n_pairs))
        max_iter = max(min_iter, min(max_iter, time_cap))
        p=pos.copy(); GAP=0.01
        for _ in range(max_iter):
            moved=False
            for i in range(num_hard):
                for j in range(i+1,num_hard):
                    xi,yi=p[i,0],p[i,1]; xj,yj=p[j,0],p[j,1]
                    wi,hi=sizes[i,0],sizes[i,1]; wj,hj=sizes[j,0],sizes[j,1]
                    ox=(wi+wj)/2-abs(xi-xj); oy=(hi+hj)/2-abs(yi-yj)
                    if ox<=0 or oy<=0: continue
                    if ox<oy:
                        push=(ox+GAP)/2; dx_i=-push if xi<xj else +push
                        dx_j=-dx_i; dy_i=dy_j=0.0
                    else:
                        push=(oy+GAP)/2; dy_i=-push if yi<yj else +push
                        dy_j=-dy_i; dx_i=dx_j=0.0
                    if not fixed[i]:
                        p[i,0]=np.clip(xi+dx_i,wi/2,canvas_w-wi/2)
                        p[i,1]=np.clip(yi+dy_i,hi/2,canvas_h-hi/2)
                    if not fixed[j]:
                        p[j,0]=np.clip(xj+dx_j,wj/2,canvas_w-wj/2)
                        p[j,1]=np.clip(yj+dy_j,hj/2,canvas_h-hj/2)
                    moved=True
            if not moved: break
        return p

    def _resolve_fully(self, pos, num_hard, sizes, canvas_w, canvas_h, fixed, max_rounds=20):
        p=pos.copy()
        for _ in range(max_rounds):
            # min_iter=30 bypasses time cap for large benchmarks (ibm17: cap=3, but we need more)
            p=self._resolve(p,num_hard,sizes,canvas_w,canvas_h,fixed,max_iter=1000,min_iter=30)
            if self._total_overlap(p,num_hard,sizes)<1e-6: break
            for i in range(num_hard):
                if fixed[i]: continue
                wi,hi=sizes[i,0],sizes[i,1]
                p[i,0]=np.clip(p[i,0]+np.random.uniform(-wi*0.1,wi*0.1),wi/2,canvas_w-wi/2)
                p[i,1]=np.clip(p[i,1]+np.random.uniform(-hi*0.1,hi*0.1),hi/2,canvas_h-hi/2)
        return p

    def _try_load_plc(self, benchmark):
        try:
            from macro_place._plc import PlacementCost
            for d in [
                f"external/MacroPlacement/Testcases/ICCAD04/{benchmark.name}",
                f"external/MacroPlacement/Flows/NanGate45/{benchmark.name}/netlist/output_CT_Grouping",
            ]:
                netlist=f"{d}/netlist.pb.txt"
                if os.path.exists(netlist):
                    plc=PlacementCost(netlist.replace("\\","/"))
                    plc_f=f"{d}/initial.plc"
                    if os.path.exists(plc_f):
                        plc.restore_placement(plc_f,ifInital=True,ifReadComment=True)
                    return plc
        except Exception: pass
        return None

    def _true_cost(self, pos, benchmark, plc, return_components=False):
        if plc is None:
            return (float('inf'), None) if return_components else float('inf')
        try:
            from macro_place.objective import compute_proxy_cost
            r = compute_proxy_cost(torch.from_numpy(pos).float(), benchmark, plc)
            if return_components:
                return float(r['proxy_cost']), r
            return float(r['proxy_cost'])
        except Exception:
            return (float('inf'), None) if return_components else float('inf')

    def _all_pos(self, pos, port_pos):
        return np.vstack([pos,port_pos]) if port_pos.shape[0]>0 else pos

    def _spectral_start(self, pos_init, movable_idx, sizes, nets_np, net_weights_np,
                         num_hard, canvas_w, canvas_h, fixed_np, deadline):
        """Perturb along the 1st nontrivial eigenvector of the macro Laplacian.

        The first spectral mode captures the primary bisection of the netlist —
        the structurally most-different direction from initial.plc. Sampling along
        this direction explores topology changes that are connectivity-coherent,
        unlike pure Gaussian noise.
        Returns None if scipy unavailable or matrix build too slow.
        """
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
        except ImportError:
            return None

        n_mov = len(movable_idx)
        if n_mov < 2 or time.time() > deadline - 10:
            return None

        # Build sparse Laplacian over movable hard macros
        idx_map = {idx: k for k, idx in enumerate(movable_idx)}
        rows, cols, vals = [], [], []
        for ni, nodes in enumerate(nets_np):
            hards = [idx_map[int(v)] for v in nodes if idx_map.get(int(v)) is not None]
            if len(hards) < 2:
                continue
            w = float(net_weights_np[ni]) if ni < len(net_weights_np) else 1.0
            ew = w / max(1, len(hards) - 1)
            for i in range(len(hards)):
                for j in range(i + 1, len(hards)):
                    rows += [hards[i], hards[j], hards[i], hards[j]]
                    cols += [hards[j], hards[i], hards[i], hards[j]]
                    vals += [-ew, -ew, ew, ew]

        if not rows:
            return None

        L = csr_matrix((vals, (rows, cols)), shape=(n_mov, n_mov))
        try:
            # Use a random mode from the top-4 non-trivial eigenvectors for diversity.
            n_modes = min(4, n_mov - 1)
            _, vecs = eigsh(L, k=n_modes + 1, which='SM', tol=1e-3, maxiter=500)
            mode_idx = random.randint(1, n_modes)
            v1 = np.real(vecs[:, mode_idx])
            v1 = v1 / (np.std(v1) + 1e-9)
        except Exception:
            return None

        amplitude = (canvas_w + canvas_h) * 0.20
        sign = 1.0 if random.random() < 0.5 else -1.0
        axis = random.randint(0, 1)  # perturb x or y
        p = pos_init.copy()
        for k, idx in enumerate(movable_idx):
            if k >= len(v1):
                break
            w, h = sizes[idx, 0], sizes[idx, 1]
            delta = sign * amplitude * v1[k]
            if axis == 0:
                p[idx, 0] = np.clip(pos_init[idx, 0] + delta, w / 2, canvas_w - w / 2)
            else:
                p[idx, 1] = np.clip(pos_init[idx, 1] + delta, h / 2, canvas_h - h / 2)
        return p

    def _spectral_cem(self, pos_init, movable_idx, sizes, nets_np, net_weights_np,
                       num_hard, canvas_w, canvas_h, fixed_np,
                       n_pop=2, cem_budget=60.0, cheap_cost_fn=None):
        """Multi-spectral CEM using eigenvectors 2-8 of the macro Laplacian.

        Each eigenvector is an orthogonal Fourier mode of the netlist graph —
        a different frequency of connectivity structure. Seeds along mode k
        place macros according to that topological decomposition, exploring
        regions of the landscape that Fiedler-only seeding cannot reach.

        Two strategies per mode:
          (a) Spectral-sorted: rank macros by mode value, assign uniform positions
              along that axis (pure topological layout, no WL bias).
          (b) Large-amplitude perturbation: shift macros along the mode vector
              by ±15-35% of canvas diagonal (landscape escaping).
        """
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
        except ImportError:
            return [pos_init.copy()]

        t0_cem = time.time()
        n_mov = len(movable_idx)
        if n_mov < 3:
            return [pos_init.copy()]

        # Build sparse Laplacian over movable hard macros
        idx_map = {idx: k for k, idx in enumerate(movable_idx)}
        rows, cols, vals = [], [], []
        for ni, nodes in enumerate(nets_np):
            hards = [idx_map[int(v)] for v in nodes if idx_map.get(int(v)) is not None]
            if len(hards) < 2:
                continue
            w = float(net_weights_np[ni]) if ni < len(net_weights_np) else 1.0
            ew = w / max(1, len(hards) - 1)
            for i in range(len(hards)):
                for j in range(i + 1, len(hards)):
                    rows += [hards[i], hards[j], hards[i], hards[j]]
                    cols += [hards[j], hards[i], hards[i], hards[j]]
                    vals += [-ew, -ew, ew, ew]

        if not rows:
            return [pos_init.copy()]

        L = csr_matrix((vals, (rows, cols)), shape=(n_mov, n_mov))
        # Request up to 8 non-trivial eigenvectors (modes 1-8 of graph spectrum).
        # More modes = more orthogonal topological regions explored.
        n_modes = min(8, n_mov - 1)
        try:
            _, vecs = eigsh(L, k=n_modes + 1, which='SM', tol=1e-3, maxiter=1000)
            modes = []
            for i in range(1, n_modes + 1):
                v = np.real(vecs[:, i])
                modes.append(v / (np.std(v) + 1e-9))
        except Exception:
            return [pos_init.copy()]

        if time.time() - t0_cem > cem_budget - 3:
            return [pos_init.copy()]

        base_cost = cheap_cost_fn(pos_init) if cheap_cost_fn else float('inf')
        candidates = [(base_cost, pos_init.copy())]
        canvas_diag = canvas_w + canvas_h

        # Strategy A: Spectral-sorted seeding for all mode pairs (vi→x, vj→y).
        # Each pair represents a different "spectral coordinate frame" — macros
        # are laid out according to their graph-frequency coordinates in that frame.
        # This is the strongest diversity mechanism: seeds are structurally orthogonal.
        for i in range(len(modes)):
            for j in range(len(modes)):
                if time.time() - t0_cem > cem_budget * 0.65:
                    break
                p = self._spectral_sorted_placement(
                    pos_init, movable_idx, sizes, modes[i], modes[j],
                    num_hard, canvas_w, canvas_h, fixed_np)
                p = self._resolve(p, num_hard, sizes, canvas_w, canvas_h, fixed_np)
                c = cheap_cost_fn(p) if cheap_cost_fn else float('inf')
                candidates.append((c, p))

        # Strategy B: Large-amplitude perturbations along each mode.
        # Amplitudes ±0.15 and ±0.30 give near and far explorations.
        for mi, mode in enumerate(modes[:4]):
            for amp in [-0.30, -0.15, 0.15, 0.30]:
                if time.time() - t0_cem > cem_budget - 1:
                    break
                p = pos_init.copy()
                for k, idx in enumerate(movable_idx):
                    if k >= len(mode):
                        break
                    w_s, h_s = sizes[idx, 0], sizes[idx, 1]
                    delta = canvas_diag * amp * mode[k]
                    if mi % 2 == 0:
                        p[idx, 0] = np.clip(pos_init[idx, 0] + delta, w_s / 2, canvas_w - w_s / 2)
                    else:
                        p[idx, 1] = np.clip(pos_init[idx, 1] + delta, h_s / 2, canvas_h - h_s / 2)
                p = self._resolve(p, num_hard, sizes, canvas_w, canvas_h, fixed_np)
                c = cheap_cost_fn(p) if cheap_cost_fn else float('inf')
                candidates.append((c, p))

        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates[:n_pop]]

    def _random_start(self, pos_init, movable_idx, sizes, canvas_w, canvas_h, fixed_np):
        """Quadrant-shuffle placement: divide canvas into 4 quadrants, shuffle which
        macro group goes where. Gives topology diversity while staying partially legal.
        Each macro is placed uniformly within its destination quadrant."""
        p = pos_init.copy()
        n_mov = len(movable_idx)
        if n_mov == 0:
            return p

        cx, cy = canvas_w / 2, canvas_h / 2
        # Assign movable macros to their current quadrant
        quads: list[list[int]] = [[], [], [], []]
        for idx in movable_idx:
            x, y = pos_init[idx, 0], pos_init[idx, 1]
            q = (1 if x > cx else 0) + (2 if y > cy else 0)
            quads[q].append(idx)

        # Shuffle which source quadrant maps to which destination
        quad_order = list(range(4))
        random.shuffle(quad_order)

        quad_bounds = [
            (0.0, 0.0, cx, cy),
            (cx, 0.0, canvas_w, cy),
            (0.0, cy, cx, canvas_h),
            (cx, cy, canvas_w, canvas_h),
        ]

        for src_q, dst_q in enumerate(quad_order):
            xlo, ylo, xhi, yhi = quad_bounds[dst_q]
            for idx in quads[src_q]:
                w, h = sizes[idx, 0], sizes[idx, 1]
                xl = xlo + w / 2; xr = xhi - w / 2
                yl = ylo + h / 2; yr = yhi - h / 2
                if xl >= xr: nx = (xlo + xhi) / 2
                else: nx = np.random.uniform(xl, xr)
                if yl >= yr: ny = (ylo + yhi) / 2
                else: ny = np.random.uniform(yl, yr)
                p[idx, 0] = np.clip(nx, w / 2, canvas_w - w / 2)
                p[idx, 1] = np.clip(ny, h / 2, canvas_h - h / 2)
        return p

    def _spectral_sorted_placement(self, pos_init, movable_idx, sizes, vx, vy,
                                    num_hard, canvas_w, canvas_h, fixed_np):
        """Rank-based spectral placement: sort macros by eigenvector rank, assign
        uniformly-spaced x from vx-rank and y from vy-rank. Preserves spectral
        connectivity ordering without density explosion (rank = uniform spread)."""
        p = pos_init.copy()
        n_mov = len(movable_idx)
        if n_mov < 3:
            return p
        vx_order = np.argsort(vx[:n_mov])
        vy_order = np.argsort(vy[:n_mov])
        for rank, k in enumerate(vx_order):
            idx = movable_idx[k]
            w_s = sizes[idx, 0]
            x = (rank + 0.5) / n_mov * canvas_w
            p[idx, 0] = float(np.clip(x, w_s / 2, canvas_w - w_s / 2))
        for rank, k in enumerate(vy_order):
            idx = movable_idx[k]
            h_s = sizes[idx, 1]
            y = (rank + 0.5) / n_mov * canvas_h
            p[idx, 1] = float(np.clip(y, h_s / 2, canvas_h - h_s / 2))
        return p

    def _perturb(self, pos, movable_idx, sizes, canvas_w, canvas_h, noise):
        p=pos.copy()
        if noise<=0: return p
        for idx in movable_idx:
            w,h=sizes[idx,0],sizes[idx,1]
            p[idx,0]=np.clip(p[idx,0]+np.random.normal(0,noise),w/2,canvas_w-w/2)
            p[idx,1]=np.clip(p[idx,1]+np.random.normal(0,noise),h/2,canvas_h-h/2)
        return p

    def _spatial_crossover(self, pa, pb, movable_idx, canvas_w, canvas_h):
        """Spatial block crossover: take macros in a random half of the canvas
        from parent A, macros in the other half from parent B.
        Preserves spatial clusters from each parent."""
        child = pa.copy()
        # Random split: horizontal or vertical at a random position
        if random.random() < 0.5:
            # Vertical split: x < split_x from pa, x >= split_x from pb
            split = canvas_w * (0.3 + random.random() * 0.4)  # 30%-70% of canvas
            for idx in movable_idx:
                if pb[idx, 0] < split:
                    child[idx] = pb[idx].copy()
        else:
            # Horizontal split: y < split_y from pa, y >= split_y from pb
            split = canvas_h * (0.3 + random.random() * 0.4)
            for idx in movable_idx:
                if pb[idx, 1] < split:
                    child[idx] = pb[idx].copy()
        return child

    @staticmethod
    def _hpwl_vec(ap, safe_nnp, nnmask):
        all_x=ap[safe_nnp,0]; all_y=ap[safe_nnp,1]
        INF=1e15
        mx=np.where(nnmask,all_x,-INF); mn=np.where(nnmask,all_x,INF)
        ry=np.where(nnmask,all_y,-INF); ly=np.where(nnmask,all_y,INF)
        return float((mx.max(1)-mn.min(1)+ry.max(1)-ly.min(1)).sum())

    def _density_grid(self,pos,num_hard,sizes,grid_rows,grid_cols,canvas_w,canvas_h):
        cell_w=canvas_w/grid_cols; cell_h=canvas_h/grid_rows; cell_a=cell_w*cell_h
        dens=np.zeros((grid_rows,grid_cols))
        for i in range(num_hard):
            xi,yi=pos[i,0],pos[i,1]; wi,hi=sizes[i,0],sizes[i,1]
            lx_m=xi-wi/2; rx_m=xi+wi/2; ly_m=yi-hi/2; ry_m=yi+hi/2
            c_lo=max(0,int(lx_m/cell_w)); c_hi=min(grid_cols-1,int(rx_m/cell_w))
            r_lo=max(0,int(ly_m/cell_h)); r_hi=min(grid_rows-1,int(ry_m/cell_h))
            for r in range(r_lo,r_hi+1):
                for c in range(c_lo,c_hi+1):
                    ox=max(0.,min(rx_m,(c+1)*cell_w)-max(lx_m,c*cell_w))
                    oy=max(0.,min(ry_m,(r+1)*cell_h)-max(ly_m,r*cell_h))
                    dens[r,c]+=(ox*oy)/cell_a
        return dens

    def _add_macro_to_dens(self,dens,xi,yi,wi,hi,cell_w,cell_h,grid_rows,grid_cols,cell_a,sign=1.):
        lx_m=xi-wi/2; rx_m=xi+wi/2; ly_m=yi-hi/2; ry_m=yi+hi/2
        c_lo=max(0,int(lx_m/cell_w)); c_hi=min(grid_cols-1,int(rx_m/cell_w))
        r_lo=max(0,int(ly_m/cell_h)); r_hi=min(grid_rows-1,int(ry_m/cell_h))
        for r in range(r_lo,r_hi+1):
            for c in range(c_lo,c_hi+1):
                ox=max(0.,min(rx_m,(c+1)*cell_w)-max(lx_m,c*cell_w))
                oy=max(0.,min(ry_m,(r+1)*cell_h)-max(ly_m,r*cell_h))
                dens[r,c]+=sign*(ox*oy)/cell_a

    @staticmethod
    def _top_k_mean(grid, frac):
        flat=grid.ravel(); top_k=max(1,int(len(flat)*frac))
        return float(np.partition(flat,-top_k)[-top_k:].mean())

    def _total_overlap(self,pos,num_hard,sizes):
        lx=pos[:num_hard,0]-sizes[:num_hard,0]/2; rx=pos[:num_hard,0]+sizes[:num_hard,0]/2
        ly=pos[:num_hard,1]-sizes[:num_hard,1]/2; ry=pos[:num_hard,1]+sizes[:num_hard,1]/2
        total=0.
        for i in range(num_hard):
            ox=np.maximum(0.,np.minimum(rx[i],rx)-np.maximum(lx[i],lx))
            oy=np.maximum(0.,np.minimum(ry[i],ry)-np.maximum(ly[i],ly))
            row=ox*oy; total+=row.sum()-row[i]
        return total/2
