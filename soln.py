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

        pos_init = benchmark.macro_positions.numpy().astype(np.float64).copy()
        pos_init = self._resolve(pos_init, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)

        plc = self._try_load_plc(benchmark)

        def cheap_cost(p):
            ap   = self._all_pos(p, port_pos)
            wl   = self._hpwl_vec(ap, safe_nnp, nnmask) / hpwl_norm
            dens = self._density_grid(p, num_hard, sizes_np, grid_rows, grid_cols, canvas_w, canvas_h)
            den  = self._top_k_mean(dens, 0.10)
            ov   = self._total_overlap(p, num_hard, sizes_np)
            return wl + 0.5 * den + OVERLAP_WEIGHT * ov

        # ── Phase 1: Cheap SA warm-up (10% of budget) ─────────────────────────
        # Fast WL+density SA from initial.plc and one random-perturbed start.
        # Goal: find a better starting point for the oracle SA phase.
        sa_end = t0 + (TIME_BUDGET - 30) * 0.10

        islands = []
        for i in range(N_ISLANDS):
            p = self._perturb(pos_init, movable_idx, sizes_np, canvas_w, canvas_h,
                              0.0 if i == 0 else 0.15)
            p = self._resolve(p, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
            islands.append(p)

        costs    = [cheap_cost(p) for p in islands]
        best_cheap_pos  = islands[int(np.argmin(costs))].copy()
        best_cheap_cost = float(min(costs))

        while time.time() < sa_end:
            elapsed_frac = min(1.0, (time.time() - t0) / max(1.0, (TIME_BUDGET - 30) * 0.10))
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

        # ── Phase 2: Oracle SA with temperature (80% of budget) ───────────────
        # Use PlacementCost as the exact cost function in a SA loop.
        # Temperature enables escaping initial.plc's local minimum.
        # Net-centroid-guided moves provide smarter WL-aware proposals.
        # Hard deadline: 10s before TIME_BUDGET to leave room for final resolution.
        hard_deadline = t0 + TIME_BUDGET - 10
        oracle_end    = min(hard_deadline - 5, t0 + (TIME_BUDGET - 30) * 0.90)
        best_pos      = best_cheap_pos.copy()

        if plc is not None:
            # Evaluate pos_init as the reference baseline.
            init_r    = self._resolve_fully(pos_init, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
            init_cost = self._true_cost(init_r, benchmark, plc)

            # Collect starting points for multi-start oracle SA:
            # 1. Best of cheap SA vs pos_init (connectivity-aware)
            # 2. Moderately perturbed pos_init (30% noise) — diversifies search
            # 3. Heavily perturbed pos_init (60% noise) — explores distant basins
            # Each start gets 1/3 of oracle SA budget; global best is tracked.
            starts = [init_r]
            starts_cost = [init_cost]
            start_r_cheap = self._resolve_fully(best_cheap_pos, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
            sc_cheap = self._true_cost(start_r_cheap, benchmark, plc)
            if sc_cheap < init_cost:
                starts[0] = start_r_cheap
                starts_cost[0] = sc_cheap

            # Add perturbed starts for diversity.
            # Use _resolve (not _resolve_fully) to avoid 40s+ resolve time on tangled positions.
            # Set initial cost = inf so oracle SA accepts any improvement freely from the start.
            for noise in [0.30, 0.60]:
                p_pert = self._perturb(pos_init, movable_idx, sizes_np, canvas_w, canvas_h, noise)
                p_pert = self._resolve(p_pert, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                starts.append(p_pert)
                starts_cost.append(float('inf'))  # oracle SA will accept any finite cost

            best_global_cost = min(starts_cost)
            best_pos = starts[int(np.argmin(starts_cost))].copy()

            # Divide oracle SA budget equally among starts
            n_starts  = len(starts)
            slot_dur  = (oracle_end - time.time()) / n_starts
            for si, (s_pos, s_cost) in enumerate(zip(starts, starts_cost)):
                slot_end = time.time() + slot_dur if si < n_starts - 1 else oracle_end
                result_s = self._plc_oracle_sa(
                    s_pos, movable_idx, sizes_np, port_pos, canvas_w, canvas_h,
                    num_hard, fixed_np, benchmark, plc, slot_end, s_cost,
                    macro_to_nets, nets_np,
                )
                # Evaluate result in resolve_fully'd form to get accurate cost
                result_sr = self._resolve_fully(result_s, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                cost_s = self._true_cost(result_sr, benchmark, plc)
                if cost_s < best_global_cost:
                    best_global_cost = cost_s; best_pos = result_sr.copy()

        # ── Phase 3: Final selection — oracle result vs pos_init ──────────────
        # Always resolve_fully best_pos first so phase 3 comparison is valid.
        best_pos = self._resolve_fully(best_pos, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
        if plc is not None and time.time() < hard_deadline - 3:
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
        best_pos32 = self._resolve(best_pos32, num_hard, sizes_np, canvas_w, canvas_h, fixed_np, max_iter=200)
        result[:] = torch.from_numpy(best_pos32).float()
        return result

    # ── Oracle SA with temperature ────────────────────────────────────────────

    def _plc_oracle_sa(self, pos, movable_idx, sizes, port_pos, canvas_w, canvas_h,
                       num_hard, fixed, benchmark, plc, deadline, init_cost,
                       macro_to_nets, nets_np):
        """
        SA using exact PlacementCost as the objective function.

        With ~200 oracle evals in budget, uses a geometric temperature schedule
        to escape local minima (especially initial.plc's basin).

        Move types (probability-weighted):
        - 25% swap: exchange positions of two macros
        - 25% net-centroid: move macro toward weighted centroid of its net endpoints
                            (force-directed step — connectivity-aware)
        - 25% small Gaussian: fine-tune individual macro position
        - 15% large jump: escape basin (20% of canvas)
        - 10% cluster shuffle: permute positions of 3-5 connected macros
        """
        best_pos  = pos.copy()
        cur_pos   = pos.copy()
        best_cost = init_cost
        cur_cost  = init_cost
        n_mov     = len(movable_idx)
        step_small = (canvas_w + canvas_h) * 0.03
        step_large = (canvas_w + canvas_h) * 0.18

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

            cand = cur_pos.copy()
            move_type = random.random()

            if move_type < 0.25:
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

            elif move_type < 0.50:
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

            elif move_type < 0.75:
                # Small Gaussian: fine-tune
                idx = movable_idx[random.randrange(n_mov)]
                w, h = sizes[idx, 0], sizes[idx, 1]
                cand[idx, 0] = float(np.clip(
                    cur_pos[idx,0] + np.random.normal(0, step_small), w/2, canvas_w-w/2))
                cand[idx, 1] = float(np.clip(
                    cur_pos[idx,1] + np.random.normal(0, step_small), h/2, canvas_h-h/2))

            elif move_type < 0.90:
                # Large jump: escape basin
                idx = movable_idx[random.randrange(n_mov)]
                w, h = sizes[idx, 0], sizes[idx, 1]
                cand[idx, 0] = float(np.clip(
                    cur_pos[idx,0] + np.random.normal(0, step_large), w/2, canvas_w-w/2))
                cand[idx, 1] = float(np.clip(
                    cur_pos[idx,1] + np.random.normal(0, step_large), h/2, canvas_h-h/2))

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
            if move_type < 0.25:
                moved_set = [idx, idx2]
            elif move_type < 0.90:
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

    def _resolve(self, pos, num_hard, sizes, canvas_w, canvas_h, fixed, max_iter=500):
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
            p=self._resolve(p,num_hard,sizes,canvas_w,canvas_h,fixed,max_iter=1000)
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

    def _true_cost(self, pos, benchmark, plc):
        if plc is None: return float('inf')
        try:
            from macro_place.objective import compute_proxy_cost
            r=compute_proxy_cost(torch.from_numpy(pos).float(),benchmark,plc)
            return float(r['proxy_cost'])
        except Exception: return float('inf')

    def _all_pos(self, pos, port_pos):
        return np.vstack([pos,port_pos]) if port_pos.shape[0]>0 else pos

    def _perturb(self, pos, movable_idx, sizes, canvas_w, canvas_h, noise):
        p=pos.copy()
        if noise<=0: return p
        for idx in movable_idx:
            w,h=sizes[idx,0],sizes[idx,1]
            p[idx,0]=np.clip(p[idx,0]+np.random.normal(0,noise),w/2,canvas_w-w/2)
            p[idx,1]=np.clip(p[idx,1]+np.random.normal(0,noise),h/2,canvas_h-h/2)
        return p

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
