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
TIME_BUDGET     = int(os.environ.get("PLACE_TIME_BUDGET", 3300))

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

        def cheap_components(p):
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
            return wl, den, cong, ov

        def cheap_cost(p):
            wl, den, cong, ov = cheap_components(p)
            return wl + 0.5 * den + 0.5 * cong + OVERLAP_WEIGHT * ov

        # ── Phase 0: Multi-world topology sweep (30% of budget) ─────────────────
        # "Multiple worlds": each gradient start uses a different λ_den (density–WL
        # balance), creating topologically distinct force models from the same random
        # position. WL-dominant (low λ) → macros cluster near connections.
        # Density-dominant (high λ) → macros spread globally.
        # Spectral starts (Laplacian eigenvector–sorted positions) give connected macros
        # spatial locality → gradient converges faster from near-topology initializations.
        grad_end_total = t0 + (TIME_BUDGET - 30) * 0.30
        topo_budget    = min(300.0, max(30.0, grad_end_total - t0))

        # Scale per-start time with sqrt(n_mov/20): 4s for 20 macros, ~25s for 760
        _n_mov_factor = max(1.0, len(movable_idx) / 20.0)
        per_start_s   = max(4.0, min(60.0, 4.0 * (_n_mov_factor ** 0.5)))
        n_topo        = max(5, min(60, int(topo_budget / per_start_s)))

        # λ_den "worlds": different density–WL trade-offs → different topologies
        # Include high-λ chains (3.0, 4.0) to explore aggressively-spread configurations.
        # For high-density benchmarks (ibm17: oracle den=0.945), λ=0.25/0.70 give the
        # same basin as λ=0.5/1.0. Replacing them with 3.0/4.0 finds genuinely different
        # lower-density basins that oracle SA cannot reach from moderate-density starts.
        _world_lam_dens = [0.05, 0.20, 0.60, 1.50, 0.10, 0.40, 1.00, 2.00,
                           0.08, 0.30, 0.80, 0.15, 0.50, 1.20, 3.00, 4.00]

        topology_seeds: list = []
        topology_costs: list = []
        topology_tags:  list = []

        # Include initial.plc as baseline seed
        _p_init_r = self._resolve(pos_init.copy(), num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
        topology_seeds.append(_p_init_r)
        topology_costs.append(cheap_cost(_p_init_r))
        topology_tags.append('init.plc')

        # B2B quadratic analytical seeds: produce topologically distinct
        # WL-optimal starts independent of initial.plc's basin. These will be
        # injected as gradient-phase starting points (NOT compared raw — the
        # raw B2B output has high density before gradient spreads it).
        _b2b_pool: list = []
        _b2b_tags: list = []
        _t_b2b0 = time.time()
        try:
            _b2b_budget = min(25.0, max(5.0, len(movable_idx) * 0.05))
            # Five distinct B2B variants:
            #  0: ports+soft anchors,  center init       — standard analytical
            #  1: ports-only anchors,  center init       — strips soft-macro topology
            #  2: ports+soft anchors,  random init       — different basin
            #  3: ports+soft anchors,  corners init      — radically different topology
            #  4: ports+soft, short-net boost, random    — cong-aware variant
            _b2b_variants = [
                ('ports_soft', 'center',  None),
                ('ports_only', 'center',  None),
                ('ports_soft', 'random',  None),
                ('ports_soft', 'corners', None),
                ('ports_soft', 'random',
                    lambda ni, k: 2.0 if k <= 4 else 0.5),  # short-net emphasis
            ]
            for _bi, (_am, _im, _wfn) in enumerate(_b2b_variants):
                if time.time() - _t_b2b0 > _b2b_budget:
                    break
                np.random.seed(20260428 + _bi * 7919)
                _p_b2b = self._b2b_quadratic_seed(
                    pos_init, movable_idx, num_hard, nets_np, net_weights_np,
                    port_pos, sizes_np, canvas_w, canvas_h, fixed_np,
                    n_iters=4, anchor_mode=_am, init_mode=_im, net_weight_fn=_wfn,
                )
                _p_b2b = self._resolve(_p_b2b, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                _b2b_pool.append(_p_b2b)
                _b2b_tags.append(f'b2b-{_bi}-{_am}-{_im}')
        except Exception:
            _b2b_pool = []
            _b2b_tags = []


        # GPU-batched multi-world sweep: run N_PAR gradient starts simultaneously.
        # On GPU: true batched execution (all N in one forward/backward pass) → N_PAR× more
        # starts in the same budget vs sequential. On CPU: sequential (N_PAR=1).
        _use_gpu_par = torch.cuda.is_available()
        _vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                    if _use_gpu_par else 0.0)
        # Scale N_PAR and SA chains with available VRAM (8GB→16, 16GB→32, 48GB→64).
        _n_par = (64 if _vram_gb > 40 else 32 if _vram_gb > 16 else 16) if _use_gpu_par else 1
        # Total starts scales with parallelism: same wall-time, N_PAR× more exploration
        _max_topo = n_topo * _n_par

        _topo_done = 0
        while _topo_done < _max_topo:
            if time.time() >= t0 + topo_budget - per_start_s - 2:
                break
            _slot_end = min(time.time() + per_start_s, t0 + topo_budget - 2, grad_end_total - 5)
            if _slot_end - time.time() < 2:
                break

            # Generate a batch of N_PAR initial positions (main thread, sequential)
            _batch_p, _batch_lam, _batch_tags = [], [], []
            for _bi in range(_n_par):
                if _topo_done >= _max_topo:
                    break
                _ti = _topo_done
                # B2B injection: in the very first batch, replace the first
                # len(_b2b_pool) starts with B2B analytical placements. These
                # then get gradient-refined like any other start, giving them
                # a fair chance to compete instead of being judged on raw cost.
                _use_b2b = (_topo_done < len(_b2b_pool))
                if _use_b2b:
                    _p_new = _b2b_pool[_topo_done].copy()
                    _btag = _b2b_tags[_topo_done] if _topo_done < len(_b2b_tags) else f'b2b-{_topo_done}'
                    _batch_p.append(_p_new)
                    _batch_lam.append(_world_lam_dens[_ti % len(_world_lam_dens)])
                    _batch_tags.append(_btag)
                    _topo_done += 1
                    continue
                _p_new = pos_init.copy()
                for _idx in movable_idx:
                    _w2, _h2 = sizes_np[_idx, 0], sizes_np[_idx, 1]
                    _p_new[_idx, 0] = float(np.random.uniform(_w2 / 2, canvas_w - _w2 / 2))
                    _p_new[_idx, 1] = float(np.random.uniform(_h2 / 2, canvas_h - _h2 / 2))
                _p_new = self._resolve(_p_new, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                _batch_p.append(_p_new)
                _batch_lam.append(_world_lam_dens[_ti % len(_world_lam_dens)])
                _batch_tags.append('rand')
                _topo_done += 1

            # Run batch gradient phases: parallel on GPU, sequential on CPU
            # Mode distribution within each 16-batch (richer than prior 8):
            #  - 6/16: normal       (lc=0.30, csf=0.60) — WL+density baseline
            #  - 2/16: cong_mid     (lc=1.00, csf=0.30) — moderate cong push
            #  - 2/16: cong_hard    (lc=3.00, csf=0.05) — force low-cong topology
            #  - 4/16: density_first(ld×3, lc=0.1, csf=0.05) — over-spread first
            #          → addresses density-dominated gap diagnosed across all benchmarks
            #  - 2/16: balanced_strong (ld×2, lc=2.0, csf=0.20) — push both equally
            _batch_modes = [(_topo_done - len(_batch_p) + i) % 16
                             for i in range(len(_batch_p))]
            _batch_lams  = []
            _batch_csfrac = []
            _batch_lam_cong = []
            _batch_tds = []  # target_den_start per chain: low → ePlace continuation
            for _mi, _lam in zip(_batch_modes, _batch_lam):
                if _mi < 6:           # normal
                    _batch_lams.append(_lam);        _batch_csfrac.append(0.60); _batch_lam_cong.append(None); _batch_tds.append(None)
                elif _mi < 8:         # cong_mid
                    _batch_lams.append(_lam * 0.7);  _batch_csfrac.append(0.30); _batch_lam_cong.append(1.00); _batch_tds.append(None)
                elif _mi < 10:        # cong_hard + density continuation: global spread first, then cong focus
                    _batch_lams.append(_lam * 0.4);  _batch_csfrac.append(0.05); _batch_lam_cong.append(3.00); _batch_tds.append(0.05)
                elif _mi < 14:        # density_first: ePlace continuation, target_den starts at 0.01
                    _batch_lams.append(min(4.0, _lam * 3.0));
                    _batch_csfrac.append(0.05);      _batch_lam_cong.append(0.10); _batch_tds.append(0.01)
                else:                 # balanced_strong: mild ePlace continuation
                    _batch_lams.append(min(4.0, _lam * 2.0));
                    _batch_csfrac.append(0.20);      _batch_lam_cong.append(2.00); _batch_tds.append(0.20)

            if _use_gpu_par and len(_batch_p) > 1:
                # True GPU batched: all N positions in one forward/backward pass
                _grad_res = self._gradient_phase_batched(
                    _batch_p, _batch_lams, movable_idx, sizes_np, port_pos,
                    canvas_w, canvas_h, num_hard, grad_safe_nnp, grad_nnmask,
                    grid_rows, grid_cols, grad_hpwl_norm, _slot_end,
                    fast_eng=fast_eng, cong_start_fracs=_batch_csfrac,
                    lam_cong_overrides=_batch_lam_cong,
                    target_den_starts=_batch_tds,
                )
            else:
                _grad_res = []
                for _p, _lam, _csf, _lco, _tdi in zip(_batch_p, _batch_lams, _batch_csfrac, _batch_lam_cong, _batch_tds):
                    try:
                        _pg = self._gradient_phase(
                            _p, movable_idx, sizes_np, port_pos, canvas_w, canvas_h,
                            num_hard, grad_safe_nnp, grad_nnmask, grid_rows, grid_cols,
                            grad_hpwl_norm, _slot_end, fast_eng=fast_eng,
                            lam_den_override=_lam, cong_start_frac=_csf,
                            lam_cong_override=_lco,
                            target_den_start=_tdi,
                            lam_den_start_frac=1.0 if _tdi is not None else 0.05,
                        )
                        _grad_res.append(_pg)
                    except Exception:
                        _grad_res.append(_p.copy())

            # Resolve and score in main thread; record probe data per output
            for _bi_out, _pg in enumerate(_grad_res):
                _tag_out = _batch_tags[_bi_out] if _bi_out < len(_batch_tags) else 'unk'
                # B2B seeds start from analytically collapsed positions; gradient pushes
                # macros outward but leaves residual overlaps that _resolve (iteration-capped)
                # cannot fully clean. Use _resolve_fully for B2B-seeded results so they
                # compete fairly with init.plc and random seeds. Random/init seeds have
                # much smaller post-gradient overlaps and need only fast _resolve.
                if _tag_out.startswith('b2b'):
                    _pg = self._resolve_fully(_pg, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                else:
                    _pg = self._resolve(_pg, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                _wl, _de, _co, _ov = cheap_components(_pg)
                _proxy = _wl + 0.5 * _de + 0.5 * _co + OVERLAP_WEIGHT * _ov
                topology_seeds.append(_pg)
                topology_costs.append(_proxy)
                topology_tags.append(_tag_out)

        # Keep top-5 by cheap_cost (scores may include overlap penalty from limited _resolve)
        _topo_rank     = sorted(range(len(topology_seeds)), key=lambda i: topology_costs[i])
        topology_seeds = [topology_seeds[i] for i in _topo_rank[:5]]
        topology_costs = [topology_costs[i] for i in _topo_rank[:5]]
        topology_tags  = [topology_tags[i]  for i in _topo_rank[:5]]

        # Post-pass: apply _resolve_fully to any top-5 seed with residual overlaps.
        # The topo loop used fast _resolve (iteration-capped); gradient output can have
        # residual overlaps that inflate the proxy by 200× per overlap unit. Fixing them
        # here (outside the timing-sensitive gradient loop) costs ~1-5s per dirty seed
        # but gives the correct proxy score for ranking and for Phase 1/2 starts.
        for _ci in range(len(topology_seeds)):
            _, _, _, _ov_ci = cheap_components(topology_seeds[_ci])
            if _ov_ci > 1e-3:
                topology_seeds[_ci] = self._resolve_fully(
                    topology_seeds[_ci], num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                _wl_ci, _de_ci, _co_ci, _ov_ci = cheap_components(topology_seeds[_ci])
                topology_costs[_ci] = _wl_ci + 0.5*_de_ci + 0.5*_co_ci + OVERLAP_WEIGHT*_ov_ci
        _topo_rank2    = sorted(range(len(topology_seeds)), key=lambda i: topology_costs[i])
        topology_seeds = [topology_seeds[i] for i in _topo_rank2]
        topology_costs = [topology_costs[i] for i in _topo_rank2]
        topology_tags  = [topology_tags[i]  for i in _topo_rank2]

        best_cheap_cost = topology_costs[0]
        best_cheap_pos  = topology_seeds[0].copy()
        print(f"[PHASE0_DONE] t={time.time()-t0:.1f}s best={best_cheap_cost:.4f} "
              f"top5={[(topology_tags[i], round(topology_costs[i],4)) for i in range(min(5,len(topology_costs)))]}", flush=True)

        # ── Phase 1: Cheap SA warm-up (skipped — sa_end expires before Phase 1 starts at 300s) ──
        hard_deadline = t0 + TIME_BUDGET - 10
        print(f"[PH1_SKIP] t={time.time()-t0:.1f}s best_cheap={best_cheap_cost:.4f}", flush=True)

        # ── Phase 2: Oracle SA + surrogate calibration ────────────────────────
        # oracle_end is set dynamically after measuring oracle_call_secs so
        # SOFT_REOPT always has budget: oracle_end = hard_deadline - (secs + 35).
        oracle_end     = hard_deadline - 45  # preliminary; updated after oracle calibration
        best_pos              = best_cheap_pos.copy()
        _oracle_calls_possible = 0  # updated after oracle calibration

        if plc is not None:
            # Evaluate pos_init as the reference baseline and calibrate oracle call time.
            init_r    = self._resolve_fully(pos_init, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
            t_oc0 = time.time()
            init_cost, init_comps = self._true_cost(init_r, benchmark, plc, return_components=True)
            oracle_call_secs = time.time() - t_oc0
            # Reserve for: last SA slot's post-SA oracle call + SOFT_REOPT (gradient+oracle) + buffer.
            oracle_end = min(hard_deadline - 5, hard_deadline - int(2 * oracle_call_secs + 35))
            # oracle SA is viable when ≥3 calls fit in remaining budget.
            _oracle_calls_possible = int(
                max(0, oracle_end - time.time()) / max(1e-3, oracle_call_secs)
            ) if oracle_call_secs < TIME_BUDGET * 0.60 else 0

            # Best of gradient result vs pos_init as primary start.
            starts = [init_r]
            starts_cost = [init_cost]
            if _oracle_calls_possible >= 3:
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

            # DIAG: print oracle components of initial oracle-SA start
            _diag_comps = ref_comps or {}
            print(f"[DIAG] pre-SA oracle: proxy={starts_cost[0]:.4f}  "
                  f"den={_diag_comps.get('density_cost', 0.0):.3f}  "
                  f"cong={_diag_comps.get('congestion_cost', 0.0):.3f}", flush=True)
            print(f"[PH2_ENTER] oracle_calls_possible={_oracle_calls_possible} oracle_call_secs={oracle_call_secs:.1f}", flush=True)

            # Build oracle SA start pool from oracle-evaluated topology seeds.
            if _oracle_calls_possible >= 3:
                # Oracle-evaluate Phase 0 topology seeds so they enter the SA pool with
                # known costs (prioritized over pure-random inf-cost starts).
                for _p_topo in topology_seeds:
                    if oracle_end - time.time() < 20:
                        starts.append(_p_topo.copy())
                        starts_cost.append(float('inf'))
                        continue
                    try:
                        _p_topo_r = self._resolve_fully(_p_topo.copy(), num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                        _sc_topo = self._true_cost(_p_topo_r, benchmark, plc)
                        starts.append(_p_topo_r)
                        starts_cost.append(_sc_topo if np.isfinite(_sc_topo) else float('inf'))
                        if np.isfinite(_sc_topo) and _sc_topo < starts_cost[0]:
                            starts[0] = _p_topo_r
                            starts_cost[0] = _sc_topo
                    except Exception:
                        starts.append(_p_topo.copy())
                        starts_cost.append(float('inf'))

            # Diverse start augmentation: add varied-perturbation + random starts (inf cost).
            # These sort after oracle-evaluated seeds but expose SA to fresh basins when
            # the top gradient starts all converge to the same local optimum.
            # K+L ablation (300s) showed ~0% effect; at 3300s they matter because
            # ORA_SA_EXIT fires only after exhausting this pool — more starts → more SA slots.
            if _oracle_calls_possible >= 6:
                _best_start_idx = int(np.argmin(starts_cost))
                for _pi in range(3):
                    _sigma = 0.20 + _pi * 0.15  # 0.20, 0.35, 0.50 — increasing exploration
                    _p_pert = self._perturb(starts[_best_start_idx], movable_idx, sizes_np,
                                            canvas_w, canvas_h, _sigma)
                    _p_pert = self._resolve(_p_pert, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                    starts.append(_p_pert)
                    starts_cost.append(float('inf'))
                for _ri in range(2):
                    _p_rand = pos_init.copy()
                    _rng_r = np.random.default_rng(20260504 + _ri * 13337)
                    for _idx in movable_idx:
                        _w2, _h2 = sizes_np[_idx, 0], sizes_np[_idx, 1]
                        _p_rand[_idx, 0] = float(_rng_r.uniform(_w2 / 2, canvas_w - _w2 / 2))
                        _p_rand[_idx, 1] = float(_rng_r.uniform(_h2 / 2, canvas_h - _h2 / 2))
                    _p_rand = self._resolve(_p_rand, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                    starts.append(_p_rand)
                    starts_cost.append(float('inf'))

            best_global_cost = min(starts_cost)
            best_pos = starts[int(np.argmin(starts_cost))].copy()

            slot_dur = 180.0     # default; overridden below for fast oracles
            _ora_t_scale = 1.0   # default; overridden for slow oracles

            if _oracle_calls_possible >= 3:
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
                _n_known_starts = len(known)   # oracle-evaluated starts (gradient-initialized)

                # Slow oracle (ibm17: ~30-50s/call) → very few evals per slot.
                # Scale up SA temperature so each expensive eval explores more broadly.
                _ora_t_scale = max(1.0, oracle_call_secs / 12.0)
                _ora_no_improve = 0  # consecutive non-improving slots
                for si, (s_cost_init, s_pos) in enumerate(sorted_starts):
                    if time.time() >= oracle_end - 5:
                        break
                    # Exit as soon as 2 consecutive slots fail to improve.
                    # si >= 2 ensures we always try at least 2 slots before deciding.
                    # Removed _n_known_starts floor: data shows SA contributes 0% on
                    # 15/17 benchmarks — forcing 6 slots wastes ~780s for no gain.
                    if _ora_no_improve >= 2 and si >= 2:
                        print(f"[ORA_SA_EXIT] {_ora_no_improve} consecutive non-improving slots, exiting early at si={si}", flush=True)
                        break
                    slot_end = min(time.time() + slot_dur, oracle_end)
                    result_s = self._plc_oracle_sa(
                        s_pos, movable_idx, sizes_np, port_pos, canvas_w, canvas_h,
                        num_hard, fixed_np, benchmark, plc, slot_end, s_cost_init,
                        macro_to_nets, nets_np,
                        fast_eng=fast_eng, grid_rows=grid_rows, grid_cols=grid_cols,
                        t_scale=_ora_t_scale,
                    )
                    result_sr = self._resolve_fully(result_s, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                    cost_s = self._true_cost(result_sr, benchmark, plc)
                    print(f"[ORA_SA_SLOT {si}] cost={cost_s:.4f} best={best_global_cost:.4f} t={time.time()-t0:.0f}s")
                    if cost_s < best_global_cost:
                        best_global_cost = cost_s; best_pos = result_sr.copy()
                        _ora_no_improve = 0
                    else:
                        _ora_no_improve += 1
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

        # ── Soft macro density-aware re-optimization (post-oracle SA) ────────────
        # Oracle SA only moves hard macros; soft macros stay at gradient positions.
        # Soft macros cluster near hard macros, contributing to density cost (top-10%).
        # Re-running with lam_den=5 spreads soft macros to reduce combined density.
        _sr_cong_w = 0.0  # default; set below when plc available
        if (soft_movable_idx and plc is not None and
                _oracle_calls_possible >= 3 and
                time.time() < hard_deadline - 12):
            # cong_w=2.0 for dense benchmarks (ibm14/17: oracle den>0.88 → congestion also over capacity)
            _is_hi_den = (ref_comps or {}).get('density_cost', 0.0) > 0.88
            _sr_cong_w = 2.0 if _is_hi_den else 0.0
            try:
                _prev_oracle_sr = best_global_cost
                # Use remaining oracle budget for gradient (up to 120s cap).
                # When ORA_SA_EXIT fires early (ibm01: 2600s freed), 25s was wasted budget.
                _sr_extra = max(0.0, oracle_end - time.time() - oracle_call_secs - 10)
                _sr_grad_t = max(25.0, min(120.0, _sr_extra))
                _soft_ddl = min(hard_deadline - 8, time.time() + _sr_grad_t)
                _best_pos_sr = self._gradient_phase_soft(
                    best_pos, soft_movable_idx, sizes_np, port_pos,
                    canvas_w, canvas_h, num_hard,
                    grad_safe_nnp, grad_nnmask, grid_rows, grid_cols, grad_hpwl_norm,
                    _soft_ddl,
                    cong_weight=_sr_cong_w,
                    fast_eng=fast_eng,
                )
                _sc_sr = self._true_cost(_best_pos_sr, benchmark, plc)
                _sr_mode = f"cong_w={_sr_cong_w:.0f}" if _sr_cong_w > 0 else "den_only"
                print(f"[SOFT_REOPT] {_sr_mode} oracle: {_sc_sr:.4f}  (prev: {_prev_oracle_sr:.4f})", flush=True)
                if np.isfinite(_sc_sr) and _sc_sr < _prev_oracle_sr:
                    best_pos = _best_pos_sr
                    best_global_cost = _sc_sr  # keep best_global_cost in sync for refinement loop
            except Exception as _sr_e:
                print(f"[SOFT_REOPT] failed: {_sr_e}", flush=True)

        # ── Phase 3: Final selection — oracle result vs pos_init ──────────────
        # Always resolve_fully best_pos first so phase 3 comparison is valid.
        best_pos = self._resolve_fully(best_pos, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
        # Only run oracle comparison in Phase 3 if oracle is fast enough (skip for ibm17-scale).
        _oracle_fast_p3 = _oracle_calls_possible >= 3 and plc is not None
        if _oracle_fast_p3 and time.time() < hard_deadline - 3:
            best_final_cost = self._true_cost(best_pos, benchmark, plc) if plc else float('inf')
            if time.time() < hard_deadline - 2:
                init_rr = self._resolve_fully(pos_init, num_hard, sizes_np, canvas_w, canvas_h, fixed_np)
                tc_init = self._true_cost(init_rr, benchmark, plc)
                if tc_init < best_final_cost:
                    best_pos = init_rr.copy()
        result = benchmark.macro_positions.clone()
        result[:] = torch.from_numpy(best_pos).float()
        # Float32 quantization can reintroduce tiny overlaps.  Resolve *inside float32 space*:
        # each round converts back to float32 between resolve passes so the check and the
        # final output agree (the plain _total_overlap check uses float64 and can miss tiny
        # overlaps that are only visible after float32 rounding, causing persistent INVALID).
        best_pos32 = result.numpy().astype(np.float64)
        for _rnd32 in range(30):
            best_pos32 = self._resolve(best_pos32, num_hard, sizes_np, canvas_w, canvas_h, fixed_np, max_iter=500, min_iter=50)
            # Re-quantise to float32 and check *in float32 space*
            _q32 = torch.from_numpy(best_pos32).float().numpy().astype(np.float64)
            if self._total_overlap(_q32, num_hard, sizes_np) < 1e-10:
                best_pos32 = _q32
                break
            best_pos32 = _q32  # next resolve starts from float32-aligned positions
        # After SOFT_REOPT, a soft macro may overlap a hard macro due to float32 quantization.
        # Fix: targeted push — move only the soft macro out of any hard macro it touches.
        # Do NOT resolve soft-soft overlaps (gradient handles those; mass-resolve destroys placement).
        _n_total = best_pos32.shape[0]
        if _n_total > num_hard and soft_movable_idx:
            _soft_idx_list = list(soft_movable_idx)
            for _pass in range(5):
                _any_fix = False
                for _hi in range(num_hard):
                    _xh = best_pos32[_hi, 0]; _yh = best_pos32[_hi, 1]
                    _wh = sizes_np[_hi, 0];   _hh = sizes_np[_hi, 1]
                    for _si_fix in _soft_idx_list:
                        _xs = best_pos32[_si_fix, 0]; _ys = best_pos32[_si_fix, 1]
                        _ws = sizes_np[_si_fix, 0];   _hs = sizes_np[_si_fix, 1]
                        ox = (_ws + _wh) * 0.5 - abs(_xs - _xh)
                        oy = (_hs + _hh) * 0.5 - abs(_ys - _yh)
                        if ox > 1e-8 and oy > 1e-8:
                            _any_fix = True
                            if ox < oy:
                                best_pos32[_si_fix, 0] = float(np.clip(
                                    _xs + (ox + 0.001) * (1.0 if _xs >= _xh else -1.0),
                                    _ws / 2, canvas_w - _ws / 2))
                            else:
                                best_pos32[_si_fix, 1] = float(np.clip(
                                    _ys + (oy + 0.001) * (1.0 if _ys >= _yh else -1.0),
                                    _hs / 2, canvas_h - _hs / 2))
                if not _any_fix:
                    break
        result[:] = torch.from_numpy(best_pos32).float()
        return result

    # ── Oracle SA with temperature ────────────────────────────────────────────

    def _plc_oracle_sa(self, pos, movable_idx, sizes, port_pos, canvas_w, canvas_h,
                       num_hard, fixed, benchmark, plc, deadline, init_cost,
                       macro_to_nets, nets_np,
                       fast_eng=None, grid_rows=1, grid_cols=1, t_scale=1.0):
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

        # Density hotspot tracking: record the least-dense canvas region so we
        # can propose moves that explicitly bring over-packed macros to open space.
        # This targets the density floor (den 0.81-0.95) that SA can't escape.
        _den_low_cell = None   # (x, y) center of least-dense region
        _den_refresh  = 40

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
        _conv_t_last_best = t_run_start  # time of last best_cost improvement (for convergence exit)

        # Cached combined position array (hard macros + ports), updated incrementally
        cur_all = self._all_pos(cur_pos, port_pos)

        # Pre-compute 1-hop movable-macro adjacency for cluster-translate moves.
        # Time-guarded at 3s to handle very large benchmarks safely.
        _macro_set_ora = set(movable_idx)
        _adj_sets_ora: dict[int, set] = {idx: set() for idx in movable_idx}
        _t_adj0 = time.time()
        for _nodes_a in nets_np:
            if time.time() - _t_adj0 > 3.0: break
            _mov_on_net = [int(_v) for _v in _nodes_a if int(_v) in _macro_set_ora]
            for _mi in _mov_on_net:
                _adj_sets_ora[_mi].update(_mj for _mj in _mov_on_net if _mj != _mi)
        _macro_adj = {k: list(v) for k, v in _adj_sets_ora.items()}

        # SA epoch restarts: 3 epochs, each restarting from best_pos with half temperature.
        # SA at low T wastes time in a frozen landscape — restart with fresh temperature
        # from current best to keep exploring new neighborhoods.
        _n_ep = 3
        _total_sa_dur = max(1.0, deadline - 1.5 - t_run_start)
        _ep_dur = _total_sa_dur / _n_ep
        _ep_t0 = [t_run_start + _ep_dur * k for k in range(_n_ep)]
        _ep_t1 = [t_run_start + _ep_dur * (k + 1) for k in range(_n_ep)]
        _ep_idx = 0

        while time.time() < deadline - 1.5:
            # Epoch boundary: restart from best_pos with halved temperature scale
            if _ep_idx < _n_ep - 1 and time.time() >= _ep_t1[_ep_idx]:
                _ep_idx += 1
                cur_pos = best_pos.copy()
                cur_all[:len(cur_pos)] = cur_pos
                cur_cost = best_cost
                _den_low_cell = None  # force density refresh

            # Epoch-local temperature: full cooling arc within each epoch
            _ep_elapsed = max(0.0, time.time() - _ep_t0[_ep_idx])
            _ep_scale = 1.0 / (2.0 ** _ep_idx)   # 1.0 → 0.5 → 0.25
            _ep_frac = min(1.0, _ep_elapsed / max(1.0, _ep_t1[_ep_idx] - _ep_t0[_ep_idx]))
            # t_scale >1 for slow-oracle benchmarks (ibm17: ~30-50s/call → t_scale≈2-4)
            # Fewer evals per budget → need higher T to escape basins in each slot.
            T = T_ORC_START * t_scale * _ep_scale * (T_ORC_END / T_ORC_START) ** _ep_frac

            elapsed     = time.time() - t_run_start
            time_budget = max(1.0, deadline - 1.5 - t_run_start)
            frac        = min(1.0, elapsed / time_budget)

            # Periodically refresh congestion hotspot using fast surrogate
            if fast_eng is not None and step % hotspot_refresh_steps == 0:
                cg = self._fast_cong_grid(self._all_pos(cur_pos, port_pos), num_hard, sizes,
                                          fast_eng, grid_rows, grid_cols, canvas_w, canvas_h)
                r_hot, c_hot = np.unravel_index(cg.argmax(), cg.shape)
                gw = canvas_w / grid_cols; gh = canvas_h / grid_rows
                hotspot_cell = ((c_hot + 0.5) * gw, (r_hot + 0.5) * gh)

            # Periodically refresh density hotspot to track low-density empty cells.
            if step % _den_refresh == 0:
                dg = self._density_grid(cur_pos, num_hard, sizes, grid_rows, grid_cols,
                                        canvas_w, canvas_h)
                _gw_d = canvas_w / grid_cols; _gh_d = canvas_h / grid_rows
                # Smoothed density: prefer cells far from macros (avoid thin sliver cells)
                _den_min_r, _den_min_c = np.unravel_index(dg.argmin(), dg.shape)
                _den_low_cell = ((_den_min_c + 0.5) * _gw_d, (_den_min_r + 0.5) * _gh_d)

            cand = cur_pos.copy()
            # Temperature-adaptive move distribution:
            # High T (frac<0.3): 30% swap + 15% large + 15% net-centroid + 15% small + 10% hotspot + 8% cluster-translate
            # Low T  (frac>0.7): 10% swap + 3% large + 35% net-centroid + 30% small + 10% hotspot + 8% cluster-translate
            _p_swap    = 0.28 - 0.18 * frac   # 0.28 → 0.10
            _p_centroid = 0.13 + 0.20 * frac  # 0.13 → 0.33
            _p_small   = 0.13 + 0.13 * frac   # 0.13 → 0.26
            _p_large   = 0.13 - 0.10 * frac   # 0.13 → 0.03
            # Early SA: boost bulk density spread to escape density basin.
            # Transfer probability from cluster_translate and density_escape
            # (less useful early) to bulk_density_spread (most useful early).
            _frac_early  = max(0.0, 1.0 - frac / 0.40)  # 1.0 → 0 over first 40%
            _clus_w      = max(0.01, 0.07 * (1.0 - 0.86 * _frac_early))  # 0.01 → 0.07
            _den_esc_w   = max(0.01, 0.07 * (1.0 - 0.86 * _frac_early))  # 0.01 → 0.07
            _bulk_w      = 0.06 + 0.12 * _frac_early                      # 0.18 → 0.06
            _b1 = _p_swap
            _b2 = _b1 + _p_centroid
            _b3 = _b2 + _p_small
            _b4 = _b3 + _p_large
            _b5 = _b4 + 0.09        # congestion hotspot escape
            _b6 = _b5 + _clus_w     # cluster translate
            _b7 = _b6 + _den_esc_w  # density escape: move dense-region macro to emptiest cell
            _b8 = _b7 + _bulk_w     # bulk density spread: 5-15 dense macros → low-density regions
            # net-cluster-shuffle occupies [_b8, 1.0]

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

            elif move_type < _b5:
                if hotspot_cell is not None:
                    # Hotspot escape: move macro nearest congestion hotspot to a random location.
                    hx, hy = hotspot_cell
                    idx = min(movable_idx, key=lambda _i: abs(cur_pos[_i,0]-hx)+abs(cur_pos[_i,1]-hy))
                    w, h = sizes[idx, 0], sizes[idx, 1]
                    cand[idx, 0] = float(np.random.uniform(w/2, canvas_w - w/2))
                    cand[idx, 1] = float(np.random.uniform(h/2, canvas_h - h/2))
                    moved_set = [idx]
                elif has_orient:
                    # Orientation flip when no hotspot tracked yet.
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
                            _conv_t_last_best = time.time()
                    else:
                        plc.update_macro_orientation(plc_idx, old_orient)
                    continue
                else:
                    continue

            elif move_type < _b6:
                # Cluster translate: translate seed + 1-hop connected neighbors as a rigid unit.
                # Preserves intra-cluster topology; changes inter-cluster WL and congestion.
                seed = movable_idx[random.randrange(n_mov)]
                nbrs = _macro_adj.get(seed, [])
                k_ct = min(3, len(nbrs))
                cluster = [seed] + (random.sample(nbrs, k_ct) if k_ct > 0 else [])
                tx = np.random.normal(0, step_large * 0.5)
                ty = np.random.normal(0, step_large * 0.5)
                valid = True
                for cidx in cluster:
                    w, h = sizes[cidx, 0], sizes[cidx, 1]
                    nx2 = cur_pos[cidx, 0] + tx
                    ny2 = cur_pos[cidx, 1] + ty
                    if nx2 < w/2 or nx2 > canvas_w - w/2 or ny2 < h/2 or ny2 > canvas_h - h/2:
                        valid = False; break
                    cand[cidx, 0] = float(nx2)
                    cand[cidx, 1] = float(ny2)
                if not valid: continue
                moved_set = cluster

            elif move_type < _b7:
                # Density-escape: move a macro from the densest region toward the
                # emptiest canvas cell. Directly attacks the density floor that random
                # SA moves cannot escape (den 0.81-0.95 vs RePlAce's 0.5-0.7).
                if _den_low_cell is None:
                    continue
                # Pick the macro closest to the densest cell (use density grid max).
                dg_cur = self._density_grid(cur_pos, num_hard, sizes, grid_rows, grid_cols,
                                             canvas_w, canvas_h)
                _r_d, _c_d = np.unravel_index(dg_cur.argmax(), dg_cur.shape)
                _gw_d = canvas_w / grid_cols; _gh_d = canvas_h / grid_rows
                _dense_x = (_c_d + 0.5) * _gw_d; _dense_y = (_r_d + 0.5) * _gh_d
                # Find the movable macro closest to the densest cell
                _dists = [(abs(cur_pos[i, 0] - _dense_x) + abs(cur_pos[i, 1] - _dense_y), i)
                          for i in movable_idx]
                _, _esc_idx = min(_dists)
                w, h = sizes[_esc_idx, 0], sizes[_esc_idx, 1]
                # Propose moving it toward the least-dense cell (with small jitter)
                _tx_d, _ty_d = _den_low_cell
                _tx_d += np.random.normal(0, _gw_d * 1.5)
                _ty_d += np.random.normal(0, _gh_d * 1.5)
                nx_d = float(np.clip(_tx_d, w / 2, canvas_w - w / 2))
                ny_d = float(np.clip(_ty_d, h / 2, canvas_h - h / 2))
                cand[_esc_idx, 0] = nx_d
                cand[_esc_idx, 1] = ny_d
                moved_set = [_esc_idx]

            elif move_type < _b8:
                # Bulk density spread: find macros in the densest grid cells, propose moving
                # 5-15 of them simultaneously to low-density regions.
                # Key insight: single-macro SA cannot coordinate the multi-macro relocation
                # needed to reduce den from 0.81 → 0.6. This move makes that coordinated jump.
                dg_cur = self._density_grid(cur_pos, num_hard, sizes, grid_rows, grid_cols,
                                             canvas_w, canvas_h)
                _gw_d = canvas_w / grid_cols; _gh_d = canvas_h / grid_rows
                _den_thresh = float(np.percentile(dg_cur, 75))
                # Collect movable macros whose cell has above-threshold density
                _dense_macros = []
                for _di in movable_idx:
                    _cx_cell = min(int(cur_pos[_di, 0] / _gw_d), grid_cols - 1)
                    _cy_cell = min(int(cur_pos[_di, 1] / _gh_d), grid_rows - 1)
                    if dg_cur[_cy_cell, _cx_cell] >= _den_thresh:
                        _dense_macros.append(_di)
                if len(_dense_macros) < 2:
                    continue
                _n_bulk = random.randint(3, min(12, len(_dense_macros)))
                _bulk_group = random.sample(_dense_macros, _n_bulk)
                # Find low-density target cells (bottom 25%)
                _den_ceil = float(np.percentile(dg_cur, 25))
                _low_rs, _low_cs = np.where(dg_cur <= _den_ceil)
                if len(_low_rs) == 0:
                    continue
                # WL-guided targeting: each dense macro goes to the low-density
                # cell closest to its WL centroid → minimises WL cost increase →
                # much higher SA acceptance probability vs random cell assignment.
                _used_cells: set = set()
                for _bidx in _bulk_group:
                    _w, _h = sizes[_bidx, 0], sizes[_bidx, 1]
                    # Compute WL centroid for this macro
                    _nets_b = macro_to_nets.get(_bidx, [])
                    _cx_wl = _cy_wl = _wt_wl = 0.0
                    for _ni_b in _nets_b:
                        _nd_b = nets_np[_ni_b]
                        _oth = [int(_v) for _v in _nd_b if int(_v) != _bidx]
                        if not _oth: continue
                        _wt = 1.0 / max(1, len(_nd_b))
                        for _n in _oth:
                            _cx_wl += _wt * cur_all[_n, 0]
                            _cy_wl += _wt * cur_all[_n, 1]
                            _wt_wl += _wt
                    if _wt_wl > 1e-9:
                        _wl_x, _wl_y = _cx_wl / _wt_wl, _cy_wl / _wt_wl
                    else:
                        _wl_x, _wl_y = cur_pos[_bidx, 0], cur_pos[_bidx, 1]
                    # Pick the low-density cell nearest the WL centroid
                    # (sample up to 15 low-density cells, take the closest unused)
                    _n_cand = min(15, len(_low_rs))
                    _indices = random.sample(range(len(_low_rs)), _n_cand)
                    _best_dist = float('inf')
                    _best_tx = _best_ty = None
                    for _li in _indices:
                        if _li in _used_cells:
                            continue
                        _ccx = (_low_cs[_li] + 0.5) * _gw_d
                        _ccy = (_low_rs[_li] + 0.5) * _gh_d
                        _d2 = (_wl_x - _ccx) ** 2 + (_wl_y - _ccy) ** 2
                        if _d2 < _best_dist:
                            _best_dist = _d2
                            _best_tx, _best_ty = _ccx, _ccy
                            _best_li = _li
                    if _best_tx is None:
                        _pick = random.randrange(len(_low_rs))
                        _best_tx = (_low_cs[_pick] + 0.5) * _gw_d
                        _best_ty = (_low_rs[_pick] + 0.5) * _gh_d
                        _best_li = _pick
                    _used_cells.add(_best_li)
                    cand[_bidx, 0] = float(np.clip(
                        _best_tx + np.random.normal(0, _gw_d * 0.5), _w / 2, canvas_w - _w / 2))
                    cand[_bidx, 1] = float(np.clip(
                        _best_ty + np.random.normal(0, _gh_d * 0.5), _h / 2, canvas_h - _h / 2))
                moved_set = _bulk_group

            else:
                # Net-cluster shuffle: pick a random net, permute positions of its movable macros.
                # Smarter than random shuffle: macros sharing nets explore each other's positions.
                _ni_r = random.randrange(len(nets_np))
                cluster = [int(_v) for _v in nets_np[_ni_r] if int(_v) in _macro_set_ora]
                if len(cluster) < 2: continue
                cluster = cluster[:5]
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
                moved_set = cluster

            # Fast local resolve: only check moved macros against all others.
            # O(k×n) instead of O(n²); k=1-5, n=num_hard. ~100x faster than full resolve.
            if move_type < _b1:
                moved_set = [idx, idx2]
            elif move_type < _b4:
                moved_set = [idx]
            # For >= _b4: moved_set is set within the branch above
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
                    _conv_t_last_best = time.time()

            # Convergence exit: free remaining slot time when SA has stalled.
            # Stall threshold = max(90s, min(50% of slot, 300s)).
            # Guard: ≥3 oracle calls and ≥25% of slot time elapsed.
            # Note: step>=10 would never fire for slow oracles (31s/call in 276s slot → 8 max steps).
            if step >= 3:
                _stall = time.time() - _conv_t_last_best
                _stall_thr = max(90.0, min(0.50 * _total_sa_dur, 300.0))
                if _stall > _stall_thr and time.time() - t_run_start > 0.25 * _total_sa_dur:
                    print(f"[SLOT_CONV] step={step} stall={_stall:.0f}s thr={_stall_thr:.0f}s best={best_cost:.4f}", flush=True)
                    break

        # Restore plc to best-found orientations before returning
        if has_orient:
            for plc_idx, orient in best_orient.items():
                plc.update_macro_orientation(plc_idx, orient)

        return best_pos

    # ── Gradient-based placement (ePlace-style smooth WL + density) ──────────

    def _gradient_phase(self, pos, movable_idx, sizes_np, port_pos,
                        canvas_w, canvas_h, num_hard,
                        safe_nnp, nnmask, grid_rows, grid_cols, hpwl_norm, deadline,
                        fast_eng=None, lam_den_override=None, lam_cong_override=None,
                        cong_start_frac=0.60,
                        target_den_start=None, lam_den_start_frac=0.05,
):
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
        # NTUplace3 / ePlace cell inflation: inflate macro sizes 1.3× in the density
        # kernel only. Inflated macros appear 69% larger in area → gradient sees
        # massive overflow even at 60-70% real utilization → forces strong global
        # spreading. The real placement uses actual sizes — so a grad-phase solution
        # that "just barely fits" at 1.3× will have healthy whitespace at 1.0×.
        _den_inflate = 1.30
        hw_den  = sizes_den_t[:, 0] * _den_inflate / 2 + cell_w / 2  # (n_den_macros,)
        hh_den  = sizes_den_t[:, 1] * _den_inflate / 2 + cell_h / 2

        # Exact oracle-aligned density: cell boundary tensors for box-overlap coverage.
        # Bell kernel optimized a different metric than the oracle; gradient got stuck.
        # Box-overlap matches oracle's exact grid-cell utilization computation.
        cell_lx = torch.arange(grid_cols, dtype=torch.float32, device=dev) * cell_w   # (grid_cols,)
        cell_rx = cell_lx + cell_w
        cell_by = torch.arange(grid_rows, dtype=torch.float32, device=dev) * cell_h   # (grid_rows,)
        cell_ty = cell_by + cell_h
        _smooth_r = int(fast_eng['smooth']) if (fast_eng is not None and 'smooth' in fast_eng) else 0

        canvas_diag  = float(canvas_w + canvas_h)
        gamma_start  = canvas_diag * 0.04
        gamma_end    = canvas_diag * 0.004
        # Adaptive density target: 5% slack above physical utilization.
        # ePlace/RePlAce only penalize cells above this threshold → macros spread
        # until no cell exceeds target. Raw penalty (no threshold) was why we
        # over-packed to 0.81-0.95 vs RePlAce's 0.5-0.7.
        _macro_area = float(np.sum(sizes_np[:num_hard, 0] * sizes_np[:num_hard, 1]))
        _canvas_area = float(canvas_w * canvas_h)
        _utilization = _macro_area / max(_canvas_area, 1e-9)
        target_den_final = float(np.clip(_utilization * 1.08, 0.50, 0.82))
        # For ePlace-continuation worlds (low density start), cap at 0.70 for sparse
        # benchmarks so macros settle in a lower-density basin (~RePlAce's 0.70-0.75).
        if target_den_start is not None and target_den_start < 0.20 and _utilization < 0.85:
            target_den_final = min(target_den_final, 0.70)
        # ePlace/RePlAce density continuation: ramp target_den from a low value up to
        # the physical utilization × 1.08. When target_den_start is near 0, every cell
        # contributes to the density penalty from step 1, forcing global spreading.
        # As target_den rises, cells below threshold stop contributing → WL clustering
        # reasserts, and macros settle at the low-density basin (~utilization × 1.0).
        # This is what produces RePlAce's den~0.73 vs our fixed-target den~0.81.
        _td_start = target_den_start if target_den_start is not None else target_den_final
        lam_den_start = (lam_den_override * lam_den_start_frac) if lam_den_override else (2.50 * lam_den_start_frac)
        lam_den_end   = lam_den_override if lam_den_override else 2.50
        lam_cong_max  = lam_cong_override if lam_cong_override else 0.30
        lr            = canvas_diag * 0.002
        t_start       = time.time()
        t_total       = max(1.0, deadline - t_start)

        # Precompute congestion tensors if fast_eng available.
        # GPU: no n_pairs limit (matmul is O(p×r + p×c) not O(p×r×c)).
        # CPU: limit to 200K pairs (memory guard for einsum).
        cong_enabled = False
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

        # Nesterov NAG with RMS-normalized step (DREAMPlace §3.3 style).
        # β=0.85 lookahead: momentum avoids oscillation in WL-density conflict ravines.
        # RMS normalization gives per-macro adaptive step without Adam's warm-up lag.
        lr_min = lr * 0.02
        _beta_nag = 0.85
        _rms_eps  = 1e-8
        x_prev_data = x_param.data.clone()

        # FFT Poisson electrostatic density (ePlace model).
        # ∇²φ = -ρ solved via FFT: φ_hat = ρ_hat / |k|². den_loss = (ρ·φ).mean().
        # Global long-range repulsion vs bell-kernel's 2-cell local support.
        _kx2 = torch.fft.fftfreq(grid_rows, device=dev).pow(2).unsqueeze(1)
        _ky2 = torch.fft.rfftfreq(grid_cols, device=dev).pow(2).unsqueeze(0)
        _k2  = (4 * math.pi ** 2 * (_kx2 + _ky2)).clamp(min=1e-6)
        phi_green_t = 1.0 / _k2
        phi_green_t[0, 0] = 0.0
        _phi_norm = float(phi_green_t[phi_green_t > 0].mean().item())
        if _phi_norm < 1e-12:
            _phi_norm = 1.0

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
        tau_den_h   = 0.5  # will be annealed inside loop

        step = 0
        t_first = None
        _grad_loss_ema = None  # exponential moving average of loss for plateau detection
        _prev_ema_check = None  # EMA value at last 50-step checkpoint
        _plateau_count = 0

        while time.time() < deadline:
            t_s  = time.time()
            frac = min(1.0, (t_s - t_start) / t_total)
            # Anneal density τ: broad early (global spreading) → sharp late (top-cell focus).
            # Smaller τ makes logsumexp ≈ top-k max → better matches oracle ABU density.
            tau_den_h = max(0.08, 0.5 - 0.42 * frac)
            # Cosine annealing LR: high early for exploration, near-zero late for refinement
            cur_lr = lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * frac))
            # Anneal: gamma decreases (sharper WL), lam_den increases (stronger spread)
            g       = gamma_start * (gamma_end / gamma_start) ** frac
            lam_den = lam_den_start + (lam_den_end - lam_den_start) * frac
            # ePlace continuation: ramp target density from _td_start to target_den_final
            target_den_cur = _td_start + (target_den_final - _td_start) * frac

            # Nesterov lookahead: compute gradient at y = x + β*(x - x_prev)
            with torch.no_grad():
                _x_saved  = x_param.data.clone()
                _momentum = x_param.data - x_prev_data
                x_param.data = x_param.data + _beta_nag * _momentum

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

            # ── Exact oracle-aligned density (box-overlap + oracle smoothing) ──
            # Bell kernel failed: gradient optimized a different metric from oracle,
            # getting stuck at den=0.812 despite 59s of Phase 2 gradient.
            # Box-overlap gives gradient aligned with what oracle actually measures.
            mx  = pos_cur[:n_den_macros, 0]
            my  = pos_cur[:n_den_macros, 1]
            left_x  = mx - sizes_den_t[:, 0] / 2       # (n_den_macros,)
            right_x = mx + sizes_den_t[:, 0] / 2
            bot_y   = my - sizes_den_t[:, 1] / 2
            top_y   = my + sizes_den_t[:, 1] / 2
            cov_x = torch.clamp(
                torch.minimum(right_x.unsqueeze(1), cell_rx.unsqueeze(0)) -
                torch.maximum(left_x.unsqueeze(1),  cell_lx.unsqueeze(0)),
                min=0.0) / cell_w                       # (n_den_macros, grid_cols)
            cov_y = torch.clamp(
                torch.minimum(top_y.unsqueeze(1),   cell_ty.unsqueeze(0)) -
                torch.maximum(bot_y.unsqueeze(1),   cell_by.unsqueeze(0)),
                min=0.0) / cell_h                       # (n_den_macros, grid_rows)
            density = torch.einsum('ic,ir->rc', cov_x, cov_y)  # (grid_rows, grid_cols)
            if _smooth_r > 0:
                _ks = 2 * _smooth_r + 1
                density = torch.nn.functional.avg_pool2d(
                    torch.nn.functional.pad(
                        density.unsqueeze(0).unsqueeze(0), [_smooth_r] * 4, mode='replicate'),
                    _ks, stride=1, padding=0).squeeze(0).squeeze(0)
            _target_t = torch.tensor(target_den_cur, dtype=torch.float32, device=dev)
            den_ovflow = torch.clamp(density - _target_t, min=0.0)
            _rho_hat = torch.fft.rfft2(den_ovflow)
            _phi_es  = torch.fft.irfft2(_rho_hat * phi_green_t, s=(grid_rows, grid_cols))
            fft_den  = (den_ovflow * _phi_es).mean() / _phi_norm
            bell_den = torch.logsumexp(den_ovflow.view(-1) / tau_den_h, 0) * tau_den_h / top_k_den_h
            den_loss = fft_den + 0.5 * bell_den

            # ── Differentiable congestion (H-then-V 2-pin model) ──────────────
            # Enabled only after 40% of gradient time, to let WL+density settle first.
            cong_loss = torch.tensor(0.0, device=dev)
            lam_cong  = 0.0
            if cong_enabled and frac > cong_start_frac:
                _cong_ramp = max(0.30, 1.0 - cong_start_frac)
                lam_cong = lam_cong_max * min(1.0, (frac - cong_start_frac) / _cong_ramp)
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
            if x_param.grad is not None:
                x_param.grad = None
            loss.backward()
            with torch.no_grad():
                # Preconditioned RMS-normalized Nesterov step
                _g     = x_param.grad.data * precond_t
                _g_rms = _g.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=_rms_eps)
                x_param.data  = _x_saved - cur_lr * _g / _g_rms
                x_prev_data   = _x_saved
                x_param.grad  = None

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

            # Plateau exit: stop gradient when EMA hasn't improved for 6 consecutive
            # 50-step windows (window-to-window comparison avoids oscillation resets).
            _loss_val = float(loss.detach())
            if _grad_loss_ema is None:
                _grad_loss_ema = _loss_val
            else:
                _grad_loss_ema = 0.99 * _grad_loss_ema + 0.01 * _loss_val
            if step % 50 == 49 and frac > 0.40:
                if _prev_ema_check is not None:
                    rel_change = (_prev_ema_check - _grad_loss_ema) / (abs(_prev_ema_check) + 1e-8)
                    if rel_change < 2e-4:
                        _plateau_count += 1
                    else:
                        _plateau_count = 0
                    if _plateau_count >= 6:
                        print(f"[GRAD_PLATEAU_EXIT] step={step} frac={frac:.2f} ema={_grad_loss_ema:.4f}", flush=True)
                        break
                _prev_ema_check = _grad_loss_ema

        # ── Density overflow pass (ePlace-style Lagrangian update) ───────────
        # After main gradient, measure density overflow. If >15% of cells overflow
        # target density, run a second pass with lam_den boosted 3× to force spreading.
        # Guard: snapshot x_param before; revert if surrogate WL+density worsens.
        if time.time() < deadline - 15:
            with torch.no_grad():
                pos_ov = pos_ref.index_put((mov_t,), x_param)
                mx_ov = pos_ov[:n_den_macros, 0]
                my_ov = pos_ov[:n_den_macros, 1]
                left_x_ov  = mx_ov - sizes_den_t[:, 0] / 2
                right_x_ov = mx_ov + sizes_den_t[:, 0] / 2
                bot_y_ov   = my_ov - sizes_den_t[:, 1] / 2
                top_y_ov   = my_ov + sizes_den_t[:, 1] / 2
                cov_x_ov = torch.clamp(
                    torch.minimum(right_x_ov.unsqueeze(1), cell_rx.unsqueeze(0)) -
                    torch.maximum(left_x_ov.unsqueeze(1),  cell_lx.unsqueeze(0)),
                    min=0.0) / cell_w
                cov_y_ov = torch.clamp(
                    torch.minimum(top_y_ov.unsqueeze(1),   cell_ty.unsqueeze(0)) -
                    torch.maximum(bot_y_ov.unsqueeze(1),   cell_by.unsqueeze(0)),
                    min=0.0) / cell_h
                dens_ov = torch.einsum('ic,ir->rc', cov_x_ov, cov_y_ov)
                if _smooth_r > 0:
                    _ks = 2 * _smooth_r + 1
                    dens_ov = torch.nn.functional.avg_pool2d(
                        torch.nn.functional.pad(
                            dens_ov.unsqueeze(0).unsqueeze(0), [_smooth_r] * 4, mode='replicate'),
                        _ks, stride=1, padding=0).squeeze(0).squeeze(0)
                _dens_ov_check = dens_ov
                overflow_frac = float((_dens_ov_check > target_den_final).float().mean())
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
                    left_x2  = mx2 - sizes_den_t[:, 0] / 2
                    right_x2 = mx2 + sizes_den_t[:, 0] / 2
                    bot_y2   = my2 - sizes_den_t[:, 1] / 2
                    top_y2   = my2 + sizes_den_t[:, 1] / 2
                    cov_x2 = torch.clamp(
                        torch.minimum(right_x2.unsqueeze(1), cell_rx.unsqueeze(0)) -
                        torch.maximum(left_x2.unsqueeze(1),  cell_lx.unsqueeze(0)),
                        min=0.0) / cell_w
                    cov_y2 = torch.clamp(
                        torch.minimum(top_y2.unsqueeze(1),   cell_ty.unsqueeze(0)) -
                        torch.maximum(bot_y2.unsqueeze(1),   cell_by.unsqueeze(0)),
                        min=0.0) / cell_h
                    dens2 = torch.einsum('ic,ir->rc', cov_x2, cov_y2)
                    if _smooth_r > 0:
                        _ks = 2 * _smooth_r + 1
                        dens2 = torch.nn.functional.avg_pool2d(
                            torch.nn.functional.pad(
                                dens2.unsqueeze(0).unsqueeze(0), [_smooth_r] * 4, mode='replicate'),
                            _ks, stride=1, padding=0).squeeze(0).squeeze(0)
                    den_loss2 = torch.logsumexp(dens2.view(-1) / tau_den_h, 0) * tau_den_h / top_k_den_h

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
                    left_x_af  = mx_af - sizes_den_t[:, 0] / 2
                    right_x_af = mx_af + sizes_den_t[:, 0] / 2
                    bot_y_af   = my_af - sizes_den_t[:, 1] / 2
                    top_y_af   = my_af + sizes_den_t[:, 1] / 2
                    cov_x_af = torch.clamp(
                        torch.minimum(right_x_af.unsqueeze(1), cell_rx.unsqueeze(0)) -
                        torch.maximum(left_x_af.unsqueeze(1),  cell_lx.unsqueeze(0)),
                        min=0.0) / cell_w
                    cov_y_af = torch.clamp(
                        torch.minimum(top_y_af.unsqueeze(1),   cell_ty.unsqueeze(0)) -
                        torch.maximum(bot_y_af.unsqueeze(1),   cell_by.unsqueeze(0)),
                        min=0.0) / cell_h
                    dens_af = torch.einsum('ic,ir->rc', cov_x_af, cov_y_af)
                    if _smooth_r > 0:
                        _ks = 2 * _smooth_r + 1
                        dens_af = torch.nn.functional.avg_pool2d(
                            torch.nn.functional.pad(
                                dens_af.unsqueeze(0).unsqueeze(0), [_smooth_r] * 4, mode='replicate'),
                            _ks, stride=1, padding=0).squeeze(0).squeeze(0)
                    den_af = torch.logsumexp(dens_af.view(-1) / tau_den_h, 0) * tau_den_h / top_k_den_h
                    cost_after = float(wl_af + lam_den_end * den_af)
                    if cost_after > cost_before:
                        x_param.data = x_param_snapshot  # revert

        result = pos.copy()
        with torch.no_grad():
            result[mov_arr] = x_param.data.cpu().numpy().astype(np.float64)
        return result

    # ── GPU-batched gradient: N positions optimised simultaneously ───────────────

    def _gradient_phase_batched(self, pos_list, lam_dens, movable_idx, sizes_np, port_pos,
                                canvas_w, canvas_h, num_hard,
                                safe_nnp, nnmask, grid_rows, grid_cols, hpwl_norm, deadline,
                                fast_eng=None, cong_start_fracs=None, lam_cong_overrides=None,
                                target_den_starts=None):
        """Run N gradient starts simultaneously on GPU with batched tensor ops.

        Instead of N serial gradient passes, batch all N positions into one
        (N, n_mov, 2) tensor and compute WL+density+cong for all N in one forward.
        N× speedup on GPU — limited only by GPU memory (N≤8 typically fine).
        """
        N = len(pos_list)
        if N == 0 or time.time() >= deadline:
            return pos_list
        if cong_start_fracs is None:
            cong_start_fracs = [0.60] * N
        if lam_cong_overrides is None:
            lam_cong_overrides = [None] * N
        # Resolve per-element lam_cong_max: use override if provided, else default 0.30
        _lam_cong_maxs = [v if v is not None else 0.30 for v in lam_cong_overrides]

        n_mov = len(movable_idx)
        n_nets, max_nsz = safe_nnp.shape
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _tds_list = target_den_starts if target_den_starts is not None else [None] * N
        if dev.type != 'cuda':
            # CPU fallback: run sequentially
            out = []
            for i, pos in enumerate(pos_list):
                _tds_i = _tds_list[i] if i < len(_tds_list) else None
                out.append(self._gradient_phase(
                    pos, movable_idx, sizes_np, port_pos, canvas_w, canvas_h,
                    num_hard, safe_nnp, nnmask, grid_rows, grid_cols, hpwl_norm,
                    deadline, fast_eng=fast_eng, lam_den_override=lam_dens[i],
                    cong_start_frac=cong_start_fracs[i],
                    lam_cong_override=lam_cong_overrides[i],
                    target_den_start=_tds_i,
                    lam_den_start_frac=1.0 if _tds_i is not None else 0.05))
            return out
        # Memory guard: N × n_nets × max_nsz floats
        batch_mem = N * n_nets * max_nsz
        if batch_mem > 200_000_000:
            # Too large for batch: fall back to parallel sequential on separate streams
            results = [None] * N
            streams = [torch.cuda.Stream() for _ in range(N)]
            def _run_i(i):
                _tds_i2 = _tds_list[i] if i < len(_tds_list) else None
                with torch.cuda.stream(streams[i]):
                    results[i] = self._gradient_phase(
                        pos_list[i], movable_idx, sizes_np, port_pos, canvas_w, canvas_h,
                        num_hard, safe_nnp, nnmask, grid_rows, grid_cols, hpwl_norm,
                        deadline, fast_eng=fast_eng, lam_den_override=lam_dens[i],
                        cong_start_frac=cong_start_fracs[i],
                        lam_cong_override=lam_cong_overrides[i],
                        target_den_start=_tds_i2,
                        lam_den_start_frac=1.0 if _tds_i2 is not None else 0.05)
            import concurrent.futures as _cff
            with _cff.ThreadPoolExecutor(max_workers=N) as pool:
                list(pool.map(_run_i, range(N)))
            torch.cuda.synchronize()
            return [results[i] if results[i] is not None else pos_list[i] for i in range(N)]

        n_ports = port_pos.shape[0]
        mov_arr = np.array(movable_idx, dtype=np.int64)

        # Build batch tensor: pos has ALL macros (hard+soft), then append ports
        if n_ports > 0:
            pos_base_np = np.vstack([pos_list[0], port_pos]).astype(np.float32)
        else:
            pos_base_np = pos_list[0].astype(np.float32)

        # Build per-start movable init: (N, n_mov, 2)
        x_init = np.stack([p[mov_arr].astype(np.float32) for p in pos_list], axis=0)  # (N, n_mov, 2)
        x_batch = torch.nn.Parameter(torch.from_numpy(x_init).to(dev))  # (N, n_mov, 2)

        # Reference tensor (shared, no grad): (n_macros + n_ports, 2)
        pos_ref = torch.from_numpy(pos_base_np).to(dev)

        # Bounds
        hw_bnd = torch.from_numpy(sizes_np[mov_arr, 0].astype(np.float32)).to(dev) / 2  # (n_mov,)
        hh_bnd = torch.from_numpy(sizes_np[mov_arr, 1].astype(np.float32)).to(dev) / 2
        cw_t = float(canvas_w); ch_t = float(canvas_h)

        # Net tensors
        safe_t = torch.from_numpy(safe_nnp.astype(np.int64)).to(dev)    # (n_nets, max_nsz)
        mask_t = torch.from_numpy(nnmask).to(dev)                       # (n_nets, max_nsz)

        # Density tensors
        cell_w = canvas_w / grid_cols; cell_h = canvas_h / grid_rows
        cx_t = torch.arange(grid_cols, dtype=torch.float32, device=dev) * cell_w + cell_w / 2
        cy_t = torch.arange(grid_rows, dtype=torch.float32, device=dev) * cell_h + cell_h / 2
        n_den = num_hard
        sizes_den_t = torch.from_numpy(sizes_np[:n_den].astype(np.float32)).to(dev)
        hw_den = sizes_den_t[:, 0] / 2 + cell_w / 2   # (n_den,)
        hh_den = sizes_den_t[:, 1] / 2 + cell_h / 2

        # Congestion
        cong_enabled = False
        if fast_eng is not None:
            n_pairs = len(fast_eng['src'])
            if n_pairs < 2_000_000:
                src_ct = torch.from_numpy(fast_eng['src'].astype(np.int64)).to(dev)
                snk_ct = torch.from_numpy(fast_eng['snk'].astype(np.int64)).to(dev)
                w_ct   = torch.from_numpy(fast_eng['w'].astype(np.float32)).to(dev)
                gw_t = canvas_w / grid_cols; gh_t = canvas_h / grid_rows
                h_cap = gh_t * float(fast_eng['hpm'])
                v_cap = gw_t * float(fast_eng['vpm'])
                r_idx_t = torch.arange(grid_rows, dtype=torch.float32, device=dev)
                c_idx_t = torch.arange(grid_cols, dtype=torch.float32, device=dev)
                cong_enabled = (h_cap > 1e-9 and v_cap > 1e-9)

        canvas_diag = canvas_w + canvas_h
        gamma_start = canvas_diag * 0.04; gamma_end = canvas_diag * 0.004
        lr = canvas_diag * 0.002; lr_min = lr * 0.02
        beta_nag = 0.85; rms_eps = 1e-8

        # Per-start λ_den schedule — start at full lam_den (not 0.5×) so density
        # force is strong from step 1, pushing macros to spread before WL clusters them.
        lam_den_starts = torch.tensor(list(lam_dens), device=dev)   # (N,) — full lam_den from start
        lam_cong_maxs  = torch.tensor(_lam_cong_maxs, device=dev)   # (N,)

        # DREAMPlace preconditioner (shared across batch)
        _mov_to_k = {idx: k for k, idx in enumerate(movable_idx)}
        _deg = np.zeros(n_mov, dtype=np.float32)
        for _v in safe_nnp.ravel():
            _k = _mov_to_k.get(int(_v))
            if _k is not None: _deg[_k] += 1.0
        _deg = np.maximum(_deg, 1.0)
        _pc = 1.0 / np.sqrt(_deg); _pc /= max(float(_pc.max()), 1e-9)
        precond = torch.from_numpy(_pc).to(dev).unsqueeze(0).unsqueeze(2)  # (1, n_mov, 1)

        x_prev = x_batch.data.clone()
        t_start = time.time(); t_total = max(1.0, deadline - t_start)
        t_first = None
        BIG = 1e10

        while time.time() < deadline:
            t_s = time.time()
            frac = min(1.0, (t_s - t_start) / t_total)
            cur_lr = lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * frac))
            g = gamma_start * (gamma_end / gamma_start) ** frac
            lam_den = lam_den_starts

            # Nesterov lookahead
            with torch.no_grad():
                x_saved = x_batch.data.clone()
                x_batch.data = x_batch.data + beta_nag * (x_batch.data - x_prev)

            # Assemble full positions for each element: index only the nodes we need.
            # Avoid expensive clone of the full (N, n_total, 2) tensor each step.
            # For WL: gather node positions using safe_t (n_nets, max_nsz) indices.
            # split into movable vs fixed using a precomputed mask.
            # Simple approach: build lookup (n_total, 2) once per step without clone.
            # Use index_put to scatter x_batch into pos_ref copy for density.
            pos_cur = pos_ref.unsqueeze(0).repeat(N, 1, 1)   # (N, n_total, 2), no grad
            pos_cur[:, mov_arr, :] = x_batch                  # scatter movable positions

            # ── Batched smooth HPWL ──────────────────────────────────────────────
            # node_x: (N, n_nets, max_nsz)
            node_x = pos_cur[:, safe_t, 0]  # (N, n_nets, max_nsz) — advanced indexing
            node_y = pos_cur[:, safe_t, 1]
            mask_exp = mask_t.unsqueeze(0).expand(N, -1, -1)
            nx_max = torch.where(mask_exp, node_x, torch.full_like(node_x, -BIG))
            nx_min = torch.where(mask_exp, node_x, torch.full_like(node_x,  BIG))
            ny_max = torch.where(mask_exp, node_y, torch.full_like(node_y, -BIG))
            ny_min = torch.where(mask_exp, node_y, torch.full_like(node_y,  BIG))
            wl_x = (torch.logsumexp(nx_max / g, 2) + torch.logsumexp(-nx_min / g, 2)) * g
            wl_y = (torch.logsumexp(ny_max / g, 2) + torch.logsumexp(-ny_min / g, 2)) * g
            wl_loss = (wl_x + wl_y).sum(dim=1) / hpwl_norm  # (N,)

            # ── Batched bell-kernel density — hard macros only ──
            # Bell kernel (tent function) guarantees nonzero gradient everywhere,
            # even for small macros fully inside a grid cell.  The box-overlap
            # kernel has zero gradient for fully-interior macros (∂overlap/∂pos=0),
            # which caused ibm17 (517 small macros in 51×44 grid) to collapse to a
            # clustering trap — only cong gradient survived, pulling all macros together.
            mx = pos_cur[:, :n_den, 0]   # (N, n_den)
            my = pos_cur[:, :n_den, 1]
            dx_b = mx.unsqueeze(2) - cx_t.unsqueeze(0).unsqueeze(0)   # (N, n_den, grid_cols)
            dy_b = my.unsqueeze(2) - cy_t.unsqueeze(0).unsqueeze(0)   # (N, n_den, grid_rows)
            kx_b = torch.clamp(1.0 - dx_b.abs() / hw_den.view(1, n_den, 1), min=0.0)
            ky_b = torch.clamp(1.0 - dy_b.abs() / hh_den.view(1, n_den, 1), min=0.0)
            density = torch.einsum('nic,nir->nrc', kx_b, ky_b) / (cell_w * cell_h)  # (N, R, C)
            dens_flat = density.view(N, -1)  # (N, n_cells)
            # Anneal τ from 0.5 → 0.08 over optimization: early steps use smooth loss
            # (global density spreading), late steps use sharp loss focused on the densest
            # cells (better approximates oracle ABU = top-10% mean density).
            # τ=0.08 makes the logsumexp ≈ top-k max, closely matching oracle objective.
            _tau_b = max(0.08, 0.5 - 0.42 * frac)
            _topk_b = max(1, int(grid_rows * grid_cols * 0.10))
            den_loss = torch.logsumexp(dens_flat / _tau_b, 1) * _tau_b / _topk_b  # (N,)

            # ── Batched congestion ───────────────────────────────────────────────
            cong_loss = torch.zeros(N, device=dev)
            lam_cong_vec = torch.zeros(N, device=dev)
            if cong_enabled:
                cong_fracs_t = torch.tensor(cong_start_fracs, device=dev)
                active = frac > cong_fracs_t  # (N,) bool
                if active.any():
                    for _ni in range(N):
                        if not active[_ni].item(): continue
                        _csf = cong_start_fracs[_ni]
                        _cong_ramp = max(0.30, 1.0 - _csf)
                        _lc = float(lam_cong_maxs[_ni].item()) * min(1.0, (frac - _csf) / _cong_ramp)
                        lam_cong_vec[_ni] = _lc
                        xs = pos_cur[_ni, src_ct, 0]; ys = pos_cur[_ni, src_ct, 1]
                        xt = pos_cur[_ni, snk_ct, 0]; yt = pos_cur[_ni, snk_ct, 1]
                        al = 10.0 / max(float(canvas_w), float(canvas_h))
                        # H routing: at source row (bilinear), cols [min(xs,xt), max(xs,xt)]
                        r_s = ys / gh_t
                        row_w_H = torch.clamp(1.0 - (r_s.unsqueeze(1) - r_idx_t).abs(), min=0.0)
                        c_lo = torch.minimum(xs, xt) / gw_t
                        c_hi = torch.maximum(xs, xt) / gw_t
                        col_H = torch.sigmoid(al * (c_idx_t - c_lo.unsqueeze(1))) * \
                                torch.sigmoid(al * (c_hi.unsqueeze(1) - c_idx_t))
                        H_cong = torch.einsum('p,pr,pc->rc', w_ct, row_w_H, col_H) / h_cap
                        # V routing: at sink col (bilinear), rows [min(ys,yt), max(ys,yt)]
                        c_t_v = xt / gw_t
                        col_w_V = torch.clamp(1.0 - (c_t_v.unsqueeze(1) - c_idx_t).abs(), min=0.0)
                        r_lo = torch.minimum(ys, yt) / gh_t
                        r_hi = torch.maximum(ys, yt) / gh_t
                        row_V = torch.sigmoid(al * (r_idx_t - r_lo.unsqueeze(1))) * \
                                torch.sigmoid(al * (r_hi.unsqueeze(1) - r_idx_t))
                        V_cong = torch.einsum('p,pr,pc->rc', w_ct, row_V, col_w_V) / v_cap
                        excess = torch.clamp(H_cong + V_cong - 1.0, min=0.0)
                        n_cells = excess.numel()
                        top_k = max(1, int(n_cells * 0.05))
                        cong_loss[_ni] = torch.logsumexp(excess.view(-1) / 0.5, 0) * 0.5 / top_k

            # Total loss: sum over batch (each element independent, gradients independent)
            total_loss = (wl_loss + lam_den * den_loss + lam_cong_vec * cong_loss).sum()

            if x_batch.grad is not None: x_batch.grad = None
            total_loss.backward()

            with torch.no_grad():
                g_v = x_batch.grad.data * precond      # (N, n_mov, 2)
                g_rms = g_v.pow(2).mean(dim=2, keepdim=True).sqrt().clamp(min=rms_eps)
                x_batch.data = x_saved - cur_lr * g_v / g_rms
                x_prev = x_saved.clone()
                x_batch.grad = None
                # Snap noise
                x_batch.data[:, :, 0] += torch.empty_like(x_batch.data[:, :, 0]).uniform_(-cell_w/4, cell_w/4)
                x_batch.data[:, :, 1] += torch.empty_like(x_batch.data[:, :, 1]).uniform_(-cell_h/4, cell_h/4)
                # Clamp to canvas bounds
                x_batch.data[:, :, 0] = x_batch.data[:, :, 0].clamp(
                    min=hw_bnd.unsqueeze(0), max=cw_t - hw_bnd.unsqueeze(0))
                x_batch.data[:, :, 1] = x_batch.data[:, :, 1].clamp(
                    min=hh_bnd.unsqueeze(0), max=ch_t - hh_bnd.unsqueeze(0))

            if t_first is None:
                t_first = time.time() - t_s
                if t_first > 30.0: break  # too slow, abort batch

        # Extract results
        results = []
        with torch.no_grad():
            x_out = x_batch.data.cpu().numpy()  # (N, n_mov, 2)
        for i, pos in enumerate(pos_list):
            r = pos.copy()
            r[mov_arr] = x_out[i].astype(np.float64)
            results.append(r)
        return results

    # ── Soft macro gradient: WL + density-all (Step C2) ────────────────────────

    def _gradient_phase_soft(self, pos, soft_movable_idx, sizes_np, port_pos,
                              canvas_w, canvas_h, num_hard,
                              safe_nnp, nnmask, grid_rows, grid_cols, hpwl_norm, deadline,
                              cong_weight=0.0, fast_eng=None):
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
        t_first = None

        # ── Congestion-aware mode (gated on cong_weight > 0 for high-cong benchmarks) ──
        _cong_ok = False
        if cong_weight > 0.0 and fast_eng is not None:
            try:
                _gw_c = float(canvas_w) / grid_cols
                _gh_c = float(canvas_h) / grid_rows
                _h_cap_c = float(_gh_c * fast_eng['hpm'])
                _v_cap_c = float(_gw_c * fast_eng['vpm'])
                _n_pairs_c = len(fast_eng['src'])
                _mem_ok = (_n_pairs_c * max(grid_rows, grid_cols) <
                           (50_000_000 if dev.type == 'cuda' else 2_000_000))
                if _h_cap_c > 1e-9 and _v_cap_c > 1e-9 and _mem_ok:
                    _h_cap_ct = torch.tensor(_h_cap_c, device=dev)
                    _v_cap_ct = torch.tensor(_v_cap_c, device=dev)
                    _halloc_c = float(fast_eng.get('halloc', 0.5))
                    _valloc_c = float(fast_eng.get('valloc', 0.5))
                    _src_cct = torch.from_numpy(fast_eng['src'].astype(np.int64)).to(dev)
                    _snk_cct = torch.from_numpy(fast_eng['snk'].astype(np.int64)).to(dev)
                    _w_cct = torch.from_numpy(fast_eng['w'].astype(np.float32)).to(dev)
                    _r_idx_c = torch.arange(grid_rows, dtype=torch.float32, device=dev) + 0.5
                    _c_idx_c = torch.arange(grid_cols, dtype=torch.float32, device=dev) + 0.5
                    _cx_edges = torch.arange(grid_cols + 1, dtype=torch.float32, device=dev) * _gw_c
                    _cy_edges = torch.arange(grid_rows + 1, dtype=torch.float32, device=dev) * _gh_c
                    # Static hard macro blockage (hard positions fixed during soft opt)
                    with torch.no_grad():
                        _mx_h = pos_ref[:num_hard, 0]; _my_h = pos_ref[:num_hard, 1]
                        _hw_h = sizes_all_t[:num_hard, 0] / 2; _hh_h = sizes_all_t[:num_hard, 1] / 2
                        _x_ov_h = torch.clamp(
                            torch.minimum((_mx_h + _hw_h).unsqueeze(1), _cx_edges[1:].unsqueeze(0)) -
                            torch.maximum((_mx_h - _hw_h).unsqueeze(1), _cx_edges[:-1].unsqueeze(0)), min=0.0)
                        _y_ov_h = torch.clamp(
                            torch.minimum((_my_h + _hh_h).unsqueeze(1), _cy_edges[1:].unsqueeze(0)) -
                            torch.maximum((_my_h - _hh_h).unsqueeze(1), _cy_edges[:-1].unsqueeze(0)), min=0.0)
                        _static_cong = (
                            torch.einsum('ic,ir->rc', _x_ov_h, _y_ov_h / _gw_c) * (_valloc_c / _v_cap_ct) +
                            torch.einsum('ic,ir->rc', _x_ov_h / _gh_c, _y_ov_h) * (_halloc_c / _h_cap_ct)
                        ).detach()
                    _cong_matmul = (_n_pairs_c * grid_rows > 50_000)
                    _cong_ok = True
            except Exception:
                pass

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
            dens_flat = density.reshape(-1)
            den_loss = torch.logsumexp(dens_flat / tau_den, dim=0) * tau_den / top_k_den

            if _cong_ok:
                _al_c = 6.0
                _xs_c = pos_cur[_src_cct, 0]; _ys_c = pos_cur[_src_cct, 1]
                _xt_c = pos_cur[_snk_cct, 0]; _yt_c = pos_cur[_snk_cct, 1]
                _row_w_H = torch.clamp(1.0 - ((_ys_c / _gh_c).unsqueeze(1) - _r_idx_c).abs(), min=0.0)
                _c_lo_c = torch.minimum(_xs_c, _xt_c) / _gw_c
                _c_hi_c = torch.maximum(_xs_c, _xt_c) / _gw_c
                _col_H_c = (torch.sigmoid(_al_c * (_c_idx_c - _c_lo_c.unsqueeze(1))) *
                            torch.sigmoid(_al_c * (_c_hi_c.unsqueeze(1) - _c_idx_c)))
                if _cong_matmul:
                    _H_net = ((_w_cct.unsqueeze(1) * _row_w_H).T @ _col_H_c) / _h_cap_ct
                else:
                    _H_net = torch.einsum('p,pr,pc->rc', _w_cct, _row_w_H, _col_H_c) / _h_cap_ct
                _col_w_V = torch.clamp(1.0 - ((_xt_c / _gw_c).unsqueeze(1) - _c_idx_c).abs(), min=0.0)
                _r_lo_c = torch.minimum(_ys_c, _yt_c) / _gh_c
                _r_hi_c = torch.maximum(_ys_c, _yt_c) / _gh_c
                _row_V_c = (torch.sigmoid(_al_c * (_r_idx_c - _r_lo_c.unsqueeze(1))) *
                            torch.sigmoid(_al_c * (_r_hi_c.unsqueeze(1) - _r_idx_c)))
                if _cong_matmul:
                    _V_net = ((_w_cct.unsqueeze(1) * _row_V_c).T @ _col_w_V) / _v_cap_ct
                else:
                    _V_net = torch.einsum('p,pr,pc->rc', _w_cct, _row_V_c, _col_w_V) / _v_cap_ct
                # Soft macro blockage (dynamic: gradients flow through soft positions)
                _mx_s = pos_cur[num_hard:n_total_macros, 0]
                _my_s = pos_cur[num_hard:n_total_macros, 1]
                _x_ov_s = torch.clamp(
                    torch.minimum((_mx_s + sizes_all_t[num_hard:, 0] / 2).unsqueeze(1),
                                  _cx_edges[1:].unsqueeze(0)) -
                    torch.maximum((_mx_s - sizes_all_t[num_hard:, 0] / 2).unsqueeze(1),
                                  _cx_edges[:-1].unsqueeze(0)), min=0.0)
                _y_ov_s = torch.clamp(
                    torch.minimum((_my_s + sizes_all_t[num_hard:, 1] / 2).unsqueeze(1),
                                  _cy_edges[1:].unsqueeze(0)) -
                    torch.maximum((_my_s - sizes_all_t[num_hard:, 1] / 2).unsqueeze(1),
                                  _cy_edges[:-1].unsqueeze(0)), min=0.0)
                _cong_all = (_static_cong + _H_net + _V_net +
                             torch.einsum('ic,ir->rc', _x_ov_s, _y_ov_s / _gw_c) * (_valloc_c / _v_cap_ct) +
                             torch.einsum('ic,ir->rc', _x_ov_s / _gh_c, _y_ov_s) * (_halloc_c / _h_cap_ct))
                _exc_c = torch.clamp(_cong_all - 1.0, min=0.0)
                _top_k_c = max(1, int(_exc_c.numel() * 0.05))
                _cong_loss_sr = torch.logsumexp(_exc_c.reshape(-1) / 0.5, 0) * 0.5 / _top_k_c
                loss = wl_loss + lam_den * den_loss + cong_weight * _cong_loss_sr
            else:
                loss = wl_loss + lam_den * den_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_param.data[:, 0] = torch.max(
                    torch.min(x_param.data[:, 0], cw_t - hw_bnd), hw_bnd)
                x_param.data[:, 1] = torch.max(
                    torch.min(x_param.data[:, 1], ch_t - hh_bnd), hh_bnd)

            if t_first is None:
                t_first = time.time() - t_s
                if t_first > 10.0:
                    break

        result = pos.copy()
        with torch.no_grad():
            result[mov_arr] = x_param.data.cpu().numpy().astype(np.float64)
        return result

    def _b2b_quadratic_seed(self, pos_init, movable_idx, num_hard, nets_np,
                             net_weights_np, port_pos, sizes_np, canvas_w,
                             canvas_h, fixed_np, n_iters=4, anchor_mode='ports_soft',
                             init_mode='center', net_weight_fn=None):
        """Bound-to-bound (B2B) analytical placement for HARD movable macros.

        Standard wirelength-driven analytical placement (RePlAce/eplace seed):
        for each net, the two extreme pins (in x then y) get spring weights
        proportional to 1/(num_pins-1)/dist; other pins are anchored to those
        bounds. Iteratively solving Lx=bx converges to a WL-optimal continuous
        placement IGNORING overlap. The legalizer (`_resolve`) then spreads
        macros while preserving the topological structure.

        This produces a seed topologically distinct from initial.plc — the
        critical missing ingredient that lets us escape the same basin every
        run lands in.
        """
        try:
            from scipy.sparse import lil_matrix
            from scipy.sparse.linalg import cg as sp_cg
        except ImportError:
            return pos_init.copy()

        n_total = pos_init.shape[0]
        # Hard movable rows in the linear system
        hard_movable = [i for i in range(num_hard) if not fixed_np[i]]
        if not hard_movable:
            return pos_init.copy()
        h2row = {idx: r for r, idx in enumerate(hard_movable)}
        n_h = len(hard_movable)

        def get_xy(p, idx):
            if idx < n_total:
                return float(p[idx, 0]), float(p[idx, 1])
            pi = idx - n_total
            if 0 <= pi < len(port_pos):
                return float(port_pos[pi, 0]), float(port_pos[pi, 1])
            return 0.0, 0.0

        cur = pos_init.copy()
        # Initialise hard movable per init_mode to break initial.plc dependence
        cx, cy = canvas_w * 0.5, canvas_h * 0.5
        if init_mode == 'random':
            for idx in hard_movable:
                w_i, h_i = sizes_np[idx, 0], sizes_np[idx, 1]
                cur[idx, 0] = float(np.random.uniform(w_i / 2, canvas_w - w_i / 2))
                cur[idx, 1] = float(np.random.uniform(h_i / 2, canvas_h - h_i / 2))
        elif init_mode == 'corners':
            # Distribute around 4 corners — gives strong topological diversity
            corners = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for k, idx in enumerate(hard_movable):
                fx, fy = corners[k % 4]
                cur[idx, 0] = fx * canvas_w + np.random.uniform(-0.05 * canvas_w, 0.05 * canvas_w)
                cur[idx, 1] = fy * canvas_h + np.random.uniform(-0.05 * canvas_h, 0.05 * canvas_h)
        else:  # 'center'
            for idx in hard_movable:
                cur[idx, 0] = cx + np.random.uniform(-0.05 * canvas_w, 0.05 * canvas_w)
                cur[idx, 1] = cy + np.random.uniform(-0.05 * canvas_h, 0.05 * canvas_h)

        # Anchor mode: which non-movable pins act as anchors
        # 'ports_soft' (default): soft macros + ports
        # 'ports_only': only ports (treat soft as movable too — but we don't move them, they get the same anchor weight as if movable=∞)
        # In 'ports_only' mode, soft-macro pins are EXCLUDED from anchor set, so the
        # B2B system only sees ports → very different topology.
        _exclude_soft = (anchor_mode == 'ports_only')

        DELTA = max(1.0, 0.001 * (canvas_w + canvas_h))

        for _it in range(n_iters):
            Lx = lil_matrix((n_h, n_h), dtype=np.float64)
            Ly = lil_matrix((n_h, n_h), dtype=np.float64)
            bx = np.zeros(n_h)
            by = np.zeros(n_h)

            for ni, nodes in enumerate(nets_np):
                k = len(nodes)
                if k < 2:
                    continue
                w = float(net_weights_np[ni]) if ni < len(net_weights_np) else 1.0
                if net_weight_fn is not None:
                    w *= float(net_weight_fn(ni, k))
                # Collect pin coordinates
                pins = [int(v) for v in nodes]
                if _exclude_soft:
                    # Drop soft-macro pins: keep only hard-movable, hard-fixed, and port pins
                    pins = [p for p in pins
                            if (p < num_hard) or (p >= n_total)]
                    if len(pins) < 2:
                        continue
                xs = np.array([get_xy(cur, v)[0] for v in pins])
                ys = np.array([get_xy(cur, v)[1] for v in pins])
                # Find bound pins (min, max) in each axis
                xi_min = int(np.argmin(xs)); xi_max = int(np.argmax(xs))
                yi_min = int(np.argmin(ys)); yi_max = int(np.argmax(ys))
                # B2B model: each non-bound pin i has 2 springs (to min and max bound).
                # Spring weight = 2*w / ((k-1) * dist_to_bound).
                # Bound pins also have spring connecting them.
                wnorm = 2.0 * w / max(1, k - 1)

                def _add_spring(axis_lo, axis_hi, coords, L, b):
                    a, c = pins[axis_lo], pins[axis_hi]
                    ca, cc = coords[axis_lo], coords[axis_hi]
                    d = max(DELTA, abs(cc - ca))
                    sw = wnorm / d
                    # spring between a (lo) and c (hi)
                    self._b2b_add_edge(L, b, h2row, a, c, ca, cc, sw)
                    # other pins → both bounds
                    for j in range(len(pins)):
                        if j == axis_lo or j == axis_hi:
                            continue
                        p = pins[j]; cp = coords[j]
                        d_lo = max(DELTA, abs(cp - ca))
                        d_hi = max(DELTA, abs(cc - cp))
                        sw_lo = wnorm / d_lo
                        sw_hi = wnorm / d_hi
                        self._b2b_add_edge(L, b, h2row, p, a, cp, ca, sw_lo)
                        self._b2b_add_edge(L, b, h2row, p, c, cp, cc, sw_hi)

                _add_spring(xi_min, xi_max, xs, Lx, bx)
                _add_spring(yi_min, yi_max, ys, Ly, by)

            # Tikhonov regularization toward current pos to keep solver bounded
            reg = 1e-3
            for i in range(n_h):
                if Lx[i, i] < 1e-6: Lx[i, i] = 1e-6
                if Ly[i, i] < 1e-6: Ly[i, i] = 1e-6
                Lx[i, i] += reg; Ly[i, i] += reg
                bx[i] += reg * cur[hard_movable[i], 0]
                by[i] += reg * cur[hard_movable[i], 1]

            Lx = Lx.tocsr(); Ly = Ly.tocsr()
            x0 = np.array([cur[i, 0] for i in hard_movable])
            y0 = np.array([cur[i, 1] for i in hard_movable])
            sol_x, _ = sp_cg(Lx, bx, x0=x0, maxiter=150, atol=1e-5)
            sol_y, _ = sp_cg(Ly, by, x0=by * 0 + y0, maxiter=150, atol=1e-5)

            for r, idx in enumerate(hard_movable):
                w_i, h_i = sizes_np[idx, 0], sizes_np[idx, 1]
                cur[idx, 0] = float(np.clip(sol_x[r], w_i / 2, canvas_w - w_i / 2))
                cur[idx, 1] = float(np.clip(sol_y[r], h_i / 2, canvas_h - h_i / 2))

        return cur

    @staticmethod
    def _b2b_add_edge(L, b, h2row, u, v, cu, cv, sw):
        ru = h2row.get(u); rv = h2row.get(v)
        if ru is None and rv is None:
            return
        if ru is not None and rv is not None:
            L[ru, ru] += sw; L[rv, rv] += sw
            L[ru, rv] -= sw; L[rv, ru] -= sw
        elif ru is not None:
            L[ru, ru] += sw
            b[ru] += sw * cv
        else:
            L[rv, rv] += sw
            b[rv] += sw * cu

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
        p = pos.copy(); GAP = 0.01
        sw = sizes[:num_hard, 0]; sh = sizes[:num_hard, 1]
        for _ in range(max_iter):
            moved = False
            for i in range(num_hard - 1):
                xi, yi = p[i, 0], p[i, 1]
                wi, hi = sw[i], sh[i]
                xj = p[i+1:num_hard, 0]; yj = p[i+1:num_hard, 1]
                wj = sw[i+1:num_hard];   hj = sh[i+1:num_hard]
                ox = (wi + wj) * 0.5 - np.abs(xi - xj)
                oy = (hi + hj) * 0.5 - np.abs(yi - yj)
                mask = (ox > 0) & (oy > 0)
                if not np.any(mask):
                    continue
                moved = True
                px_m = mask & (ox < oy)
                py_m = mask & ~px_m
                sx = np.where(xi < xj, -1.0, 1.0)
                sy = np.where(yi < yj, -1.0, 1.0)
                if not fixed[i]:
                    dxi = np.sum(np.where(px_m, sx * (ox + GAP) * 0.5, 0.0))
                    dyi = np.sum(np.where(py_m, sy * (oy + GAP) * 0.5, 0.0))
                    p[i, 0] = np.clip(xi + dxi, wi * 0.5, canvas_w - wi * 0.5)
                    p[i, 1] = np.clip(yi + dyi, hi * 0.5, canvas_h - hi * 0.5)
                fj = fixed[i+1:num_hard]
                dxj = np.where(px_m & ~fj, -sx * (ox + GAP) * 0.5, 0.0)
                dyj = np.where(py_m & ~fj, -sy * (oy + GAP) * 0.5, 0.0)
                p[i+1:num_hard, 0] = np.clip(p[i+1:num_hard, 0] + dxj, wj * 0.5, canvas_w - wj * 0.5)
                p[i+1:num_hard, 1] = np.clip(p[i+1:num_hard, 1] + dyj, hj * 0.5, canvas_h - hj * 0.5)
            if not moved:
                break
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
