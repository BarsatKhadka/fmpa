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
from pathlib import Path
from itertools import permutations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _isfinite(x: float) -> bool:
    try:
        return bool(math.isfinite(float(x)))
    except Exception:
        return False

# Competition evaluation uses up to 3300s/benchmark. Keep that as the default so
# `uv run evaluate new-soln.py` reflects the intended regime unless overridden.
TIME_BUDGET = int(os.environ.get("PLACE_TIME_BUDGET", 3300))
TIME_GUARD = float(os.environ.get("PLACE_TIME_GUARD", "10.0"))
SEED = int(os.environ.get("PLACE_SEED", 20260417))
PROFILE = int(os.environ.get("PLACE_PROFILE", 0)) != 0
TRACE_DIR = os.environ.get("PLACE_TRACE_DIR", None)
EXACT_PARALLEL = int(os.environ.get("PLACE_EXACT_PARALLEL", "0")) != 0
ORACLE_THREADS = int(os.environ.get("PLACE_ORACLE_THREADS", "0"))
SURROGATE_LS = int(os.environ.get("PLACE_SURROGATE_LS", "0")) != 0
TOPO_SWEEP = int(os.environ.get("PLACE_TOPO_SWEEP", "0")) != 0
SMALL_ORACLE = int(os.environ.get("PLACE_SMALL_ORACLE", "0")) != 0
CALIB_ORACLE = int(os.environ.get("PLACE_CALIB_ORACLE", "0")) != 0

WL_ALPHA = 10.0
BASE_DENSITY_TARGET = 0.88
SPIRAL_POINTS = 20
_LEGAL_GAP_OVERRIDE = os.environ.get("PLACE_LEGAL_GAP", None)
LEGAL_GAP = float(_LEGAL_GAP_OVERRIDE) if _LEGAL_GAP_OVERRIDE is not None else 0.0
CONG_ALPHA = 8.0
TWO_PI = 2.0 * math.pi


# --- Optional exact-eval parallelism (small benchmarks only) -------------------
_EXACT_WORKER_BENCH = None
_EXACT_WORKER_PLC = None


def _exact_worker_init(netlist_file: str, plc_file: str, name: str) -> None:
    # Initialized inside subprocess.
    from macro_place.loader import load_benchmark

    bench, plc = load_benchmark(netlist_file, plc_file, name=name)
    globals()["_EXACT_WORKER_BENCH"] = bench
    globals()["_EXACT_WORKER_PLC"] = plc


def _exact_worker_eval(payload: Tuple[np.ndarray, np.ndarray]) -> float:
    hard, soft = payload
    bench = globals().get("_EXACT_WORKER_BENCH")
    plc = globals().get("_EXACT_WORKER_PLC")
    if bench is None or plc is None:
        return float("inf")
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    placement = bench.macro_positions.clone()
    placement[: bench.num_hard_macros] = torch.from_numpy(hard).float()
    if bench.num_soft_macros > 0:
        placement[bench.num_hard_macros : bench.num_macros] = torch.from_numpy(soft).float()
    ok, _ = validate_placement(placement, bench)
    if not ok:
        return float("inf")
    result = compute_proxy_cost(placement, bench, plc)
    return float(result["proxy_cost"])


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
    grad_hpwl_norm: float
    safe_nnp_np: np.ndarray
    nnmask_np: np.ndarray
    safe_nnp_t: torch.Tensor
    nnmask_t: torch.Tensor
    grad_safe_nnp_t: torch.Tensor
    grad_nnmask_t: torch.Tensor
    grad_net_weights_t: torch.Tensor
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
    def _trace_open(self, benchmark: Benchmark) -> None:
        if TRACE_DIR is None and not PROFILE:
            self._trace_fp = None
            return
        try:
            out_dir = Path(TRACE_DIR) if TRACE_DIR is not None else Path("runs")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            path = out_dir / f"trace-{benchmark.name}-seed{SEED}-tb{TIME_BUDGET}-{ts}.log"
            self._trace_fp = open(path, "w", encoding="utf-8")
            self._trace(f"trace_open path={str(path)}")
        except Exception:
            self._trace_fp = None

    def _trace(self, msg: str) -> None:
        fp = getattr(self, "_trace_fp", None)
        if fp is None:
            return
        try:
            dt = time.time() - float(getattr(self, "_place_t0", time.time()))
            left = float(self._time_left(getattr(self, "_place_t0", time.time())))
            fp.write(f"{dt:9.3f}s left={left:9.3f}s {msg}\n")
            fp.flush()
        except Exception:
            return

    def _trace_close(self) -> None:
        fp = getattr(self, "_trace_fp", None)
        if fp is None:
            return
        try:
            self._trace("trace_close")
            fp.close()
        except Exception:
            pass
        self._trace_fp = None

    def _legal_gap(self, data: "PreparedData") -> float:
        if _LEGAL_GAP_OVERRIDE is not None:
            return float(LEGAL_GAP)
        # Safety margin: keep a tiny non-zero gap for *all* designs to prevent float-epsilon
        # overlaps from slipping past one overlap test and triggering another (e.g. validate vs
        # compute_overlap_metrics). Small designs can afford a slightly larger buffer.
        if data.num_hard <= 320:
            return 5e-4
        return 1e-4

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        self._place_t0 = t0
        self._deadline = float(t0) + float(TIME_BUDGET) - float(TIME_GUARD)
        self._prof = {"oracle_calls": 0, "oracle_sec": 0.0}
        self._oracle_call_sec = None
        did_oracle_sa = False
        self._trace_open(benchmark)
        select_reserve = float(min(60.0, max(20.0, 0.12 * float(TIME_BUDGET))))
        # Reserve budget for final oracle-based selection; some stages run large GPU/CPU kernels
        # that must not starve the last-mile exact evaluation.
        self._select_reserve = float(select_reserve)
        small_budget = TIME_BUDGET <= 240

        def _plog(msg: str) -> None:
            if not PROFILE:
                return
            try:
                import sys

                dt = time.time() - self._place_t0
                left = self._time_left(self._place_t0)
                print(f"[PROFILE] {benchmark.name}: t={dt:.1f}s left={left:.1f}s {msg}", file=sys.stderr)
            except Exception:
                return

        def _tlog(msg: str) -> None:
            try:
                self._trace(msg)
            except Exception:
                return
        self._seed_everything(benchmark.name)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        _plog(f"start device={self._device.type}")
        self._trace(f"start device={self._device.type}")
        data = self._prepare(benchmark)
        _plog("prepared")
        self._trace("prepared")
        _plog(f"num_hard={int(getattr(data, 'num_hard', 0))} movable_hard={int(np.sum(getattr(data, 'movable_hard', np.zeros(0, dtype=bool))))}")
        self._trace(f"num_hard={int(getattr(data, 'num_hard', 0))} movable_hard={int(np.sum(getattr(data, 'movable_hard', np.zeros(0, dtype=bool))))}")
        plc = self._try_load_plc(benchmark)
        _plog(f"plc={'yes' if plc is not None else 'no'}")
        self._trace(f"plc={'yes' if plc is not None else 'no'}")
        if self._time_left(t0) <= 2.0:
            # Hard stop: budget exhausted during initialization (e.g. slow PLC load).
            placement = benchmark.macro_positions.clone()
            placement[: benchmark.num_hard_macros] = benchmark.macro_positions[: benchmark.num_hard_macros]
            self._trace("early_exit init_budget_exhausted")
            self._trace_close()
            return placement
        self._oracle_plcs = None
        if plc is not None and TIME_BUDGET >= 300:
            # Parallel oracle evaluation helps a lot on small/medium benchmarks (oracle is fast there).
            # Default behavior: auto-pick a small threadpool unless user overrides PLACE_ORACLE_THREADS.
            max_threads = 4 if data.num_hard <= 420 else 2
            if int(ORACLE_THREADS) > 0:
                n_threads = int(max(1, min(int(ORACLE_THREADS), int(max_threads))))
            else:
                try:
                    cpu = int(os.cpu_count() or 2)
                except Exception:
                    cpu = 2
                # Keep initialization overhead low on short budgets; PLC load dominates.
                if TIME_BUDGET < 900:
                    n_threads = 1
                else:
                    n_threads = int(max(1, min(int(max_threads), max(2, cpu // 4))))
            if n_threads > 1 and getattr(self, "_plc_netlist_path", None):
                self._oracle_plcs = self._make_plc_pool(benchmark, n_threads)
        _plog(f"oracle_threads={0 if self._oracle_plcs is None else len(self._oracle_plcs)}")
        self._trace(f"oracle_threads={0 if self._oracle_plcs is None else len(self._oracle_plcs)}")
        data.fast_cong_engine = self._build_fast_cong_engine(benchmark, plc)
        _plog(f"fast_cong={'yes' if data.fast_cong_engine is not None else 'no'}")
        self._trace(f"fast_cong={'yes' if data.fast_cong_engine is not None else 'no'}")
        if data.fast_cong_engine is not None:
            # Build a stable sample of routing pairs for fast congestion guidance in local search.
            try:
                src_len = int(len(data.fast_cong_engine["src"]))
                if src_len > 60_000 and "sample_idx" not in data.fast_cong_engine:
                    rng = np.random.default_rng((SEED ^ self._stable_seed(f"{benchmark.name}-cong-sample")) & 0xFFFFFFFF)
                    k = int(min(60_000, max(20_000, src_len // 6)))
                    data.fast_cong_engine["sample_idx"] = rng.choice(src_len, size=k, replace=False).astype(np.int64)
            except Exception:
                pass

        candidates: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        did_gpu_pop = False
        base_soft = self._clamp_soft_copy(data.init_soft, data)
        _plog("seeded")
        self._trace("seeded")
        # Always include the raw initial placement candidate (often surprisingly strong).
        init_hard = data.init_hard.copy().astype(np.float32)
        self._clamp_hard(init_hard, data)
        candidates.append((init_hard, base_soft.copy(), "initial-raw", self._cheap_score(init_hard, base_soft, data)))
        init_gap_seed = None
        # If the initial placement is near-legal but fails strict overlap checks due to float-epsilon,
        # apply a minimal gap-fix (preserves topology far better than a full re-legalization).
        if self._time_left(t0) >= 3:
            try:
                init_gap = self._tiny_fix_hard(init_hard, data, rounds=120)
                init_gap_seed = init_gap.astype(np.float32, copy=False)
                candidates.append(
                    (
                        init_gap.astype(np.float32),
                        base_soft.copy(),
                        "initial-gapfix",
                        self._cheap_score(init_gap.astype(np.float32), base_soft, data),
                    )
                )
            except Exception:
                pass
        if data.num_hard <= 320 and TIME_BUDGET <= 240 and self._time_left(t0) >= 20:
            # Small designs: the stronger O(N^2) legalizer preserves structure better and is still fast enough.
            legacy_hard = self._legalize_hard(data.init_hard, data)
        else:
            legacy_hard = self._fast_legalize_hard(
                data.init_hard, data, sweeps=6 if data.num_hard <= 420 else 5
            )
        _plog("legacy_legalized")
        self._trace("legacy_legalized")
        legacy_hard_fast = legacy_hard.copy().astype(np.float32, copy=False)
        legacy_valid = self._placement_is_valid(legacy_hard, base_soft, benchmark, data)
        legacy_fast_valid = bool(legacy_valid)
        oracle_fast = False
        if (
            plc is not None
            and legacy_valid
            and TIME_BUDGET >= 300
            and data.num_hard <= 420
            and self._time_left(t0) >= (select_reserve + 70)
        ):
            try:
                # One probe to estimate oracle call time; drives scheduling decisions.
                _ = self._proxy_cost_if_valid(legacy_hard, base_soft, benchmark, plc, data)
                est = float(getattr(self, "_oracle_call_sec", 99.0) or 99.0)
                oracle_fast = bool(est <= 3.0)
                _plog(f"oracle_fast={int(oracle_fast)} est={est:.2f}")
                _tlog(f"oracle_fast={int(oracle_fast)} est={est:.2f}")
            except Exception:
                oracle_fast = False

        # Soft refinement can materially improve proxy on some designs even when oracle is fast.
        # Keep it short and selection-reserve safe.
        if (
            self._device.type == "cuda"
            and data.num_soft > 0
            and TIME_BUDGET >= 300
            and self._time_left(t0) >= (select_reserve + 45)
        ):
            soft_t0 = time.time()
            steps = 20 if oracle_fast else 40
            refined_soft = self._gpu_soft_refine(legacy_hard, base_soft, data, soft_t0, steps=int(steps))
            candidates.append(
                (
                    legacy_hard.copy(),
                    refined_soft.copy(),
                    "legacy-soft-gpu",
                    self._cheap_score(legacy_hard, refined_soft, data),
                )
            )
        # Large designs: if fast legalization fails, optionally add a guaranteed-legal grid-pack candidate
        # (but do not replace the fast seed; we still want topology-preserving starts for latent search).
        if not legacy_valid and data.num_hard > 520 and self._time_left(t0) >= 18:
            try:
                legacy_gp = self._grid_pack_legalize_hard(data.init_hard, data)
                legacy_gp = self._fast_legalize_hard(legacy_gp, data, sweeps=5)
                gp_valid = self._placement_is_valid(legacy_gp, base_soft, benchmark, data)
                _plog(f"legacy_gridpack_valid={int(gp_valid)}")
                if gp_valid:
                    candidates.append(
                        (
                            legacy_gp.astype(np.float32),
                            base_soft.copy(),
                            "legacy-gridpack",
                            self._cheap_score(legacy_gp.astype(np.float32), base_soft, data),
                        )
                    )
            except Exception:
                pass
        _plog(f"legacy_valid={int(legacy_valid)}")
        self._trace(f"legacy_valid={int(legacy_valid)} oracle_fast={int(oracle_fast)}")
        boost_seed_hard: Optional[np.ndarray] = None

        # Benchmark-calibrated surrogate weighting (at most one exact call).
        # This improves cheap-score ranking when WL/density/congestion scales differ a lot.
        self._surrogate_calib = None
        if plc is not None and legacy_valid:
            base_cheap_comps = self._cheap_components(legacy_hard, base_soft, data)
            base_exact_comps = None
            if (
                TIME_BUDGET <= 240
                and data.num_hard >= 200
                and data.num_hard <= 420
                and self._time_left(t0) >= 120
                and CALIB_ORACLE
            ):
                base_exact_comps = self._proxy_components_if_valid(legacy_hard, base_soft, benchmark, plc, data)
            self._surrogate_calib = self._init_surrogate_calib(base_cheap_comps, base_exact_comps)
            # Adaptive density target: spread more aggressively on congestion-heavy cases.
            cong_ref = float(base_exact_comps["cong"]) if base_exact_comps is not None else float(base_cheap_comps.get("cong", 0.0))
            dens_t = float(BASE_DENSITY_TARGET)
            if cong_ref > 1.60:
                dens_t = 0.70
            elif cong_ref > 1.35:
                dens_t = 0.76
            elif cong_ref > 1.18:
                dens_t = 0.82
            self._density_target = float(np.clip(dens_t, 0.62, 0.92))
            # Rescore existing candidates under the calibrated surrogate.
            rescored = []
            for h, s, n, _ in candidates:
                rescored.append((h, s, n, float(self._cheap_score(h, s, data))))
            candidates = rescored

        # Make sure the "legacy" legalized seed is always in the candidate set before any
        # oracle-driven search (otherwise SA can miss a very strong basin).
        candidates.append(
            (legacy_hard, base_soft.copy(), "legacy-base", self._cheap_score(legacy_hard, base_soft, data))
        )

        # IncreMacro-style blockage breaking: diagnose poorly-supported central macros and
        # evacuate a subset toward the periphery while preserving order on each side.
        if (
            TIME_BUDGET >= 300
            and data.num_hard >= 120
            and self._time_left(t0) >= (select_reserve + 26)
        ):
            try:
                periphery_candidates: List[Tuple[np.ndarray, float]] = []
                periph_inputs = [(0.60, 0.75)]
                cong_hint = None
                if self._surrogate_calib is not None or plc is not None:
                    try:
                        cong_hint = float(self._cheap_components(legacy_hard, base_soft, data).get("cong", 0.0))
                    except Exception:
                        cong_hint = None
                if cong_hint is not None and cong_hint >= 1.20:
                    periph_inputs.append((0.78, 1.05))
                for strength, hotspot_weight in periph_inputs:
                    if self._time_left(t0) < (select_reserve + 12):
                        break
                    evac = self._periphery_evacuate(
                        legacy_hard,
                        base_soft,
                        data,
                        strength=float(strength),
                        hotspot_weight=float(hotspot_weight),
                    )
                    if evac is None:
                        continue
                    candidates.append(
                        (
                            evac.astype(np.float32),
                            base_soft.copy(),
                            f"periphery-s{strength:.2f}",
                            self._cheap_score(evac.astype(np.float32), base_soft, data),
                        )
                    )
                    periphery_candidates.append((evac.astype(np.float32), float(self._cheap_score(evac.astype(np.float32), base_soft, data))))
                if (
                    plc is not None
                    and legacy_valid
                    and periphery_candidates
                    and getattr(self, "_oracle_call_sec", None) is not None
                    and float(getattr(self, "_oracle_call_sec", 99.0) or 99.0) <= 8.5
                    and self._time_left(t0) >= (select_reserve + 18)
                ):
                    best_exact_h = None
                    best_exact_proxy = float("inf")
                    best_exact_cheap = float("inf")
                    for cand_h, cand_cheap in periphery_candidates[:2]:
                        exact = self._proxy_cost_if_valid(cand_h, base_soft, benchmark, plc, data)
                        if math.isfinite(exact) and float(exact) < best_exact_proxy:
                            best_exact_proxy = float(exact)
                            best_exact_h = cand_h.copy()
                            best_exact_cheap = float(cand_cheap)
                    if best_exact_h is not None:
                        candidates.append(
                            (
                                best_exact_h.astype(np.float32),
                                base_soft.copy(),
                                "periphery-exact-best",
                                float(best_exact_cheap - 1e-3),
                            )
                        )
                _plog("periphery_candidates_done")
                _tlog("periphery_candidates_done")
            except Exception:
                pass

        # Cheap global topology perturbations (affine flips/rotations) to escape stubborn basins.
        if TIME_BUDGET >= 300 and self._time_left(t0) >= (select_reserve + 22):
            try:
                for kind in ["flipx", "flipy", "rot180", "rot90"]:
                    if self._time_left(t0) < (select_reserve + 12):
                        break
                    h_aff = self._affine_transform_world(legacy_hard_fast, data, kind)
                    h_aff = self._fast_legalize_hard(h_aff, data, sweeps=3)
                    candidates.append(
                        (h_aff, base_soft.copy(), f"affine-{kind}", self._cheap_score(h_aff, base_soft, data))
                    )
            except Exception:
                pass

        # When oracle is fast, do a *small* early GPU population pass before oracle SA so we get
        # genuinely different basins without spending most of the budget inside SA.
        if (
            plc is not None
            and legacy_valid
            and oracle_fast
            and self._device.type == "cuda"
            and TIME_BUDGET >= 300
            and TIME_BUDGET <= 600
            and self._time_left(t0) >= (select_reserve + 150)
        ):
            try:
                seeds = [legacy_hard_fast]
                if init_gap_seed is not None:
                    seeds.append(init_gap_seed)
                pop = self._gpu_population_eplace(
                    data,
                    t0,
                    soft_seed=base_soft.copy(),
                    k=2,
                    seed_hards=seeds,
                    iters_scale=0.70,
                )
                for hard, soft, name, score in pop:
                    candidates.append((hard, soft, f"{name}-preosa", score))
                did_gpu_pop = True if pop else did_gpu_pop
                _plog("gpu_pop_preosa300_done")
                _tlog("gpu_pop_preosa300_done")
            except Exception:
                pass

        # Oracle-first scheduling: if oracle is fast, do an early multistart oracle SA sweep and
        # use its winner as the latent seed for downstream stages.
        latent_seed_hard = legacy_hard_fast.copy()
        latent_seed_soft = base_soft.copy()
        # Exploration-before-exploitation: when oracle is fast, spend a small chunk early on
        # global GPU exploration to find new basins, then let oracle SA exploit the best starts.
        if (
            plc is not None
            and legacy_valid
            and oracle_fast
            and self._device.type == "cuda"
            and TIME_BUDGET >= 600
            and data.num_hard >= 200
            and self._time_left(t0) >= (select_reserve + 260)
        ):
            try:
                seeds = [legacy_hard_fast]
                if init_gap_seed is not None:
                    seeds.append(init_gap_seed)
                seeds.append(self._fast_legalize_hard(self._quadrant_permute_world(legacy_hard_fast, data), data, sweeps=3))
                pop = self._gpu_population_eplace(
                    data,
                    t0,
                    soft_seed=base_soft.copy(),
                    k=3,
                    seed_hards=seeds,
                    iters_scale=0.9,
                )
                for hard, soft, name, score in pop:
                    candidates.append((hard, soft, f"{name}-preosa", score))
                _plog("gpu_pop_preosa_done")
                _tlog("gpu_pop_preosa_done")
            except Exception:
                pass

        if (
            plc is not None
            and legacy_valid
            and oracle_fast
            and TIME_BUDGET >= 300
            and data.num_hard <= 420
            and self._time_left(t0) >= (select_reserve + 120)
        ):
            try:
                # Cap SA stage budget so we don't starve downstream stages on short/medium budgets.
                est = float(getattr(self, "_oracle_call_sec", 2.0) or 2.0)
                stage_budget = float(min(self._time_left(t0) - select_reserve, 0.48 * float(TIME_BUDGET - TIME_GUARD)))
                max_calls = int(min(2400, max(120, stage_budget / max(est, 0.4))))
                if TIME_BUDGET <= 600:
                    # For 5-minute budgets, keep SA from monopolizing the run (ibm01 tends to plateau under SA).
                    max_calls = int(min(int(max_calls), 70 if data.num_hard <= 280 else 55))
                sa = self._oracle_sa_multistart(
                    candidates,
                    benchmark,
                    plc,
                    data,
                    t0,
                    max_calls=max_calls,
                )
                if sa is not None:
                    sa_h, sa_s, sa_name, sa_cheap = sa
                    candidates.append((sa_h, sa_s, sa_name, sa_cheap))
                    latent_seed_hard = sa_h.copy()
                    latent_seed_soft = sa_s.copy()
                    boost_seed_hard = sa_h.copy()
                    did_oracle_sa = True
                    _plog("oracle_sa_early_done")
                    _tlog("oracle_sa_early_done")
            except Exception:
                pass
        if not small_budget and self._time_left(t0) >= 80:
            pair_hard = self._legalize_hard(data.init_hard, data)
        else:
            pair_hard = self._fast_legalize_hard(data.init_hard, data, sweeps=5)
        pair_hard_fast = pair_hard.copy().astype(np.float32, copy=False)
        if data.num_hard > 520 and not self._placement_is_valid(pair_hard, base_soft, benchmark, data) and self._time_left(t0) >= 18:
            try:
                pair_gp = self._grid_pack_legalize_hard(pair_hard, data)
                pair_gp = self._fast_legalize_hard(pair_gp, data, sweeps=5)
                if self._placement_is_valid(pair_gp, base_soft, benchmark, data):
                    candidates.append(
                        (
                            pair_gp.astype(np.float32),
                            base_soft.copy(),
                            "pair-gridpack",
                            self._cheap_score(pair_gp.astype(np.float32), base_soft, data),
                        )
                    )
            except Exception:
                pass
        if not small_budget and self._time_left(t0) >= 28:
            shelf_hard = self._tiny_fix_hard(self._shelf_legalize_hard(data.init_hard, data), data)
        else:
            shelf_hard = pair_hard.copy()
        shelf_hard_fast = shelf_hard.copy().astype(np.float32, copy=False)
        if data.num_hard > 520:
            # Keep a "fast/structure-preserving" seed for latent modes even if we later grid-pack for legality.
            shelf_hard_fast = pair_hard_fast.copy().astype(np.float32, copy=False)
        pair_candidate = (pair_hard, base_soft.copy(), "pair-base", self._cheap_score(pair_hard, base_soft, data))
        shelf_candidate = (
            shelf_hard,
            base_soft.copy(),
            "shelf-base",
            self._cheap_score(shelf_hard, base_soft, data),
        )
        extra_hard_seeds: List[np.ndarray] = [
            legacy_hard_fast.copy(),
            pair_hard_fast.copy(),
            shelf_hard_fast.copy(),
            pair_hard,
            shelf_hard,
        ]
        if boost_seed_hard is not None:
            extra_hard_seeds.append(boost_seed_hard)
        if legacy_valid:
            candidates.append(pair_candidate if pair_candidate[3] <= shelf_candidate[3] else shelf_candidate)
        else:
            candidates.append(pair_candidate)
            candidates.append(shelf_candidate)

        # Topology sweep (short budgets): quick GPU gradient "worlds" with different density weights.
        # Goal: generate structurally different seeds before analytical/global refinement.
        if (
            self._device.type == "cuda"
            and plc is not None
            and TIME_BUDGET <= 240
            and TOPO_SWEEP
            and data.num_hard <= 260
            and self._time_left(t0) >= 55
        ):
            topo_lams = [0.25, 1.00]
            ti = 0
            for lam in topo_lams:
                if self._time_left(t0) < 22:
                    break
                anchor = legacy_hard.copy()
                hard_t = self._gradient_stage(
                    legacy_hard.copy(),
                    base_soft.copy(),
                    anchor,
                    data,
                    t0,
                    steps=12,
                    lr=0.060,
                    density_weight=float(lam),
                    overlap_weight=0.0,
                    anchor_weight=0.0,
                    snap_noise=0.60,
                    cong_weight=0.0,
                )
                hard_t = self._fast_legalize_hard(hard_t, data, sweeps=3)
                candidates.append((hard_t, base_soft.copy(), f"topo-{ti}-lam{lam:.2f}", self._cheap_score(hard_t, base_soft, data)))
                ti += 1
            _plog("topo_sweep_done")

        # Quick GPU global refinement for short budgets: cheap, coordinated moves that can beat
        # "just legalization" on congestion-heavy cases without committing to long analytical phases.
        if (
            self._device.type == "cuda"
            and plc is not None
            and TIME_BUDGET <= 240
            and data.num_hard <= 260
            and self._time_left(t0) >= 32
        ):
            quick = self._gpu_population_eplace(data, t0, soft_seed=base_soft.copy(), k=2)
            for hard, soft, name, score in quick[:2]:
                candidates.append((hard, soft, f"{name}-quick", score))
            _plog("gpu_pop_quick_done")

        # soln.py-style multi-world topology sweep (GPU-batched): different density weights
        # create distinct topologies, then we rely on cheap-score selection.
        if (
            self._device.type == "cuda"
            and plc is not None
            and TIME_BUDGET <= 240
            and 260 < data.num_hard <= 520
            and self._time_left(t0) >= 60
        ):
            dens_target = float(getattr(self, "_density_target", BASE_DENSITY_TARGET))
            if dens_target <= 0.76:
                inflate = 1.12
            elif dens_target <= 0.82:
                inflate = 1.08
            else:
                inflate = 1.03
            world_lams = [0.25, 0.55, 1.00, 1.60, 0.35]
            seeds = [
                legacy_hard.copy(),
                pair_hard.copy(),
                self._fast_legalize_hard(self._quadrant_permute_world(legacy_hard, data), data, sweeps=4),
            ]
            pop = self._gpu_population_eplace(
                data,
                t0,
                soft_seed=base_soft.copy(),
                k=5,
                seed_hards=seeds,
                world_den_w=world_lams,
                den_schedule=True,
                inflate_factor=float(inflate),
            )
            for hard, soft, name, score in pop:
                candidates.append((hard, soft, f"{name}-mw", score))
            _plog("gpu_multiworld_done")

            # Operator scheduling / decomposition (bandit-like): allocate remaining time across
            # a few global operators that generate genuinely different topologies.
            mw = [c for c in candidates if isinstance(c[2], str) and "-mw" in c[2]]
            def _best_cheap_now() -> float:
                try:
                    return float(min(candidates, key=lambda c: float(c[3]))[3])
                except Exception:
                    return float("inf")

            op_stats = {"conggrad": [0, 0.0], "stress": [0, 0.0], "bin": [0, 0.0]}
            for _round in range(3):
                if self._time_left(t0) < 18:
                    break
                base_before = _best_cheap_now()

                # Choose op by optimistic UCB on improvement-per-try (cheap-score delta).
                total_tries = sum(int(v[0]) for v in op_stats.values()) + 1
                def ucb(name: str) -> float:
                    tries, tot = op_stats[name]
                    avg = (tot / max(1, tries))
                    prior = 0.015  # small optimistic prior
                    bonus = 0.015 * math.sqrt(math.log(float(total_tries)) / float(tries + 1))
                    return float((avg if tries > 0 else prior) + bonus)

                # Feasibility masks.
                feasible = []
                if data.fast_cong_engine is not None and self._time_left(t0) >= 22 and mw:
                    feasible.append("conggrad")
                if self._time_left(t0) >= 22:
                    feasible.append("stress")
                if self._time_left(t0) >= 26:
                    feasible.append("bin")
                if not feasible:
                    break

                op = max(feasible, key=ucb)

                if op == "conggrad" and mw:
                    hard_b, soft_b, name_b, _ = min(mw, key=lambda c: float(c[3]))
                    anchor = hard_b.copy()
                    hard_c = self._gradient_stage(
                        hard_b.copy(),
                        soft_b.copy(),
                        anchor,
                        data,
                        t0,
                        steps=22,
                        lr=0.032,
                        density_weight=0.22,
                        overlap_weight=0.0,
                        anchor_weight=0.010,
                        snap_noise=0.25,
                        cong_weight=0.28,
                    )
                    hard_c = self._fast_legalize_hard(hard_c, data, sweeps=4)
                    if data.num_hard <= 420 and self._time_left(t0) >= 10:
                        try:
                            if self._exact_hard_overlap_area(hard_c.astype(np.float64), data) > 1e-10:
                                hard_c = self._legalize_hard(hard_c, data)
                        except Exception:
                            pass
                    candidates.append((hard_c, soft_b.copy(), f"{name_b}-conggrad", self._cheap_score(hard_c, soft_b, data)))
                    extra_hard_seeds.append(hard_c.copy())
                    _plog("mw_conggrad_done")

                elif op == "stress":
                    base_h = min(candidates, key=lambda c: float(c[3]))[0]
                    se = self._stress_embed_project(base_h, base_soft, data, t0, iters=14)
                    if se is not None and self._placement_is_valid(se, base_soft, benchmark, data):
                        candidates.append((se, base_soft.copy(), "stress-embed", self._cheap_score(se, base_soft, data)))
                        extra_hard_seeds.append(se.copy())
                        _plog("stress_embed_done")

                elif op == "bin":
                    base_h = min(candidates, key=lambda c: float(c[3]))[0]
                    all_pos = self._all_pos_np(base_h, base_soft, data)
                    net_centers = self._net_box_centers_np(all_pos, data)
                    tgt = np.zeros((data.num_hard, 2), dtype=np.float32)
                    for i in range(data.num_hard):
                        tgt[i] = self._macro_net_target(int(i), net_centers, data).astype(np.float32)
                    bsh = self._capacitated_bin_shuffle(
                        base_h,
                        base_soft,
                        data,
                        t0,
                        bin_rows=8,
                        bin_cols=8,
                        target=tgt,
                    )
                    if bsh is not None and self._placement_is_valid(bsh, base_soft, benchmark, data):
                        candidates.append((bsh, base_soft.copy(), "bin-shuffle", self._cheap_score(bsh, base_soft, data)))
                        extra_hard_seeds.append(bsh.copy())
                        _plog("bin_shuffle_done")

                base_after = _best_cheap_now()
                impr = max(0.0, float(base_before - base_after))
                op_stats[op][0] += 1
                op_stats[op][1] += float(impr)
                # If nothing improves, don't thrash; leave time for downstream stages.
                if impr < 1e-4 and _round >= 1:
                    break

        # Large designs: run a lighter multiworld GPU pass (helps escape bad legalization basins).
        if (
            self._device.type == "cuda"
            and plc is not None
            and TIME_BUDGET <= 240
            and data.num_hard > 520
            and self._time_left(t0) >= 55
        ):
            dens_target = float(getattr(self, "_density_target", BASE_DENSITY_TARGET))
            inflate = 1.10 if dens_target <= 0.86 else 1.06
            world_lams = [0.30, 0.85, 1.40]
            seeds = [
                legacy_hard.copy(),
                self._fast_legalize_hard(self._quadrant_permute_world(legacy_hard, data), data, sweeps=4),
                self._fast_legalize_hard(data.spectral_xy, data, sweeps=4),
            ]
            pop = self._gpu_population_eplace(
                data,
                t0,
                soft_seed=base_soft.copy(),
                k=3,
                seed_hards=seeds,
                world_den_w=world_lams,
                den_schedule=True,
                inflate_factor=float(inflate),
            )
            for hard, soft, name, score in pop:
                candidates.append((hard, soft, f"{name}-lg", score))
            _plog("gpu_large_multiworld_done")

            # Facility-location style reshuffle on large designs: coarse capacitated assignment + packing.
            if data.fast_cong_engine is not None and self._time_left(t0) >= 30:
                all_pos = self._all_pos_np(legacy_hard_fast, base_soft, data)
                net_centers = self._net_box_centers_np(all_pos, data)
                tgt = np.zeros((data.num_hard, 2), dtype=np.float32)
                for i in range(data.num_hard):
                    tgt[i] = self._macro_net_target(int(i), net_centers, data).astype(np.float32)
                bsh = self._capacitated_bin_shuffle(
                    legacy_hard_fast,
                    base_soft,
                    data,
                    t0,
                    bin_rows=10,
                    bin_cols=10,
                    target=tgt,
                )
                if bsh is not None and self._placement_is_valid(bsh, base_soft, benchmark, data):
                    candidates.append((bsh, base_soft.copy(), "bin-shuffle-lg", self._cheap_score(bsh, base_soft, data)))
                    extra_hard_seeds.append(bsh.copy())
                    _plog("bin_shuffle_lg_done")

        # If we still have plenty of budget (small/medium), run a larger batched GPU pass.
        if (
            self._device.type == "cuda"
            and plc is not None
            and TIME_BUDGET <= 240
            and data.num_hard <= 260
            and self._time_left(t0) >= 85
        ):
            dens_target = float(getattr(self, "_density_target", BASE_DENSITY_TARGET))
            inflate = 1.10 if dens_target <= 0.82 else 1.06 if dens_target <= 0.88 else 1.03
            # Small-design multiworld: include both "cluster" and "spread" worlds.
            world_lams = [0.08, 0.25, 0.60, 1.40]
            seeds = [
                legacy_hard.copy(),
                pair_hard.copy(),
                self._fast_legalize_hard(self._quadrant_permute_world(legacy_hard, data), data, sweeps=4),
            ]
            pop = self._gpu_population_eplace(
                data,
                t0,
                soft_seed=base_soft.copy(),
                k=4,
                seed_hards=seeds,
                world_den_w=world_lams,
                den_schedule=True,
                inflate_factor=float(inflate),
            )
            for hard, soft, name, score in pop:
                candidates.append((hard, soft, f"{name}-mid", score))
            _plog("gpu_pop_mid_done")

            # Small-design operator scheduling: try a couple of global reshuffles, but keep
            # enough time for final selection.
            if self._time_left(t0) >= 40:
                def _best_cheap_now() -> float:
                    try:
                        return float(min(candidates, key=lambda c: float(c[3]))[3])
                    except Exception:
                        return float("inf")

                for _ in range(2):
                    if self._time_left(t0) < 28:
                        break
                    before = _best_cheap_now()
                    # Try stress embed first (often cheap and topology-changing).
                    base_h = min(candidates, key=lambda c: float(c[3]))[0]
                    se = self._stress_embed_project(base_h, base_soft, data, t0, iters=10)
                    if se is not None and self._placement_is_valid(se, base_soft, benchmark, data):
                        candidates.append((se, base_soft.copy(), "stress-embed-sm", self._cheap_score(se, base_soft, data)))
                        _plog("stress_embed_sm_done")
                    # Then try capacitated bin shuffle (facility-location).
                    if self._time_left(t0) >= 22:
                        all_pos = self._all_pos_np(base_h, base_soft, data)
                        net_centers = self._net_box_centers_np(all_pos, data)
                        tgt = np.zeros((data.num_hard, 2), dtype=np.float32)
                        for i in range(data.num_hard):
                            tgt[i] = self._macro_net_target(int(i), net_centers, data).astype(np.float32)
                        bsh = self._capacitated_bin_shuffle(base_h, base_soft, data, t0, bin_rows=7, bin_cols=7, target=tgt)
                        if bsh is not None and self._placement_is_valid(bsh, base_soft, benchmark, data):
                            candidates.append((bsh, base_soft.copy(), "bin-shuffle-sm", self._cheap_score(bsh, base_soft, data)))
                            _plog("bin_shuffle_sm_done")
                    after = _best_cheap_now()
                    if float(before - after) < 1e-4:
                        break

            # One more cheap coordinate-descent style refinement on the best current candidate.
            if self._time_left(t0) >= 45:
                hard_b, soft_b, name_b, _ = min(candidates, key=lambda c: float(c[3]))
                anchor = hard_b.copy()
                hard_cd = self._gradient_stage(
                    hard_b.copy(),
                    soft_b.copy(),
                    anchor,
                    data,
                    t0,
                    steps=45,
                    lr=0.030,
                    density_weight=0.40,
                    overlap_weight=0.10 if data.num_hard <= 450 else 0.0,
                    anchor_weight=0.02,
                    snap_noise=0.35,
                )
                hard_cd = self._fast_legalize_hard(hard_cd, data, sweeps=4)
                if data.num_hard <= 420 and self._time_left(t0) >= 12:
                    try:
                        if self._exact_hard_overlap_area(hard_cd.astype(np.float64), data) > 1e-10:
                            hard_cd = self._legalize_hard(hard_cd, data)
                    except Exception:
                        pass
                soft_cd = self._clamp_soft_copy(soft_b, data)
                if data.num_soft > 0 and self._time_left(t0) >= 12:
                    soft_cd = self._gpu_soft_refine(hard_cd, soft_cd, data, t0, steps=10)
                candidates.append((hard_cd, soft_cd, f"{name_b}-cd", self._cheap_score(hard_cd, soft_cd, data)))
                _plog("gpu_cd_done")

        # Auction-based slot permutation: globally reassign macros to existing "slot" positions.
        # This is an unconventional but very powerful move: it changes topology without introducing
        # new geometry, then relies on a light legalizer to remove residual overlaps.
        if (
            data.num_hard >= 520
            and TIME_BUDGET >= 300
            and self._time_left(t0) >= 26
            and (not legacy_valid or data.num_hard >= 420)
        ):
            # Only attempt if we can find a legal starting point (permute preserves geometry, not legality).
            base_h = None
            base_s = None
            for h_try, s_try, _, _ in sorted(candidates, key=lambda c: float(c[3]))[: min(4, len(candidates))]:
                if self._placement_is_valid(h_try, s_try, benchmark, data):
                    base_h = h_try
                    base_s = s_try
                    break
            auc = None if base_h is None else self._auction_slot_permute(base_h, base_s, benchmark, data, t0)
            if auc is not None:
                h_a, s_a, n_a, sc_a = auc
                candidates.append((h_a, s_a, n_a, sc_a))
                extra_hard_seeds.append(h_a.copy())
                _plog("auction_perm_done")

        large_global_first = data.num_hard >= 520
        did_exact_parallel = False

        # GPU parallel electrostatic population: global coordinated moves (beyond 1-macro-at-a-time refinement).
        # Only worth it when we have enough budget to amortize FFT + legalization.
        if (
            self._device.type == "cuda"
            and plc is not None
            and TIME_BUDGET >= 300
            and data.num_hard >= 140
            and self._time_left(t0) >= (select_reserve + (60 if small_budget else 80))
            and (not did_gpu_pop)
        ):
            _tlog("gpu_pop_start")
            if PROFILE:
                import sys

                print(f"[PROFILE] {benchmark.name}: starting gpu-pop-eplace", file=sys.stderr)
            scale = 1.0
            if TIME_BUDGET >= 1800:
                scale = 1.8 if data.num_hard <= 420 else 1.4
            elif TIME_BUDGET >= 900:
                scale = 1.35
            pop = self._gpu_population_eplace(
                data,
                t0,
                soft_seed=base_soft.copy(),
                k=6 if (TIME_BUDGET >= 1800 and data.num_hard <= 420) else (4 if data.num_hard <= 350 else 3),
                iters_scale=float(scale),
            )
            _tlog(f"gpu_pop_out={len(pop)}")
            if PROFILE:
                import sys

                print(f"[PROFILE] {benchmark.name}: finished gpu-pop-eplace", file=sys.stderr)
            for hard, soft, name, score in pop:
                candidates.append((hard, soft, name, score))
            _plog("gpu_pop_done")
            _tlog("gpu_pop_done")

        # OT/Sinkhorn global reassignment: a globally coordinated "macro shuffle" (not physics-based).
        # Useful when oracle is slow and we still want a coordinated escape move.
        if (
            plc is not None
            and self._device.type == "cuda"
            and TIME_BUDGET >= 300
            and data.num_hard >= 360
            and self._time_left(t0) >= (select_reserve + (26 if small_budget else 34))
        ):
            # Use the best *currently valid* seed as the base for OT so the target geometry is sensible.
            ot_seed_h = legacy_hard
            ot_seed_s = base_soft
            for h_try, s_try, _, _ in sorted(candidates, key=lambda c: float(c[3]))[: min(6, len(candidates))]:
                if self._placement_is_valid(h_try, s_try, benchmark, data):
                    ot_seed_h = h_try
                    ot_seed_s = s_try
                    break
            ot = self._ot_sinkhorn_reassign(ot_seed_h, ot_seed_s, benchmark, plc, data, t0)
            if ot is not None:
                h_ot, s_ot, n_ot, sc_ot = ot
                candidates.append((h_ot, s_ot, n_ot, sc_ot))
                _plog("ot_reassign_done")
                _tlog("ot_reassign_done")

        # Mid-run oracle SA: for moderately slow oracles, we still want exact-guided discrete search,
        # but we must start it early enough to matter.
        if (
            plc is not None
            and TIME_BUDGET >= 300
            and (not did_oracle_sa)
            and (not oracle_fast)
            and data.num_hard <= 420
            and self._time_left(t0) >= (select_reserve + 140)
        ):
            try:
                est = float(getattr(self, "_oracle_call_sec", 99.0) or 99.0)
                reserve = float(min(60.0, max(20.0, 0.12 * float(TIME_BUDGET))))
                eff_left = float(max(0.0, self._time_left(t0) - reserve))
                can_calls = eff_left / max(est, 0.8)
                if est <= 25.0 and can_calls >= 18.0:
                    sa = self._oracle_sa_multistart(
                        candidates,
                        benchmark,
                        plc,
                        data,
                        t0,
                        max_calls=int(min(2200, max(60, can_calls))),
                    )
                    if sa is not None:
                        sa_h, sa_s, sa_name, sa_cheap = sa
                        candidates.append((sa_h, sa_s, sa_name, sa_cheap))
                        did_oracle_sa = True
                        _plog("oracle_sa_mid_done")
                        _tlog("oracle_sa_mid_done")
            except Exception:
                pass

        # Short-budget surrogate local search (no oracle; improves mid benchmarks).
        if (
            plc is not None
            and TIME_BUDGET <= 240
            and SURROGATE_LS
            and data.fast_cong_engine is not None
            and 300 <= data.num_hard <= 520
            and self._time_left(t0) >= 28
        ):
            _plog("surrogate_ls_start")
            # Start from best cheap legal candidate we already have.
            seed = None
            # Prefer a GPU-generated coordinated candidate as the seed if available.
            gpu_legal = [
                c
                for c in candidates
                if isinstance(c[2], str) and c[2].startswith("gpu-pop-eplace") and self._placement_is_valid(c[0], c[1], benchmark, data)
            ]
            if gpu_legal:
                seed = min(gpu_legal, key=lambda c: float(c[3]))
            else:
                for h_try, s_try, n_try, sc_try in sorted(candidates, key=lambda c: float(c[3]))[: min(8, len(candidates))]:
                    if self._placement_is_valid(h_try, s_try, benchmark, data):
                        seed = (h_try, s_try, n_try, sc_try)
                        break
            if seed is not None:
                h0, s0, n0, seed_cheap = seed
                _plog(f"surrogate_ls_seed={n0}")
                ls = self._surrogate_local_search(h0, s0, data, t0, seconds=14.0)
                if ls is not None:
                    h_ls, s_ls, n_ls, sc_ls = ls
                    cheap_ls = float(self._cheap_score(h_ls, s_ls, data))
                    if cheap_ls + 1e-6 < float(seed_cheap):
                        candidates.append((h_ls, s_ls, f"{n0}-{n_ls}", cheap_ls))
                        _plog("surrogate_ls_done")
                    else:
                        _plog("surrogate_ls_reject")
                else:
                    _plog("surrogate_ls_none")

        # Exact-oracle batched search (small/medium designs where oracle is fast enough).
        if (
            plc is not None
            and legacy_valid
            and TIME_BUDGET >= 300
            and data.num_hard <= 380
            and self._time_left(t0) >= (select_reserve + (45 if small_budget else 75))
        ):
            start_h, start_s, start_name, _ = min(candidates, key=lambda c: float(c[3]))
            improved = self._oracle_batch_search(
                start_h,
                start_s,
                benchmark,
                plc,
                data,
                t0,
                label=f"{start_name}-oracle-batch",
            )
            if improved is not None:
                hard_i, soft_i, name_i, cheap_i = improved
                candidates.append((hard_i, soft_i, name_i, cheap_i))
                latent_seed_hard = hard_i.copy()
                latent_seed_soft = soft_i.copy()
                extra_hard_seeds.append(hard_i.copy())
                _plog("oracle_batch_done")

        # Small-benchmark exact parallel search in latent space (CPU parallelism; very high leverage).
        # Run it early (before expensive heuristic refinements) while we still have budget.
        if (
            plc is not None
            and EXACT_PARALLEL
            and oracle_fast
            and data.num_hard <= 320
            and self._time_left(t0) >= (select_reserve + (55 if TIME_BUDGET <= 240 else 120))
        ):
            _plog("exact_parallel_start")
            did_exact_parallel = True
            base_h, base_s, _, _ = min(candidates, key=lambda c: float(c[3])) if candidates else (
                legacy_hard,
                base_soft,
                "legacy",
                0.0,
            )
            refined = self._exact_parallel_latent_search(
                base_h,
                base_s,
                benchmark,
                plc,
                data,
                t0,
            )
            if refined is not None:
                r_hard, r_soft, r_name, r_score = refined
                candidates.append((r_hard, r_soft, r_name, r_score))
            _plog("exact_parallel_done")

        # GPU multi-start analytical placement: explores distinct basins cheaply.
        # Only run when budget is large enough that it won't crowd out population exploration.
        if (
            not small_budget
            and self._device.type == "cuda"
            and self._time_left(t0) >= 70
            and (plc is None or float(getattr(self, "_oracle_call_sec", 99.0) or 99.0) >= 9.0)
        ):
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

        # GPU random-restart global refinement: escapes legacy basin on high-congestion cases.
        if (
            not small_budget
            and self._device.type == "cuda"
            and plc is not None
            and self._time_left(t0) >= 85
            and data.num_hard <= 520
            and float(getattr(self, "_oracle_call_sec", 99.0) or 99.0) >= 10.0
        ):
            rr = self._gpu_random_restart_gradient(
                data, t0, base_soft.copy(), k=3 if (TIME_BUDGET >= 1800 and data.num_hard <= 420) else (2 if data.num_hard <= 320 else 1)
            )
            for hard, soft, name, score in rr:
                candidates.append((hard, soft, name, score))

        if plc is not None and large_global_first and self._time_left(t0) >= 26 and not did_exact_parallel:
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

        if plc is not None and self._time_left(t0) >= 25 and large_global_first and not did_exact_parallel:
            if legacy_fast_valid:
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
                if TIME_BUDGET >= 300 and data.num_hard <= 520 and self._time_left(stage_t0) >= 18:
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
                        if plc is not None and TIME_BUDGET >= 300 and data.num_hard <= 520 and self._time_left(local_t0) >= 14:
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
                        if TIME_BUDGET >= 300 and self._time_left(local_t0) >= 18:
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

        # Keep time for final exact selection; skip long refinements if we're close to the end.
        # If oracle is fast (small/medium benchmarks), do an oracle-driven multistart sweep *before*
        # latent search so downstream stages start from an actually good basin.
        if (
            plc is not None
            and not small_budget
            and not large_global_first
            and data.num_hard <= 420
            and self._time_left(t0) >= (select_reserve + 160)
        ):
            try:
                # Initialize oracle call-time estimate on the current best cheap legal candidate.
                seed_h, seed_s, _, _ = min(candidates, key=lambda c: float(c[3]))
                _ = self._proxy_cost_if_valid(seed_h, seed_s, benchmark, plc, data)
                est = float(getattr(self, "_oracle_call_sec", 99.0) or 99.0)
                if est <= 3.25 and self._time_left(t0) >= (select_reserve + 140):
                    sa = self._oracle_sa_multistart(
                        candidates,
                        benchmark,
                        plc,
                        data,
                        t0,
                        max_calls=220,
                    )
                    if sa is not None:
                        sa_h, sa_s, sa_name, sa_cheap = sa
                        candidates.append((sa_h, sa_s, sa_name, sa_cheap))
                        latent_seed_hard = sa_h.copy()
                        latent_seed_soft = sa_s.copy()
                        did_oracle_sa = True
                        _plog("oracle_sa_early_done")
            except Exception:
                pass

        if (
            plc is not None
            and self._time_left(t0) >= (select_reserve + 22)
            and legacy_valid
            and not large_global_first
            and not did_exact_parallel
            and not small_budget
        ):
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

        if (
            plc is not None
            and self._time_left(t0) >= (select_reserve + 30)
            and not large_global_first
            and not did_exact_parallel
            and not small_budget
        ):
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
        run_expensive_worlds = (
            plc is not None
            and TIME_BUDGET >= 300
            and data.num_hard <= 600
            and not did_exact_parallel
            and self._time_left_for_work() >= (
                36 if data.num_hard <= 350 else 44 if data.num_hard <= 650 else 52
            )
        )
        if run_expensive_worlds:
            for hard_seed, soft_seed, world_name in worlds[:max_worlds]:
                if self._time_left_for_work() < 20:
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
                if self._time_left_for_work() >= 14:
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
        _plog(f"candidates={len(candidates)}")
        self._trace(f"candidates={len(candidates)}")
        if PROFILE:
            import sys

            # Keep profiling output cheap; avoid burning selection-reserve time on expensive surrogate components.
            max_show = 6 if self._time_left_for_work() >= 30 else 3
            top = sorted(candidates, key=lambda c: float(c[3]))[: min(int(max_show), len(candidates))]
            for h, s, n, sc in top:
                if self._time_left_for_work() < 8:
                    break
                try:
                    comps = self._cheap_components(h, s, data)
                    msg = (
                        f"[PROFILE] {benchmark.name}: cand {n} cheap={float(sc):.4f} "
                        f"(wl={comps['wl']:.3f} den={comps['den']:.3f} cong={comps['cong']:.3f} ov={comps['overlap']:.2e})"
                    )
                except Exception:
                    msg = f"[PROFILE] {benchmark.name}: cand {n} cheap={float(sc):.4f}"
                print(msg, file=sys.stderr)

        # Oracle SA refinement for small/medium designs when oracle is fast enough.
        # Exact objective + discrete coordinated moves (cluster translations + swaps).
        run_oracle_sa = (
            plc is not None
            and (
                (TIME_BUDGET <= 240 and data.num_hard <= 280 and self._time_left(t0) >= 30)
                or (TIME_BUDGET > 240 and data.num_hard <= 420 and self._time_left(t0) >= 140)
            )
        )
        if run_oracle_sa and not did_oracle_sa:
            # Initialize oracle call time estimate.
            seed_h, seed_s, seed_name, _ = min(candidates, key=lambda c: float(c[3]))
            _ = self._proxy_cost_if_valid(seed_h, seed_s, benchmark, plc, data)
            est = float(getattr(self, "_oracle_call_sec", 2.0) or 2.0)
            if TIME_BUDGET <= 240:
                if data.num_hard <= 320:
                    if data.num_hard <= 280:
                        est_ok = est <= 3.40
                        calls_cap = 20
                        need_left = 28
                    else:
                        est_ok = est <= 2.20
                        calls_cap = 24
                        need_left = 45
                else:
                    # Medium designs: oracle can be slower, still worth a few exact-guided moves.
                    est_ok = est <= 8.00 and data.num_hard <= 420
                    calls_cap = 12
                    need_left = 60
            else:
                # Long-budget regime: allow slower oracle calls; still valuable for escaping basins.
                # Decide by how many exact evaluations we can actually afford, not by a fixed latency.
                eff_left = float(max(0.0, self._time_left(t0) - reserve))
                can_calls = eff_left / max(est, 0.8)
                est_ok = (est <= 25.0) and (can_calls >= 30.0) and (self._time_left(t0) >= 120.0)
                # Calls cap scales with budget and oracle speed.
                calls_cap = int(min(3000, max(80, can_calls)))
                need_left = 200 if calls_cap >= 600 else 160 if calls_cap >= 240 else 120
            _plog(f"oracle_sa_gate est={est:.2f} ok={int(est_ok)} left={self._time_left(t0):.1f}")
            if est_ok and self._time_left(t0) >= need_left:
                sa = self._oracle_sa_multistart(
                    candidates,
                    benchmark,
                    plc,
                    data,
                    t0,
                    max_calls=calls_cap,
                )
                if sa is not None:
                    sa_h, sa_s, sa_name, sa_cheap = sa
                    candidates.append((sa_h, sa_s, sa_name, sa_cheap))
                    did_oracle_sa = True
                    _plog("oracle_sa_done")
                    _tlog("oracle_sa_done")

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
        self._trace("select_best_done")

        # Final safety: enforce legality before returning (some legalization paths can leave
        # tiny but non-zero overlaps that slip past fast checks).
        if plc is not None and self._time_left(t0) >= 8:
            try:
                if not self._placement_is_valid(best_hard, best_soft, benchmark, data):
                    hard_fix = best_hard.copy()
                    if data.num_hard <= 420:
                        hard_fix = self._legalize_hard(hard_fix, data)
                    else:
                        hard_fix = self._fast_legalize_hard(hard_fix, data, sweeps=6)
                    if not self._placement_is_valid(hard_fix, best_soft, benchmark, data) and self._time_left(t0) >= 6:
                        hard_fix = self._grid_pack_legalize_hard(hard_fix, data)
                        hard_fix = self._fast_legalize_hard(hard_fix, data, sweeps=5)
                    if self._placement_is_valid(hard_fix, best_soft, benchmark, data):
                        best_hard = hard_fix.astype(np.float32)
            except Exception:
                pass

        placement = benchmark.macro_positions.clone()
        placement[: data.num_hard] = torch.from_numpy(best_hard).float()
        if data.num_soft > 0:
            placement[data.num_hard : data.num_macros] = torch.from_numpy(best_soft).float()

        if PROFILE:
            import sys

            oracle_calls = int(self._prof.get("oracle_calls", 0))
            oracle_sec = float(self._prof.get("oracle_sec", 0.0))
            per_call = oracle_sec / max(oracle_calls, 1)
            print(
                f"[PROFILE] {benchmark.name}: oracle_calls={oracle_calls} oracle_sec={oracle_sec:.2f} "
                f"oracle_call_sec={per_call:.2f} device={self._device.type}",
                file=sys.stderr,
            )
        self._trace(f"done oracle_calls={int(self._prof.get('oracle_calls', 0))} oracle_sec={float(self._prof.get('oracle_sec', 0.0)):.3f}")
        self._trace_close()
        return placement

    def _exact_parallel_latent_search(
        self,
        base_hard: np.ndarray,
        base_soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, float]]:
        # Exact-driven CEM in latent space; uses multiprocessing when available.
        if plc is None or self._time_left(t0) < 35:
            return None
        reserve = float(min(60.0, max(20.0, 0.12 * float(TIME_BUDGET))))

        # Determine benchmark directory for subprocess init.
        bench_dir = None
        for path in [
            f"external/MacroPlacement/Testcases/ICCAD04/{benchmark.name}",
            f"external/MacroPlacement/Flows/NanGate45/{benchmark.name}/netlist/output_CT_Grouping",
        ]:
            netlist = f"{path}/netlist.pb.txt"
            plc_path = f"{path}/initial.plc"
            if os.path.exists(netlist) and os.path.exists(plc_path):
                bench_dir = path
                netlist_file = netlist.replace("\\", "/")
                plc_file = plc_path.replace("\\", "/")
                break
        if bench_dir is None:
            return None

        modes = self._build_latent_modes(base_hard, base_soft, data, extra_hard_seeds=None)
        if len(modes) == 0:
            return None

        dim = len(modes)
        unit = float(max(data.cell_w, data.cell_h))

        # Exact baseline.
        base_exact = self._proxy_cost_if_valid(base_hard, base_soft, benchmark, plc, data)
        if not math.isfinite(base_exact):
            return None

        # Parallel config.
        nprocs = int(min(max((os.cpu_count() or 4) - 1, 2), 6))
        # Windows `spawn` pool startup is too expensive for per-benchmark budgets; keep it in-process.
        use_pool = nprocs >= 3 and os.name == "posix"

        rng = np.random.default_rng((SEED ^ self._stable_seed(f"{benchmark.name}-exactcem")) & 0xFFFFFFFF)
        mu = np.zeros(dim, dtype=np.float32)
        sigma = np.full(dim, 0.10 * unit, dtype=np.float32)
        best_cost = float(base_exact)
        best_hard = base_hard.copy().astype(np.float32)
        best_soft = base_soft.copy().astype(np.float32)
        best_alpha = mu.copy()

        def build(alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # Fast legalization only (hash + tiny-fix). Full O(N^2) legalizer is too slow inside CEM.
            hard = base_hard.copy().astype(np.float32)
            for a, mode in zip(alpha.astype(np.float32), modes):
                hard += float(a) * mode
            self._clamp_hard(hard, data)
            hard = self._hash_resolve_hard(hard, data, sweeps=2)
            hard = self._tiny_fix_hard(hard, data, rounds=60)
            soft = base_soft.copy().astype(np.float32)
            if data.num_soft > 0:
                soft = self._relax_soft(hard, soft, data, sweeps=1, damping=0.86)
            return hard.astype(np.float32), soft.astype(np.float32)

        def eval_batch(payloads: List[Tuple[np.ndarray, np.ndarray]]) -> List[float]:
            if not use_pool or self._time_left(t0) < (reserve + 18):
                # Sequential exact eval with local plc.
                out = []
                for h, s in payloads:
                    if self._time_left(t0) < (reserve + 10):
                        out.append(float("inf"))
                        continue
                    out.append(self._proxy_cost_if_valid(h, s, benchmark, plc, data))
                return out

            try:
                import multiprocessing as mp

                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=nprocs,
                    initializer=_exact_worker_init,
                    initargs=(netlist_file, plc_file, benchmark.name),
                ) as pool:
                    # Keep batches small to react to time budget.
                    return pool.map(_exact_worker_eval, payloads, chunksize=1)
            except Exception:
                out = []
                for h, s in payloads:
                    out.append(self._proxy_cost_if_valid(h, s, benchmark, plc, data))
                return out

        # Budgeted exact CEM loop.
        generations = 4
        samples = 14
        elite = 5
        for gen in range(generations):
            if self._time_left(t0) < (reserve + 16):
                break
            alphas = [rng.normal(mu, sigma).astype(np.float32) for _ in range(samples)]
            # Always include current best and a few coordinate nudges.
            alphas[0] = best_alpha.copy()
            if dim >= 1:
                alphas[1] = best_alpha.copy()
                alphas[1][0] += 0.08 * unit
            if dim >= 2:
                alphas[2] = best_alpha.copy()
                alphas[2][1] -= 0.08 * unit

            payloads = [build(a) for a in alphas]
            costs = eval_batch(payloads)
            order = np.argsort(np.asarray(costs, dtype=np.float64))
            if not np.isfinite(costs[int(order[0])]):
                continue

            # Update best.
            bi = int(order[0])
            if float(costs[bi]) + 1e-9 < best_cost:
                best_cost = float(costs[bi])
                best_hard, best_soft = payloads[bi]
                best_alpha = alphas[bi].copy()

            # Update distribution.
            elite_idx = [int(i) for i in order[:elite]]
            elite_alphas = np.stack([alphas[i] for i in elite_idx], axis=0)
            mu = elite_alphas.mean(axis=0).astype(np.float32)
            sigma = elite_alphas.std(axis=0).astype(np.float32)
            sigma = np.clip(sigma, 0.02 * unit, 0.18 * unit).astype(np.float32)
            mu = 0.60 * mu + 0.40 * best_alpha

        if best_cost + 1e-6 < base_exact:
            return best_hard, best_soft, "exact-latent-cem", self._cheap_score(best_hard, best_soft, data)
        return None

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
                (60, 0.045, 0.40, 0.18, 0.65),
                (45, 0.030, 0.62, 0.10, 0.35),
            ]
        elif data.num_hard <= 450:
            schedule = [
                (50, 0.040, 0.45, 0.16, 0.55),
                (40, 0.026, 0.68, 0.09, 0.28),
            ]
        else:
            # Large designs: avoid the O(N^2) overlap term by using overlap_weight=0.
            schedule = [
                (35, 0.034, 0.55, 0.00, 0.45),
                (30, 0.022, 0.78, 0.00, 0.22),
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

    def _gpu_random_restart_gradient(
        self,
        data: PreparedData,
        t0: float,
        soft_seed: np.ndarray,
        k: int,
    ) -> List[Tuple[np.ndarray, np.ndarray, str, float]]:
        out: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        if self._device.type != "cuda" or self._time_left(t0) < 25 or k <= 0:
            return out

        rng = np.random.default_rng((SEED ^ (self._stable_seed("rr") << 1)) & 0xFFFFFFFF)
        movable = data.movable_hard
        lo_x = data.sizes_hard[:, 0] * 0.5
        hi_x = data.canvas_w - lo_x
        lo_y = data.sizes_hard[:, 1] * 0.5
        hi_y = data.canvas_h - lo_y

        for idx in range(int(k)):
            if self._time_left(t0) < 22:
                break
            hard = data.init_hard.copy().astype(np.float32)
            hard[movable, 0] = rng.uniform(lo_x[movable], hi_x[movable]).astype(np.float32)
            hard[movable, 1] = rng.uniform(lo_y[movable], hi_y[movable]).astype(np.float32)
            hard = self._fast_legalize_hard(hard, data, sweeps=4)
            soft = self._clamp_soft_copy(soft_seed, data)
            anchor = hard.copy()

            # Short, aggressive schedule (bias toward congestion on later half).
            hard = self._gradient_stage(
                hard,
                soft,
                anchor,
                data,
                t0,
                steps=50,
                lr=0.040,
                density_weight=0.55,
                overlap_weight=0.12 if data.num_hard <= 450 else 0.0,
                anchor_weight=0.05,
                snap_noise=0.55,
            )
            hard = self._fast_legalize_hard(hard, data, sweeps=4)
            score = self._cheap_score(hard, soft, data)
            out.append((hard.copy(), soft.copy(), f"gpu-rr-{idx}", score))

        out.sort(key=lambda x: x[3])
        return out[:2]

    def _gpu_population_eplace(
        self,
        data: PreparedData,
        t0: float,
        soft_seed: np.ndarray,
        k: int,
        seed_hards: Optional[Sequence[np.ndarray]] = None,
        world_den_w: Optional[Sequence[float]] = None,
        den_schedule: bool = False,
        inflate_factor: float = 1.0,
        iters_scale: float = 1.0,
    ) -> List[Tuple[np.ndarray, np.ndarray, str, float]]:
        # Batched global analytical placement: WL (autograd) + electrostatic density force (FFT),
        # optimized with Nesterov momentum. Produces k diverse candidates in one GPU run.
        out: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        if self._device.type != "cuda" or self._time_left(t0) < 30 or k <= 0 or data.num_hard <= 1:
            return out

        device = self._device
        k = int(max(2, min(int(k), 8)))
        unit = float(max(data.cell_w, data.cell_h))

        # Build diverse hard seeds.
        hard_seeds: List[np.ndarray] = []
        if seed_hards is not None:
            for h in list(seed_hards)[: max(1, k)]:
                try:
                    hard_seeds.append(np.asarray(h, dtype=np.float32).copy())
                except Exception:
                    continue
        if len(hard_seeds) == 0:
            hard_seeds = [
                data.init_hard.copy(),
                data.spectral_xy.copy(),
                self._quadrant_permute_world(data.init_hard, data),
            ]
        rng = np.random.default_rng((SEED ^ self._stable_seed("pop")) & 0xFFFFFFFF)
        while len(hard_seeds) < k:
            noisy = data.init_hard.copy().astype(np.float32)
            noise = rng.uniform(-1.0, 1.0, size=noisy.shape).astype(np.float32)
            noise[:, 0] *= 0.9 * unit
            noise[:, 1] *= 0.9 * unit
            noise[~data.movable_hard] = 0.0
            noisy += noise
            hard_seeds.append(noisy)
        hard_seeds = hard_seeds[:k]

        # One-time legalization to keep the optimization numerically sane.
        hard0 = []
        for seed in hard_seeds:
            hard0.append(self._fast_legalize_hard(seed, data, sweeps=4))

        fixed = torch.from_numpy(hard0[0].astype(np.float32)).to(device=device)
        movable_idx = torch.from_numpy(np.flatnonzero(data.movable_hard).astype(np.int64)).to(device=device)
        if movable_idx.numel() == 0:
            return out

        # Batch params: (B, M, 2)
        params0 = torch.stack(
            [torch.from_numpy(h[movable_idx.cpu().numpy()].astype(np.float32)) for h in hard0],
            dim=0,
        ).to(device=device)
        params = params0.clone()
        velocity = torch.zeros_like(params)

        hard_sizes = data.hard_sizes_t.to(device=device, dtype=torch.float32)
        hard_sizes_den = hard_sizes
        if float(inflate_factor) > 1.001:
            hard_sizes_den = hard_sizes.clone()
            hard_sizes_den[:, 0] *= float(inflate_factor)
            hard_sizes_den[:, 1] *= float(inflate_factor)
        port_t = data.port_t.to(device=device, dtype=torch.float32)
        soft_t = (
            torch.from_numpy(self._clamp_soft_copy(soft_seed, data)).to(device=device, dtype=torch.float32)
            if data.num_soft > 0
            else None
        )

        # Preconditioning by connectivity degree (stabilizes big-degree macros).
        deg = torch.from_numpy((data.hard_degree + 1.0).astype(np.float32)).to(device=device)
        pre = (1.0 / torch.sqrt(deg)).clamp(min=0.25, max=2.5)
        pre = pre[movable_idx].view(1, -1, 1)

        # Bounds for movable params.
        lo_x = torch.from_numpy(data.sizes_hard[:, 0] * 0.5).to(device=device, dtype=torch.float32)[movable_idx]
        hi_x = (data.canvas_w - lo_x).to(device=device)
        lo_y = torch.from_numpy(data.sizes_hard[:, 1] * 0.5).to(device=device, dtype=torch.float32)[movable_idx]
        hi_y = (data.canvas_h - lo_y).to(device=device)

        safe_nnp = data.grad_safe_nnp_t.to(device=device)
        nnmask = data.grad_nnmask_t.to(device=device)

        # EPlace-style schedule.
        cong_start_frac = 0.65
        if TIME_BUDGET <= 240:
            # Short-budget: spend time on the core batched WL+density loop.
            # Avoid expensive RUDY shaping here; selection uses a congestion-aware cheap score.
            iters = 30 if data.num_hard <= 400 else 26
            lr = 0.043 if data.num_hard <= 350 else 0.038
            mu = 0.87
            dens_target = float(getattr(self, "_density_target", BASE_DENSITY_TARGET))
            if dens_target <= 0.76:
                den_w = 1.35
            elif dens_target <= 0.82:
                den_w = 1.20
            else:
                den_w = 1.00
            cong_w = 0.0
        elif data.num_hard <= 350:
            iters = 42
            lr = 0.040
            mu = 0.88
            den_w = 0.80
            cong_w = 0.35
        else:
            iters = 35
            lr = 0.036
            mu = 0.86
            den_w = 0.85
            cong_w = 0.30

        # Scale iterations up when we have substantial budget (kept general, time-bounded by deadline).
        try:
            iters = int(max(10, min(180, round(float(iters) * float(max(0.5, min(3.0, iters_scale)))))))
        except Exception:
            pass

        # Congestion-heavy cases: start shaping earlier and push harder (general rule via density target).
        dens_target = float(getattr(self, "_density_target", BASE_DENSITY_TARGET))
        if TIME_BUDGET > 240 and dens_target <= 0.76:
            cong_w = float(max(float(cong_w), 0.55))
            cong_start_frac = 0.52
        elif TIME_BUDGET > 240 and dens_target <= 0.82:
            cong_w = float(max(float(cong_w), 0.42))
            cong_start_frac = 0.58

        # Long budgets: diversify the population by default using different density weights.
        # This tends to create different topologies that the oracle can later select from.
        if TIME_BUDGET > 240 and world_den_w is None and int(k) >= 4:
            base_w = float(den_w)
            mult = [0.55, 0.80, 1.05, 1.35, 0.65, 1.20, 0.90, 1.50]
            world_den_w = [base_w * float(mult[i % len(mult)]) for i in range(int(k))]
            den_schedule = True

        # When running multi-world batches, keep iterations tighter so we can afford follow-ups.
        if world_den_w is not None and TIME_BUDGET <= 240 and data.num_hard > 260:
            iters = int(min(int(iters), 22))
            lr = float(min(float(lr), 0.040))

        # Safety: bound iterations when gradient net tensor is huge (prevents runaway runtime).
        try:
            grad_nets = int(safe_nnp.shape[0])
            grad_max = int(safe_nnp.shape[2])
            complexity = float(grad_nets) * float(grad_max)
            if complexity > 6_000_000:
                iters = int(min(int(iters), 16))
                lr = float(min(float(lr), 0.040))
                mu = float(min(float(mu), 0.86))
            elif complexity > 3_500_000:
                iters = int(min(int(iters), 20))
        except Exception:
            pass

        deadline = float(getattr(self, "_place_t0", t0)) + float(max(0.0, float(TIME_BUDGET) - float(TIME_GUARD))) - 1.5
        try:
            deadline = float(min(float(deadline), float(self._work_deadline(safety=1.5))))
        except Exception:
            pass
        # Prevent this stage from eating the entire work budget on large netlists (ibm01-like):
        # we want multiple stages + a real final selection, not one long GPU kernel.
        try:
            work_left = float(self._time_left_for_work())
            if TIME_BUDGET > 240 and work_left > 0.0:
                # Leave time for legalization/scoring and downstream discrete search.
                leave = 18.0
                max_sec = min(180.0, 0.30 * float(TIME_BUDGET), max(30.0, 0.60 * work_left))
                max_sec = float(min(max_sec, max(0.0, work_left - leave)))
                deadline = float(min(float(deadline), float(time.time() + max_sec)))
        except Exception:
            pass

        # Multi-world density weights (per-batch). If not provided, use the scalar den_w.
        if world_den_w is None:
            den_w_b = torch.full((int(params.shape[0]), 1, 1), float(den_w), device=device, dtype=torch.float32)
        else:
            w_list = [float(x) for x in list(world_den_w)]
            if len(w_list) < int(params.shape[0]):
                reps = int(math.ceil(int(params.shape[0]) / max(1, len(w_list))))
                w_list = (w_list * reps)[: int(params.shape[0])]
            else:
                w_list = w_list[: int(params.shape[0])]
            den_w_b = torch.tensor(w_list, device=device, dtype=torch.float32).view(-1, 1, 1)

        # Time-bounded loop.
        for it in range(int(iters)):
            if time.time() > deadline:
                break
            if it % (6 if TIME_BUDGET <= 240 else 6) == 0:
                torch.cuda.synchronize()
                if self._time_left_for_work() < 10:
                    break

            params = params.detach().requires_grad_(True)
            fixed_b = fixed.unsqueeze(0).expand(params.shape[0], -1, -1).contiguous()
            cur_hard = fixed_b.clone()
            cur_hard[:, movable_idx, :] = params

            # WL gradient (batched).
            if data.num_soft > 0 and soft_t is not None:
                all_pos = torch.cat(
                    [cur_hard, soft_t.unsqueeze(0).expand(cur_hard.shape[0], -1, -1), port_t.unsqueeze(0).expand(cur_hard.shape[0], -1, -1)],
                    dim=1,
                )
            else:
                all_pos = torch.cat([cur_hard, port_t.unsqueeze(0).expand(cur_hard.shape[0], -1, -1)], dim=1)

            xs = all_pos[:, safe_nnp, 0]
            ys = all_pos[:, safe_nnp, 1]
            neg_inf = torch.full_like(xs, -1e9)
            xs_p = torch.where(nnmask, xs, neg_inf)
            xs_n = torch.where(nnmask, -xs, neg_inf)
            ys_p = torch.where(nnmask, ys, neg_inf)
            ys_n = torch.where(nnmask, -ys, neg_inf)
            wl = (
                torch.logsumexp(WL_ALPHA * xs_p, dim=2)
                + torch.logsumexp(WL_ALPHA * xs_n, dim=2)
                + torch.logsumexp(WL_ALPHA * ys_p, dim=2)
                + torch.logsumexp(WL_ALPHA * ys_n, dim=2)
            ).sum(dim=1) / (WL_ALPHA * data.grad_hpwl_norm)
            loss = wl.mean()

            # Congestion shaping via differentiable RUDY+ABU.
            if cong_w > 0.0 and it >= int(iters * float(cong_start_frac)) and self._time_left(t0) >= 16:
                cong_terms = []
                for b in range(int(all_pos.shape[0])):
                    cong_map = self._rudy_congestion_map(all_pos[b], data, device=device, alpha=CONG_ALPHA)
                    cong_terms.append(self._abu_logsumexp(cong_map, frac=0.05))
                cong = torch.stack(cong_terms, dim=0).mean()
                loss = loss + cong_w * cong

            loss.backward()
            grad_wl = params.grad.detach()

            # Electrostatic density force (batched, no autograd).
            with torch.no_grad():
                # Two-phase inflation: use inflated footprints early, then revert for later refinement.
                if hard_sizes_den is not hard_sizes and it >= int(iters * 0.60):
                    sizes_force = hard_sizes
                else:
                    sizes_force = hard_sizes_den
                force_den = self._electrostatic_density_force(cur_hard, sizes_force, data, device=device)
                force_den = force_den[:, movable_idx, :]
                # Normalize density force scale across benchmarks.
                rms = torch.sqrt((force_den.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-12))
                force_den = force_den / rms.clamp(min=1e-3, max=50.0)

            if den_schedule:
                frac = float(it) / float(max(1, int(iters) - 1))
                if TIME_BUDGET > 240:
                    # For long budgets, start with stronger spreading then taper to refine WL.
                    den_scale = (1.00 - 0.45 * frac)
                else:
                    den_scale = (0.50 + 0.50 * frac)
                den_eff = den_w_b * float(den_scale)
            else:
                den_eff = den_w_b
            grad = grad_wl - den_eff * force_den  # subtract because force points away from overflow

            with torch.no_grad():
                velocity.mul_(mu).add_(-lr * grad * pre)
                params = params + velocity
                params[:, :, 0].clamp_(lo_x, hi_x)
                params[:, :, 1].clamp_(lo_y, hi_y)

        # Convert candidates back to numpy, legalize, and score.
        torch.cuda.synchronize()
        final = params.detach().cpu().numpy().astype(np.float32)
        for b in range(final.shape[0]):
            if self._time_left_for_work() < 8:
                break
            hard = hard0[b].copy().astype(np.float32)
            hard[data.movable_hard] = final[b]
            hard = self._fast_legalize_hard(hard, data, sweeps=4)
            if data.num_hard <= 420 and self._time_left_for_work() >= 6:
                if self._exact_hard_overlap_area(hard.astype(np.float64), data) > 1e-10:
                    hard = self._legalize_hard(hard, data)
            soft = self._clamp_soft_copy(soft_seed, data)
            if data.num_soft > 0 and self._time_left_for_work() >= (14 if TIME_BUDGET <= 240 else 16):
                soft_steps = 10 if TIME_BUDGET <= 240 else 22
                soft = self._gpu_soft_refine(hard, soft, data, t0, steps=soft_steps)
            score = self._cheap_score(hard, soft, data)
            out.append((hard.copy(), soft.copy(), f"gpu-pop-eplace-{b}", score))

        out.sort(key=lambda x: x[3])
        return out[:3]

    def _gpu_soft_refine(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
        t0: float,
        steps: int,
    ) -> np.ndarray:
        # Short GPU refinement for soft macros only (hard fixed).
        if data.num_soft == 0 or self._device.type != "cuda" or self._time_left_for_work() < 8:
            return self._clamp_soft_copy(soft, data)

        device = self._device
        hard_t = torch.from_numpy(hard.astype(np.float32)).to(device=device)
        soft_t = torch.from_numpy(self._clamp_soft_copy(soft, data)).to(device=device)
        movable = torch.from_numpy(data.movable_soft.astype(np.bool_)).to(device=device)
        if movable.sum().item() == 0:
            return self._clamp_soft_copy(soft, data)

        params = soft_t.clone()
        params.requires_grad_(True)
        opt = torch.optim.Adam([params], lr=0.050)

        hard_sizes = data.hard_sizes_t.to(device=device, dtype=torch.float32)
        soft_sizes = data.soft_sizes_t.to(device=device, dtype=torch.float32)
        port_t = data.port_t.to(device=device, dtype=torch.float32)
        safe_nnp = data.grad_safe_nnp_t.to(device=device)
        nnmask = data.grad_nnmask_t.to(device=device)

        hard_density = self._density_map(hard_t, hard_sizes, data, device=device).detach()

        lo_x = torch.from_numpy(data.sizes_soft[:, 0] * 0.5).to(device=device, dtype=torch.float32)
        hi_x = data.canvas_w - lo_x
        lo_y = torch.from_numpy(data.sizes_soft[:, 1] * 0.5).to(device=device, dtype=torch.float32)
        hi_y = data.canvas_h - lo_y

        for step in range(int(steps)):
            if step % 8 == 0:
                torch.cuda.synchronize()
                if self._time_left_for_work() < 6:
                    break
            opt.zero_grad(set_to_none=True)

            cur_soft = torch.where(movable.view(-1, 1), params, soft_t)
            all_pos = torch.cat([hard_t, cur_soft, port_t], dim=0)

            xs = all_pos[safe_nnp, 0]
            ys = all_pos[safe_nnp, 1]
            neg_inf = torch.full_like(xs, -1e9)
            xs_p = torch.where(nnmask, xs, neg_inf)
            xs_n = torch.where(nnmask, -xs, neg_inf)
            ys_p = torch.where(nnmask, ys, neg_inf)
            ys_n = torch.where(nnmask, -ys, neg_inf)
            wl = (
                torch.logsumexp(WL_ALPHA * xs_p, dim=1)
                + torch.logsumexp(WL_ALPHA * xs_n, dim=1)
                + torch.logsumexp(WL_ALPHA * ys_p, dim=1)
                + torch.logsumexp(WL_ALPHA * ys_n, dim=1)
            ).sum() / (WL_ALPHA * data.grad_hpwl_norm)

            soft_density = self._density_map(cur_soft, soft_sizes, data, device=device)
            dens_target = float(getattr(self, "_density_target", BASE_DENSITY_TARGET))
            overflow = (hard_density + soft_density - dens_target).clamp(min=0.0)
            den = overflow.pow(2).mean() + 0.15 * overflow.max()

            loss = wl + 0.65 * den
            if step >= int(steps * 0.70):
                cong_map = self._rudy_congestion_map(all_pos, data, device=device, alpha=CONG_ALPHA)
                cong = self._abu_logsumexp(cong_map, frac=0.05)
                loss = loss + 0.20 * cong

            loss.backward()
            opt.step()

            with torch.no_grad():
                params[:, 0].clamp_(lo_x, hi_x)
                params[:, 1].clamp_(lo_y, hi_y)

        return params.detach().cpu().numpy().astype(np.float32)

    def _electrostatic_density_force(
        self,
        hard_b: torch.Tensor,
        hard_sizes: torch.Tensor,
        data: PreparedData,
        device: torch.device,
    ) -> torch.Tensor:
        # Returns per-macro force (B, N, 2) pushing away from overflow regions.
        # Uses bilinear splat onto grid + Poisson solve via FFT.
        B, N, _ = hard_b.shape
        R = int(data.grid_rows)
        C = int(data.grid_cols)
        gw = float(data.cell_w)
        gh = float(data.cell_h)
        cell_area = max(gw * gh, 1e-6)

        # Map to cell-center index space.
        u = hard_b[:, :, 0] / gw - 0.5
        v = hard_b[:, :, 1] / gh - 0.5
        i0 = torch.floor(u).to(torch.int64)
        j0 = torch.floor(v).to(torch.int64)
        fx = (u - i0.to(u.dtype)).clamp(0.0, 1.0)
        fy = (v - j0.to(v.dtype)).clamp(0.0, 1.0)

        i1 = i0 + 1
        j1 = j0 + 1

        i0c = i0.clamp(0, C - 1)
        i1c = i1.clamp(0, C - 1)
        j0c = j0.clamp(0, R - 1)
        j1c = j1.clamp(0, R - 1)

        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        area = (hard_sizes[:, 0] * hard_sizes[:, 1]).view(1, N).to(device=device, dtype=torch.float32)
        q = area / cell_area

        rho = torch.zeros((B, R * C), device=device, dtype=torch.float32)
        idx00 = (j0c * C + i0c).view(B, N)
        idx10 = (j0c * C + i1c).view(B, N)
        idx01 = (j1c * C + i0c).view(B, N)
        idx11 = (j1c * C + i1c).view(B, N)

        rho.scatter_add_(1, idx00, q * w00)
        rho.scatter_add_(1, idx10, q * w10)
        rho.scatter_add_(1, idx01, q * w01)
        rho.scatter_add_(1, idx11, q * w11)
        rho = rho.view(B, R, C)

        # Only overflow repels; do not attract into underfull regions.
        dens_target = float(getattr(self, "_density_target", BASE_DENSITY_TARGET))
        rho = (rho - dens_target).clamp(min=0.0)

        rho_hat = torch.fft.fft2(rho)
        ky = TWO_PI * torch.fft.fftfreq(R, d=1.0, device=device).view(R, 1)
        kx = TWO_PI * torch.fft.fftfreq(C, d=1.0, device=device).view(1, C)
        k2 = (kx * kx + ky * ky).to(torch.float32)
        inv_k2 = torch.where(k2 > 0, 1.0 / k2, torch.zeros_like(k2))

        phi_hat = rho_hat * inv_k2
        ex_hat = phi_hat * (1j * kx)
        ey_hat = phi_hat * (1j * ky)
        ex = -torch.fft.ifft2(ex_hat).real
        ey = -torch.fft.ifft2(ey_hat).real

        exf = ex.reshape(B, -1)
        eyf = ey.reshape(B, -1)
        fx = fx.to(torch.float32)
        fy = fy.to(torch.float32)
        f00 = (1.0 - fx) * (1.0 - fy)
        f10 = fx * (1.0 - fy)
        f01 = (1.0 - fx) * fy
        f11 = fx * fy

        ex_m = (
            exf.gather(1, idx00) * f00
            + exf.gather(1, idx10) * f10
            + exf.gather(1, idx01) * f01
            + exf.gather(1, idx11) * f11
        )
        ey_m = (
            eyf.gather(1, idx00) * f00
            + eyf.gather(1, idx10) * f10
            + eyf.gather(1, idx01) * f01
            + eyf.gather(1, idx11) * f11
        )

        force = torch.stack([ex_m, ey_m], dim=2)
        # Fixed macros should not move: zero their force.
        mov = torch.from_numpy(data.movable_hard.astype(np.float32)).to(device=device).view(1, N, 1)
        return force * mov

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

        # Gradient-friendly net tensors: filter huge fan-out nets to keep GPU stages fast/stable.
        # This is a standard trick used in analytical placers (large nets contribute tiny per-pin force).
        MAX_GRAD_FANOUT = 64
        if max_nsz > MAX_GRAD_FANOUT:
            grad_keep = [i for i, net in enumerate(nets_np) if len(net) <= MAX_GRAD_FANOUT]
            grad_nets = [nets_np[i] for i in grad_keep]
            grad_n_nets = len(grad_nets)
            grad_max_nsz = max((len(net) for net in grad_nets), default=1)
            grad_safe_nnp_np = np.zeros((grad_n_nets, grad_max_nsz), dtype=np.int64)
            grad_nnmask_np = np.zeros((grad_n_nets, grad_max_nsz), dtype=bool)
            for i, nodes in enumerate(grad_nets):
                grad_safe_nnp_np[i, : len(nodes)] = nodes
                grad_nnmask_np[i, : len(nodes)] = True
            grad_net_weights_t = net_weights_t[torch.as_tensor(grad_keep, dtype=torch.int64)].clone()
            grad_hpwl_norm = max(1.0, float(grad_n_nets) * (canvas_w + canvas_h))
        else:
            grad_safe_nnp_np = safe_nnp_np
            grad_nnmask_np = nnmask_np
            grad_net_weights_t = net_weights_t
            grad_hpwl_norm = max(1.0, float(n_nets) * (canvas_w + canvas_h))

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
            grad_hpwl_norm=grad_hpwl_norm,
            safe_nnp_np=safe_nnp_np,
            nnmask_np=nnmask_np,
            safe_nnp_t=torch.from_numpy(safe_nnp_np),
            nnmask_t=torch.from_numpy(nnmask_np),
            grad_safe_nnp_t=torch.from_numpy(grad_safe_nnp_np),
            grad_nnmask_t=torch.from_numpy(grad_nnmask_np),
            grad_net_weights_t=grad_net_weights_t,
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

    def _affine_transform_world(self, hard: np.ndarray, data: PreparedData, kind: str) -> np.ndarray:
        # Cheap global topology change: flips/rotations in normalized canvas space, then clamp.
        out = np.asarray(hard, dtype=np.float32).copy()
        if data.num_hard == 0:
            return out
        W = float(max(data.canvas_w, 1e-6))
        H = float(max(data.canvas_h, 1e-6))
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        xn = np.clip(x / W, 0.0, 1.0)
        yn = np.clip(y / H, 0.0, 1.0)

        if kind == "flipx":
            xn2, yn2 = (1.0 - xn), yn
        elif kind == "flipy":
            xn2, yn2 = xn, (1.0 - yn)
        elif kind == "rot180":
            xn2, yn2 = (1.0 - xn), (1.0 - yn)
        elif kind == "rot90":
            # Rotate in normalized space, then fit back into (W,H).
            xn2, yn2 = yn, (1.0 - xn)
        elif kind == "rot270":
            xn2, yn2 = (1.0 - yn), xn
        else:
            return out

        out[:, 0] = xn2 * W
        out[:, 1] = yn2 * H
        # Keep fixed macros unchanged.
        try:
            out[~data.movable_hard] = np.asarray(hard, dtype=np.float32)[~data.movable_hard]
        except Exception:
            pass
        # Clamp to within boundaries considering sizes.
        try:
            for i in range(int(data.num_hard)):
                w, h = data.sizes_hard[i]
                out[i, 0] = float(np.clip(out[i, 0], 0.5 * float(w), float(data.canvas_w) - 0.5 * float(w)))
                out[i, 1] = float(np.clip(out[i, 1], 0.5 * float(h), float(data.canvas_h) - 0.5 * float(h)))
        except Exception:
            pass
        return out.astype(np.float32)

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
                    cand_hard = self._fast_legalize_hard(cand_hard, data, sweeps=4)
                    if jitter_trials > 0:
                        cheap = self._robust_latent_score(
                            cand_hard,
                            base_soft,
                            data,
                            rng,
                            calib,
                        jitter_trials=jitter_trials,
                    )
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
        cong_weight: float = 0.55,
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
            if device.type == "cuda" and step % 10 == 0:
                torch.cuda.synchronize()
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

            xs = all_pos[data.grad_safe_nnp_t.to(device), 0]
            ys = all_pos[data.grad_safe_nnp_t.to(device), 1]
            neg_inf = torch.full_like(xs, -1e9)
            nnmask = data.grad_nnmask_t.to(device)
            xs_p = torch.where(nnmask, xs, neg_inf)
            xs_n = torch.where(nnmask, -xs, neg_inf)
            ys_p = torch.where(nnmask, ys, neg_inf)
            ys_n = torch.where(nnmask, -ys, neg_inf)
            wl = (
                torch.logsumexp(WL_ALPHA * xs_p, dim=1)
                + torch.logsumexp(WL_ALPHA * xs_n, dim=1)
                + torch.logsumexp(WL_ALPHA * ys_p, dim=1)
                + torch.logsumexp(WL_ALPHA * ys_n, dim=1)
            ).sum() / (WL_ALPHA * data.grad_hpwl_norm)

            hard_density = self._density_map(cur_hard, hard_sizes, data, device=device)
            density = hard_density if soft_density is None else hard_density + soft_density
            dens_target = float(getattr(self, "_density_target", BASE_DENSITY_TARGET))
            overflow = (density - dens_target).clamp(min=0.0)
            density_cost = overflow.pow(2).mean() + 0.15 * overflow.max()

            cong_cost = xs.new_tensor(0.0)
            if cong_weight > 1e-9 and step >= int(steps * 0.55):
                cong_map = self._rudy_congestion_map(all_pos, data, device=device, alpha=CONG_ALPHA)
                cong_cost = self._abu_logsumexp(cong_map, frac=0.05)

            overlap_cost = xs.new_tensor(0.0)
            if overlap_weight > 1e-9 and data.num_hard <= 520:
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
            loss = loss + cong_weight * cong_cost
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
        xs = all_pos[data.grad_safe_nnp_t.to(device), 0]
        ys = all_pos[data.grad_safe_nnp_t.to(device), 1]
        mask = data.grad_nnmask_t.to(device)
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
        w = data.grad_net_weights_t.to(device)
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

    def _density_map_batched(
        self,
        pos_b: torch.Tensor,
        sizes_t: torch.Tensor,
        data: PreparedData,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # Batched version of _density_map for latent/batch optimization.
        # pos_b: (B, N, 2), sizes_t: (N, 2) on any device.
        if pos_b.numel() == 0:
            return torch.zeros(
                (pos_b.shape[0], data.grid_rows, data.grid_cols),
                dtype=torch.float32,
                device=pos_b.device,
            )
        if device is None:
            device = pos_b.device

        cell_cx = data.cell_cx_g.to(device)[None, None, :, :]  # (1,1,R,C)
        cell_cy = data.cell_cy_g.to(device)[None, None, :, :]
        sizes = sizes_t.to(device)
        sigma_x = sizes[:, 0].view(1, -1, 1, 1) * 0.5 + data.cell_w * 0.5
        sigma_y = sizes[:, 1].view(1, -1, 1, 1) * 0.5 + data.cell_h * 0.5
        dx = (pos_b[:, :, 0].view(pos_b.shape[0], -1, 1, 1) - cell_cx) / sigma_x
        dy = (pos_b[:, :, 1].view(pos_b.shape[0], -1, 1, 1) - cell_cy) / sigma_y
        kernel = (1.0 - dx.abs()).clamp(min=0.0) * (1.0 - dy.abs()).clamp(min=0.0)
        area_scale = (sizes[:, 0] * sizes[:, 1]).view(1, -1, 1, 1) / max(
            data.cell_w * data.cell_h, 1e-6
        )
        kernel_sum = kernel.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return (kernel / kernel_sum * area_scale).sum(dim=1)

    def _hash_resolve_hard(self, hard: np.ndarray, data: PreparedData, sweeps: int = 4) -> np.ndarray:
        # Large-N overlap reduction with spatial hashing (avoids O(N^2) loops).
        pos = hard.copy().astype(np.float32)
        self._clamp_hard(pos, data)
        gap = float(self._legal_gap(data))

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
                                + gap
                                - abs(dx)
                            )
                            oy = float(
                                (data.sizes_hard[i, 1] + data.sizes_hard[j, 1]) * 0.5
                                + gap
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

    def _hash_has_any_overlap(self, hard: np.ndarray, data: PreparedData) -> bool:
        # Conservative, hashing-based overlap check (avoids O(N^2) exact scan).
        pos = hard.astype(np.float32, copy=False)
        if data.num_hard <= 1:
            return False
        gap = float(self._legal_gap(data))

        bin_w = float(max(np.median(data.sizes_hard[:, 0]), data.cell_w, 1.0))
        bin_h = float(max(np.median(data.sizes_hard[:, 1]), data.cell_h, 1.0))
        inv_w = 1.0 / max(bin_w, 1e-6)
        inv_h = 1.0 / max(bin_h, 1e-6)

        bins: dict[tuple[int, int], list[int]] = {}
        for i in range(data.num_hard):
            bx = int(pos[i, 0] * inv_w)
            by = int(pos[i, 1] * inv_h)
            bins.setdefault((bx, by), []).append(i)

        for i in range(data.num_hard):
            bx = int(pos[i, 0] * inv_w)
            by = int(pos[i, 1] * inv_h)
            wi, hi = float(data.sizes_hard[i, 0]), float(data.sizes_hard[i, 1])
            for dxbin in (-1, 0, 1):
                for dybin in (-1, 0, 1):
                    cell = (bx + dxbin, by + dybin)
                    if cell not in bins:
                        continue
                    for j in bins[cell]:
                        if j <= i:
                            continue
                        wj, hj = float(data.sizes_hard[j, 0]), float(data.sizes_hard[j, 1])
                        dx = float(abs(pos[i, 0] - pos[j, 0]))
                        dy = float(abs(pos[i, 1] - pos[j, 1]))
                        if dx >= 0.5 * (wi + wj) + gap:
                            continue
                        if dy >= 0.5 * (hi + hj) + gap:
                            continue
                        return True
        return False

    def _greedy_reinsert_overlaps(
        self,
        hard: np.ndarray,
        data: PreparedData,
        max_movers: int,
        max_radius: int,
    ) -> np.ndarray:
        # Greedy "remove and reinsert" legalization using the same spatial hash as _hash_resolve_hard.
        # Designed to clean up a small number of stubborn overlaps without O(N^2) passes.
        pos = hard.copy().astype(np.float32)
        self._clamp_hard(pos, data)
        if data.num_hard <= 1:
            return pos

        bin_w = float(max(np.median(data.sizes_hard[:, 0]), data.cell_w, 1.0))
        bin_h = float(max(np.median(data.sizes_hard[:, 1]), data.cell_h, 1.0))
        inv_w = 1.0 / max(bin_w, 1e-6)
        inv_h = 1.0 / max(bin_h, 1e-6)
        step = float(max(data.cell_w, data.cell_h, 1e-3))
        gap = float(self._legal_gap(data))

        bins: dict[tuple[int, int], list[int]] = {}
        for i in range(data.num_hard):
            bx = int(pos[i, 0] * inv_w)
            by = int(pos[i, 1] * inv_h)
            bins.setdefault((bx, by), []).append(i)

        def _collides(idx: int, x: float, y: float) -> bool:
            bx = int(x * inv_w)
            by = int(y * inv_h)
            wi, hi = float(data.sizes_hard[idx, 0]), float(data.sizes_hard[idx, 1])
            for dxbin in (-1, 0, 1):
                for dybin in (-1, 0, 1):
                    cell = (bx + dxbin, by + dybin)
                    if cell not in bins:
                        continue
                    for j in bins[cell]:
                        if j == idx:
                            continue
                        wj, hj = float(data.sizes_hard[j, 0]), float(data.sizes_hard[j, 1])
                        dx = float(abs(x - float(pos[j, 0])))
                        dy = float(abs(y - float(pos[j, 1])))
                        if dx < 0.5 * (wi + wj) + gap and dy < 0.5 * (hi + hj) + gap:
                            return True
            return False

        # Find a set of overlapping movable macros to reinsert.
        colliders: List[Tuple[float, int]] = []
        for i in range(data.num_hard):
            if not data.movable_hard[i]:
                continue
            if _collides(i, float(pos[i, 0]), float(pos[i, 1])):
                key = float(data.hard_areas[i] + 0.10 * data.hard_degree[i])
                colliders.append((key, int(i)))
        colliders.sort(reverse=True, key=lambda it: it[0])
        movers = [idx for _, idx in colliders[: int(max(0, max_movers))]]

        for idx in movers:
            if not data.movable_hard[idx]:
                continue
            # Remove from bins.
            bx0 = int(pos[idx, 0] * inv_w)
            by0 = int(pos[idx, 1] * inv_h)
            cell0 = (bx0, by0)
            if cell0 in bins:
                try:
                    bins[cell0].remove(idx)
                except ValueError:
                    pass

            x0 = float(pos[idx, 0])
            y0 = float(pos[idx, 1])
            best_x = x0
            best_y = y0
            if _collides(idx, x0, y0):
                found = False
                for r in range(1, int(max_radius) + 1):
                    for dxm in range(-r, r + 1):
                        for dym in range(-r, r + 1):
                            if abs(dxm) != r and abs(dym) != r:
                                continue
                            x = float(np.clip(
                                x0 + dxm * step,
                                float(data.sizes_hard[idx, 0]) * 0.5,
                                float(data.canvas_w) - float(data.sizes_hard[idx, 0]) * 0.5,
                            ))
                            y = float(np.clip(
                                y0 + dym * step,
                                float(data.sizes_hard[idx, 1]) * 0.5,
                                float(data.canvas_h) - float(data.sizes_hard[idx, 1]) * 0.5,
                            ))
                            if not _collides(idx, x, y):
                                best_x, best_y = x, y
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

            pos[idx, 0] = best_x
            pos[idx, 1] = best_y

            # Reinsert into bins at new location.
            bx1 = int(pos[idx, 0] * inv_w)
            by1 = int(pos[idx, 1] * inv_h)
            bins.setdefault((bx1, by1), []).append(idx)

        self._clamp_hard(pos, data)
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
        est = getattr(self, "_oracle_call_sec", None)
        if est is not None and self._time_left(time.time()) < est + 2.5:
            return float("inf")

        from macro_place.objective import compute_proxy_cost
        from macro_place.utils import validate_placement

        placement = self._placement_tensor(hard, soft, benchmark, data)
        ok, _ = validate_placement(placement, benchmark)
        if not ok:
            return float("inf")
        t_eval = time.time()
        result = compute_proxy_cost(placement, benchmark, plc)
        dt = time.time() - t_eval
        self._prof["oracle_calls"] = int(self._prof.get("oracle_calls", 0)) + 1
        self._prof["oracle_sec"] = float(self._prof.get("oracle_sec", 0.0)) + float(dt)
        if getattr(self, "_oracle_call_sec", None) is None:
            self._oracle_call_sec = float(dt)
        else:
            self._oracle_call_sec = float(0.65 * self._oracle_call_sec + 0.35 * dt)
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
        est = getattr(self, "_oracle_call_sec", None)
        if est is not None and self._time_left(time.time()) < est + 3.0:
            return None

        from macro_place.objective import compute_proxy_cost
        from macro_place.utils import validate_placement

        placement = self._placement_tensor(hard, soft, benchmark, data)
        ok, _ = validate_placement(placement, benchmark)
        if not ok:
            return None
        t_eval = time.time()
        result = compute_proxy_cost(placement, benchmark, plc)
        dt = time.time() - t_eval
        self._prof["oracle_calls"] = int(self._prof.get("oracle_calls", 0)) + 1
        self._prof["oracle_sec"] = float(self._prof.get("oracle_sec", 0.0)) + float(dt)
        if getattr(self, "_oracle_call_sec", None) is None:
            self._oracle_call_sec = float(dt)
        else:
            self._oracle_call_sec = float(0.65 * self._oracle_call_sec + 0.35 * dt)
        return {
            "proxy": float(result["proxy_cost"]),
            "wl": float(result["wirelength_cost"]),
            "den": float(result["density_cost"]),
            "cong": float(result["congestion_cost"]),
        }

    def _proxy_components_if_valid_with_plc(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
    ) -> Tuple[Optional[dict], float]:
        if plc is None:
            return None, 0.0

        from macro_place.objective import compute_proxy_cost
        from macro_place.utils import validate_placement

        placement = self._placement_tensor(hard, soft, benchmark, data)
        ok, _ = validate_placement(placement, benchmark)
        if not ok:
            return None, 0.0
        t_eval = time.time()
        result = compute_proxy_cost(placement, benchmark, plc)
        dt = time.time() - t_eval
        return (
            {
                "proxy": float(result["proxy_cost"]),
                "wl": float(result["wirelength_cost"]),
                "den": float(result["density_cost"]),
                "cong": float(result["congestion_cost"]),
            },
            float(dt),
        )

    def _oracle_batch_search(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
        label: str,
        max_calls: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, float]]:
        if plc is None or self._time_left(t0) < 25:
            return None

        base = self._proxy_components_if_valid(hard, soft, benchmark, plc, data)
        if base is None:
            return None

        best_cost = float(base["proxy"])
        best_hard = hard.copy().astype(np.float32)
        best_soft = soft.copy().astype(np.float32)
        cur_cost = best_cost
        cur_hard = best_hard.copy()
        cur_soft = best_soft.copy()

        est = float(getattr(self, "_oracle_call_sec", 1.5) or 1.5)
        reserve = float(min(60.0, max(20.0, 0.12 * float(TIME_BUDGET))))
        # Budget oracle calls proportional to remaining time and observed call latency.
        # Avoid artificially tiny caps here; long budgets should be able to use hundreds/thousands
        # of oracle calls on small benchmarks.
        max_calls_calc = int(
            max(10, min(3000, (self._time_left(t0) - reserve) / max(est, 0.4)))
        )
        if max_calls is None:
            max_calls = max_calls_calc
        else:
            max_calls = int(max(10, min(int(max_calls), int(max_calls_calc))))
        if max_calls <= 10:
            return None

        rng = np.random.default_rng((SEED ^ self._stable_seed(f"{benchmark.name}-obatch")) & 0xFFFFFFFF)
        t_start = 0.040
        t_end = 0.004

        def accept(new_cost: float, cur: float, frac: float) -> bool:
            if new_cost + 1e-9 < cur:
                return True
            temp = t_start * (1.0 - frac) + t_end * frac
            if temp <= 1e-9:
                return False
            return rng.random() < math.exp(-(new_cost - cur) / temp)

        def cluster_labels() -> np.ndarray:
            k = 6 if data.num_hard <= 280 else 7 if data.num_hard <= 340 else 8
            return self._kmeans_labels(data.spectral_xy, k)

        def quick_legalize(candidate: np.ndarray) -> Optional[np.ndarray]:
            cand = candidate.astype(np.float32, copy=True)
            self._clamp_hard(cand, data)
            if self._exact_hard_overlap_area(cand.astype(np.float64), data) > 1e-6:
                cand = self._hash_resolve_hard(cand, data, sweeps=2)
                cand = self._tiny_fix_hard(cand, data, rounds=40)
                if self._exact_hard_overlap_area(cand.astype(np.float64), data) > 1e-6:
                    return None
            return cand.astype(np.float32)

        def quick_legalize_aggressive(candidate: np.ndarray) -> Optional[np.ndarray]:
            # Heavier legalization for rare large moves only.
            if self._time_left(t0) < reserve + est + 8:
                return None
            cand = candidate.astype(np.float32, copy=True)
            self._clamp_hard(cand, data)
            cand = self._hash_resolve_hard(cand, data, sweeps=3)
            cand = self._tiny_fix_hard(cand, data, rounds=80)
            if self._exact_hard_overlap_area(cand.astype(np.float64), data) > 1e-6:
                cand = self._fast_legalize_hard(cand, data, sweeps=3)
                if self._exact_hard_overlap_area(cand.astype(np.float64), data) > 1e-6:
                    return None
            return cand.astype(np.float32)

        labels = cluster_labels()
        unit = float(max(data.cell_w, data.cell_h))
        calls = 1

        while calls < max_calls and self._time_left(t0) > reserve + est + 6:
            frac = calls / max(max_calls - 1, 1)
            if calls % 25 == 0:
                try:
                    self._trace(f"oracle_batch label={label} calls={calls}/{max_calls} best={best_cost:.4f} cur={cur_cost:.4f}")
                except Exception:
                    pass

            proposals: List[Tuple[np.ndarray, np.ndarray, str]] = []

            # Hotspot-driven cluster translations (cong + density).
            all_pos = self._all_pos_np(cur_hard, cur_soft, data)
            net_centers = self._net_box_centers_np(all_pos, data)
            cong_grid = self._fast_cong_grid_np(all_pos, data)
            den_grid = self._density_grid_np(np.vstack([cur_hard, cur_soft]), data)

            modes: List[np.ndarray] = []
            if cong_grid is not None:
                m = self._hotspot_escape_mode(np.maximum(cong_grid - 1.0, 0.0), cur_hard, data)
                if m is not None:
                    modes.append(m)
            m = self._hotspot_escape_mode(np.maximum(den_grid - BASE_DENSITY_TARGET, 0.0), cur_hard, data)
            if m is not None:
                modes.append(m)

            for mode_i, mode in enumerate(modes[:2]):
                # Prefer larger coordinated moves for the top mode on small benchmarks.
                steps = (0.7, 1.2, 2.2) if (mode_i == 0 and data.num_hard <= 320) else (0.7, 1.2)
                for cid in range(int(labels.max()) + 1):
                    if len(proposals) >= 90 or self._time_left(t0) < reserve + est + 10:
                        break
                    members = np.flatnonzero((labels == cid) & data.movable_hard)
                    if members.size < 6:
                        continue
                    v = mode[members].mean(axis=0)
                    vn = float(np.linalg.norm(v))
                    if vn < 1e-6:
                        continue
                    v = (v / vn).astype(np.float32)
                    for step in steps:
                        delta = (step * unit) * v
                        for sign in (-1.0, 1.0):
                            cand_h = cur_hard.copy().astype(np.float32)
                            cand_h[members] += sign * delta[None, :]
                            cand_h = quick_legalize(cand_h)
                            if cand_h is None:
                                continue
                            cand_s = cur_soft.copy().astype(np.float32)
                            if data.num_soft > 0:
                                cand_s = self._relax_soft(cand_h, cand_s, data, sweeps=1, damping=0.88)
                            proposals.append((cand_h, cand_s, "cluster-shift"))

            # Rare large random cluster jumps (helps escape initial basin).
            if data.num_hard <= 350:
                for _ in range(2):
                    cid = int(rng.integers(0, int(labels.max()) + 1))
                    members = np.flatnonzero((labels == cid) & data.movable_hard)
                    if members.size < 8:
                        continue
                    delta = rng.normal(0.0, 1.0, size=(2,)).astype(np.float32)
                    dn = float(np.linalg.norm(delta))
                    if dn < 1e-6:
                        continue
                    delta = (delta / dn) * float(unit) * float(rng.uniform(2.5, 5.5))
                    cand_h = cur_hard.copy().astype(np.float32)
                    cand_h[members] += delta[None, :]
                    cand_h = quick_legalize_aggressive(cand_h)
                    if cand_h is None:
                        continue
                    proposals.append((cand_h, cur_soft.copy().astype(np.float32), "cluster-jump"))

            # Connected swaps.
            for _ in range(6 if data.num_hard <= 320 else 4):
                i = int(rng.integers(0, data.num_hard))
                neigh = data.neighbor_lists[i]
                if neigh.size == 0:
                    continue
                j = int(neigh[int(rng.integers(0, neigh.size))])
                if not (data.movable_hard[i] and data.movable_hard[j]):
                    continue
                cand_h = cur_hard.copy().astype(np.float32)
                cand_h[i], cand_h[j] = cand_h[j].copy(), cand_h[i].copy()
                self._clamp_hard(cand_h, data)
                cand_h = self._local_legalize_indices(cand_h, [i, j], data, passes=3, max_radius=20)
                if self._hard_overlaps_any(i, cand_h[i], cand_h, data) or self._hard_overlaps_any(j, cand_h[j], cand_h, data):
                    cand_h = quick_legalize(cand_h)
                    if cand_h is None:
                        continue
                proposals.append((cand_h, cur_soft.copy().astype(np.float32), "swap"))

            # A few global random swaps (not only graph-neighbors).
            for _ in range(4 if data.num_hard <= 320 else 2):
                i = int(rng.integers(0, data.num_hard))
                j = int(rng.integers(0, data.num_hard))
                if i == j or not (data.movable_hard[i] and data.movable_hard[j]):
                    continue
                cand_h = cur_hard.copy().astype(np.float32)
                cand_h[i], cand_h[j] = cand_h[j].copy(), cand_h[i].copy()
                self._clamp_hard(cand_h, data)
                cand_h = self._local_legalize_indices(cand_h, [i, j], data, passes=3, max_radius=20)
                if self._hard_overlaps_any(i, cand_h[i], cand_h, data) or self._hard_overlaps_any(j, cand_h[j], cand_h, data):
                    cand_h = quick_legalize(cand_h)
                    if cand_h is None:
                        continue
                proposals.append((cand_h, cur_soft.copy().astype(np.float32), "swap-global"))

            # Single-macro port pull for poor macros.
            poor = self._diagnose_poor_macros(cur_hard, cur_soft, data)
            order = np.argsort(-poor)[: min(12, data.num_hard)]
            for idx in order[:6]:
                idx = int(idx)
                if not data.movable_hard[idx]:
                    continue
                direction = (data.port_pull[idx] - cur_hard[idx]).astype(np.float32)
                dn = float(np.linalg.norm(direction))
                if dn < 1e-6:
                    continue
                direction /= dn
                step = unit * (0.7 + 1.2 * float(rng.random()))
                cand_h = cur_hard.copy().astype(np.float32)
                cand_h[idx] += direction * step
                self._clamp_hard(cand_h, data)
                cand_h = self._local_legalize_indices(cand_h, [idx], data, passes=2, max_radius=16)
                if self._hard_overlaps_any(idx, cand_h[idx], cand_h, data):
                    cand_h = quick_legalize(cand_h)
                    if cand_h is None:
                        continue
                proposals.append((cand_h, cur_soft.copy().astype(np.float32), "port-pull"))

            # Single-macro net-centroid pull (more direct WL reduction than port-only).
            for idx in order[:6]:
                idx = int(idx)
                if not data.movable_hard[idx]:
                    continue
                target = self._macro_net_target(idx, net_centers, data).astype(np.float32)
                direction = (target - cur_hard[idx]).astype(np.float32)
                dn = float(np.linalg.norm(direction))
                if dn < 1e-6:
                    continue
                direction /= dn
                step = float(unit) * float(0.6 + 2.0 * float(rng.random()))
                cand_h = cur_hard.copy().astype(np.float32)
                cand_h[idx] += direction * step
                self._clamp_hard(cand_h, data)
                cand_h = self._local_legalize_indices(cand_h, [idx], data, passes=2, max_radius=18)
                if self._hard_overlaps_any(idx, cand_h[idx], cand_h, data):
                    cand_h = quick_legalize_aggressive(cand_h)
                    if cand_h is None:
                        continue
                proposals.append((cand_h, cur_soft.copy().astype(np.float32), "net-pull"))

            # Subset greedy reinsertion: jointly relocate a handful of poor macros toward net targets.
            if self._time_left(t0) > reserve + 2.0 * est + 8 and poor.size >= 12:
                try:
                    kk = 10 if data.num_hard <= 280 else 12
                    subset = [int(i) for i in order[:kk] if data.movable_hard[int(i)]]
                    if len(subset) >= 6:
                        targets = np.zeros((data.num_hard, 2), dtype=np.float32)
                        for ii in range(data.num_hard):
                            targets[ii] = self._macro_net_target(int(ii), net_centers, data).astype(np.float32)
                        cand_h = self._greedy_reinsert_subset(
                            cur_hard,
                            subset=subset,
                            targets=targets,
                            data=data,
                            rng=rng,
                            max_radius=24,
                            restarts=2,
                        )
                        if cand_h is not None:
                            proposals.append((cand_h, cur_soft.copy().astype(np.float32), "subset-reinsert"))
                except Exception:
                    pass

            # Cluster exchange: swap two clusters' centroids (large coordinated move).
            if self._time_left(t0) > reserve + 2.0 * est + 8 and frac < 0.65:
                try:
                    lab = labels
                    uniq = np.unique(lab)
                    if uniq.size >= 2:
                        a = int(uniq[int(rng.integers(0, uniq.size))])
                        b = int(uniq[int(rng.integers(0, uniq.size))])
                        if a != b:
                            ma = np.flatnonzero(lab == a)
                            mb = np.flatnonzero(lab == b)
                            ma = ma[data.movable_hard[ma]]
                            mb = mb[data.movable_hard[mb]]
                            if ma.size >= 4 and mb.size >= 4:
                                ca = cur_hard[ma].mean(axis=0).astype(np.float32)
                                cb = cur_hard[mb].mean(axis=0).astype(np.float32)
                                delta_a = (cb - ca).astype(np.float32)
                                delta_b = (ca - cb).astype(np.float32)
                                cand_h = cur_hard.copy().astype(np.float32)
                                cand_h[ma] = (cand_h[ma] + delta_a[None, :]).astype(np.float32)
                                cand_h[mb] = (cand_h[mb] + delta_b[None, :]).astype(np.float32)
                                cand_h = quick_legalize_aggressive(cand_h)
                                if cand_h is not None:
                                    proposals.append((cand_h, cur_soft.copy().astype(np.float32), "cluster-xchg"))
                except Exception:
                    pass

            # Random swaps (macro placement is highly discrete; swaps are a strong operator).
            # Prefer swapping "poor" macros to escape local congestion/density traps.
            movable_idx = np.flatnonzero(data.movable_hard)
            if movable_idx.size >= 2:
                swap_trials = 10 if frac < 0.5 else 6
                for _ in range(int(swap_trials)):
                    max_seed = int(min(20, int(order.size))) if hasattr(order, "size") else 0
                    if max_seed <= 0:
                        break
                    i = int(order[int(rng.integers(0, max_seed))])
                    j = int(movable_idx[int(rng.integers(0, movable_idx.size))])
                    if i == j or not (data.movable_hard[i] and data.movable_hard[j]):
                        continue
                    cand_h = cur_hard.copy().astype(np.float32)
                    cand_h[i], cand_h[j] = cand_h[j].copy(), cand_h[i].copy()
                    cand_h = quick_legalize(cand_h)
                    if cand_h is None:
                        continue
                    proposals.append((cand_h, cur_soft.copy().astype(np.float32), "swap"))

            # Subset OT/LNS: jointly reassign a small subset of "bad" macros to a small set of
            # candidate slots near their net targets (facility-location flavored move).
            if (
                self._device.type == "cuda"
                and movable_idx.size >= 32
                and data.num_hard <= 420
                and frac < 0.80
                and self._time_left(t0) > reserve + 2.0 * est + 10
            ):
                try:
                    k = 28 if data.num_hard <= 280 else 32
                    k = int(min(k, int(order.size)))
                    subset = []
                    for idx in order[:k]:
                        idx = int(idx)
                        if data.movable_hard[idx]:
                            subset.append(idx)
                    if len(subset) >= 16:
                        subset = np.array(subset, dtype=np.int64)
                        # Build candidate "slots": blend current positions with net targets and add small jitters.
                        tgt = np.zeros((subset.size, 2), dtype=np.float32)
                        for ii, mi in enumerate(subset.tolist()):
                            tgt[ii] = self._macro_net_target(int(mi), net_centers, data).astype(np.float32)
                        slots = (0.55 * cur_hard[subset] + 0.45 * tgt).astype(np.float32)
                        unit_j = float(max(data.cell_w, data.cell_h))
                        slots += rng.normal(0.0, 0.45 * unit_j, size=slots.shape).astype(np.float32)
                        # Clamp slots to canvas for each macro size.
                        halfw = data.sizes_hard[subset, 0] * 0.5
                        halfh = data.sizes_hard[subset, 1] * 0.5
                        slots[:, 0] = np.clip(slots[:, 0], halfw, data.canvas_w - halfw)
                        slots[:, 1] = np.clip(slots[:, 1], halfh, data.canvas_h - halfh)

                        device = self._device
                        tgt_t = torch.from_numpy(tgt).to(device=device, dtype=torch.float32)
                        slot_t = torch.from_numpy(slots).to(device=device, dtype=torch.float32)
                        dx = tgt_t[:, None, 0] - slot_t[None, :, 0]
                        dy = tgt_t[:, None, 1] - slot_t[None, :, 1]
                        scale = float(max(data.canvas_w, data.canvas_h, 1e-6))
                        cost = (dx * dx + dy * dy) / float(scale * scale)
                        cost = cost - cost.amin()
                        eps = 0.012
                        K = torch.exp((-cost / max(eps, 1e-6)).clamp(min=-40.0, max=40.0))
                        a = torch.full((subset.size,), 1.0 / float(subset.size), device=device, dtype=torch.float32)
                        b = torch.full((subset.size,), 1.0 / float(subset.size), device=device, dtype=torch.float32)
                        u = torch.ones_like(a)
                        v = torch.ones_like(b)
                        for _ in range(18):
                            Kv = torch.matmul(K, v) + 1e-9
                            u = a / Kv
                            Ktu = torch.matmul(K.t(), u) + 1e-9
                            v = b / Ktu
                        P = (u[:, None] * K) * v[None, :]
                        flat = P.flatten()
                        order_pairs = torch.argsort(flat, descending=True).detach().cpu().numpy()
                        assigned_m = np.full(int(subset.size), -1, dtype=np.int32)
                        assigned_s = np.full(int(subset.size), -1, dtype=np.int32)
                        used_m = np.zeros(int(subset.size), dtype=np.bool_)
                        used_s = np.zeros(int(subset.size), dtype=np.bool_)
                        for idx_flat in order_pairs[: int(subset.size * subset.size)]:
                            i = int(idx_flat // subset.size)
                            j = int(idx_flat - i * subset.size)
                            if used_m[i] or used_s[j]:
                                continue
                            used_m[i] = True
                            used_s[j] = True
                            assigned_m[i] = i
                            assigned_s[i] = j
                            if used_m.all():
                                break
                        if (assigned_s >= 0).all():
                            cand_h = cur_hard.copy().astype(np.float32)
                            cand_h[subset] = slots[assigned_s]
                            cand_h = quick_legalize_aggressive(cand_h)
                            if cand_h is not None:
                                proposals.append((cand_h, cur_soft.copy().astype(np.float32), "subset-ot"))
                except Exception:
                    pass

            # Gaussian perturbations (small late, larger early).
            sigma_small = float(unit) * float(0.35 + 0.90 * (1.0 - frac))
            sigma_large = float(unit) * float(1.10 + 2.30 * (1.0 - frac))
            for _ in range(8):
                max_seed = int(min(24, int(order.size))) if hasattr(order, "size") else 0
                if max_seed <= 0:
                    break
                idx = int(order[int(rng.integers(0, max_seed))])
                if not data.movable_hard[idx]:
                    continue
                cand_h = cur_hard.copy().astype(np.float32)
                cand_h[idx, 0] += float(rng.normal(0.0, sigma_small))
                cand_h[idx, 1] += float(rng.normal(0.0, sigma_small))
                self._clamp_hard(cand_h, data)
                cand_h = self._local_legalize_indices(cand_h, [idx], data, passes=2, max_radius=16)
                if self._hard_overlaps_any(idx, cand_h[idx], cand_h, data):
                    cand_h = quick_legalize(cand_h)
                    if cand_h is None:
                        continue
                proposals.append((cand_h, cur_soft.copy().astype(np.float32), "gauss-sm"))
            if frac < 0.40 and self._time_left(t0) > reserve + 2.0 * est + 8:
                for _ in range(4):
                    max_seed = int(min(18, int(order.size))) if hasattr(order, "size") else 0
                    if max_seed <= 0:
                        break
                    idx = int(order[int(rng.integers(0, max_seed))])
                    if not data.movable_hard[idx]:
                        continue
                    cand_h = cur_hard.copy().astype(np.float32)
                    cand_h[idx, 0] += float(rng.normal(0.0, sigma_large))
                    cand_h[idx, 1] += float(rng.normal(0.0, sigma_large))
                    self._clamp_hard(cand_h, data)
                    cand_h = self._local_legalize_indices(cand_h, [idx], data, passes=3, max_radius=22)
                    if self._hard_overlaps_any(idx, cand_h[idx], cand_h, data):
                        cand_h = quick_legalize_aggressive(cand_h)
                        if cand_h is None:
                            continue
                    proposals.append((cand_h, cur_soft.copy().astype(np.float32), "gauss-lg"))

            if not proposals:
                break

            # Cheap filter.
            ranked = []
            for cand_h, cand_s, reason in proposals:
                ranked.append((self._cheap_score(cand_h, cand_s, data), cand_h, cand_s, reason))
            ranked.sort(key=lambda x: x[0])

            best_local = None
            best_local_cost = float("inf")
            to_eval = ranked[: min(10, len(ranked))]
            if calls + len(to_eval) > max_calls:
                to_eval = to_eval[: max(0, max_calls - calls)]
            if not to_eval:
                break

            # Optional parallel oracle evaluation using independent plc clones.
            oracle_pool = getattr(self, "_oracle_plcs", None)
            can_parallel = (
                oracle_pool is not None
                and isinstance(oracle_pool, list)
                and len(oracle_pool) >= 2
                and len(to_eval) >= 4
                and float(est) >= 0.30
                and self._time_left(t0) > reserve + float(est) + 4
            )

            if can_parallel:
                try:
                    import concurrent.futures as cf

                    with cf.ThreadPoolExecutor(max_workers=len(oracle_pool)) as ex:
                        futs = []
                        for idx, (_, cand_h, cand_s, _) in enumerate(to_eval):
                            plc_local = oracle_pool[idx % len(oracle_pool)]
                            fut = ex.submit(
                                self._proxy_components_if_valid_with_plc,
                                cand_h,
                                cand_s,
                                benchmark,
                                plc_local,
                                data,
                            )
                            futs.append((cand_h, cand_s, fut))

                        for cand_h, cand_s, fut in futs:
                            if calls >= max_calls or self._time_left(t0) <= reserve + float(est) + 2:
                                break
                            comps, dt = fut.result(timeout=max(2.0, 1.6 * float(est)))
                            calls += 1
                            if comps is None:
                                continue
                            self._prof["oracle_calls"] = int(self._prof.get("oracle_calls", 0)) + 1
                            self._prof["oracle_sec"] = float(self._prof.get("oracle_sec", 0.0)) + float(dt)
                            if float(dt) > 1e-6:
                                if getattr(self, "_oracle_call_sec", None) is None:
                                    self._oracle_call_sec = float(dt)
                                else:
                                    self._oracle_call_sec = float(0.65 * self._oracle_call_sec + 0.35 * dt)
                            cost = float(comps["proxy"])
                            if cost + 1e-9 < best_local_cost:
                                best_local_cost = cost
                                best_local = (cand_h, cand_s)
                except Exception:
                    can_parallel = False

            if not can_parallel:
                for _, cand_h, cand_s, _ in to_eval:
                    if calls >= max_calls or self._time_left(t0) <= reserve + float(est) + 2:
                        break
                    comps = self._proxy_components_if_valid(cand_h, cand_s, benchmark, plc, data)
                    calls += 1
                    if comps is None:
                        continue
                    cost = float(comps["proxy"])
                    if cost + 1e-9 < best_local_cost:
                        best_local_cost = cost
                        best_local = (cand_h, cand_s)

            if best_local is None:
                break

            cand_h, cand_s = best_local
            if accept(best_local_cost, cur_cost, frac):
                cur_hard = cand_h.copy()
                cur_soft = cand_s.copy()
                cur_cost = float(best_local_cost)
                if cur_cost + 1e-9 < best_cost:
                    best_cost = cur_cost
                    best_hard = cur_hard.copy()
                    best_soft = cur_soft.copy()
                    # Refresh clusters around improved state.
                    labels = cluster_labels()

        if best_cost + 1e-6 < float(base["proxy"]):
            return best_hard, best_soft, label, self._cheap_score(best_hard, best_soft, data)
        return None

    def _oracle_sa_multistart(
        self,
        candidates: Sequence[Tuple[np.ndarray, np.ndarray, str, float]],
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
        max_calls: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, float]]:
        if plc is None or not candidates or self._time_left(t0) < 35:
            return None

        # Choose diverse starts: don't trust the surrogate ordering too much.
        # Always consider a few named anchors (initial/gapfix/legacy) plus family diversity.
        pool = sorted(list(candidates), key=lambda c: float(c[3]))
        starts: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        seen_prefix: set = set()
        # Named anchors (if present).
        name_index = {str(n): (h, s, n, float(sc)) for (h, s, n, sc) in pool if isinstance(n, str)}
        for key in ["initial-gapfix", "initial-raw", "legacy-base", "pair-base", "shelf-base"]:
            if key in name_index:
                starts.append(name_index[key])
                seen_prefix.add(key.split("-")[0])
        best_periphery = None
        for cand in pool:
            nm = str(cand[2]) if cand[2] is not None else ""
            if nm.startswith("periphery-exact"):
                best_periphery = cand
                break
        if best_periphery is None:
            for cand in pool:
                nm = str(cand[2]) if cand[2] is not None else ""
                if nm.startswith("periphery-"):
                    best_periphery = cand
                    break
        if best_periphery is not None:
            starts.append(best_periphery)
            seen_prefix.add("periphery")
        # Best cheap always.
        if pool:
            starts.append(pool[0])
            if pool[0][2]:
                seen_prefix.add(str(pool[0][2]).split("-")[0])

        for hard, soft, name, cheap in pool:
            prefix = name.split("-")[0] if name else ""
            if prefix in seen_prefix and len(starts) >= 3:
                continue
            starts.append((hard, soft, name, cheap))
            seen_prefix.add(prefix)
            if len(starts) >= 6:
                break
        if not starts:
            return None

        est = float(getattr(self, "_oracle_call_sec", 1.5) or 1.5)
        reserve = float(min(60.0, max(20.0, 0.12 * float(TIME_BUDGET))))
        calls_budget = int(
            max(10, min(int(max_calls), (self._time_left(t0) - reserve) / max(est, 0.4)))
        )
        if calls_budget <= 10:
            return None

        # Distribute oracle budget across starts, but allow deeper per-start search when oracle is fast.
        n_starts = int(max(1, min(len(starts), 5 if data.num_hard <= 280 else 4 if data.num_hard <= 320 else 3)))
        calls_per = int(max(18, min(120, calls_budget // max(n_starts, 1))))

        if PROFILE:
            import sys

            print(
                f"[PROFILE] {benchmark.name}: oracle_sa_multistart n_starts={n_starts} calls_per={calls_per} calls_budget={calls_budget}",
                file=sys.stderr,
            )

        improved: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        for i, (hard, soft, name, _) in enumerate(starts[:n_starts]):
            if self._time_left(t0) < reserve + est + 10:
                break
            label = f"{name}-osa{i+1}"
            out = self._oracle_batch_search(
                hard,
                soft,
                benchmark,
                plc,
                data,
                t0,
                label=label,
                max_calls=calls_per,
            )
            if out is not None:
                improved.append(out)

        if not improved:
            if PROFILE:
                import sys

                print(f"[PROFILE] {benchmark.name}: oracle_sa_multistart no_improve", file=sys.stderr)
            return None

        # If we can afford a couple exact checks, pick the best by oracle.
        improved = sorted(improved, key=lambda c: float(c[3]))
        best = improved[0]
        best_cost = float("inf")
        checks = min(2, len(improved))
        for cand_h, cand_s, cand_name, cand_cheap in improved[:checks]:
            if self._time_left(t0) < reserve + est + 4:
                break
            cost = self._proxy_cost_if_valid(cand_h, cand_s, benchmark, plc, data)
            if math.isfinite(cost) and cost + 1e-9 < best_cost:
                best_cost = float(cost)
                best = (cand_h, cand_s, cand_name, cand_cheap)
        return best

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

    def _ot_sinkhorn_reassign(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        plc,
        data: PreparedData,
        t0: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, float]]:
        # Globally coordinated reassignment using entropically-regularized OT (Sinkhorn),
        # then greedy rounding to a permutation. This is a "macro shuffle" move that can
        # escape local basins when oracle is slow, while staying GPU-friendly.
        if self._device.type != "cuda" or plc is None or self._time_left(t0) < 18:
            return None

        if PROFILE:
            import sys

            print(f"[PROFILE] {benchmark.name}: ot_reassign_start", file=sys.stderr)

        movable_idx = np.flatnonzero(data.movable_hard)
        m = int(movable_idx.size)
        if m < 64 or m > 520:
            return None
        if self._time_left(t0) < (22 if m <= 420 else 28):
            return None

        all_pos = self._all_pos_np(hard, soft, data)
        net_centers = self._net_box_centers_np(all_pos, data)
        targets = np.zeros((m, 2), dtype=np.float32)
        for k, idx in enumerate(movable_idx):
            targets[k] = self._macro_net_target(int(idx), net_centers, data).astype(np.float32)
        # Blend toward current to reduce destructiveness.
        targets = (0.65 * hard[movable_idx] + 0.35 * targets).astype(np.float32)

        # Build a slot grid (size m) with aspect ratio similar to canvas.
        aspect = float(max(data.canvas_w, 1e-6) / max(data.canvas_h, 1e-6))
        nx = int(max(4, math.ceil(math.sqrt(float(m) * aspect))))
        ny = int(max(4, math.ceil(float(m) / float(nx))))
        nx = int(min(nx, max(4, m)))
        ny = int(min(ny, max(4, m)))
        # Keep a bit of padding for large macros.
        pad = 0.5 * float(max(np.percentile(data.sizes_hard[movable_idx, 0], 75), data.cell_w))
        pad_y = 0.5 * float(max(np.percentile(data.sizes_hard[movable_idx, 1], 75), data.cell_h))
        xs = np.linspace(pad, float(data.canvas_w) - pad, nx, dtype=np.float32)
        ys = np.linspace(pad_y, float(data.canvas_h) - pad_y, ny, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        slots = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
        if slots.shape[0] < m:
            return None
        slots = slots[:m].astype(np.float32, copy=False)

        device = self._device
        tgt_t = torch.from_numpy(targets).to(device=device, dtype=torch.float32)
        slot_t = torch.from_numpy(slots).to(device=device, dtype=torch.float32)

        # Cost = normalized squared distance.
        dx = tgt_t[:, None, 0] - slot_t[None, :, 0]
        dy = tgt_t[:, None, 1] - slot_t[None, :, 1]
        scale = float(max(data.canvas_w, data.canvas_h, 1e-6))
        cost = (dx * dx + dy * dy) / float(scale * scale)
        cost = cost - cost.amin()

        eps = 0.010 if m <= 420 else 0.016
        K = torch.exp((-cost / max(eps, 1e-6)).clamp(min=-40.0, max=40.0))
        a = torch.full((m,), 1.0 / float(m), device=device, dtype=torch.float32)
        b = torch.full((m,), 1.0 / float(m), device=device, dtype=torch.float32)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(32):
            if self._time_left(t0) < 10:
                return None
            Kv = torch.matmul(K, v) + 1e-9
            u = a / Kv
            Ktu = torch.matmul(K.t(), u) + 1e-9
            v = b / Ktu

        P = (u[:, None] * K) * v[None, :]
        flat = P.flatten()
        order = torch.argsort(flat, descending=True).detach().cpu().numpy()
        assigned_macro = np.full(m, -1, dtype=np.int32)
        assigned_slot = np.full(m, -1, dtype=np.int32)
        for idx in order:
            i = int(idx // m)
            j = int(idx - i * m)
            if assigned_macro[i] != -1 or assigned_slot[j] != -1:
                continue
            assigned_macro[i] = j
            assigned_slot[j] = i
            if int((assigned_macro != -1).sum()) >= m:
                break
        if np.any(assigned_macro < 0):
            # Fallback: fill remaining with first free slots.
            free_slots = list(np.flatnonzero(assigned_slot < 0).astype(np.int32))
            for i in range(m):
                if assigned_macro[i] >= 0:
                    continue
                if not free_slots:
                    break
                assigned_macro[i] = int(free_slots.pop())

        hard_new = hard.copy().astype(np.float32)
        hard_new[movable_idx] = slots[assigned_macro]
        self._clamp_hard(hard_new, data)
        # Keep this stage light; full legalization is expensive and can dominate the whole budget.
        hard_new = self._hash_resolve_hard(hard_new, data, sweeps=4)
        if data.num_hard <= 420:
            hard_new = self._tiny_fix_hard(hard_new, data, rounds=25)
        if not self._placement_is_valid(hard_new, soft, benchmark, data):
            if PROFILE:
                import sys

                print(f"[PROFILE] {benchmark.name}: ot_reassign_invalid", file=sys.stderr)
            return None

        soft_new = soft.copy().astype(np.float32)
        if data.num_soft > 0 and self._time_left(t0) >= 16:
            soft_new = self._relax_soft(hard_new, soft_new, data, sweeps=1, damping=0.90)
        if PROFILE:
            import sys

            print(f"[PROFILE] {benchmark.name}: ot_reassign_done", file=sys.stderr)
        return hard_new.astype(np.float32), soft_new.astype(np.float32), "ot-reassign", self._cheap_score(hard_new, soft_new, data)

    def _surrogate_local_search(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
        t0: float,
        seconds: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, float]]:
        # Fast heuristic search purely on the cheap surrogate (wl + density + optional fast congestion).
        # Purpose: improve mid-size benchmarks when oracle is slow and budgets are short.
        if seconds <= 0.0 or self._time_left(t0) < seconds + 6:
            return None
        if data.num_hard < 300 or data.num_hard > 520:
            return None

        cur_h = hard.copy().astype(np.float32)
        cur_s = soft.copy().astype(np.float32)
        self._clamp_hard(cur_h, data)
        if self._exact_hard_overlap_area(cur_h.astype(np.float64), data) > 1e-8:
            return None

        best_h = cur_h.copy()
        best_s = cur_s.copy()

        def score_fast(hh: np.ndarray) -> float:
            # wl + density + overlap (no congestion) for speed.
            all_pos = self._all_pos_np(hh, cur_s, data)
            xs = all_pos[data.safe_nnp_np, 0]
            ys = all_pos[data.safe_nnp_np, 1]
            inf = 1e15
            wl = (
                np.where(data.nnmask_np, xs, -inf).max(axis=1)
                - np.where(data.nnmask_np, xs, inf).min(axis=1)
                + np.where(data.nnmask_np, ys, -inf).max(axis=1)
                - np.where(data.nnmask_np, ys, inf).min(axis=1)
            ).sum() / data.hpwl_norm
            density = self._density_grid_np(np.vstack([hh, cur_s]), data)
            den = self._top_k_mean(density, 0.10)
            overlap = self._exact_hard_overlap_area(hh.astype(np.float64), data) / max(data.canvas_area, 1e-6)
            return float(wl + 0.50 * den + 250.0 * overlap)

        start_fast = float(score_fast(best_h))
        best_score = float(start_fast)
        cur_score = float(start_fast)

        rng = np.random.default_rng((SEED ^ self._stable_seed("surrogate")) & 0xFFFFFFFF)
        unit = float(max(data.cell_w, data.cell_h))
        t_end = time.time() + float(min(seconds, max(0.0, self._time_left(t0) - 6.0)))

        def quick_legalize(candidate: np.ndarray) -> Optional[np.ndarray]:
            cand = candidate.astype(np.float32, copy=True)
            self._clamp_hard(cand, data)
            cand = self._hash_resolve_hard(cand, data, sweeps=2)
            if data.num_hard <= 420:
                cand = self._tiny_fix_hard(cand, data, rounds=35)
            if self._exact_hard_overlap_area(cand.astype(np.float64), data) > 1e-8:
                return None
            return cand.astype(np.float32)

        # Precompute clusters for coordinated moves.
        labels = self._kmeans_labels(data.spectral_xy, k=8 if data.num_hard <= 380 else 10, iters=8)

        # Annealing schedule (on fast surrogate deltas).
        t_start = 0.020
        t_final = 0.003
        iter_count = 0
        sample_idx = None
        try:
            if data.fast_cong_engine is not None:
                sample_idx = data.fast_cong_engine.get("sample_idx", None)
        except Exception:
            sample_idx = None

        while time.time() < t_end and self._time_left(t0) > 6:
            iter_count += 1
            frac = float(min(1.0, (time.time() - (t_end - seconds)) / max(seconds, 1e-6)))
            temp = t_start * (1.0 - frac) + t_final * frac

            poor = self._diagnose_poor_macros(cur_h, cur_s, data)
            order = np.argsort(-poor)
            seeds = order[: min(12, data.num_hard)]

            proposals: List[np.ndarray] = []

            # Single-macro net pull.
            all_pos = self._all_pos_np(cur_h, cur_s, data)
            net_centers = self._net_box_centers_np(all_pos, data)
            for idx in seeds[:6]:
                idx = int(idx)
                if not data.movable_hard[idx]:
                    continue
                target = self._macro_net_target(idx, net_centers, data).astype(np.float32)
                direction = (target - cur_h[idx]).astype(np.float32)
                dn = float(np.linalg.norm(direction))
                if dn < 1e-6:
                    continue
                direction /= dn
                step = float(unit) * float(rng.uniform(0.6, 2.4))
                cand = cur_h.copy()
                cand[idx] = cur_h[idx] + direction * step
                proposals.append(cand.astype(np.float32))

            # Neighbor swaps (local topology changes).
            for _ in range(10):
                i = int(seeds[int(rng.integers(0, max(1, len(seeds))))])
                neigh = data.neighbor_lists[i]
                if neigh.size == 0:
                    continue
                j = int(neigh[int(rng.integers(0, neigh.size))])
                if i == j or not (data.movable_hard[i] and data.movable_hard[j]):
                    continue
                cand = cur_h.copy().astype(np.float32)
                cand[i], cand[j] = cand[j].copy(), cand[i].copy()
                proposals.append(cand)

            # Density hotspot escape (cheap, coordinated by clusters).
            den = self._density_grid_np(np.vstack([cur_h, cur_s]), data)
            den_over = np.maximum(den - BASE_DENSITY_TARGET, 0.0)
            mode = self._hotspot_escape_mode(den_over, cur_h, data)
            if mode is not None:
                for _ in range(6):
                    cid = int(rng.integers(0, int(labels.max()) + 1))
                    members = np.flatnonzero((labels == cid) & data.movable_hard)
                    if members.size < 10:
                        continue
                    v = mode[members].mean(axis=0).astype(np.float32)
                    vn = float(np.linalg.norm(v))
                    if vn < 1e-6:
                        continue
                    v = v / vn
                    step = float(unit) * float(rng.uniform(0.8, 2.2))
                    cand = cur_h.copy().astype(np.float32)
                    cand[members] += step * v[None, :]
                    proposals.append(cand)

            # Congestion hotspot escape (sampled routing pairs for speed).
            if sample_idx is not None and isinstance(sample_idx, np.ndarray) and sample_idx.size > 0:
                cong_grid = self._fast_cong_grid_np_sample(all_pos, data, sample_idx)
                if cong_grid is not None:
                    cong_over = np.maximum(cong_grid - 1.0, 0.0)
                    mode = self._hotspot_escape_mode(cong_over, cur_h, data)
                    if mode is not None:
                        for _ in range(5):
                            cid = int(rng.integers(0, int(labels.max()) + 1))
                            members = np.flatnonzero((labels == cid) & data.movable_hard)
                            if members.size < 10:
                                continue
                            v = mode[members].mean(axis=0).astype(np.float32)
                            vn = float(np.linalg.norm(v))
                            if vn < 1e-6:
                                continue
                            v = v / vn
                            step = float(unit) * float(rng.uniform(0.8, 2.0))
                            cand = cur_h.copy().astype(np.float32)
                            cand[members] += step * v[None, :]
                            proposals.append(cand)

            if not proposals:
                break

            # Evaluate a subset, accept with annealing.
            rng.shuffle(proposals)
            best_local = None
            best_local_score = cur_score
            eval_budget = 10 if TIME_BUDGET <= 240 else 14
            for cand_h in proposals[:eval_budget]:
                if self._time_left(t0) < 6:
                    break
                cand_h2 = quick_legalize(cand_h)
                if cand_h2 is None:
                    continue
                cand_s2 = cur_s
                cand_score = float(score_fast(cand_h2))
                if cand_score + 1e-9 < best_local_score:
                    best_local_score = cand_score
                    best_local = cand_h2

            if best_local is None:
                # Occasionally accept a random legal proposal to escape local minima.
                accepted = False
                for cand_h in proposals[: min(6, len(proposals))]:
                    cand_h2 = quick_legalize(cand_h)
                    if cand_h2 is None:
                        continue
                    cand_score = float(score_fast(cand_h2))
                    if cand_score + 1e-9 < cur_score:
                        cur_h = cand_h2
                        cur_score = cand_score
                        accepted = True
                        break
                    if temp > 1e-9 and rng.random() < math.exp(-(cand_score - cur_score) / temp):
                        cur_h = cand_h2
                        cur_score = cand_score
                        accepted = True
                        break
                if not accepted:
                    # If we can't move for a while, stop early.
                    if iter_count >= 40:
                        break
            else:
                cur_h = best_local
                cur_score = best_local_score

            if cur_score + 1e-9 < best_score:
                best_score = cur_score
                best_h = cur_h.copy()
                best_s = cur_s.copy()

        # Return if we improved the fast surrogate by a meaningful margin; selection later uses exact proxy anyway.
        if best_score + 1e-4 < start_fast:
            return best_h.astype(np.float32), best_s.astype(np.float32), "surrogate-ls", float(best_score)
        return None

    def _auction_slot_permute(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        benchmark: Benchmark,
        data: PreparedData,
        t0: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, float]]:
        # Parallel-auction assignment over a fixed set of slot coordinates.
        # Slot coords are the current positions of movable macros, so we preserve geometry
        # but change which macro occupies which "role" to reduce WL/congestion basins.
        if self._time_left(t0) < 18 or data.num_hard <= 1:
            return None

        movable_idx = np.flatnonzero(data.movable_hard).astype(np.int64)
        m = int(movable_idx.size)
        if m < 40 or m > 900:
            return None

        # Build targets (net-centroid) using current placement for all nodes.
        all_pos = self._all_pos_np(hard, soft, data)
        net_centers = self._net_box_centers_np(all_pos, data)
        targets = np.zeros((m, 2), dtype=np.float32)
        for k, idx in enumerate(movable_idx):
            targets[k] = self._macro_net_target(int(idx), net_centers, data).astype(np.float32)

        slots = hard[movable_idx].astype(np.float32, copy=True)
        curpos = slots.copy()

        device = self._device if hasattr(self, "_device") else torch.device("cpu")
        if device.type != "cuda":
            return None
        if self._time_left(t0) < 16:
            return None

        if PROFILE:
            import sys

            print(f"[PROFILE] {benchmark.name}: auction_permute_start m={m}", file=sys.stderr)

        slots_t_full = torch.from_numpy(slots).to(device=device, dtype=torch.float32)
        cur_t_full = torch.from_numpy(curpos).to(device=device, dtype=torch.float32)
        tgt_t_full = torch.from_numpy(targets).to(device=device, dtype=torch.float32)
        sizes_t_full = torch.from_numpy(data.sizes_hard[movable_idx].astype(np.float32)).to(device=device)

        scale = float(max(data.canvas_w, data.canvas_h, 1e-6))
        mean_area = float(max(float(data.hard_areas[movable_idx].mean()), 1e-6))

        # Cost weights (tuned to be conservative).
        w_t = 1.00  # toward net target
        w_c = 0.18  # stay near current role
        w_o = 0.60  # avoid size-incompatible local packing

        def auction_assign(cost_np: np.ndarray) -> np.ndarray:
            # Parallel auction (Jacobi-style bidding) on a sparse candidate set.
            m_local = int(cost_np.shape[0])
            L_local = 64 if m_local >= 220 else 48
            L_local = int(min(max(16, L_local), m_local))
            cand = np.argpartition(cost_np, kth=L_local - 1, axis=1)[:, :L_local].astype(np.int32, copy=False)

            eps = 1e-4
            price = np.zeros(m_local, dtype=np.float32)
            owner = np.full(m_local, -1, dtype=np.int32)
            assign = np.full(m_local, -1, dtype=np.int32)

            max_iter = 800
            for _ in range(max_iter):
                if self._time_left(t0) < 6:
                    break
                unassigned = np.flatnonzero(assign < 0)
                if unassigned.size == 0:
                    break

                best_bid = np.full(m_local, -np.inf, dtype=np.float32)
                bidder = np.full(m_local, -1, dtype=np.int32)
                inc = np.zeros(m_local, dtype=np.float32)

                for i in unassigned:
                    ci = cand[i]
                    # benefit = -cost - price
                    vals = -cost_np[i, ci] - price[ci]
                    if vals.size == 0:
                        continue
                    # top-2
                    j1 = int(ci[int(np.argmax(vals))])
                    v1 = float(vals[int(np.argmax(vals))])
                    # second best
                    if vals.size >= 2:
                        # cheap second best without full sort
                        tmp = vals.copy()
                        tmp[int(np.argmax(tmp))] = -np.inf
                        v2 = float(tmp.max())
                    else:
                        v2 = v1 - 1.0
                    bid_inc = float((v1 - v2) + eps)
                    bid_price = float(price[j1] + bid_inc)
                    if bid_price > float(best_bid[j1]):
                        best_bid[j1] = bid_price
                        bidder[j1] = int(i)
                        inc[j1] = bid_inc

                any_bid = False
                for j in range(m_local):
                    i = int(bidder[j])
                    if i < 0:
                        continue
                    any_bid = True
                    prev = int(owner[j])
                    owner[j] = i
                    price[j] = float(price[j] + inc[j])
                    assign[i] = j
                    if prev >= 0:
                        assign[prev] = -1
                if not any_bid:
                    break

            # Fill remaining greedily.
            if np.any(assign < 0):
                free_slots = set(int(x) for x in np.flatnonzero(owner < 0))
                for i in np.flatnonzero(assign < 0):
                    ci = cand[i]
                    best_j = None
                    best_c = float("inf")
                    for j in ci:
                        j = int(j)
                        if j in free_slots and float(cost_np[i, j]) < best_c:
                            best_c = float(cost_np[i, j])
                            best_j = j
                    if best_j is None and free_slots:
                        best_j = next(iter(free_slots))
                    if best_j is not None:
                        assign[int(i)] = int(best_j)
                        free_slots.discard(int(best_j))
            return assign.astype(np.int32)

        def build_cost_matrix(
            slots_t: torch.Tensor,
            cur_t: torch.Tensor,
            tgt_t: torch.Tensor,
            sizes_t: torch.Tensor,
        ) -> np.ndarray:
            n = int(slots_t.shape[0])
            if n <= 1:
                return np.zeros((n, n), dtype=np.float32)

            # Neighbor list per slot (geometric kNN) for overlap-compatibility penalty.
            k_nn = 12
            slots_np = slots_t.detach().cpu().numpy().astype(np.float32, copy=False)
            dxy = slots_np[:, None, :] - slots_np[None, :, :]
            dist2 = (dxy[:, :, 0] ** 2 + dxy[:, :, 1] ** 2).astype(np.float32)
            np.fill_diagonal(dist2, np.inf)
            neigh_idx = np.argpartition(dist2, kth=min(k_nn, n - 1) - 1, axis=1)[:, : min(k_nn, n - 1)]
            neigh_t = torch.from_numpy(neigh_idx.astype(np.int64, copy=False)).to(device=device)

            cost = torch.empty((n, n), device=device, dtype=torch.float32)
            B = 32
            for j0 in range(0, n, B):
                if self._time_left(t0) < 8:
                    break
                j1 = min(n, j0 + B)
                sb = slots_t[j0:j1]  # (B,2)

                dt = (tgt_t[:, None, :] - sb[None, :, :]).pow(2).sum(dim=2) / float(scale * scale)
                dc = (cur_t[:, None, :] - sb[None, :, :]).pow(2).sum(dim=2) / float(scale * scale)

                nb = neigh_t[j0:j1]
                npos = slots_t[nb]
                nsiz = sizes_t[nb]
                dx = (sb[:, None, 0] - npos[:, :, 0]).abs().view(1, j1 - j0, -1)
                dy = (sb[:, None, 1] - npos[:, :, 1]).abs().view(1, j1 - j0, -1)
                wi = sizes_t[:, 0].view(n, 1, 1)
                hi = sizes_t[:, 1].view(n, 1, 1)
                wj = nsiz[:, :, 0].view(1, j1 - j0, -1)
                hj = nsiz[:, :, 1].view(1, j1 - j0, -1)
                ox = (0.5 * (wi + wj) - dx).clamp(min=0.0)
                oy = (0.5 * (hi + hj) - dy).clamp(min=0.0)
                ov = (ox * oy).sum(dim=2) / float(mean_area)

                cost[:, j0:j1] = w_t * dt + w_c * dc + w_o * ov

            return cost.detach().cpu().numpy().astype(np.float32, copy=False)

        hard_new = hard.copy().astype(np.float32)

        # For very large macro sets, restrict permutations within clusters to preserve packability.
        if m > 560:
            k = 10 if m <= 720 else 12
            labels = self._kmeans_labels(slots, k=k, iters=8)
            for cid in range(int(labels.max()) + 1):
                idxs = np.flatnonzero(labels == cid).astype(np.int64)
                n = int(idxs.size)
                if n < 28:
                    continue
                if self._time_left(t0) < 10:
                    break

                # Further restrict permutations by size buckets to preserve local packability.
                areas = data.hard_areas[movable_idx[idxs]].astype(np.float32, copy=False)
                if areas.size >= 12:
                    q1, q2 = np.quantile(areas, [0.33, 0.66]).astype(np.float32)
                else:
                    q1, q2 = float(np.median(areas)), float(np.median(areas))
                buckets = [
                    idxs[areas <= q1],
                    idxs[(areas > q1) & (areas <= q2)],
                    idxs[areas > q2],
                ]
                for b in buckets:
                    nb = int(b.size)
                    if nb < 20:
                        continue
                    if self._time_left(t0) < 9:
                        break
                    s_c = slots_t_full[b]
                    c_c = cur_t_full[b]
                    t_c = tgt_t_full[b]
                    z_c = sizes_t_full[b]
                    cost_np = build_cost_matrix(s_c, c_c, t_c, z_c)
                    if cost_np.shape[0] != nb:
                        continue
                    assign = auction_assign(cost_np)
                    if assign.shape[0] != nb or np.any(assign < 0):
                        continue
                    hard_new[movable_idx[b]] = slots[b][assign]
        else:
            cost_np = build_cost_matrix(slots_t_full, cur_t_full, tgt_t_full, sizes_t_full)
            assign = auction_assign(cost_np)
            if assign.shape[0] != m or np.any(assign < 0):
                return None
            hard_new[movable_idx] = slots[assign]

        hard_new = self._fast_legalize_hard(hard_new, data, sweeps=4)
        if not self._placement_is_valid(hard_new, soft, benchmark, data):
            if PROFILE:
                import sys

                print(f"[PROFILE] {benchmark.name}: auction_permute_invalid", file=sys.stderr)
            return None

        soft_new = soft.copy().astype(np.float32)
        if data.num_soft > 0 and self._time_left(t0) >= 12:
            soft_new = self._relax_soft(hard_new, soft_new, data, sweeps=1, damping=0.90)
        if PROFILE:
            import sys

            print(f"[PROFILE] {benchmark.name}: auction_permute_done", file=sys.stderr)
        return hard_new.astype(np.float32), soft_new.astype(np.float32), "auction-permute", self._cheap_score(hard_new, soft_new, data)

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

        # Exact proxy calls can be extremely slow on large designs; keep this stage cheap-only
        # unless we have a long budget and a moderate macro count.
        exact_enabled = (
            plc is not None
            and TIME_BUDGET >= 300
            and data.num_hard <= 520
            and self._time_left(t0) >= 90
        )
        if not exact_enabled:
            exact_elites = 0

        rng = np.random.default_rng((SEED ^ ((self._stable_seed(benchmark.name) << 1) & 0xFFFFFFFF)) & 0xFFFFFFFF)
        mode_scale = 0.08 * max(data.cell_w, data.cell_h)
        dim = len(modes)
        mu = np.zeros(dim, dtype=np.float32)
        sigma = np.full(dim, mode_scale, dtype=np.float32)

        base_cheap_comps = self._cheap_components(base_hard, base_soft, data)
        base_exact_comps = (
            self._proxy_components_if_valid(base_hard, base_soft, benchmark, plc, data) if exact_elites > 0 else None
        )
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

        if exact_elites > 0:
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

            gd_hard, gd_soft, gd_alpha, gd_score = self._latent_gradient_refine(
                base_hard,
                base_soft,
                modes,
                best_alpha,
                benchmark,
                plc,
                data,
                t0,
            )
            if gd_hard is not None and gd_score + 1e-6 < best_exact:
                best_hard = gd_hard
                best_soft = gd_soft
                best_alpha = gd_alpha
                best_exact = gd_score

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

    def _latent_gradient_refine(
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
        # Optimize alpha in the latent basis directly with GPU autograd.
        if plc is None or self._time_left(t0) < 12 or len(modes) == 0:
            return None, None, start_alpha.copy(), float("inf")

        device = getattr(self, "_device", torch.device("cpu"))
        if device.type != "cuda" and data.num_hard > 420:
            return None, None, start_alpha.copy(), float("inf")

        if data.num_hard <= 220:
            batch = 6 if device.type == "cuda" else 2
            steps = 60
            lr = 0.12
        elif data.num_hard <= 450:
            batch = 5 if device.type == "cuda" else 2
            steps = 55
            lr = 0.10
        else:
            batch = 4 if device.type == "cuda" else 2
            steps = 45
            lr = 0.085

        unit = float(max(data.cell_w, data.cell_h))
        modes_t = torch.from_numpy(np.stack(list(modes), axis=0)).to(device=device, dtype=torch.float32)
        base_hard_t = torch.from_numpy(base_hard.astype(np.float32)).to(device=device)
        base_soft_t = (
            torch.from_numpy(base_soft.astype(np.float32)).to(device=device) if data.num_soft > 0 else None
        )
        port_t = data.port_t.to(device=device, dtype=torch.float32)
        movable = torch.from_numpy(data.movable_hard.astype(np.float32)).to(device=device).view(1, -1, 1)

        start = torch.zeros((batch, len(modes)), device=device, dtype=torch.float32)
        start[0, : len(start_alpha)] = torch.from_numpy(start_alpha.astype(np.float32)).to(device=device)
        if batch >= 2:
            start[1].zero_()
        if batch > 2:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self._stable_seed(benchmark.name)) & 0xFFFFFFFF)
            start[2:] = torch.randn((batch - 2, len(modes)), generator=gen, device=device) * (0.10 * unit)

        alpha = start.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([alpha], lr=lr)

        hard_sizes = data.hard_sizes_t.to(device=device, dtype=torch.float32)
        safe_nnp = data.grad_safe_nnp_t.to(device=device)
        nnmask = data.grad_nnmask_t.to(device=device)

        lo_x = torch.from_numpy(data.sizes_hard[:, 0] * 0.5).to(device=device, dtype=torch.float32).view(1, -1)
        hi_x = data.canvas_w - lo_x
        lo_y = torch.from_numpy(data.sizes_hard[:, 1] * 0.5).to(device=device, dtype=torch.float32).view(1, -1)
        hi_y = data.canvas_h - lo_y

        best_loss = float("inf")
        best_alpha_t = alpha.detach()[0].clone()

        for step in range(steps):
            if device.type == "cuda" and step % 12 == 0:
                torch.cuda.synchronize()
            if step % 12 == 0 and self._time_left(t0) < 10:
                break
            optimizer.zero_grad(set_to_none=True)

            delta = torch.einsum("bm,mnd->bnd", alpha, modes_t)
            hard_b = base_hard_t.unsqueeze(0) + delta * movable
            hard_b = torch.stack(
                [
                    hard_b[:, :, 0].clamp(lo_x, hi_x),
                    hard_b[:, :, 1].clamp(lo_y, hi_y),
                ],
                dim=2,
            )

            if data.num_soft > 0 and base_soft_t is not None:
                soft_b = base_soft_t.unsqueeze(0).expand(hard_b.shape[0], -1, -1)
                all_pos = torch.cat(
                    [hard_b, soft_b, port_t.unsqueeze(0).expand(hard_b.shape[0], -1, -1)], dim=1
                )
            else:
                all_pos = torch.cat([hard_b, port_t.unsqueeze(0).expand(hard_b.shape[0], -1, -1)], dim=1)

            xs = all_pos[:, safe_nnp, 0]
            ys = all_pos[:, safe_nnp, 1]
            neg_inf = torch.full_like(xs, -1e9)
            xs_p = torch.where(nnmask, xs, neg_inf)
            xs_n = torch.where(nnmask, -xs, neg_inf)
            ys_p = torch.where(nnmask, ys, neg_inf)
            ys_n = torch.where(nnmask, -ys, neg_inf)
            wl = (
                torch.logsumexp(WL_ALPHA * xs_p, dim=2)
                + torch.logsumexp(WL_ALPHA * xs_n, dim=2)
                + torch.logsumexp(WL_ALPHA * ys_p, dim=2)
                + torch.logsumexp(WL_ALPHA * ys_n, dim=2)
            ).sum(dim=1) / (WL_ALPHA * data.grad_hpwl_norm)

            hard_density = self._density_map_batched(hard_b, hard_sizes, data, device=device)
            overflow = (hard_density - BASE_DENSITY_TARGET).clamp(min=0.0)
            density_cost = overflow.pow(2).mean(dim=(1, 2)) + 0.15 * overflow.amax(dim=(1, 2))

            cong_cost = wl.new_zeros((hard_b.shape[0],))
            if step >= int(steps * 0.65):
                cong_cost = torch.stack(
                    [
                        self._abu_logsumexp(
                            self._rudy_congestion_map(all_pos[b], data, device=device, alpha=CONG_ALPHA),
                            frac=0.05,
                        )
                        for b in range(hard_b.shape[0])
                    ],
                    dim=0,
                )

            alpha_reg = (alpha.pow(2).mean(dim=1) / max(unit * unit, 1e-6)).clamp(min=0.0)
            loss = wl + 0.70 * density_cost + 0.40 * cong_cost + 0.03 * alpha_reg
            loss.mean().backward()
            optimizer.step()

            with torch.no_grad():
                max_step = 0.40 * unit
                alpha.clamp_(min=-max_step, max=max_step)

            cur_best = float(loss.min().detach())
            if cur_best + 1e-6 < best_loss:
                best_loss = cur_best
                best_alpha_t = alpha.detach()[int(loss.argmin())].clone()

        best_alpha_np = best_alpha_t.detach().cpu().numpy().astype(np.float32)
        cand_hard = self._apply_latent_alpha(base_hard, modes, best_alpha_np, data)
        cand_score = self._proxy_cost_if_valid(cand_hard, base_soft, benchmark, plc, data)
        if math.isfinite(cand_score):
            return (
                cand_hard.astype(np.float32),
                base_soft.copy().astype(np.float32),
                best_alpha_np,
                float(cand_score),
            )
        return None, None, start_alpha.copy(), float("inf")

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
                        cand_hard = self._fast_legalize_hard(cand_hard, data, sweeps=3)
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
        cand = self._fast_legalize_hard(cand, data, sweeps=5 if data.num_hard <= 450 else 4)
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
            noisy = self._fast_legalize_hard(noisy, data, sweeps=2)
            total += self._surrogate_score_from_components(self._cheap_components(noisy, soft, data), calib)
        return total / float(jitter_trials + 1)

    def _fast_legalize_hard(self, hard: np.ndarray, data: PreparedData, sweeps: int) -> np.ndarray:
        # Fast, robust hard-macro legalization used across all stages.
        # Uses spatial hashing to avoid O(N^2) nested Python loops.
        pos = hard.copy().astype(np.float32)
        self._clamp_hard(pos, data)

        # Hash-based resolution (cheap) + greedy reinsertion (targeted) for stubborn overlaps.
        if self._hash_has_any_overlap(pos, data):
            pos = self._hash_resolve_hard(pos, data, sweeps=max(1, int(sweeps)))

            # For larger designs, avoid O(N^2) cleanup; use greedy reinsertion on a small subset.
            if data.num_hard > 320 and self._hash_has_any_overlap(pos, data):
                pos = self._greedy_reinsert_overlaps(
                    pos,
                    data,
                    max_movers=80 if data.num_hard <= 420 else 120,
                    max_radius=24 if data.num_hard <= 420 else 18,
                )
                if self._hash_has_any_overlap(pos, data):
                    pos = self._greedy_reinsert_overlaps(
                        pos,
                        data,
                        max_movers=200 if data.num_hard <= 520 else 240,
                        max_radius=60 if data.num_hard <= 520 else 48,
                    )

            # Small designs can afford a few rounds of exact cleanup.
            if data.num_hard <= 320 and self._hash_has_any_overlap(pos, data):
                pos = self._tiny_fix_hard(pos, data, rounds=80)

            # Final light hashing pass.
            if self._hash_has_any_overlap(pos, data):
                pos = self._hash_resolve_hard(pos, data, sweeps=3)

            # Last resort: grid packing (always legal, but can increase displacement).
            if self._hash_has_any_overlap(pos, data):
                pos = self._grid_pack_legalize_hard(pos, data)

        # Medium designs can still afford a final exact cleanup if needed.
        if data.num_hard <= 420:
            total = self._exact_hard_overlap_area(pos.astype(np.float64), data)
            if total > 1e-10:
                pos = self._tiny_fix_hard(pos, data, rounds=160)
                pos = self._legalize_hard(pos, data)

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

    def _grid_pack_legalize_hard(self, hard: np.ndarray, data: PreparedData) -> np.ndarray:
        # Coarse grid-based packer: guarantees no hard-macro overlaps by construction.
        # Used as a last-resort legalizer on dense/large benchmarks.
        pos = hard.copy().astype(np.float32)
        self._clamp_hard(pos, data)
        gap = float(self._legal_gap(data))

        pitch_x = float(max(data.cell_w, 1e-3))
        pitch_y = float(max(data.cell_h, 1e-3))
        nx = int(max(1, math.floor(float(data.canvas_w) / pitch_x)))
        ny = int(max(1, math.floor(float(data.canvas_h) / pitch_y)))
        occ = np.zeros((nx, ny), dtype=np.bool_)

        def mark_rect(cx: float, cy: float, w: float, h: float) -> None:
            w_eff = float(w + 2.0 * gap)
            h_eff = float(h + 2.0 * gap)
            wx = int(max(1, math.ceil(w_eff / pitch_x)))
            hy = int(max(1, math.ceil(h_eff / pitch_y)))
            llx = int(round(cx / pitch_x - 0.5 * wx))
            lly = int(round(cy / pitch_y - 0.5 * hy))
            llx = int(np.clip(llx, 0, max(0, nx - wx)))
            lly = int(np.clip(lly, 0, max(0, ny - hy)))
            occ[llx : llx + wx, lly : lly + hy] = True

        # Pre-occupy fixed macros (if any).
        for i in range(data.num_hard):
            if not data.movable_hard[i]:
                mark_rect(float(pos[i, 0]), float(pos[i, 1]), float(data.sizes_hard[i, 0]), float(data.sizes_hard[i, 1]))

        order = np.argsort(-(data.hard_areas + 0.01 * data.hard_degree))
        for idx in order:
            idx = int(idx)
            if not data.movable_hard[idx]:
                continue

            w = float(data.sizes_hard[idx, 0])
            h = float(data.sizes_hard[idx, 1])
            w_eff = float(w + 2.0 * gap)
            h_eff = float(h + 2.0 * gap)
            wx = int(max(1, math.ceil(w_eff / pitch_x)))
            hy = int(max(1, math.ceil(h_eff / pitch_y)))
            if wx > nx or hy > ny:
                continue

            cx0 = float(pos[idx, 0])
            cy0 = float(pos[idx, 1])
            llx0 = int(round(cx0 / pitch_x - 0.5 * wx))
            lly0 = int(round(cy0 / pitch_y - 0.5 * hy))
            llx0 = int(np.clip(llx0, 0, nx - wx))
            lly0 = int(np.clip(lly0, 0, ny - hy))

            placed = False
            best_ll = (llx0, lly0)
            if not occ[llx0 : llx0 + wx, lly0 : lly0 + hy].any():
                placed = True
            else:
                max_r = int(min(max(nx, ny), 160))
                for r in range(1, max_r + 1):
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            if abs(dx) != r and abs(dy) != r:
                                continue
                            llx = llx0 + dx
                            lly = lly0 + dy
                            if llx < 0 or lly < 0 or llx + wx > nx or lly + hy > ny:
                                continue
                            if occ[llx : llx + wx, lly : lly + hy].any():
                                continue
                            best_ll = (llx, lly)
                            placed = True
                            break
                        if placed:
                            break
                    if placed:
                        break

            if placed:
                llx, lly = best_ll
                occ[llx : llx + wx, lly : lly + hy] = True
                pos[idx, 0] = float((llx + 0.5 * wx) * pitch_x)
                pos[idx, 1] = float((lly + 0.5 * hy) * pitch_y)

        self._clamp_hard(pos, data)
        return pos.astype(np.float32)

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
        gap = float(self._legal_gap(data))
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
                    # Touching is legal; only act on *strict* overlaps, then add a tiny gap
                    # to avoid float-epsilon sign flips across overlap computations.
                    if lix >= ujx or uix <= ljx or liy >= ujy or uiy <= ljy:
                        continue
                    moved = True
                    ox = (min(uix, ujx) - max(lix, ljx)) + gap
                    oy = (min(uiy, ujy) - max(liy, ljy)) + gap
                    if ox < oy:
                        push = (ox + 1e-6) * 0.5
                        if not (data.movable_hard[i] or data.movable_hard[j]):
                            continue
                        if data.movable_hard[i] and data.movable_hard[j]:
                            di = push
                            dj = push
                        elif data.movable_hard[i]:
                            di = 2.0 * push
                            dj = 0.0
                        else:
                            di = 0.0
                            dj = 2.0 * push
                        if p[i, 0] < p[j, 0]:
                            p[i, 0] -= di
                            p[j, 0] += dj
                        else:
                            p[i, 0] += di
                            p[j, 0] -= dj
                    else:
                        push = (oy + 1e-6) * 0.5
                        if not (data.movable_hard[i] or data.movable_hard[j]):
                            continue
                        if data.movable_hard[i] and data.movable_hard[j]:
                            di = push
                            dj = push
                        elif data.movable_hard[i]:
                            di = 2.0 * push
                            dj = 0.0
                        else:
                            di = 0.0
                            dj = 2.0 * push
                        if p[i, 1] < p[j, 1]:
                            p[i, 1] -= di
                            p[j, 1] += dj
                        else:
                            p[i, 1] += di
                            p[j, 1] -= dj
            if not moved:
                break
        self._clamp_hard(p, data)
        return p.astype(np.float32)

    def _legalize_hard(self, hard: np.ndarray, data: PreparedData) -> np.ndarray:
        pos = hard.copy().astype(np.float32)
        self._clamp_hard(pos, data)
        gap = float(self._legal_gap(data))

        for _ in range(18):
            moved = False
            for i in range(data.num_hard):
                for j in range(i + 1, data.num_hard):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    ox = (
                        (data.sizes_hard[i, 0] + data.sizes_hard[j, 0]) * 0.5
                        + gap
                        - abs(dx)
                    )
                    oy = (
                        (data.sizes_hard[i, 1] + data.sizes_hard[j, 1]) * 0.5
                        + gap
                        - abs(dy)
                    )
                    if ox <= 0 or oy <= 0:
                        continue
                    if ox < oy:
                        push = (ox + gap) * 0.5
                        shift_i = -push if dx >= 0 else push
                        shift_j = -shift_i
                        if data.movable_hard[i]:
                            pos[i, 0] += shift_i
                        if data.movable_hard[j]:
                            pos[j, 0] += shift_j
                    else:
                        push = (oy + gap) * 0.5
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
        gap = float(self._legal_gap(data))
        for i in range(data.num_hard):
            dx = np.abs(pos[i, 0] - pos[:, 0])
            dy = np.abs(pos[i, 1] - pos[:, 1])
            ox = np.maximum(
                0.0,
                (data.sizes_hard[i, 0] + data.sizes_hard[:, 0]) * 0.5 + gap - dx,
            )
            oy = np.maximum(
                0.0,
                (data.sizes_hard[i, 1] + data.sizes_hard[:, 1]) * 0.5 + gap - dy,
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
        max_radius: int = 41,
    ) -> np.ndarray:
        best = pos[idx].copy()
        best_score = float("inf")
        step = max(data.cell_w, data.cell_h, float(max(data.sizes_hard[idx]) * 0.25))

        candidates: List[np.ndarray] = [target_xy.astype(np.float32), pos[idx].copy()]
        candidates.append(data.port_pull[idx].astype(np.float32))

        for radius in range(1, int(max_radius) + 1):
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
        gap = float(self._legal_gap(data))
        dx = np.abs(cand[0] - pos[:, 0])
        dy = np.abs(cand[1] - pos[:, 1])
        ox = (data.sizes_hard[idx, 0] + data.sizes_hard[:, 0]) * 0.5 + gap - dx
        oy = (data.sizes_hard[idx, 1] + data.sizes_hard[:, 1]) * 0.5 + gap - dy
        overlap = (ox > 0.0) & (oy > 0.0)
        overlap[idx] = False
        return bool(overlap.any())

    def _local_legalize_indices(
        self,
        pos: np.ndarray,
        indices: Sequence[int],
        data: PreparedData,
        passes: int = 2,
        max_radius: int = 18,
    ) -> np.ndarray:
        # Fast local legalization for proposals that only touch a small number of macros.
        # Uses spiral-search nearest legal points instead of full hashing/tiny-fix passes.
        out = pos.astype(np.float32, copy=True)
        self._clamp_hard(out, data)
        idxs = [int(i) for i in indices]
        for _ in range(int(max(1, passes))):
            changed = False
            for idx in idxs:
                if idx < 0 or idx >= data.num_hard:
                    continue
                if not data.movable_hard[idx]:
                    continue
                cand = out[idx].copy().astype(np.float32)
                if not self._hard_overlaps_any(idx, cand, out, data):
                    continue
                out[idx] = self._nearest_legal_point(idx, cand, out, data, max_radius=max_radius)
                changed = True
            if not changed:
                break
        return out.astype(np.float32)

    def _greedy_reinsert_subset(
        self,
        hard: np.ndarray,
        subset: Sequence[int],
        targets: np.ndarray,
        data: PreparedData,
        rng: np.random.Generator,
        max_radius: int = 26,
        restarts: int = 2,
    ) -> Optional[np.ndarray]:
        # Greedy facility-location style reinsertion: move a subset of macros toward per-macro
        # target points, projecting each macro to the nearest legal position via spiral search.
        if hard.size == 0:
            return None
        idxs = [int(i) for i in subset if 0 <= int(i) < data.num_hard and data.movable_hard[int(i)]]
        if len(idxs) < 4:
            return None
        tgt = np.asarray(targets, dtype=np.float32)
        if tgt.shape != (data.num_hard, 2):
            return None

        best = None
        best_disp = float("inf")
        for _ in range(int(max(1, restarts))):
            out = hard.astype(np.float32, copy=True)
            self._clamp_hard(out, data)
            order = np.array(idxs, dtype=np.int64)
            rng.shuffle(order)
            for idx in order.tolist():
                desired = tgt[int(idx)].astype(np.float32, copy=True)
                desired[0] = float(np.clip(desired[0], data.sizes_hard[idx, 0] * 0.5, data.canvas_w - data.sizes_hard[idx, 0] * 0.5))
                desired[1] = float(np.clip(desired[1], data.sizes_hard[idx, 1] * 0.5, data.canvas_h - data.sizes_hard[idx, 1] * 0.5))
                out[idx] = desired
                if self._hard_overlaps_any(idx, out[idx], out, data):
                    out[idx] = self._nearest_legal_point(idx, out[idx], out, data, max_radius=max_radius)
            # Verify legality for the subset (full scan avoided).
            ok = True
            for idx in idxs:
                if self._hard_overlaps_any(int(idx), out[int(idx)], out, data):
                    ok = False
                    break
            if not ok:
                continue
            disp = float(np.square(out[idxs] - hard[idxs]).sum())
            if disp < best_disp:
                best_disp = disp
                best = out

        return None if best is None else best.astype(np.float32)

    def _diagnose_poor_macros_periphery(self, hard: np.ndarray, data: PreparedData, k: int = 8) -> np.ndarray:
        # Lightweight IncreMacro-style diagnosis: a macro is "regular" if it is close to a boundary
        # or has nearby neighbors in all four cardinal directions. Others are treated as blockage seeds.
        n = int(data.num_hard)
        if n <= 0:
            return np.zeros(0, dtype=bool)
        pos = np.asarray(hard, dtype=np.float32)
        poor = np.zeros(n, dtype=bool)
        boundary_thr = 1.5 * max(float(data.cell_w), float(data.cell_h))
        for idx in range(n):
            if not data.movable_hard[idx]:
                continue
            x, y = float(pos[idx, 0]), float(pos[idx, 1])
            w, h = float(data.sizes_hard[idx, 0]), float(data.sizes_hard[idx, 1])
            boundary_dist = min(x - 0.5 * w, y - 0.5 * h, data.canvas_w - x - 0.5 * w, data.canvas_h - y - 0.5 * h)
            if boundary_dist <= boundary_thr:
                continue
            dx = pos[:, 0] - x
            dy = pos[:, 1] - y
            dist2 = dx * dx + dy * dy
            dist2[idx] = np.inf
            nn = np.argsort(dist2)[: max(4, min(int(k), max(1, n - 1)))]
            north = False
            south = False
            east = False
            west = False
            for j in nn.tolist():
                sep_x = 0.35 * float(data.sizes_hard[idx, 0] + data.sizes_hard[j, 0])
                sep_y = 0.35 * float(data.sizes_hard[idx, 1] + data.sizes_hard[j, 1])
                if dy[j] >= sep_y:
                    north = True
                if dy[j] <= -sep_y:
                    south = True
                if dx[j] >= sep_x:
                    east = True
                if dx[j] <= -sep_x:
                    west = True
            poor[idx] = not (north and south and east and west)
        return poor

    def _ordered_axis_targets_periphery(
        self,
        desired: np.ndarray,
        extents: np.ndarray,
        lo: float,
        hi: float,
        gap: float,
    ) -> np.ndarray:
        n = int(len(desired))
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        order = np.argsort(desired.astype(np.float32))
        out = desired.astype(np.float32, copy=True)[order]
        ext = extents.astype(np.float32, copy=False)[order]
        lo_arr = lo + 0.5 * ext
        hi_arr = hi - 0.5 * ext
        out = np.clip(out, lo_arr, hi_arr)
        for i in range(1, n):
            min_sep = 0.5 * float(ext[i - 1] + ext[i]) + float(gap)
            out[i] = max(out[i], out[i - 1] + min_sep)
        for i in range(n - 2, -1, -1):
            min_sep = 0.5 * float(ext[i + 1] + ext[i]) + float(gap)
            out[i] = min(out[i], out[i + 1] - min_sep)
        out = np.clip(out, lo_arr, hi_arr)
        # If the chain is still infeasible, fall back to evenly spaced slots in-order.
        feasible = True
        for i in range(1, n):
            need = 0.5 * float(ext[i - 1] + ext[i]) + float(gap)
            if out[i] < out[i - 1] + need - 1e-4:
                feasible = False
                break
        if not feasible:
            span_lo = float(np.min(lo_arr))
            span_hi = float(np.max(hi_arr))
            out = np.linspace(span_lo, span_hi, num=n, dtype=np.float32)
            out = np.clip(out, lo_arr, hi_arr)
        final = np.zeros(n, dtype=np.float32)
        final[order] = out.astype(np.float32)
        return final

    def _periphery_evacuate(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
        strength: float,
        hotspot_weight: float,
    ) -> Optional[np.ndarray]:
        if data.num_hard < 80:
            return None
        poor = self._diagnose_poor_macros_periphery(hard, data, k=8 if data.num_hard <= 320 else 10)
        if int(poor.sum()) < 4:
            return None
        all_pos = self._all_pos_np(hard, soft, data)
        net_centers = self._net_box_centers_np(all_pos, data)
        hotspot = np.array([0.5 * data.canvas_w, 0.5 * data.canvas_h], dtype=np.float32)
        cong_grid = self._fast_cong_grid_np(all_pos, data)
        if cong_grid is not None and cong_grid.size > 0:
            rr, cc = np.unravel_index(int(np.argmax(cong_grid)), cong_grid.shape)
            hotspot = np.array([(cc + 0.5) * data.cell_w, (rr + 0.5) * data.cell_h], dtype=np.float32)

        pos = np.asarray(hard, dtype=np.float32)
        boundary_dist = np.minimum.reduce(
            [
                pos[:, 0] - 0.5 * data.sizes_hard[:, 0],
                data.canvas_w - pos[:, 0] - 0.5 * data.sizes_hard[:, 0],
                pos[:, 1] - 0.5 * data.sizes_hard[:, 1],
                data.canvas_h - pos[:, 1] - 0.5 * data.sizes_hard[:, 1],
            ]
        )
        center_scale = float(max(data.canvas_w, data.canvas_h, 1e-6))
        hot_dist = np.sqrt(np.square(pos[:, 0] - hotspot[0]) + np.square(pos[:, 1] - hotspot[1]))
        hot_term = 1.0 - np.clip(hot_dist / center_scale, 0.0, 1.0)
        deg = np.asarray(data.hard_degree, dtype=np.float32)
        if deg.size > 0:
            deg = deg / max(float(np.max(deg)), 1.0)
        priority = (
            poor.astype(np.float32)
            * (0.65 * np.clip(boundary_dist / center_scale, 0.0, 1.0) + float(hotspot_weight) * hot_term + 0.25 * deg)
        )
        movable_idx = np.flatnonzero(data.movable_hard & (priority > 0.0))
        if movable_idx.size < 4:
            return None
        subset_cap = min(36, max(8, data.num_hard // 7))
        subset = movable_idx[np.argsort(priority[movable_idx])[::-1][:subset_cap]]
        if subset.size < 4:
            return None

        gap = float(self._legal_gap(data)) + 0.35 * max(float(data.cell_w), float(data.cell_h))
        inset = 0.75 * max(float(data.cell_w), float(data.cell_h))
        targets = pos.copy().astype(np.float32)
        side_groups = {"left": [], "right": [], "bottom": [], "top": []}
        desired_axis = {"left": [], "right": [], "bottom": [], "top": []}

        for idx in subset.tolist():
            net_t = self._macro_net_target(int(idx), net_centers, data).astype(np.float32)
            vec = pos[idx] - hotspot
            if abs(float(vec[0])) >= abs(float(vec[1])):
                side = "right" if float(vec[0]) >= 0.0 else "left"
                axis = float((1.0 - strength) * pos[idx, 1] + strength * net_t[1])
            else:
                side = "top" if float(vec[1]) >= 0.0 else "bottom"
                axis = float((1.0 - strength) * pos[idx, 0] + strength * net_t[0])
            if max(abs(float(vec[0])), abs(float(vec[1]))) < 0.08 * center_scale:
                dleft = float(net_t[0])
                dright = float(data.canvas_w - net_t[0])
                dbottom = float(net_t[1])
                dtop = float(data.canvas_h - net_t[1])
                choice = min(
                    [("left", dleft), ("right", dright), ("bottom", dbottom), ("top", dtop)],
                    key=lambda x: x[1],
                )[0]
                side = choice
                axis = float(net_t[1] if side in ("left", "right") else net_t[0])
            side_groups[side].append(int(idx))
            desired_axis[side].append(axis)

        for side in ["left", "right", "bottom", "top"]:
            idxs = side_groups[side]
            if not idxs:
                continue
            idx_arr = np.asarray(idxs, dtype=np.int64)
            if side in ("left", "right"):
                ext = data.sizes_hard[idx_arr, 1]
                axis = self._ordered_axis_targets_periphery(
                    np.asarray(desired_axis[side], dtype=np.float32),
                    ext.astype(np.float32),
                    0.0,
                    float(data.canvas_h),
                    gap=float(gap),
                )
                x_fixed = (
                    0.5 * data.sizes_hard[idx_arr, 0] + inset
                    if side == "left"
                    else data.canvas_w - 0.5 * data.sizes_hard[idx_arr, 0] - inset
                )
                for local_i, idx in enumerate(idx_arr.tolist()):
                    targets[idx, 0] = float(x_fixed[local_i])
                    targets[idx, 1] = float(axis[local_i])
            else:
                ext = data.sizes_hard[idx_arr, 0]
                axis = self._ordered_axis_targets_periphery(
                    np.asarray(desired_axis[side], dtype=np.float32),
                    ext.astype(np.float32),
                    0.0,
                    float(data.canvas_w),
                    gap=float(gap),
                )
                y_fixed = (
                    0.5 * data.sizes_hard[idx_arr, 1] + inset
                    if side == "bottom"
                    else data.canvas_h - 0.5 * data.sizes_hard[idx_arr, 1] - inset
                )
                for local_i, idx in enumerate(idx_arr.tolist()):
                    targets[idx, 0] = float(axis[local_i])
                    targets[idx, 1] = float(y_fixed[local_i])

        rng = np.random.default_rng((SEED ^ self._stable_seed(f"periphery-{int(100*strength)}")) & 0xFFFFFFFF)
        return self._greedy_reinsert_subset(
            hard,
            subset.tolist(),
            targets.astype(np.float32),
            data,
            rng,
            max_radius=32,
            restarts=3,
        )

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

    def _capacitated_bin_shuffle(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
        t0: float,
        bin_rows: int,
        bin_cols: int,
        target: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        # Facility-location / capacitated assignment inspired global reshuffle:
        # assign macros to coarse bins with area capacities, minimizing distance to a target
        # (net centroid) while respecting capacity, then pack within bins and legalize.
        if self._time_left(t0) < 10 or data.num_hard <= 1:
            return None
        bin_rows = int(max(2, min(int(bin_rows), 16)))
        bin_cols = int(max(2, min(int(bin_cols), 16)))

        movable = data.movable_hard
        if movable is None:
            return None
        m_idx = np.flatnonzero(movable).astype(np.int64)
        if m_idx.size < 10:
            return None

        # Target positions (defaults: net-centroid targets).
        if target is None:
            all_pos = self._all_pos_np(hard, soft, data)
            net_centers = self._net_box_centers_np(all_pos, data)
            tgt = np.zeros((data.num_hard, 2), dtype=np.float32)
            for idx in range(data.num_hard):
                tgt[idx] = self._macro_net_target(int(idx), net_centers, data).astype(np.float32)
            target = tgt

        bw = float(data.canvas_w) / float(bin_cols)
        bh = float(data.canvas_h) / float(bin_rows)
        bin_area = bw * bh
        dens_target = float(getattr(self, "_density_target", BASE_DENSITY_TARGET))
        cap = np.full((bin_rows, bin_cols), dens_target * bin_area, dtype=np.float32)

        # Remove "fixed" macro area from capacity (treat fixed macros as occupying bins).
        fixed_idx = np.flatnonzero(~movable).astype(np.int64)
        if fixed_idx.size > 0:
            fx = np.clip((hard[fixed_idx, 0] / bw).astype(np.int32), 0, bin_cols - 1)
            fy = np.clip((hard[fixed_idx, 1] / bh).astype(np.int32), 0, bin_rows - 1)
            farea = (data.sizes_hard[fixed_idx, 0] * data.sizes_hard[fixed_idx, 1]).astype(np.float32)
            for r, c, a in zip(fy.tolist(), fx.tolist(), farea.tolist()):
                cap[int(r), int(c)] -= float(a)
        cap = np.maximum(cap, 0.10 * bin_area).astype(np.float32)

        # Macro areas and assignment.
        areas = (data.sizes_hard[m_idx, 0] * data.sizes_hard[m_idx, 1]).astype(np.float32)
        order = m_idx[np.argsort(-areas)]  # big-first
        assign_r = np.full(data.num_hard, -1, dtype=np.int32)
        assign_c = np.full(data.num_hard, -1, dtype=np.int32)

        # Precompute bin centers.
        bc_x = (np.arange(bin_cols, dtype=np.float32) + 0.5) * bw
        bc_y = (np.arange(bin_rows, dtype=np.float32) + 0.5) * bh
        centers = np.stack(np.meshgrid(bc_x, bc_y), axis=-1).reshape(-1, 2)  # (B,2)

        # Greedy capacitated assignment with shortlist of nearest bins to each target.
        for idx in order.tolist():
            if self._time_left(t0) < 6:
                break
            a = float(data.sizes_hard[idx, 0] * data.sizes_hard[idx, 1])
            tx, ty = target[int(idx)]
            # shortlist: nearest bins by L2 to target (k=8)
            dx = centers[:, 0] - float(tx)
            dy = centers[:, 1] - float(ty)
            dist = dx * dx + dy * dy
            k = 8 if centers.shape[0] >= 8 else int(centers.shape[0])
            cand = np.argpartition(dist, kth=k - 1)[:k]
            chosen = None
            for b in cand.tolist():
                r = int(b // bin_cols)
                c = int(b % bin_cols)
                if float(cap[r, c]) >= a:
                    chosen = (r, c)
                    break
            if chosen is None:
                # fallback: pick bin with most remaining capacity, tie-break by distance
                flat_cap = cap.reshape(-1)
                best_bins = np.argpartition(-flat_cap, kth=min(8, flat_cap.size - 1))[: min(8, flat_cap.size)]
                best = None
                best_score = float("inf")
                for b in best_bins.tolist():
                    r = int(b // bin_cols)
                    c = int(b % bin_cols)
                    if float(flat_cap[b]) < 0.8 * a:
                        continue
                    sc = float(dist[b]) / max(unit := float(max(data.cell_w, data.cell_h)), 1e-6)
                    if sc < best_score:
                        best_score = sc
                        best = (r, c)
                chosen = best if best is not None else (int(np.argmax(flat_cap) // bin_cols), int(np.argmax(flat_cap) % bin_cols))
            r, c = chosen
            assign_r[int(idx)] = int(r)
            assign_c[int(idx)] = int(c)
            cap[int(r), int(c)] -= float(a)

        if np.any(assign_r[m_idx] < 0):
            return None

        # Pack within each bin.
        out = hard.copy().astype(np.float32)
        gap = float(max(0.0, min(0.15 * float(max(data.cell_w, data.cell_h)), 0.25 * float(max(data.cell_w, data.cell_h)))))
        for r in range(bin_rows):
            for c in range(bin_cols):
                idxs = m_idx[(assign_r[m_idx] == r) & (assign_c[m_idx] == c)]
                if idxs.size == 0:
                    continue
                # Sort by height then area.
                hs = data.sizes_hard[idxs, 1].astype(np.float32)
                ars = (data.sizes_hard[idxs, 0] * data.sizes_hard[idxs, 1]).astype(np.float32)
                idxs = idxs[np.lexsort((-ars, -hs))]
                # Bin bounds.
                lx = c * bw
                ux = lx + bw
                ly = r * bh
                uy = ly + bh
                cur_x = lx + gap
                cur_y = ly + gap
                row_h = 0.0
                for idx in idxs.tolist():
                    w, h = map(float, data.sizes_hard[int(idx)])
                    if cur_x + w + gap > ux:
                        cur_x = lx + gap
                        cur_y = cur_y + row_h + gap
                        row_h = 0.0
                    if cur_y + h + gap > uy:
                        # spill: restart at top-left; let global legalizer fix.
                        cur_x = lx + gap
                        cur_y = ly + gap
                        row_h = 0.0
                    x = cur_x + 0.5 * w
                    y = cur_y + 0.5 * h
                    out[int(idx), 0] = float(np.clip(x, 0.5 * w, float(data.canvas_w) - 0.5 * w))
                    out[int(idx), 1] = float(np.clip(y, 0.5 * h, float(data.canvas_h) - 0.5 * h))
                    cur_x = cur_x + w + gap
                    row_h = max(row_h, h)

        out = self._fast_legalize_hard(out, data, sweeps=5 if data.num_hard <= 420 else 4)
        if data.num_hard <= 420 and self._time_left(t0) >= 10:
            try:
                if self._exact_hard_overlap_area(out.astype(np.float64), data) > 1e-10:
                    out = self._legalize_hard(out, data)
            except Exception:
                pass
        return out.astype(np.float32)

    def _stress_embed_project(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
        t0: float,
        iters: int = 18,
        rest_len: Optional[float] = None,
        anchor_w: float = 0.15,
    ) -> Optional[np.ndarray]:
        # Constrained graph drawing / stress majorization inspired operator:
        # use neighbor graph (spectral adjacency) with a rest-length spring model,
        # do a few Jacobi-style stress updates, then project via legalization.
        if data.num_hard < 40 or self._time_left(t0) < 10:
            return None
        neigh = getattr(data, "neighbor_lists", None)
        if neigh is None:
            return None
        unit = float(max(data.cell_w, data.cell_h))
        if rest_len is None:
            rest_len = float(2.2 * unit)
        rest_len = float(max(0.8 * unit, min(rest_len, 6.0 * unit)))

        cur = hard.copy().astype(np.float32)
        anchor = hard.copy().astype(np.float32)

        # Small set of "attractors": net-centroid targets for connected macros.
        all_pos = self._all_pos_np(cur, soft, data)
        net_centers = self._net_box_centers_np(all_pos, data)
        tgt = np.zeros((data.num_hard, 2), dtype=np.float32)
        for i in range(data.num_hard):
            tgt[i] = self._macro_net_target(int(i), net_centers, data).astype(np.float32)

        for it in range(int(iters)):
            if self._time_left(t0) < 6:
                break
            nxt = cur.copy()
            # Jacobi update: x_i <- weighted average of neighbor "desired" positions.
            for i in range(data.num_hard):
                if not data.movable_hard[i]:
                    continue
                ni = neigh[i]
                if ni.size == 0:
                    continue
                xi = cur[i]
                acc = np.zeros(2, dtype=np.float32)
                wsum = 1e-6
                for j in ni.tolist()[:12]:
                    j = int(j)
                    xj = cur[j]
                    d = xi - xj
                    dist = float(np.linalg.norm(d))
                    if dist < 1e-6:
                        continue
                    # desired position is xj + rest_len * (xi-xj)/||xi-xj||
                    des = xj + (rest_len / dist) * d.astype(np.float32)
                    w = 1.0
                    acc += float(w) * des.astype(np.float32)
                    wsum += float(w)
                # add target + anchor to keep it from drifting
                acc += 0.25 * tgt[i]
                wsum += 0.25
                acc += float(anchor_w) * anchor[i]
                wsum += float(anchor_w)
                nxt[i] = acc / wsum
            cur = nxt
            self._clamp_hard(cur, data)
            if it % 6 == 5:
                cur = self._fast_legalize_hard(cur, data, sweeps=4)

        cur = self._fast_legalize_hard(cur, data, sweeps=5 if data.num_hard <= 420 else 4)
        if data.num_hard <= 420 and self._time_left(t0) >= 10:
            try:
                if self._exact_hard_overlap_area(cur.astype(np.float64), data) > 1e-10:
                    cur = self._legalize_hard(cur, data)
            except Exception:
                pass
        return cur.astype(np.float32)

    def _count_overlaps_all(self, hard: np.ndarray, soft: np.ndarray, data: PreparedData, gap: float = 0.0) -> int:
        # Count AABB overlaps among all macros (hard+soft). This is a fast consistency check
        # to prevent rare PLC overlap_count mismatches.
        n_h = int(data.num_hard)
        n_s = int(data.num_soft)
        if n_s <= 0:
            return 0
        pos = np.vstack([hard, soft]).astype(np.float32, copy=False)
        sizes = data.sizes_all.astype(np.float32, copy=False)
        n = int(pos.shape[0])
        g = float(max(0.0, gap))
        count = 0
        for i in range(n):
            xi, yi = float(pos[i, 0]), float(pos[i, 1])
            wi, hi = float(sizes[i, 0]), float(sizes[i, 1])
            for j in range(i + 1, n):
                xj, yj = float(pos[j, 0]), float(pos[j, 1])
                wj, hj = float(sizes[j, 0]), float(sizes[j, 1])
                if abs(xi - xj) < 0.5 * (wi + wj) + g and abs(yi - yj) < 0.5 * (hi + hj) + g:
                    count += 1
        return int(count)

    def _soft_legalize_quick(
        self,
        hard: np.ndarray,
        soft: np.ndarray,
        data: PreparedData,
        t0: float,
        gap: float = 1e-3,
    ) -> np.ndarray:
        # Quick soft-macro de-overlap against hard+soft using spiral search.
        if int(data.num_soft) <= 0 or self._time_left(t0) < 6:
            return soft
        unit = float(max(data.cell_w, data.cell_h))
        gap = float(max(0.0, gap))
        n_h = int(data.num_hard)
        n_s = int(data.num_soft)
        sizes = data.sizes_all.astype(np.float32, copy=False)

        out = soft.copy().astype(np.float32)
        placed_pos = [hard[i].astype(np.float32) for i in range(n_h)]
        placed_sizes = [sizes[i].astype(np.float32) for i in range(n_h)]

        # Sort soft by area descending (harder first).
        s_idx = np.arange(n_s, dtype=np.int64)
        s_area = (sizes[n_h:, 0] * sizes[n_h:, 1]).astype(np.float32)
        s_idx = s_idx[np.argsort(-s_area)]

        # Precompute spiral offsets.
        offsets = []
        for r in range(1, 18):
            rad = float(r) * 0.7 * unit
            for k in range(16):
                ang = TWO_PI * (float(k) / 16.0)
                offsets.append((rad * math.cos(ang), rad * math.sin(ang)))

        def ok_pos(x: float, y: float, w: float, h: float) -> bool:
            for (p, sz) in zip(placed_pos, placed_sizes):
                if abs(x - float(p[0])) < 0.5 * (w + float(sz[0])) + gap and abs(y - float(p[1])) < 0.5 * (
                    h + float(sz[1])
                ) + gap:
                    return False
            return True

        for si in s_idx.tolist():
            if self._time_left(t0) < 4:
                break
            w, h = map(float, sizes[n_h + int(si)])
            x0, y0 = map(float, out[int(si)])
            # Clamp baseline.
            x0 = float(np.clip(x0, 0.5 * w, float(data.canvas_w) - 0.5 * w))
            y0 = float(np.clip(y0, 0.5 * h, float(data.canvas_h) - 0.5 * h))
            if ok_pos(x0, y0, w, h):
                x_best, y_best = x0, y0
            else:
                x_best, y_best = x0, y0
                found = False
                for dx, dy in offsets:
                    x = float(np.clip(x0 + dx, 0.5 * w, float(data.canvas_w) - 0.5 * w))
                    y = float(np.clip(y0 + dy, 0.5 * h, float(data.canvas_h) - 0.5 * h))
                    if ok_pos(x, y, w, h):
                        x_best, y_best = x, y
                        found = True
                        break
                if not found:
                    # Last resort: random-ish jitter within one cell.
                    x_best = float(np.clip(x0 + 0.25 * unit, 0.5 * w, float(data.canvas_w) - 0.5 * w))
                    y_best = float(np.clip(y0 + 0.25 * unit, 0.5 * h, float(data.canvas_h) - 0.5 * h))
            out[int(si), 0] = x_best
            out[int(si), 1] = y_best
            placed_pos.append(np.array([x_best, y_best], dtype=np.float32))
            placed_sizes.append(np.array([w, h], dtype=np.float32))

        return out.astype(np.float32)

    def _fast_cong_grid_np_sample(self, all_pos: np.ndarray, data: PreparedData, sample_idx: np.ndarray) -> Optional[np.ndarray]:
        engine = data.fast_cong_engine
        if engine is None:
            return None
        try:
            idx = sample_idx.astype(np.int64, copy=False)
            src = engine["src"][idx]
            snk = engine["snk"][idx]
            w = engine["w"][idx]
        except Exception:
            return None

        # Reuse the same congestion grid routine but with sampled routing pairs.
        hpm = float(engine["hpm"])
        vpm = float(engine["vpm"])
        halloc = float(engine["halloc"])
        valloc = float(engine["valloc"])
        smooth = int(engine["smooth"])

        rows = int(data.grid_rows)
        cols = int(data.grid_cols)
        cell_w = float(data.cell_w)
        cell_h = float(data.cell_h)
        canvas_w = float(data.canvas_w)
        canvas_h = float(data.canvas_h)

        h_cap = max(1e-6, hpm * cell_h)
        v_cap = max(1e-6, vpm * cell_w)

        h_grid = np.zeros((rows, cols), dtype=np.float32)
        v_grid = np.zeros((rows, cols), dtype=np.float32)

        for s, t, weight in zip(src, snk, w):
            x1, y1 = all_pos[int(s)]
            x2, y2 = all_pos[int(t)]
            lx, ux = (x1, x2) if x1 <= x2 else (x2, x1)
            ly, uy = (y1, y2) if y1 <= y2 else (y2, y1)
            # Clip.
            lx = float(np.clip(lx, 0.0, canvas_w))
            ux = float(np.clip(ux, 0.0, canvas_w))
            ly = float(np.clip(ly, 0.0, canvas_h))
            uy = float(np.clip(uy, 0.0, canvas_h))

            c_lo = int(max(0, min(cols - 1, lx / cell_w)))
            c_hi = int(max(0, min(cols - 1, ux / cell_w)))
            r_lo = int(max(0, min(rows - 1, ly / cell_h)))
            r_hi = int(max(0, min(rows - 1, uy / cell_h)))

            if c_lo == c_hi and r_lo == r_hi:
                h_grid[r_lo, c_lo] += float(weight) * float(halloc)
                v_grid[r_lo, c_lo] += float(weight) * float(valloc)
                continue

            for row in range(r_lo, r_hi + 1):
                y0 = row * cell_h
                y1c = y0 + cell_h
                oy = max(0.0, min(uy, y1c) - max(ly, y0))
                if oy <= 0.0:
                    continue
                for col in range(c_lo, c_hi + 1):
                    x0 = col * cell_w
                    x1c = x0 + cell_w
                    ox = max(0.0, min(ux, x1c) - max(lx, x0))
                    if ox <= 0.0:
                        continue
                    v_grid[row, col] += float(weight) * float(ox) * float(valloc)
                    h_grid[row, col] += float(weight) * float(oy) * float(halloc)

        if smooth > 0:
            # Simple box smoothing.
            pad = int(smooth)
            h_p = np.pad(h_grid, ((pad, pad), (pad, pad)), mode="edge")
            v_p = np.pad(v_grid, ((pad, pad), (pad, pad)), mode="edge")
            kernel = (2 * pad + 1) ** 2
            h_s = np.zeros_like(h_grid)
            v_s = np.zeros_like(v_grid)
            for r in range(rows):
                for c in range(cols):
                    h_s[r, c] = float(h_p[r : r + 2 * pad + 1, c : c + 2 * pad + 1].sum()) / float(kernel)
                    v_s[r, c] = float(v_p[r : r + 2 * pad + 1, c : c + 2 * pad + 1].sum()) / float(kernel)
            h_grid = h_s
            v_grid = v_s

        h_grid /= float(h_cap)
        v_grid /= float(v_cap)
        return (h_grid + v_grid).astype(np.float32)

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
        if data.fast_cong_engine is not None:
            if len(data.fast_cong_engine["src"]) <= 250_000:
                cong = self._fast_cong_np(all_pos, data)
            else:
                sample_idx = data.fast_cong_engine.get("sample_idx", None)
                if isinstance(sample_idx, np.ndarray) and sample_idx.size > 0:
                    sampled = self._fast_cong_grid_np_sample(all_pos, data, sample_idx)
                    if sampled is not None:
                        cong = self._top_k_mean(sampled, 0.05)
        return {
            "wl": float(wl),
            "den": float(topk),
            "cong": float(cong),
            "overlap": float(overlap / max(data.canvas_area, 1e-6)),
        }

    def _pareto_filter(
        self,
        items: Sequence[Tuple[np.ndarray, np.ndarray, str, float, dict]],
        max_keep: int,
    ) -> List[Tuple[np.ndarray, np.ndarray, str, float, dict]]:
        # Keep a small non-dominated set in (wl, den, cong, overlap) space.
        # Lower is better on all axes.
        keep: List[Tuple[np.ndarray, np.ndarray, str, float, dict]] = []
        for hard, soft, name, cheap, comps in items:
            v = (
                float(comps.get("wl", float("inf"))),
                float(comps.get("den", float("inf"))),
                float(comps.get("cong", float("inf"))),
                float(comps.get("overlap", float("inf"))),
            )
            if not all(_isfinite(x) for x in v):
                continue
            dominated = False
            new_keep = []
            for kh, ks, kn, kc, kcomps in keep:
                kv = (
                    float(kcomps.get("wl", float("inf"))),
                    float(kcomps.get("den", float("inf"))),
                    float(kcomps.get("cong", float("inf"))),
                    float(kcomps.get("overlap", float("inf"))),
                )
                # k dominates v?
                if (kv[0] <= v[0] and kv[1] <= v[1] and kv[2] <= v[2] and kv[3] <= v[3]) and (
                    kv[0] < v[0] or kv[1] < v[1] or kv[2] < v[2] or kv[3] < v[3]
                ):
                    dominated = True
                    break
                # v dominates k?
                if (v[0] <= kv[0] and v[1] <= kv[1] and v[2] <= kv[2] and v[3] <= kv[3]) and (
                    v[0] < kv[0] or v[1] < kv[1] or v[2] < kv[2] or v[3] < kv[3]
                ):
                    continue
                new_keep.append((kh, ks, kn, kc, kcomps))
            if dominated:
                continue
            new_keep.append((hard, soft, name, float(cheap), comps))
            keep = new_keep

        # Cap size by best surrogate score (cheap).
        keep.sort(key=lambda x: float(x[3]))
        return keep[: int(max(1, min(int(max_keep), len(keep) if keep else 1)))]

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
        calib = getattr(self, "_surrogate_calib", None)
        return self._surrogate_score_from_components(self._cheap_components(hard, soft, data), calib=calib)

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

        # For short budgets, mostly avoid internal exact evaluation:
        # the evaluator will run once anyway. Exception: small/medium designs where
        # oracle calls are cheap and selecting the best candidate matters a lot.
        legal: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        invalid_names: List[str] = []
        for hard, soft, name, cheap in candidates:
            placement = benchmark.macro_positions.clone()
            placement[: data.num_hard] = torch.from_numpy(hard).float()
            if data.num_soft > 0:
                placement[data.num_hard : data.num_macros] = torch.from_numpy(soft).float()
            ok, _ = validate_placement(placement, benchmark)
            if not ok:
                if PROFILE and len(invalid_names) < 8:
                    invalid_names.append(name)
                continue
            legal.append((hard, soft, name, cheap))

        if PROFILE:
            import sys

            print(
                f"[PROFILE] {benchmark.name}: candidates={len(candidates)} legal={len(legal)} invalid_sample={invalid_names}",
                file=sys.stderr,
            )

        pool = legal if legal else list(candidates)
        pool = sorted(pool, key=lambda item: float(item[3]))

        def _left() -> float:
            try:
                return float(self._time_left(getattr(self, "_place_t0", time.time())))
            except Exception:
                return 0.0

        def _oracle_eval(hard: np.ndarray, soft: np.ndarray, name: str) -> Optional[dict]:
            # Returns proxy components dict and updates surrogate calibration online.
            try:
                placement = benchmark.macro_positions.clone()
                placement[: data.num_hard] = torch.from_numpy(hard).float()
                if data.num_soft > 0:
                    placement[data.num_hard : data.num_macros] = torch.from_numpy(soft).float()
            except Exception:
                return None
            t_eval = time.time()
            result = compute_proxy_cost(placement, benchmark, plc)
            dt = time.time() - t_eval
            self._prof["oracle_calls"] = int(self._prof.get("oracle_calls", 0)) + 1
            self._prof["oracle_sec"] = float(self._prof.get("oracle_sec", 0.0)) + float(dt)
            if getattr(self, "_oracle_call_sec", None) is None:
                self._oracle_call_sec = float(dt)
            else:
                self._oracle_call_sec = float(0.65 * self._oracle_call_sec + 0.35 * dt)
            exact = {
                "proxy": float(result["proxy_cost"]),
                "wl": float(result["wirelength_cost"]),
                "den": float(result["density_cost"]),
                "cong": float(result["congestion_cost"]),
            }
            # Online calibration update.
            try:
                cheap_comps = self._cheap_components(hard, soft, data)
                calib = getattr(self, "_surrogate_calib", None)
                if calib is None:
                    calib = self._init_surrogate_calib(cheap_comps, exact)
                else:
                    calib = self._update_surrogate_calib(calib, cheap_comps, exact, eta=0.22)
                self._surrogate_calib = calib
            except Exception:
                pass
            if PROFILE:
                import sys

                print(f"[PROFILE] {benchmark.name}: eval {name} proxy={exact['proxy']:.4f}", file=sys.stderr)
            return exact

        if TIME_BUDGET <= 240:
            # Small designs: oracle is often fast enough; evaluate a few top candidates.
            if SMALL_ORACLE and plc is not None and data.num_hard <= 280 and len(pool) >= 2:
                top = pool[: min(6, len(pool))]
                best_cost = float("inf")
                best_pair = (top[0][0], top[0][1])
                best_name = top[0][2]
                evaluated = 0
                for hard, soft, name, _ in top:
                    est = getattr(self, "_oracle_call_sec", None)
                    if evaluated >= 3 and est is not None and self._time_left(time.time()) < float(est) + 3.0:
                        break
                    placement = benchmark.macro_positions.clone()
                    placement[: data.num_hard] = torch.from_numpy(hard).float()
                    if data.num_soft > 0:
                        placement[data.num_hard : data.num_macros] = torch.from_numpy(soft).float()
                    t_eval = time.time()
                    result = compute_proxy_cost(placement, benchmark, plc)
                    dt = time.time() - t_eval
                    self._prof["oracle_calls"] = int(self._prof.get("oracle_calls", 0)) + 1
                    self._prof["oracle_sec"] = float(self._prof.get("oracle_sec", 0.0)) + float(dt)
                    if getattr(self, "_oracle_call_sec", None) is None:
                        self._oracle_call_sec = float(dt)
                    else:
                        self._oracle_call_sec = float(0.65 * self._oracle_call_sec + 0.35 * dt)
                    cost = float(result["proxy_cost"])
                    evaluated += 1
                    if PROFILE:
                        import sys

                        print(f"[PROFILE] {benchmark.name}: eval {name} proxy={cost:.4f}", file=sys.stderr)
                    if cost + 1e-9 < best_cost:
                        best_cost = cost
                        best_pair = (hard.copy(), soft.copy())
                        best_name = name
                if PROFILE:
                    import sys

                    print(
                        f"[PROFILE] {benchmark.name}: select_best short_budget oracle evaluated={evaluated}/{len(top)} best={best_name} cost={best_cost:.4f}",
                        file=sys.stderr,
                    )
                return best_pair

            # Build Pareto front and pick a few leaders; this is our "multi-objective" selector.
            view = pool[: min(18, len(pool))]
            with_comps = []
            for hard, soft, name, cheap in view:
                try:
                    comps = self._cheap_components(hard, soft, data)
                except Exception:
                    comps = {"wl": float("inf"), "den": float("inf"), "cong": float("inf"), "overlap": float("inf")}
                with_comps.append((hard, soft, name, float(cheap), comps))
            pareto = self._pareto_filter(with_comps, max_keep=10)
            if pareto:
                pareto_sorted = sorted(pareto, key=lambda x: float(x[3]))
                best = pareto_sorted[0]
                if PROFILE:
                    import sys

                    print(
                        f"[PROFILE] {benchmark.name}: pareto_keep={len(pareto_sorted)} best={best[2]} cheap={float(best[3]):.4f}",
                        file=sys.stderr,
                    )
                # Prefer Pareto-best when we don't do oracle.
                pool = [(h, s, n, sc) for (h, s, n, sc, _) in pareto_sorted] + [p for p in pool if p[2] not in {x[2] for x in pareto_sorted}]

            # Leader oracle selection (online-calibrated surrogate):
            # evaluate a few Pareto/objective leaders if oracle is plausibly affordable,
            # update surrogate calib on the fly, then return best exact.
            est0 = getattr(self, "_oracle_call_sec", None)
            if (
                plc is not None
                and 180 <= TIME_BUDGET <= 240
                and data.num_hard <= 520
                and len(pool) >= 2
                and _left() >= 24
                and (est0 is None or float(est0) <= 12.0)
            ):
                view = pool[: min(12, len(pool))]
                comps = []
                for hard, soft, name, cheap in view:
                    try:
                        c = self._cheap_components(hard, soft, data)
                    except Exception:
                        c = {"wl": float("inf"), "den": float("inf"), "cong": float("inf"), "overlap": float("inf")}
                    comps.append((hard, soft, name, cheap, c))

                def pick_best(metric: str):
                    return min(comps, key=lambda x: float(x[4].get(metric, float("inf"))))

                shortlist = [view[0]]
                for metric in ["cong", "wl", "den"]:
                    cand = pick_best(metric)
                    shortlist.append((cand[0], cand[1], cand[2], cand[3]))
                for hard, soft, name, cheap in view:
                    if isinstance(name, str) and "latent-cem" in name:
                        shortlist.append((hard, soft, name, cheap))
                        break

                seen = set()
                leaders = []
                for item in shortlist:
                    nm = item[2]
                    if nm in seen:
                        continue
                    seen.add(nm)
                    leaders.append(item)
                    if len(leaders) >= 4:
                        break

                best_cost = float("inf")
                best_pair = (leaders[0][0], leaders[0][1])
                best_name = leaders[0][2]
                evaluated = 0
                for hard, soft, name, _ in leaders:
                    est = getattr(self, "_oracle_call_sec", None)
                    if evaluated >= 2 and est is not None and _left() < float(est) + 8:
                        break
                    if _left() < 14:
                        break
                    exact = _oracle_eval(hard, soft, name)
                    if exact is None:
                        continue
                    evaluated += 1
                    cost = float(exact["proxy"])
                    if cost + 1e-9 < best_cost:
                        best_cost = cost
                        best_pair = (hard.copy(), soft.copy())
                        best_name = name
                    # Stop early if oracle is slow and we're close to the end.
                    est = getattr(self, "_oracle_call_sec", None)
                    if est is not None and _left() < float(est) + 8:
                        break
                if evaluated > 0:
                    if PROFILE:
                        import sys

                        print(
                            f"[PROFILE] {benchmark.name}: leader_oracle evaluated={evaluated}/{len(leaders)} best={best_name} cost={best_cost:.4f}",
                            file=sys.stderr,
                        )
                    return best_pair

            best = pool[0]
            if PROFILE:
                import sys

                print(
                    f"[PROFILE] {benchmark.name}: select_best short_budget no_oracle best={best[2]} cheap={float(best[3]):.4f}",
                    file=sys.stderr,
                )
            return best[0], best[1]

        if TIME_BUDGET >= 180:
            top_k = 20 if data.num_hard <= 350 else 12 if data.num_hard <= 520 else 8
        elif TIME_BUDGET >= 90:
            top_k = 10 if data.num_hard <= 350 else 7 if data.num_hard <= 520 else 5
        else:
            top_k = 6 if data.num_hard <= 350 else 4 if data.num_hard <= 520 else 3
        # Fast oracle: selection matters; consider more candidates.
        try:
            est_sel = getattr(self, "_oracle_call_sec", None)
            if est_sel is not None and float(est_sel) <= 2.0 and TIME_BUDGET >= 300:
                top_k = int(min(32, max(int(top_k), 14 if data.num_hard <= 420 else 10)))
        except Exception:
            pass

        # Diversity-first shortlist: include a few structurally different candidates even when their
        # surrogate score is slightly worse (surrogate misranking is common, especially on ibm01-like).
        leaders: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        if pool:
            leaders.append(pool[0])

        def _best_named(pred) -> Optional[Tuple[np.ndarray, np.ndarray, str, float]]:
            best = None
            best_sc = float("inf")
            for h, s, n, sc in pool[: min(40, len(pool))]:
                try:
                    nm = str(n)
                    if not pred(nm):
                        continue
                except Exception:
                    continue
                if float(sc) < best_sc:
                    best_sc = float(sc)
                    best = (h, s, n, sc)
            return best

        for pick in [
            _best_named(lambda n: "osa" in n),
            _best_named(lambda n: n.startswith("periphery-exact")),
            _best_named(lambda n: n.startswith("gpu-pop-eplace")),
            _best_named(lambda n: n.startswith("periphery-")),
            _best_named(lambda n: n.startswith("affine-")),
            _best_named(lambda n: "latent" in n),
            _best_named(lambda n: "consensus" in n),
        ]:
            if pick is not None:
                leaders.append(pick)

        seen = set()
        ordered: List[Tuple[np.ndarray, np.ndarray, str, float]] = []
        for item in leaders + pool:
            nm = str(item[2])
            if nm in seen:
                continue
            seen.add(nm)
            ordered.append(item)
        pool = ordered[: min(int(top_k), len(ordered))]

        best_cost = float("inf")
        best_pair = (pool[0][0], pool[0][1])
        best_name = pool[0][2] if len(pool[0]) >= 3 else "unknown"
        evaluated = 0
        est = getattr(self, "_oracle_call_sec", None)
        slow_oracle = est is not None and float(est) > (6.0 if TIME_BUDGET <= 240 else 10.0)
        if slow_oracle:
            min_exact = 2
        else:
            min_exact = 3 if TIME_BUDGET <= 240 else 1
        # If oracle is fast and we have time, do a small exact "leaderboard" to avoid
        # picking the wrong candidate due to surrogate misranking.
        try:
            if est is not None and float(est) <= 2.0:
                min_exact = int(max(min_exact, 3))
                if self._time_left(time.time()) >= 12.0:
                    min_exact = int(max(min_exact, 5 if data.num_hard <= 420 else 4))
        except Exception:
            pass
        for hard, soft, name, _ in pool:
            est = getattr(self, "_oracle_call_sec", None)
            if evaluated >= min_exact and est is not None and self._time_left(time.time()) < float(est) + 3.0:
                break
            placement = benchmark.macro_positions.clone()
            placement[: data.num_hard] = torch.from_numpy(hard).float()
            if data.num_soft > 0:
                placement[data.num_hard : data.num_macros] = torch.from_numpy(soft).float()
            t_eval = time.time()
            result = compute_proxy_cost(placement, benchmark, plc)
            dt = time.time() - t_eval
            self._prof["oracle_calls"] = int(self._prof.get("oracle_calls", 0)) + 1
            self._prof["oracle_sec"] = float(self._prof.get("oracle_sec", 0.0)) + float(dt)
            if getattr(self, "_oracle_call_sec", None) is None:
                self._oracle_call_sec = float(dt)
            else:
                self._oracle_call_sec = float(0.65 * self._oracle_call_sec + 0.35 * dt)
            cost = float(result["proxy_cost"])
            evaluated += 1
            if PROFILE:
                import sys

                print(f"[PROFILE] {benchmark.name}: eval {name} proxy={cost:.4f}", file=sys.stderr)
            if cost + 1e-9 < best_cost:
                best_cost = cost
                best_pair = (hard.copy(), soft.copy())
                best_name = name

        if PROFILE:
            import sys

            print(
                f"[PROFILE] {benchmark.name}: select_best evaluated={evaluated}/{len(pool)} best={best_name} cost={best_cost:.4f}",
                file=sys.stderr,
            )
        return best_pair

    def _time_left_for_work(self) -> float:
        # Time remaining excluding the reserve we keep for final candidate selection.
        try:
            reserve = float(getattr(self, "_select_reserve", 0.0) or 0.0)
        except Exception:
            reserve = 0.0
        return max(0.0, float(self._time_left(getattr(self, "_place_t0", time.time()))) - float(reserve))

    def _work_deadline(self, safety: float = 0.0) -> float:
        # Absolute wall-clock deadline for "work" phases; selection is allowed to use the reserved tail.
        try:
            deadline = getattr(self, "_deadline", None)
            if deadline is None:
                base = getattr(self, "_place_t0", time.time())
                deadline = float(base) + float(TIME_BUDGET) - float(TIME_GUARD)
            reserve = float(getattr(self, "_select_reserve", 0.0) or 0.0)
            return float(deadline) - float(reserve) - float(safety)
        except Exception:
            return float(time.time())

    def _time_left(self, t0: float) -> float:
        deadline = getattr(self, "_deadline", None)
        if deadline is None:
            base = getattr(self, "_place_t0", t0)
            budget = max(0.0, float(TIME_BUDGET) - float(TIME_GUARD))
            return max(0.0, budget - (time.time() - base))
        return max(0.0, float(deadline) - float(time.time()))

    def _try_load_plc(self, benchmark: Benchmark):
        try:
            from macro_place._plc import PlacementCost

            self._plc_netlist_path = None
            self._plc_init_plc_path = None
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
                    self._plc_netlist_path = netlist.replace("\\", "/")
                    self._plc_init_plc_path = plc_path.replace("\\", "/") if os.path.exists(plc_path) else None
                    return plc
        except Exception:
            return None
        return None

    def _make_plc_pool(self, benchmark: Benchmark, n: int):
        if n <= 1:
            return None
        netlist = getattr(self, "_plc_netlist_path", None)
        if not netlist:
            return None
        try:
            from macro_place._plc import PlacementCost

            pool = []
            init_plc = getattr(self, "_plc_init_plc_path", None)
            for _ in range(int(n)):
                plc = PlacementCost(str(netlist))
                if init_plc:
                    try:
                        plc.restore_placement(str(init_plc), ifInital=True, ifReadComment=True)
                    except Exception:
                        pass
                pool.append(plc)
            return pool
        except Exception:
            return None
