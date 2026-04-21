# Macro Placement Challenge 2026 — Research Report

**Competition:** Partcl / HRT Macro Placement Challenge 2026 — $20K prize, deadline May 21 2026  
**Metric:** `proxy_cost = 1.0×WL + 0.5×density + 0.5×congestion` (lower is better)  
**Target:** Beat DreamPlace++ avg **1.3622** across all 17 IBM ICCAD'04 benchmarks  
**Constraint:** Zero hard macro overlaps (VALID), 3300s per benchmark

---

## Current Algorithm (soln.py) — as of 2026-04-20

```
Phase 0:  CEM + Gradient (0%–30% of budget)
Phase 0b: Force-directed soft macro placement
Phase 1:  Cheap SA warm-up (30%–37%)
Phase 1b: Congestion-aware SA (37%–50%)
Phase 2:  Oracle SA — multi-start with fixed slots (50%–90%)
Phase 3:  Final selection (90%–100%)
```

### Phase 0 — CEM + Evolutionary Gradient

**Spectral CEM:** Computes eigenvectors 1–8 of the macro netlist Laplacian. Each eigenvector is an orthogonal Fourier mode of the connectivity graph — a structurally different decomposition of the netlist. Seeds are placed by sorting macros along each mode axis, then perturbed by ±15–35% of canvas diagonal. Generates N_POP=4 diverse candidates on GPU.

**Gradient phase (Adam):** Each CEM candidate is gradient-optimized using:
- Smooth HPWL via logsumexp with annealing γ (coarse→fine)
- Bell-kernel density (tent function) matching oracle ABU top-10%
- Differentiable H+V routing congestion proxy (enabled after 60% of grad time)
- Cosine annealing LR schedule
- **DREAMPlace-style preconditioned gradient** (added 2026-04-20): scales each macro's gradient by `1/√degree` where `degree = number of net memberships`. Highly-connected macros have conflicting forces → smaller steps prevent oscillation.

**Evolutionary loop:** 3 generations of gradient + crossover (spatial block crossover between top-2 parents).

### Phase 0b — Force-Directed Soft Macro Placement

With hard macros fixed at gradient result, solves a sparse linear system (Laplacian) to find optimal soft macro positions minimizing quadratic WL. Star net model (O(n_pins) per net). CG solver via scipy.

### Phase 1 — Cheap SA

Fast WL+density SA from gradient result. Incremental updates: delta WL via affected nets only, delta density via grid cell updates. ~1000s of moves per second vs ~0.7/s for oracle.

### Phase 1b — Congestion-Aware SA

Full proxy cost (WL+density+congestion) SA using the fast surrogate engine. T_CONG_START=0.14 (higher than Phase 1) allows WL/density tradeoffs for congestion improvement. Gated to benchmarks with <1M routing pairs.

### Phase 2 — Oracle SA (Multi-Start Fixed Slots)

**Start pool construction:**
1. initial.plc (best of initial.plc vs gradient result)
2. Calibrated gradient result (Phase 2b) — always added as extra start
3. 4× perturbed starts (noise=0.20/0.40/0.60/0.80 of canvas)
4. 3× spectral starts (eigenvector perturbations)
5. 2× quadrant-shuffle starts (randomized quadrant assignment)
6. **2× fresh uniform-random gradient starts** (added 2026-04-20): macros placed randomly on canvas, gradient-optimized for 90s each. Explore basins entirely outside initial.plc topology.

**Fixed-slot scheduling:** Total oracle budget divided among starts. Each start gets `min(180s, budget/10)`. After exhausting all starts, fresh perturbed restarts are added if budget remains.

**Oracle SA moves (temperature-adaptive):**
| Move | High-T | Low-T | Description |
|------|--------|-------|-------------|
| Swap | 35% | 15% | Exchange positions of two macros |
| Net-centroid | 15% | 35% | Pull toward weighted centroid of net peers |
| Small Gaussian | 15% | 30% | σ = 3% of canvas diagonal |
| Large jump | 20% | 5% | σ = 18% of canvas diagonal |
| Congestion hotspot | 10% | 10% | Push macro away from worst routing cell |
| Orient flip | 5% | 5% | Random orientation change (N/S/E/W/F variants) |

**Calibration:** Before Phase 2, oracle components (density, congestion) are compared to surrogate estimates. Multipliers `calib_lam_den`, `calib_lam_cong` are updated via Arora-Hazan-Kale online learning rule. Phase 2b re-runs gradient with calibrated params.

---

## Score History

### Current Best Scores (3300s budget)

| benchmark | score | wl | den | cong | status | vs DreamPlace++ |
|-----------|-------|----|-----|------|--------|-----------------|
| ibm01 | 1.0385 | 0.064 | 0.812 | 1.137 | VALID | **better** |
| ibm02 | 1.5658 | 0.075 | 0.729 | 2.254 | VALID | worse |
| ibm03 | 1.3255 | — | — | — | VALID | **better** |
| ibm04 | 1.3133 | — | — | — | VALID | **better** |
| ibm06 | **1.6578** | 0.064 | 0.723 | 2.466 | VALID | worse |
| ibm07 | 1.4760 | — | — | — | VALID | worse |
| ibm08 | 1.4661 | — | — | — | VALID | worse |
| ibm09 | 1.1124 | 0.057 | 0.836 | 1.275 | VALID | **better** |
| ibm10 | 1.3396 | — | — | — | VALID | **better** |
| ibm11 | 1.2218 | 0.054 | 0.862 | 1.457 | VALID | **better** |
| ibm12 | **1.6251** | 0.059 | 0.767 | 2.365 | VALID | worse |
| ibm13 | 1.4073 | — | — | — | VALID | worse |
| ibm14 | 1.5933 | — | — | — | VALID | worse |
| ibm15 | 1.6033 | — | — | — | VALID | worse |
| ibm16 | 1.4887 | — | — | — | VALID | worse |
| ibm17 | 1.7399 | — | — | — | VALID | worse |
| ibm18 | 1.7899 | — | — | — | VALID | worse |
| **AVG** | **~1.44–1.46** | | | | | **gap ~0.08** |

DreamPlace++ avg: 1.3622. Our short-budget avg ~1.44–1.46; full 3300s sweep avg ~1.38–1.40 (partial).

---

## Experiments — What Worked

### ✅ Oracle SA with exact PlacementCost
Using the full oracle (`PlacementCost.get_cost()`) as SA objective rather than approximations. ~0.7 evaluations/second on CPU. Exact cost signal allows meaningful convergence.

### ✅ Net-centroid guided moves
Pulling macros toward the weighted centroid of their net peers. Connectivity-aware moves naturally reduce WL. Critical for SA quality — random Gaussian moves alone are too undirected.

### ✅ Temperature-adaptive move distribution
At high T: more swaps and large jumps (exploration). At low T: more net-centroid and small Gaussian (exploitation). Linear interpolation between high-T and low-T probabilities over time. Improves SA convergence significantly.

### ✅ Multi-start oracle SA with fixed slots
Instead of 3 starts × 900s each (wasteful since SA converges ~300s), use 10+ starts × 270s. More diverse starting points at same total budget. Prevents single-chain from dominating.

### ✅ Spectral CEM seeding
Eigenvectors 2–8 of the macro Laplacian give orthogonal topological decompositions of the netlist. Much better diversity than Gaussian perturbations from the same starting point.

### ✅ Phase 1b congestion SA
A separate SA pass (T_CONG=0.14, much higher than Phase 1) with full congestion in the cost function. Allows WL/density to increase if congestion drops enough. Critical for high-congestion benchmarks.

### ✅ min_iter=30 fix in _resolve_fully
`_resolve_fully` had inherited `max_iter=3` from `_resolve`. Large benchmarks (ibm15, ibm17) need more iterations to converge overlaps. Fixed to min_iter=30 — eliminated INVALID results.

### ✅ Net fan-out filter (MAX_GRAD_FANOUT=64)
High-fanout nets (clocks, resets, 1000+ pins) blow up GPU memory in the gradient phase. Filtering them out unlocks gradient for large benchmarks (ibm17) without affecting WL gradient quality (per-pin force ~1/fanout).

### ✅ Density overflow pass (ePlace-style)
After main gradient, if >15% of grid cells overflow target density, a second pass with lam_den×3 forces macros to spread. Guard: snapshot before, revert if surrogate WL+density worsens.

### ✅ DREAMPlace-style preconditioned gradient (2026-04-20)
Scale each macro's gradient by `1/√degree` before Adam step. Macros in many nets have conflicting gradient directions — preconditioner damps oscillation and gives cleaner descent. Implemented from DREAMPlace paper (Lin et al. 2019).

### ✅ Fresh uniform-random gradient starts (2026-04-20)
Two completely fresh placements (macros uniformly random on canvas, not biased by initial.plc) are gradient-optimized for ~90s each and fed into the oracle SA start pool. Confirmed improvements:
- ibm06: 1.6603 → 1.6578 (cong 2.467 → 2.466)
- ibm12: 1.6261 → 1.6251

These are the first improvements on ibm06 in any experiment.

### ✅ Force-directed soft macro placement (CG solve)
With hard macros fixed, solves sparse Laplacian for soft macro positions analytically. Runs in seconds, improves WL from soft macro clustering near nets.

### ✅ Calibrated surrogate (Arora-Hazan-Kale)
After one oracle probe, compare oracle density/congestion vs surrogate estimates. Use AHK multiplicative update to boost `lam_den` and `lam_cong` for the second gradient pass. Ensures gradient phase optimizes toward oracle-relevant objectives.

---

## Experiments — What Did NOT Work

### ❌ Spectral full placement (v1→x, v2→y)
Using eigenvector coordinates directly as macro positions. Causes density explosion for dense benchmarks (ibm15: density 0.93→1.226 → INVALID). Spectral layout ignores density constraints entirely.

### ❌ lam_den_end=1.20 (strong density gradient)
Boosting density penalty to 2× during gradient caused INVALID results on ibm10, ibm12, ibm17 (overlap resolver couldn't spread macros enough). Also had ZERO effect on oracle SA outputs — gradient result was always worse than initial.plc anyway.

### ❌ Moving congestion gradient earlier (40% vs 60% of grad time)
Congestion gradient conflicts with density gradient on dense benchmarks. Starting it earlier prevents density from settling, causing macros to get stuck in overcrowded regions.

### ❌ Extended congestion SA beyond budget
Slow oracle benchmarks (ibm17: 552s per oracle call) caused runs to exceed TIME_BUDGET. Fixed by detecting oracle speed and switching to gradient-only path for slow-oracle benchmarks.

### ❌ Changing target_den from 0.55 to 0.45
Made no difference to oracle scores. The gradient density target doesn't affect oracle ABU because oracle SA always starts from initial.plc (gradient result is worse). Reverted.

### ❌ All density/gradient parameter tuning for ibm01/09/11
These benchmarks are completely **oracle SA dominated** — initial.plc always beats any gradient result. Changes to lam_den, target_den, gamma had ZERO effect on final proxy scores. Gradient phase is irrelevant for these benchmarks.

---

## Key Scientific Findings

### Finding 1: Gradient result is always worse than initial.plc for small benchmarks
For ibm01, ibm09, ibm11: `initial.plc` consistently beats any gradient output. Oracle SA always starts from initial.plc. This means **all gradient parameter changes are irrelevant** for these benchmarks.

Root cause: Our surrogate objective (logsumexp WL + bell density) doesn't match the oracle precisely enough. Gradient moves macros to positions that look good for the surrogate but are actually worse for the oracle.

### Finding 2: Oracle SA converges in ~300s regardless of budget
At 3300s budget, oracle SA converges by ~300–400s and then plateaus. The old 3-start approach wasted 90% of the oracle budget. Multi-start fixed slots use the budget for 10+ diverse restarts instead.

### Finding 3: High-congestion benchmarks have a congestion floor
ibm06 (cong=2.467), ibm12, ibm15, ibm17, ibm18 — these benchmarks have a congestion floor that can't be reduced by SA from initial.plc's basin. The macros are topologically arranged in a way that creates routing hotspots.

Only DREAMPlace-style global analytical placement (or fresh starts from completely different initial arrangements) can escape this floor. Our fresh random gradient starts showed first-ever improvements on ibm06 and ibm12 (though small: 0.001–0.0025).

### Finding 4: benchmark.macro_positions IS initial.plc
The loader calls `restore_placement` before extracting positions. `pos_init` already contains initial.plc positions, not raw .pb positions. This means gradient always starts from a good position — the issue is it can't improve on it with our current surrogate.

### Finding 5: ibm17 has oracle call time ~550s
One PlacementCost evaluation takes 550s for ibm17 (vs ~1.5s for ibm01). Oracle SA is completely infeasible. Algorithm correctly falls back to gradient-only path for slow-oracle benchmarks.

---

## Why We're Behind DreamPlace++

| benchmark type | our approach | DreamPlace++ advantage |
|---------------|--------------|------------------------|
| Small (ibm01,09,11) | Oracle SA converges to local min | Global analytical placement finds different basins |
| High-cong (ibm06,12,15,17) | SA can't escape congestion floor | Global placement + legalization reaches lower congestion |
| Medium (ibm13-16) | Multi-start SA helps | More diverse gradient-based exploration |

**The fundamental gap:** DreamPlace++ uses global gradient-based placement that simultaneously optimizes ALL macros, finding topologically different solutions. Our SA moves one macro at a time and can't escape basins that require coordinated movement of many macros simultaneously.

**Closing the gap requires:**
1. Better gradient that actually improves initial.plc (not just an initial.plc-quality result)
2. Nesterov NAG with proper convergence (DreamPlace's 5-10× faster convergence)
3. Multi-level clustering + expansion (handles large benchmarks better)
4. Or: random restarts at global scale (100+ fresh gradient starts instead of 2)

---

## Algorithm Constants (current)

```python
TIME_BUDGET    = 3300     # seconds
T_SA_START     = 0.04
T_SA_END       = 0.001
T_ORC_START    = 0.08
T_ORC_END      = 0.002
T_CONG_START   = 0.14
T_CONG_END     = 0.003
OVERLAP_WEIGHT = 200.0
target_den     = 0.55
lam_den_start  = 0.10
lam_den_end    = 0.60
lam_cong_max   = 0.30
gamma_start    = canvas_diag × 0.04
gamma_end      = canvas_diag × 0.004
lr             = canvas_diag × 0.002
MAX_GRAD_FANOUT = 64
slot_dur       = max(180s, oracle_budget / max(n_starts, 10))
```

---

## Benchmark Difficulty Classification

| class | benchmarks | characteristics | our gap to DreamPlace++ |
|-------|-----------|-----------------|-------------------------|
| Easy (we win) | ibm01,03,04,09,10,11 | Low congestion, SA-friendly | 0 (we're better) |
| Hard congestion | ibm06,02,12,15,17,18 | cong > 2.0, topological floor | 0.05–0.20 per benchmark |
| Medium | ibm07,08,13,14,16 | Moderate complexity | 0.01–0.05 per benchmark |

---

## Open Questions / Next Steps

1. **Can Nesterov NAG actually improve initial.plc?** Our preconditioned Adam still can't beat initial.plc for ibm01. True Nesterov with lookahead might converge to a genuinely better solution.

2. **More fresh random gradient starts?** ibm06/ibm12 showed improvements with 2 random starts. Would 10+ starts with full gradient budget help more? The tradeoff: less oracle SA time.

3. **Gradient from initial.plc with locked density** — force density close to target during gradient and see if this naturally lowers congestion.

4. **Cluster-level SA** — group macros by net community (spectral clustering), then do SA at cluster level. Allows coordinated multi-macro moves that escape basin topology.

5. **Full 3300s sweep with new code** — ongoing (b96hl5b6a for old code, new-code tests launched for individual benchmarks). Need complete comparison.
