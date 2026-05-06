# Method Documentation — Macro Placement Challenge 2026

**Team:** Barsat Khadka  
**Email:** khadkabarsat598@gmail.com  
**Method Name:** Multi-Start Gradient SA with Oracle-Driven SOFT_REOPT  
**Short Description:** GPU-batched multi-world gradient initialization followed by oracle simulated annealing on hard macros, then density-aware gradient re-optimization of soft macros. Achieves 16/17 wins vs RePlAce on IBM ICCAD04 benchmarks. Future direction: replacing oracle SA with a GNN-based learned cost model trained on large-scale placement datasets would eliminate the wall-clock bottleneck of expensive oracle calls and enable faster, higher-quality search — not developed here due to insufficient public placement data; open datasets from ParTCL (open-source designs with placement solutions) would make this feasible.

---

## Overview

The placer runs in three sequential phases within a fixed time budget (default 3300 seconds):

```
Phase 0  (0–30% of budget):  Multi-world GPU-batched gradient sweep → top-5 seeds
Phase 2  (30–95% of budget): Oracle Simulated Annealing on hard macros
SOFT_REOPT  (post-SA):       Soft macro density/congestion gradient re-optimization
Phase 3  (final seconds):    Best-of comparison: SA result vs initial.plc oracle
```

**Proxy cost** (what we optimize): `proxy = 1.0 × WL + 0.5 × Density + 0.5 × Congestion`  
Hard constraint: zero overlap between hard macros.

---

## Phase 0 — Multi-World Gradient Initialization

### Purpose
Generate diverse, high-quality starting positions for oracle SA. The key insight is that different density-wirelength trade-off weights (λ_den) create topologically distinct force fields, causing gradient descent to converge to different local optima ("worlds"). Running many worlds in parallel on GPU finds more of the landscape.

### B2B Quadratic Analytical Seeds
Before gradient, we generate 5 bound-to-bound (B2B) analytical placements — a classical quadratic wirelength solver that collapses all macros to their net-centroid-optimal positions. Five variants are used:
- `ports_soft + center init` — standard analytical, all anchor types
- `ports_only + center init` — strips soft-macro topology to expose port connectivity
- `ports_soft + random init` — different basin than center-initialized
- `ports_soft + corners init` — radically different topology (macros spread to corners)
- `ports_soft + short-net boost` — congestion-aware variant emphasizing short nets

These B2B positions are then refined by gradient (not used raw).

### GPU-Batched Multi-World Gradient Sweep
**N_PAR** gradient chains run simultaneously as a single batched tensor `[N_PAR, n_mov, 2]` on GPU:
- 48 GB GPU → N_PAR = 64 parallel worlds
- 16 GB GPU → N_PAR = 32
- CPU → N_PAR = 1 (sequential)

Each batch of N_PAR starts uses a mix of 5 gradient "modes" to explore different parts of the landscape:

| Mode | λ_den mult | λ_cong | cong_start_frac | Description |
|------|-----------|--------|-----------------|-------------|
| normal (6/16) | 1× | none | 0.60 | Baseline WL + density |
| cong_mid (2/16) | 0.7× | 1.0 | 0.30 | Moderate congestion pressure |
| cong_hard (2/16) | 0.4× | 3.0 | 0.05 | Force low-congestion topology |
| density_first (4/16) | 3× | 0.1 | 0.05 | Over-spread first (ePlace continuation) |
| balanced_strong (2/16) | 2× | 2.0 | 0.20 | Push both density and congestion |

The **λ_den "worlds"** sweep 16 values: `[0.05, 0.20, 0.60, 1.50, 0.10, 0.40, 1.00, 2.00, 0.08, 0.30, 0.80, 0.15, 0.50, 1.20, 3.00, 4.00]`. Low λ → macros cluster near connections. High λ → macros spread globally.

### Gradient Loss Function (per-chain)
```
loss = WL_loss + λ_den × Density_loss + λ_cong × Cong_loss
```
- **WL loss**: Differentiable log-sum-exp approximation to HPWL using net bounding boxes
- **Density loss**: Bell-kernel soft density: `logsumexp(dens_flat / τ) * τ / top_k` with τ annealing 0.5→0.08, computed over top-10% grid cells
- **Cong loss**: Routing congestion approximation using macro area occupancy per routing tile

High-fanout nets (>64 pins, e.g. clocks/resets) are excluded from gradient: their per-pin force `~1/fanout` is negligible but they would dominate memory.

### Seed Selection
After gradient, each result is resolved (overlaps removed) and scored by cheap proxy. Top-5 are kept. `initial.plc` is always included in the candidate set as a guaranteed fallback.

**SEED_ORACLE_SPREAD**: Before oracle SA, all top-5 seeds are oracle-evaluated to get exact proxy costs. This calibrates the SA starting pool.

---

## Phase 2 — Oracle Simulated Annealing

### Purpose
Refine hard macro positions using the exact oracle (PlacementCost) as the energy function. Unlike the cheap proxy, the oracle uses the full TILOS congestion model and exact density computation, giving the true competition score.

### Oracle Calibration
Before SA begins, we measure how long one oracle call takes (`oracle_call_secs`). This determines:
- `oracle_calls_possible = (time_remaining) / oracle_call_secs`
- `oracle_end = hard_deadline - 2 × oracle_call_secs - 35s` (reserves budget for SOFT_REOPT)

If oracle is too slow (>60% of total budget per call), SA is skipped entirely.

### SA Configuration
- **Temperature**: Starts at `T_ORC_START = 0.08` (accepts +0.02 worse solutions ~78% of the time), anneals to `T_ORC_END = 0.002`
- **Temperature scaling**: `_ora_t_scale = max(1.0, oracle_call_secs / 12.0)` — slow oracles get proportionally higher temperature to explore more broadly per expensive evaluation
- **Move generation**: One hard macro is moved per SA step. Move distance is sampled from a Gaussian scaled by temperature and canvas size
- **Acceptance**: Standard Boltzmann: `accept = exp(-(cost_new - cost_best) / T)` if cost_new > cost_best

### Slot-Based Execution
SA is organized into "slots" — each slot runs from one starting position until convergence:
- `slot_dur = max(180s, oracle_time / num_starts)` — ensures enough starts get tried
- **SLOT_CONV** early exit: If SA hasn't improved in `max(90s, min(50% of slot, 300s))` AND ≥3 oracle steps AND ≥25% of slot elapsed → exit slot, try next start
- **ORA_SA_EXIT**: After 2 consecutive non-improving slots → stop SA entirely (leave budget for SOFT_REOPT)

### Start Pool
Sorted by oracle cost (best first), including:
1. Oracle-evaluated gradient seeds (known cost)
2. Perturbed versions of the best gradient seed (σ = 0.20, 0.35, 0.50 × canvas)
3. Fresh random starts (cost = ∞, sorted last)

This gives SA maximum diversity: it starts from the best known position then explores progressively harder restarts as the budget allows.

### Within-Slot SA
Each SA slot calls `_plc_oracle_sa()`:
- Runs net-centroid-guided moves: each macro is proposed to move toward the centroid of its nets (force-directed), then perturbed by temperature-scaled Gaussian noise
- After each accepted move, overlaps are resolved locally (only moved macro + neighbors)
- Falls back to full resolve if local resolve leaves residual overlaps

---

## SOFT_REOPT — Soft Macro Re-Optimization

### Purpose
Oracle SA only moves **hard macros**. Soft macros (standard-cell clusters) remain at their Phase 0 gradient positions, which were optimized for WL+density but may be suboptimal relative to the final hard macro configuration found by SA.

After SA, we re-run gradient optimization on **soft macros only**, with hard macros frozen at the SA result. This adjusts soft macro density and (for dense benchmarks) congestion.

### Configuration
- **Budget**: `min(120s, oracle_end - time.time() - oracle_call_secs - 10s)` — uses whatever SA left over, up to 120s
- **Loss weight**: `cong_weight = 2.0` if `oracle_density > 0.88` (dense benchmarks: ariane133-class), else `0.0` (density only)
- The density-only mode reduces clustering of soft macros around hard macros
- The congestion mode additionally pushes soft macros away from routing hot spots

### Why It Helps
Hard macros occupy specific grid cells. Soft macros cluster in the remaining space, often creating secondary density peaks. Re-running gradient with `λ_den = 5.0` (much higher than Phase 0's typical 0.5–1.5) forces soft macros to spread more uniformly, reducing the top-10% density cost in the oracle.

Typical improvement: 5–15% proxy cost reduction (e.g. oracle SA 1.38 → SOFT_REOPT 1.18 on ibm04).

---

## Phase 3 — Final Selection

After SOFT_REOPT, we compare three candidate solutions:
1. **SOFT_REOPT result** (or SA result if SOFT_REOPT didn't improve)
2. **initial.plc oracle score** — always evaluated as a sanity check

The best by oracle score is returned. This prevents rare cases where SA + SOFT_REOPT produces a result worse than the initial placement.

### Float32 Quantization Fix
The returned tensor uses float32 (required by the benchmark format), but our internal computation uses float64. Float32 rounding can reintroduce tiny overlaps. We run up to 30 rounds of overlap resolution in float32 space, checking after each round whether quantized positions are overlap-free.

Additionally, soft macros can overlap hard macros after float32 rounding. We apply a targeted fix: for each soft macro that overlaps a hard macro, push it to the nearest non-overlapping position (without disturbing soft-soft overlaps, which the gradient handled).

---

## Key Implementation Details

### Overlap Resolution (`_resolve`)
Uses an iterative force-based push: overlapping macro pairs are pushed apart proportional to their overlap area. Hard macros only (soft macros are excluded from hard-overlap resolution). Runs until total overlap < 1e-10 or maximum iterations reached.

`_resolve_fully` runs until convergence (no iteration cap) and is used when placement quality matters (before oracle evaluation). Fast `_resolve` is used inside the gradient loop where time is critical.

### Cheap Proxy Cost
For fast evaluation during gradient and SA warm-up:
```python
cheap_cost = WL + 0.5 × density + 0.5 × congestion + 200.0 × overlap
```
- WL: HPWL via log-sum-exp over net bounding boxes (differentiable, same formula as gradient)
- Density: top-10% grid cell mean occupancy
- Congestion: routing tile occupancy approximation using `fast_eng` (precomputed routing capacities from the oracle's internal state)
- Overlap: total overlapping area × 200 penalty weight

### Oracle (`_true_cost`)
Calls the PlacementCost C++ evaluator directly via the `plc` object. This is the exact same computation used in the competition's `evaluate.py`. It measures:
- WL: HPWL of all nets including port pins and macro pin offsets
- Density: exact grid cell occupancy with smooth range = 2
- Congestion: H/V routing congestion with top-5% smoothing

---

## Benchmark Performance

Results from the current implementation on 17 IBM ICCAD04 benchmarks (3300s budget):

| Benchmark | Our Score | RePlAce | vs RePlAce |
|-----------|-----------|---------|------------|
| ibm01 | ~0.96 | 0.9976 | +3.8% |
| ibm02 | ~1.52 | 1.8370 | +17.2% |
| ibm03 | ~1.19 | 1.3222 | +9.9% |
| ibm04 | ~1.20 | 1.3024 | +7.9% |
| ibm06 | ~1.53 | 1.6187 | +5.7% |
| ibm07 | ~1.36 | 1.4633 | +7.0% |
| ibm08 | ~1.36 | 1.4285 | +4.9% |
| ibm09 | ~1.04 | 1.1194 | +7.5% |
| ibm10 | ~1.32 | 1.5009 | +12.0% |
| ibm11 | ~1.15 | 1.1774 | +2.3% |
| ibm12 | ~1.57 | 1.7261 | +9.0% |
| ibm13 | ~1.27 | 1.3355 | +4.9% |
| ibm14 | ~1.51 | 1.5436 | +2.2% |
| ibm15 | ~1.49 | 1.5159 | +1.7% |
| ibm16 | ~1.49 | 1.4780 | -0.8% ✗ |
| ibm17 | ~1.63 | 1.6446 | +0.9% |
| ibm18 | ~1.72 | 1.7722 | +3.0% |
| **AVG** | **~1.38** | **1.4578** | **+5.6%** |

*(Scores are approximate from this commit. Exact scores from the running evaluation will be updated.)*

**Average runtime**: ~1000–1800s per benchmark (depends on oracle speed: fast benchmarks ~700s, large slow benchmarks like ibm16 ~2500s).

---

## WNS and Area on Ariane133 NG45

**These metrics require running the full OpenROAD PnR flow** (`scripts/evaluate_with_orfs.py`), which was not available in the submission environment. To obtain them:

```bash
# Requires OpenROAD-flow-scripts installed and in PATH
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --placer soln.py
```

The organizers will compute WNS and Area for the top 7 submissions as part of Tier 2 evaluation.

---

## What the Method Does NOT Do

- No reinforcement learning or neural networks
- No look-up tables or hardcoded positions
- No benchmark-specific parameter tuning (same code runs on all 17 benchmarks)
- No external proprietary tools — only PyTorch, NumPy, and the provided TILOS PlacementCost oracle

---

## Algorithm Summary (One Paragraph)

The placer initializes with a GPU-batched gradient sweep across 200–1000 random/analytical starting positions using 16 different density-wirelength trade-off weights ("worlds"), each refined by automatic differentiation through a differentiable WL+density+congestion objective. The top-5 cheaply-scored results are oracle-evaluated and used as seeds for oracle simulated annealing, which refines hard macro positions using the exact PlacementCost evaluator as the energy function. SA uses slot-based execution with convergence detection (SLOT_CONV) and early exit after consecutive non-improving slots (ORA_SA_EXIT). After SA, soft macros are re-optimized by gradient descent with high density weight (SOFT_REOPT), significantly reducing density concentration around hard macros. The final result is the best oracle score among all candidates, with float32 overlap correction applied.

## Future Direction

The current bottleneck is oracle SA: each call to the exact PlacementCost evaluator costs 1–170 seconds depending on benchmark size, limiting the number of configurations that can be explored in the 3300s budget. The natural next step is to replace oracle SA with a **GNN-based learned surrogate**. Given a large dataset of (placement, oracle cost) pairs, a GNN could predict cost in milliseconds rather than seconds, enabling millions of SA steps in the same budget. This would likely produce a step-change improvement in solution quality.

This version was not developed because sufficient public placement data does not currently exist. The IBM ICCAD04 benchmarks provide 17 netlists but only one placement solution each — far too few for generalization. **ParTCL providing open-source designs with placement solutions** (diverse netlists + their optimized placements) would make this GNN surrogate feasible. The gradient + SOFT_REOPT framework developed here would remain intact; only the SA oracle would be replaced by the learned model.
