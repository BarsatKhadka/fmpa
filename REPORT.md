# Macro Placement Challenge 2026 — Solution Report

## Problem Statement

Place hard macros on a chip canvas to minimize:

```
proxy_cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

Constraints: zero macro overlaps (VALID), within 300s per benchmark. Evaluated on 17 IBM ICCAD'04 benchmarks.

---

## Algorithm Overview

### Phase 1 — Cheap SA (Fast Exploration)

A fast simulated annealing pass that uses a proxy cost based on overlap penalties rather than the full `PlacementCost` oracle. This explores the space quickly and provides a good warm start for Phase 2.

- **Cost**: `wirelength_approx + OVERLAP_WEIGHT × overlap_area`
- **Temperature**: exponential schedule from `T_SA_START=0.04` to `T_SA_END=0.001`
- **Moves**: Gaussian random displacement + swap
- **Overlap resolution**: `_resolve()` — iterative pairwise repulsion (O(n²) per round, up to 20 rounds)

### Phase 2 — Multi-Start Oracle SA

Uses the exact `PlacementCost` oracle (~1.5s per evaluation) as the SA cost function. Three starting points are tried in parallel time slots:

1. **Best of initial.plc and cheap-SA result** — warm start from best known position
2. **30% perturbed** — moderate noise applied to initial positions
3. **60% perturbed** — heavy noise for broader exploration

Each start receives `1/3` of the remaining oracle time budget. The global best across all three chains is kept.

**Key SA parameters:**
- `T_ORC_START = 0.08` — accepts moves ~8% worse with 37% probability at start
- `T_ORC_END = 0.002` — nearly greedy at end
- Temperature: exponential decay over each slot's time fraction

**Move types in oracle SA:**
| Move | Probability | Description |
|------|-------------|-------------|
| Swap | 25% | Swap positions of two random macros |
| Net-centroid | 25% | Pull macro toward weighted centroid of connected net peers |
| Small Gaussian | 25% | σ = 3% of canvas diagonal |
| Large Gaussian | 15% | σ = 18% of canvas diagonal |
| Cluster | 10% | Move group of spatially nearby macros together |

**Local overlap resolution**: After each move, only the moved macros are checked against all others (`_local_resolve`, O(k×n)), with fallback to full `_resolve` if residual overlaps remain.

### Phase 3 — Final Cleanup

- `_resolve_fully`: Full overlap resolver (20 rounds, jitter, GAP=0.01) applied to best found position
- Compare with `initial.plc` as a sanity check — keep whichever is cheaper
- **Float32 safety resolve**: After converting float64 → float32 (required by Torch), a final `_resolve(max_iter=200)` eliminates any overlaps introduced by float32 quantization (~1e-4 error for large coordinates)

---

## Key Implementation Details

### `_resolve()` — Iterative Pairwise Repulsion
Iterates over all macro pairs, pushing overlapping macros apart by half the overlap in each axis. Clamps to canvas boundaries. Runs up to `max_iter` rounds (default 20).

### `_resolve_fully()` — Robust Full Resolver
`_resolve` with `max_iter=20`, plus random jitter if still overlapping after standard rounds.

### `_local_resolve()` — Fast Oracle-SA Resolver
Only resolves `moved_indices` against all `num_hard` macros. O(k×n) per iteration. Falls back to full `_resolve(max_iter=100)` if residual overlaps remain after 120 iterations.

### Net-Centroid Move
For a randomly chosen macro `i`, computes the weighted centroid of all other macros connected via shared nets, weighted by net fan-out inverse. Steps `i` toward the centroid by `step_size × (1 - frac)` where `frac` is time elapsed.

### Budget Management
```
hard_deadline = t0 + TIME_BUDGET - 10   # absolute cutoff
oracle_end    = min(hard_deadline - 5, t0 + (TIME_BUDGET - 30) × 0.90)
```
Phase 1 (cheap SA) runs until ~10% of budget used; Phase 2 runs until `oracle_end`; Phase 3 gets the remainder.

---

## Results

### Benchmark Scores (300s budget, all VALID, 0 overlaps)

Multi-start oracle SA results (3 chains: warm-start, 30% perturb, 60% perturb):

| Benchmark | Our Score | RePlAce | vs SA baseline | vs RePlAce |
|-----------|-----------|---------|----------------|------------|
| ibm01     | 1.0385    | 0.9976  | +21.1%         | -4.1%      |
| ibm02     | 1.5656    | 1.8370  | +17.9%         | **+14.8%** |
| ibm03     | 1.3252    | 1.3222  | +23.8%         | -0.2%      |
| ibm04     | 1.3133    | 1.3024  | +12.7%         | -0.8%      |
| ibm06     | 1.6577    | 1.6187  | +33.8%         | -2.4%      |
| ibm07     | 1.4760    | 1.4633  | +27.0%         | -0.9%      |
| ibm08     | 1.4664    | 1.4285  | +23.8%         | -2.7%      |
| ibm09     | 1.1126    | 1.1194  | +19.8%         | **+0.6%**  |
| ibm10     | 1.3368    | 1.5009  | +36.7%         | **+10.9%** |
| ibm11     | 1.2117    | 1.1774  | +29.2%         | -2.9%      |
| ibm12     | 1.6243    | 1.7261  | +42.5%         | **+5.9%**  |
| ibm13     | 1.3851    | 1.3355  | +27.6%         | -3.7%      |
| ibm14     | 1.5910    | 1.5436  | +30.1%         | -3.1%      |
| ibm15     | 1.6028    | 1.5159  | +30.3%         | -5.7%      |
| ibm16     | 1.4904    | 1.4780  | +33.3%         | -0.8%      |
| ibm17     | 1.7386    | 1.6446  | +52.7%         | -5.7%      |
| ibm18     | 1.7899    | 1.7722  | +35.5%         | -1.0%      |
| **AVG**   | **1.4545**| **1.4578**| **+31.6%**   | **+0.2%**  |

**+X% = our score is better (lower) than baseline**

### Wins vs RePlAce
- 4 benchmarks where we beat RePlAce: ibm02 (+14.8%), ibm10 (+10.9%), ibm12 (+5.9%), ibm09 (+0.6%)
- Overall average: 1.4545 vs 1.4578 — **+0.2% improvement over RePlAce**

### Benchmarks Needing Improvement
- ibm15 (-5.7% gap), ibm17 (-5.7% gap): RePlAce significantly better
- ibm01 (-4.1%), ibm13 (-3.7%): moderate gaps

### Effect of Multi-Start vs Single-Chain
Multi-start (3 chains) vs single-chain oracle SA — avg 1.4545 vs 1.4547 (+0.01%). Marginal improvement; the warm-start initial.plc chain dominates, perturbed chains help occasionally but hurt slightly on others (ibm09: 1.1126 vs 1.1068).

---

## Baselines

| Method | Avg Proxy Cost | Notes |
|--------|---------------|-------|
| initial.plc | ~1.4578 | Reference hand-crafted placements (INVALID — overlaps) |
| RePlAce | 1.4578 | Gradient-based analytical placer |
| **Ours** | **1.4547** | Oracle SA, all VALID, 0 overlaps |

---

## Algorithm Evolution

1. **Greedy hill-climbing with oracle**: No temperature, always accept improvements. Got stuck in local minima.
2. **Oracle SA (single chain)**: Added temperature schedule. Allowed escaping local minima. First version to beat RePlAce on multiple benchmarks.
3. **Net-centroid moves**: Added connectivity-aware moves pulling macros toward net peers. Improved wirelength.
4. **Multi-start oracle SA**: 3 parallel chains with different starting points. Marginal improvement (~0.1% avg) — the initial.plc warm start is already very strong.
5. **Float32 safety resolve**: Fixed INVALID results from float64→float32 quantization.
6. **Local resolve optimization**: Fast O(k×n) resolver for oracle SA inner loop, 2-3× more oracle evaluations per second.

---

## Constants

```python
TIME_BUDGET  = 300      # seconds (3300 for full competition)
T_SA_START   = 0.04     # cheap SA temperature start
T_SA_END     = 0.001    # cheap SA temperature end
T_ORC_START  = 0.08     # oracle SA temperature start
T_ORC_END    = 0.002    # oracle SA temperature end
OVERLAP_WEIGHT = 200.0  # penalty weight for cheap SA
GAP          = 0.01     # minimum clearance between macros
```
