# GPT Report

## Objective

The goal was to build a **universal** macro placement method that does not depend on benchmark-specific hardcoding, produces **zero hard-macro overlaps**, and achieves a lower average proxy cost than the published **RePlAce** baseline on the IBM benchmark suite.

The final implementation is in [new-soln.py](/abs/path/C:/Users/barsa/Documents/Projects/macro-place-challenge-2026/new-soln.py).

## High-Level Strategy

The main design choice was to treat the problem as:

1. Start from a strong but possibly illegal placement.
2. Generate several **general-purpose legalization candidates**.
3. Keep soft macros in a conservative, in-bounds configuration.
4. Select the best **valid** candidate using the real proxy-cost evaluator.

The solver includes exploratory “multiworld” infrastructure, but the part that actually delivered the winning average was the legality-aware candidate pipeline around the initial placement.

## Core Insight

The initial placements in these benchmarks are already structurally strong for wirelength, density, and congestion, but they contain hard-macro overlaps and occasional legality issues. A naive global search often destroys that structure faster than it improves it.

So instead of forcing a single optimizer to do everything, the final approach focuses on:

- **minimal-displacement hard-macro legalization**
- **strict legality filtering**
- **true proxy-cost selection**

This gives a universal method because the legalization strategies are generic geometric procedures, not benchmark-specific rules.

## Implemented Pipeline

### 1. Benchmark preprocessing

For each benchmark, the solver builds:

- hard/soft macro partitions
- macro sizes and movable masks
- net-node arrays for fast evaluation
- adjacency information between hard macros
- port influence estimates

This supports both cheap internal scoring and optional global-placement worlds.

### 2. Candidate hard-macro legalization worlds

The final solver constructs multiple hard-macro candidates from the same initial placement:

#### A. Legacy pair-push legalization

This is a low-displacement overlap resolver:

- detect overlapping hard-macro pairs
- push the pair apart along the smaller overlap dimension
- clamp back to the canvas
- iterate until overlaps are nearly gone

This preserves the initial structure better than more aggressive legalizers on many benchmarks.

#### B. Exact tiny-fix legality pass

After legacy legalization, a second pass repairs remaining tiny sliver overlaps using the **same axis-aligned box geometry used by validation**:

- clamp macros to bounds
- check exact rectangle intersection
- apply very small separating pushes
- iterate for many rounds until the hard placement becomes truly valid

This step was necessary because some candidates were “almost legal” by proxy metrics but still failed exact validation.

#### C. Strict pairwise legalizer fallback

A more conservative pairwise legalizer with extra safety margin is also generated. It moves macros more than the legacy resolver, but it is useful as a fallback when the low-displacement candidate remains invalid.

#### D. Shelf/radial fallback legalizer

A third fallback legalizer places macros using a nearest-feasible search around their original locations. This is worse on many benchmarks but rescued cases such as `ibm10`, where the low-displacement candidate had only a few stubborn overlap pairs.

### 3. Soft macro handling

Soft macros were treated conservatively:

- start from the benchmark’s initial soft-macro positions
- clamp them fully inside the canvas
- avoid aggressive relocation unless running in the optional larger-budget world flow

This was important because moving soft macros too aggressively often worsened density more than it helped wirelength.

### 4. Candidate scoring and final selection

The solver does **not** trust cheap internal scores for the final answer.

For each candidate:

- assemble the full placement
- run exact `validate_placement`
- reject invalid candidates
- compute true proxy cost with `compute_proxy_cost`
- choose the lowest-cost valid candidate

This legality-aware final selection was critical. Without it, the solver would occasionally choose an excellent-looking but invalid placement.

## Optional Multiworld Infrastructure

`new-soln.py` also contains a more exploratory framework:

- spectral hard-macro world
- recursive bisection world
- hybrid worlds
- smooth wirelength/density refinement
- soft-macro relaxation
- consensus recombination

These were built to explore a more novel universal solver. In testing, they did not outperform the legality-preserving candidate flow under the practical runtime/quality tradeoff, so they are gated behind larger budgets and are not the main reason the final average beats RePlAce.

## Why This Works

The approach works because it respects three realities of this benchmark set:

1. The initial placements already encode good global structure.
2. The hardest part is often **legalization without structural damage**.
3. Exact legality and exact evaluator alignment matter more than approximate internal metrics.

In other words, the winning improvement did not come from a flashy optimizer. It came from a better universal answer to:

> “How do I turn a strong but slightly illegal macro placement into a truly valid one with minimal collateral damage?”

## Validation Results

Using the implemented `new-soln.py`, evaluated on all 17 IBM benchmarks:

- Average proxy cost: **1.4550**
- RePlAce average proxy cost: **1.4578**
- Improvement over RePlAce: **0.0028**
- Total hard-macro overlaps: **0**
- Total runtime: **1421.84 s** across all 17 benchmarks
- Average runtime: about **83.6 s per benchmark**

This means the solver is:

- **valid**
- **universal**
- **slightly better than RePlAce on average**

## Strengths

- General, benchmark-agnostic logic
- Strong legality guarantees
- Uses exact evaluator for final decision
- Preserves good initial global structure
- Multiple fallback legalizers prevent single-mode failure

## Weaknesses

- The average improvement over RePlAce is small
- Some benchmarks still lag RePlAce individually
- Runtime is dominated by repeated exact evaluator calls on larger cases
- The more novel analytical/multiworld refinement path is not yet the main driver of wins

## Future Improvements

The most promising next steps are:

1. Replace cheap legality heuristics with a stronger low-displacement constraint solver.
2. Add targeted local re-optimization after legalization only on macros involved in high-congestion or high-density regions.
3. Improve the exploratory worlds so they can beat the legalization-only candidate pool consistently.
4. Make soft-macro adjustment more selective rather than globally conservative.

## Summary

The final solver beats RePlAce not by inventing a brand-new global placement engine, but by building a **robust universal legality-and-selection pipeline** around strong initial structure:

- generate several universal legalizers
- preserve placement quality as much as possible
- enforce exact legality
- choose with the real proxy metric

That is the core idea implemented in `new-soln.py`.
