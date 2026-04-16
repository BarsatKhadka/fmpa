# Macro Placement Literature Reference

> Proxy Cost = **1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion**  
> Target: beat RePlAce baseline (1.4578 avg). Top entry: 1.3255.

---

## Table of Contents

1. [Analytical Placement (Core Foundation)](#1-analytical-placement-core-foundation)
2. [Simulated Annealing](#2-simulated-annealing)
3. [RL / ML-Based Placement](#3-rl--ml-based-placement)
4. [GNN / Graph-Based Placement](#4-gnn--graph-based-placement)
5. [Force-Directed Placement](#5-force-directed-placement)
6. [Legalization / Overlap Removal](#6-legalization--overlap-removal)
7. [Congestion-Aware Placement](#7-congestion-aware-placement)
8. [Wirelength Models](#8-wirelength-models)
9. [Macro-Specific Representations](#9-macro-specific-representations)
10. [Multilevel / Hierarchical Methods](#10-multilevel--hierarchical-methods)
11. [Hybrid & Escape-from-Local-Optima](#11-hybrid--escape-from-local-optima)
12. [Benchmarking & Evaluation](#12-benchmarking--evaluation)
13. [Strategy Recommendations](#13-strategy-recommendations)

---

## 1. Analytical Placement (Core Foundation)

These are the methods behind the top leaderboard entries. All use gradient descent on a smooth objective combining wirelength and density.

---

### ePlace / ePlace-MS
**Authors:** Jingwei Lu, Pengwen Chen et al. | **Year:** 2014–2015  
**Venue:** DAC / TCAD  
**Paper:** https://cseweb.ucsd.edu/~jlu/papers/eplace-dac14/paper.pdf

**Core Idea:**  
Models every cell as a positive electric charge. Uses electrostatic potential energy as the density penalty — solves a Poisson equation via FFT to get density forces. Then runs Nesterov's accelerated gradient method to jointly minimize wirelength + density. ePlace-MS extends this to mixed-size designs (macros + standard cells).

**Why it matters:**  
ePlace-Lite on the leaderboard (score 1.3913) is directly derived from this. The FFT-based electrostatic density model is the foundation for both RePlAce and DREAMPlace.

---

### RePlAce
**Authors:** Chung-Kuan Cheng, Andrew Kahng, Ilgweon Kang, Lutong Wang | **Year:** 2018–2019  
**Venue:** TCAD  
**Paper:** https://vlsicad.ucsd.edu/Publications/Journals/j126.pdf  
**Code:** https://github.com/The-OpenROAD-Project/RePlAce

**Core Idea:**  
Non-linear analytical placer using Nesterov's method with improved preconditioning, dynamic step-size adaptation, and constraint-oriented local smoothing. Uses the WA (weighted-average) wirelength model. Directly integrated into OpenROAD.

**Why it matters:**  
This is the **competition's baseline (1.4578)**. You need to beat it. Understanding exactly where RePlAce converges poorly is key to improving on it.

---

### DREAMPlace
**Authors:** Yibo Lin, Shounak Dhar, Wuxi Li, Haoxing Ren, Brucek Khailany, David Z. Pan | **Year:** 2019  
**Venue:** DAC  
**Paper:** https://yibolin.com/publications/papers/PLACE_DAC2019_Lin.pdf  
**Code:** https://github.com/limbo018/DREAMPlace

**Core Idea:**  
Reimplements the analytical placement problem in PyTorch. Custom CUDA kernels for the WA wirelength gradient and electrostatic density gradient (FFT-based) give ~40× speedup over multi-threaded RePlAce with no quality loss.

**Why it matters:**  
DREAMPlace++ (leaderboard #2, score 1.3622) is built on this. It is the strongest open-source analytical placer. The GPU backend is reused by AutoDMP, ABCDPlace, and all top entries.

---

### DREAMPlace 3.0 — Multi-Electrostatics
**Authors:** Jiaqi Gu, Zixuan Jiang, Yibo Lin, David Z. Pan | **Year:** 2020  
**Venue:** ICCAD  
**Code:** included in DREAMPlace repo

**Core Idea:**  
Uses separate electrostatic density systems for different region types: placement blockages, preplaced cells, hard macros. Achieves more robust convergence on mixed-size designs with heterogeneous macro sizes.

**Why it matters:**  
IBM benchmarks have macros with 33× size variation — the multi-electrostatics formulation handles this much better than a single density system. Likely the exact formulation in DREAMPlace++.

---

### DREAMPlace 4.0 — Timing-Driven
**Authors:** Yibo Lin et al. | **Year:** 2022–2023  
**Venue:** DATE / TCAD  
**Paper:** https://dl.acm.org/doi/abs/10.1109/TCAD.2023.3240132  
**Code:** included in DREAMPlace repo

**Core Idea:**  
Extends DREAMPlace with timing-driven net weighting using momentum (exponential moving average) to stabilize updates, plus a Lagrangian relaxation-based detailed placement refinement stage. Significantly improves WNS/TNS.

**Why it matters:**  
While focused on timing, the net-weighting and Lagrangian refinement techniques are transferable to the proxy cost. The Grand Prize ($20K) is judged on WNS/TNS/Area in OpenROAD — so timing is ultimately what matters.

---

### AutoDMP — Automated DREAMPlace-Based Macro Placement
**Authors:** Anthony Agnesina, Puranjay Rajvanshi et al. (NVIDIA) | **Year:** 2023  
**Venue:** ISPD  
**Paper:** https://d1qx31qr3h6wln.cloudfront.net/publications/AutoDMP.pdf  
**Code:** https://github.com/NVlabs/AutoDMP

**Core Idea:**  
DREAMPlace + ABCDPlace with simultaneous macro + standard cell placement and automatic hyperparameter tuning via multi-objective Bayesian optimization (MOBO). Tunes density penalty, wirelength model parameters, and learning rate schedules automatically. Tested on 2.7M-cell designs with 320 macros.

**Why it matters:**  
Directly targets the same problem (macro placement on large mixed-size designs). The automatic hyperparameter tuning is practical if you want to squeeze the best performance out of an analytical solver across 17 diverse benchmarks without manual tuning.

---

### Stronger Mixed-Size Placement — Second-Order Information
**Authors:** Yifan Chen, Zaiwen Wen, Yun Liang, Yibo Lin | **Year:** 2023  
**Venue:** ICCAD  
**Paper:** https://yibolin.com/publications/papers/PLACE_ICCAD2023_Chen.pdf

**Core Idea:**  
Replaces the first-order Nesterov gradient in DREAMPlace with a Barzilai-Borwein quasi-Newton method that incorporates second-order curvature information. Achieves robust convergence with fewer iterations.

**Why it matters:**  
6.5–33% HPWL reduction vs. default DREAMPlace on ISPD2005/TILOS benchmarks. A **drop-in improvement** to any Nesterov-based solver with minimal implementation overhead.

---

### Moreau Envelope Wirelength Model
**Authors:** Peiyu Liao, Hongduo Liu, Yibo Lin, Bei Yu, Martin Wong | **Year:** 2023  
**Venue:** DAC  
**Paper:** https://yibolin.com/publications/papers/PLACE_DAC2023_Liao.pdf

**Core Idea:**  
Proposes the Moreau envelope as a convex, differentiable HPWL approximation replacing the standard WA/LSE models. The proximal mapping of HPWL is always convex regardless of net size, giving a tighter approximation than WA.

**Why it matters:**  
Up to 5.4% HPWL improvement over the WA model. A **drop-in wirelength model replacement** — compatible with any DREAMPlace-based solver. Directly reduces the 1.0×Wirelength term in the proxy cost.

---

### DG-RePlAce — Dataflow-Driven Analytical Placement
**Authors:** Andrew B. Kahng, Zhiang Wang | **Year:** 2024  
**Venue:** TCAD  
**Paper:** https://arxiv.org/abs/2404.13049

**Core Idea:**  
Extends RePlAce/DREAMPlace with structural awareness of datapath regularity for ML accelerator chips. Encodes macro array structure as an additional placement objective. Achieves 10% routed wirelength reduction and 31% TNS improvement vs. RePlAce.

**Why it matters:**  
Shows how structural priors about macro connectivity patterns can improve placement. The IBM benchmarks have structured connectivity — encoding that could reduce wirelength.

---

### IncreMacro — Incremental Macro Placement Refinement
**Authors:** Yuan Pu et al. (CUHK) | **Year:** 2024  
**Venue:** ISPD / TCAD  
**Paper:** https://www.cse.cuhk.edu.hk/~byu/papers/C205-ISPD2024-IncreMacro.pdf

**Core Idea:**  
Post-global-placement macro refinement. Uses:
1. kd-tree-based macro diagnosis to identify problematic macros
2. Gradient-based macro shifting for wirelength/congestion improvement
3. Constraint-graph LP for legalization (guaranteed zero-overlap)

**Why it matters:**  
Reduces routed wirelength by 6.5% on top of DREAMPlace/AutoDMP. The legalization technique is critical for the zero-overlap requirement. Works as a **post-processing step** on top of any base placer.

---

### ABCDPlace — Accelerated Batch-Based Concurrent Detailed Placement
**Authors:** Yibo Lin et al. | **Year:** 2020  
**Venue:** TCAD  
**Paper:** https://yibolin.com/publications/papers/ABCDPLACE_TCAD2020_Lin.pdf  
**Code:** included in DREAMPlace repo

**Core Idea:**  
Parallelizes detailed placement (independent set matching, global swap, local reordering) for multi-threaded CPUs and GPUs using batched concurrent algorithms. 10×+ speedup over sequential NTUplace3.

**Why it matters:**  
Used as the detailed placement stage in AutoDMP and DREAMPlace pipelines. Fast detailed placement is needed to refine within the 1-hour time limit.

---

## 2. Simulated Annealing

---

### SA Baseline (ICCAD 2004)
**Authors:** Various | **Year:** 2003–2004 (re-assessed 2022–2023 by TILOS)  
**Code:** https://github.com/TILOS-AI-Institute/MacroPlacement

**Core Idea:**  
Simulated annealing on a sequence-pair or B*-tree representation with perturbation moves (rotate, move, swap). The IBM ICCAD 2004 benchmarks were created to test SA-based floorplanners. In 2023 reassessment, SA with the proxy cost outperforms Google's Circuit Training RL on all 17 benchmarks.

**Why it matters:**  
The competition's SA baseline scores **2.1251** — far below analytical methods. However, SA is robust to local optima and guarantees zero overlaps by construction. Good starting point before you can implement analytical placement.

---

### FastSA — Modern Floorplanning Based on Fast Simulated Annealing
**Authors:** Tung-Chieh Chen, Yao-Wen Chang et al. | **Year:** 2005  
**Venue:** ISPD / TCAD  
**Paper:** https://cc.ee.ntu.edu.tw/~ywchang/Papers/ispd05-floorplanning.pdf

**Core Idea:**  
Three-stage SA temperature schedule with B*-tree representation:
- Stage 1: high-temperature random exploration
- Stage 2: focused wirelength optimization
- Stage 3: fine-tuning with small moves

For fixed-outline problems, dynamically reweights the cost function to balance area and wirelength.

**Why it matters:**  
Three-stage cooling schedules are a practical engineering improvement over vanilla SA. If you're building an SA solution, this is the schedule to use. Significantly faster convergence vs. exponential cooling.

---

## 3. RL / ML-Based Placement

---

### AlphaChip / Circuit Training
**Authors:** Azalia Mirhoseini, Anna Goldie et al. (Google Brain) | **Year:** 2021  
**Venue:** Nature  
**Paper:** https://www.nature.com/articles/s41586-021-03544-w  
**Code:** https://github.com/google-research/circuit_training

**Core Idea:**  
Frames chip floorplanning as an RL problem. GNN-based state encoder embeds macros and standard-cell clusters. Policy network places macros one-by-one on a discrete grid. Claimed to outperform human designers in <6 hours.

**Why it matters:**  
Sparked the macro placement ML explosion. However, **TILOS showed SA and RePlAce consistently outperform CT on the exact IBM benchmarks used in this competition**. Nature added an Editor's Note in Sept 2023 questioning the claims. Understanding why RL fails here informs what actually works.

---

### Assessment of RL for Macro Placement (TILOS/UCSD)
**Authors:** Cheng, Kahng et al. (TILOS) | **Year:** 2022–2023  
**Venue:** ISPD  
**Paper:** https://vlsicad.ucsd.edu/Publications/Conferences/396/c396.pdf  
**Code:** https://github.com/TILOS-AI-Institute/MacroPlacement

**Core Idea:**  
Independent reproduction and evaluation of Google's CT on all 17 IBM benchmarks. SA and commercial tools consistently outperform RL. Identified flaws in Google's experimental setup. Open-sourced the full evaluation framework (which is what this competition uses).

**Why it matters:**  
**This paper defines the exact benchmark suite and proxy cost metric used in this competition.** Essential reading to understand what scores mean and why certain approaches work or fail.

---

### MaskPlace — Fast Chip Placement via Reinforced Visual Representation
**Authors:** Yao Lai, Jinxin Liu et al. | **Year:** 2022  
**Venue:** NeurIPS  
**Paper:** https://proceedings.neurips.cc/paper_files/paper/2022/file/97c8a8eb0e5231d107d0da51b79e09cb-Paper-Conference.pdf  
**Code:** https://github.com/laiyao1/maskplace

**Core Idea:**  
Represents chip state as three visual masks:
- **Wiremask:** HPWL increment per grid cell (tells you how much wirelength placing here costs)
- **Viewmask:** global canvas state
- **Positionmask:** overlap prevention

Trains RL policy on visual representation. Guarantees zero overlaps by construction. 60–90% wirelength reduction over Google CT.

**Why it matters:**  
The **wiremask concept** is independently useful — it's a fast way to evaluate HPWL impact of a placement without recomputing from scratch. Can be used as a cheap inner-loop evaluator in SA or BBO.

---

### ChiPFormer — Offline Decision Transformer for Placement
**Authors:** Yao Lai et al. | **Year:** 2023  
**Venue:** ICML  
**Paper:** https://proceedings.mlr.press/v202/lai23c/lai23c.pdf  
**Code:** https://github.com/laiyao1/chipformer

**Core Idea:**  
GPT-based decision transformer for offline RL. Pre-trains on offline placement data from multiple circuits, then fine-tunes for new circuits. Reduces placement runtime from hours to minutes.

**Why it matters:**  
The transfer learning paradigm: train on some IBM benchmarks, generalize to the rest. If you have compute for pretraining, this could give good initializations for all 17 benchmarks.

---

### MaskRegulate — RL as Macro Regulator (Not Placer)
**Authors:** LAMDA group | **Year:** 2024  
**Venue:** NeurIPS  
**Paper:** https://papers.nips.cc/paper_files/paper/2024/file/fe224a60b878e79d5b3d79d7f113f76b-Paper-Conference.pdf  
**Code:** https://github.com/lamda-bbo/macro-regulator

**Core Idea:**  
Instead of placing from scratch, RL **refines** an existing analytically-generated placement. Policy learns small adjustments to macro positions starting from a DREAMPlace solution. Adds regularity as a reward signal. Achieves 17% routing wirelength improvement and 73% reduction in horizontal congestion overflow vs. MaskPlace.

**Why it matters:**  
The **regulator paradigm** is highly actionable: use DREAMPlace to get a baseline solution, then apply RL to fine-tune for the proxy cost. Far more sample-efficient than learning from scratch because the initial placement is already reasonable.

---

### LaMPlace — Learning Cross-Stage Metrics
**Authors:** Zijie Geng et al. (MIRA Lab/USTC) | **Year:** 2025  
**Venue:** ICLR (Oral)  
**Paper:** https://proceedings.iclr.cc/paper_files/paper/2025/file/04c0399a47ee4107cd03b08f1f8c3eeb-Paper-Conference.pdf  
**Code:** https://github.com/MIRALab-USTC/AI4EDA-LaMPlace

**Core Idea:**  
Trains a predictor to estimate post-PnR metrics (WNS, TNS) that are only measurable after full routing. Uses predictions to generate a placement mask that guides black-box optimization of macro positions. Achieves 43% WNS and 30% TNS improvement.

**Why it matters:**  
Proxy cost (Tier 1) doesn't perfectly correlate with OpenROAD WNS/TNS/Area (Tier 2). A predictor for OpenROAD outcomes would allow optimizing for the Grand Prize directly. Technique: replace timing metrics with proxy cost terms and use offline IBM benchmark data for training.

---

### Chip Placement with Diffusion Models
**Authors:** UC Berkeley group | **Year:** 2024–2025  
**Venue:** arXiv / ICML 2025  
**Paper:** https://arxiv.org/abs/2407.12282

**Core Idea:**  
Trains a denoising diffusion model to place all macros simultaneously (vs. sequential RL). Uses guided sampling to optimize placement quality. Trains on synthetic data — zero-shot on real circuits. Reduces RUDY congestion by 35%+ vs. existing methods.

**Why it matters:**  
Zero-shot capability is powerful for 17 diverse benchmarks. The 35% congestion reduction directly targets the 0.5×Congestion term in the proxy cost. Simultaneous placement avoids the ordering-dependence of sequential RL.

---

### EfficientPlace — MCTS + RL for Macro Placement
**Authors:** MIRA Lab | **Year:** 2024  
**Venue:** ICML  
**Paper:** https://icml.cc/virtual/2024/poster/34772

**Core Idea:**  
Combines Monte Carlo Tree Search (global strategy) with local RL policy learning. MCTS provides global guidance to escape poor placement regions; RL refines within subtrees.

**Why it matters:**  
Tree search can escape local optima that gradient methods get stuck in. The MCTS global search + RL local refinement is a clean separation that can work on top of any base policy.

---

## 4. GNN / Graph-Based Placement

---

### GraphPlanner — Floorplanning with GNN
**Authors:** Liu, Ju et al. | **Year:** 2022  
**Venue:** ACM TODAES  
**Paper:** https://dl.acm.org/doi/10.1145/3555804

**Core Idea:**  
Variational GCN encodes circuit connectivity into a latent distribution. Generative decoder samples placements from this distribution. Uses spectral clustering for hierarchical netlist coarsening. 25% faster and 4% lower wirelength than standalone analytical placers.

**Why it matters:**  
GNN-based **initialization** before analytical refinement is a powerful hybrid. The variational encoder can capture complex connectivity patterns in the 7K–16K nets of IBM benchmarks, providing a much better starting point than random initialization.

---

### GRPlace (Competition Entry, Score 1.4017)
**Leaderboard rank:** 4th (as of April 2026)

**Description:**  
Uses GNNs to predict macro positions, trained to minimize proxy cost or wirelength/congestion. Competitive with both analytical methods and RL-based approaches. The 4% gap vs. RePlAce baseline shows GNNs are viable here.

**Why it matters:**  
The closest publicly documented competitor to study. Likely a GNN encoder → position predictor → legalization → refinement pipeline. Getting close to this score (~1.40) is a realistic target.

---

## 5. Force-Directed Placement

---

### Kraftwerk2 — Accurate Force-Directed Quadratic Placement
**Authors:** Peter Spindler, Ulf Schlichtmann, Frank M. Johannes | **Year:** 2008  
**Venue:** TCAD  
**Paper:** https://dl.acm.org/doi/10.1109/TCAD.2008.925783

**Core Idea:**  
Force-directed placer using the "Bound2Bound" net model — accurately represents HPWL in the quadratic cost function. Separates hold forces (prevent overlap) and move forces (reduce wirelength). Converges to well-optimized HPWL.

**Why it matters:**  
Force-directed methods are fast and naturally handle mixed-size designs. Bound2Bound is a better quadratic approximation of HPWL than clique/star models. Can be implemented quickly in PyTorch with gradient descent on the quadratic wirelength.

---

### SimPL — Self-Contained Force-Directed Placement
**Authors:** Myung-Chul Kim, Dongjin Lee, Igor L. Markov | **Year:** 2010–2011  
**Venue:** ICCAD / TCAD  
**Paper:** https://web.eecs.umich.edu/~imarkov/pubs/jour/tcad11-simpl.pdf

**Core Idea:**  
Maintains converging lower-bound (unconstrained QP) and upper-bound (spread) placements. Uses preconditioned conjugate gradients for QP solving. Outperforms mPL6, NTUplace3, FastPlace3, APlace2 by 2% HPWL at 6.4× faster runtime.

**Why it matters:**  
SimPL's lower/upper bound convergence framework is an elegant alternative to Nesterov gradient descent. The conjugate gradient solver is numerically stable and relatively easy to implement in numpy/PyTorch.

---

## 6. Legalization / Overlap Removal

---

### Abacus — Fast Legalization with Minimal Movement
**Authors:** Peter Spindler, Ulf Schlichtmann | **Year:** 2008  
**Venue:** ISPD  
**Paper:** https://www.semanticscholar.org/paper/Abacus:-fast-legalization-of-standard-cell-circuits-Spindler-Schlichtmann/b7c0656875460a88616342fa9ab55da9496bd22f

**Core Idea:**  
Sorts cells by position and legalizes one at a time using dynamic programming to minimize total displacement. 30% lower average movement than Tetris, only 7% additional runtime overhead.

**Why it matters:**  
After global placement, macros must be legalized to zero overlap (hard competition requirement). Abacus is the gold standard for minimal-displacement legalization. IncreMacro uses a constraint-graph LP extension of this idea.

---

### Tetris Legalization
**Authors:** Various | **Year:** ~1997, extended through 2018

**Core Idea:**  
Greedy row-by-row legalization — places each macro at the closest legal site in the current row. Extended versions handle obstacles and irregular macro sizes.

**Why it matters:**  
Tetris is faster than Abacus but produces higher displacement. Very useful as a **fast inner-loop legalizer** when you need to evaluate thousands of candidate placements in SA or BBO loops. Use Abacus for final legalization.

---

### Constraint-Graph LP Legalization (IncreMacro)
**Authors:** Yuan Pu et al. (CUHK) | **Year:** 2024  
**Venue:** ISPD

**Core Idea:**  
Builds a constraint graph between pairs of macros that could potentially overlap, then solves an LP to find minimum-displacement positions that satisfy all non-overlap constraints simultaneously.

**Why it matters:**  
Guaranteed zero overlaps with minimum displacement. More precise than greedy Tetris/Abacus for macro-sized blocks. The constraint graph approach is also used in B*-tree-based SA to check placement validity.

---

## 7. Congestion-Aware Placement

---

### RUDY — Rectangular Uniform Wire Density
**Authors:** Peter Spindler, Frank M. Johannes | **Year:** 2007  
**Venue:** DATE  
**Paper:** https://past.date-conference.com/proceedings-archive/2007/DATE07/PDFFILES/08.7_1.PDF

**Core Idea:**  
Estimates routing demand per routing tile by distributing each net's wire uniformly over its bounding box. Independent of specific router models. Formula: for each net with bounding box (W×H), add W×H routing demand uniformly distributed.

**Why it matters:**  
**RUDY is the congestion metric in the proxy cost (×0.5)**. Understanding RUDY allows directly computing gradients with respect to macro positions and incorporating them into an analytical placer. DREAMPlace uses RUDY-based congestion gradients internally.

---

## 8. Wirelength Models

---

### LSE (Log-Sum-Exp) and WA (Weighted Average) Models
**Introduced in:** APlace (2005), ePlace (2014), RePlAce (2018)

**Core Idea:**  
Two smooth, differentiable approximations of HPWL:

- **LSE:** `Ŵ = γ · ln(Σ exp(xᵢ/γ))` — overestimates HPWL, shrinks as γ→0
- **WA:** weights each pin by `exp(xᵢ/γ) / Σ exp(xₖ/γ)` — smaller error than LSE for large nets

Both use a smoothing factor γ that decreases across iterations (coarse-to-fine).

**Why it matters:**  
The wirelength component of the proxy cost (1.0 × WL) requires a differentiable approximation for gradient-based optimization. **WA is universally preferred** over LSE in modern analytical placers. Smaller γ → more accurate but less smooth gradient.

---

### Moreau Envelope as WL Model
See [Section 1](#moreau-envelope-wirelength-model). Drop-in replacement for WA with 5.4% better HPWL.

---

## 9. Macro-Specific Representations

---

### B*-Tree
**Authors:** Yun-Chih Chang, Yao-Wen Chang et al. | **Year:** 2000  
**Venue:** DAC  
**Paper:** https://websrv.cecs.uci.edu/~papers/compendium94-03/papers/2000/dac00/pdffiles/27_1.pdf

**Core Idea:**  
Encodes non-slicing floorplans as ordered binary trees where each node is a macro. 1-to-1 correspondence between admissible placements and B*-trees (no redundancy). Tree perturbation operations (rotate, insert, delete) take O(1). SA inner loops are fast.

**Why it matters:**  
B*-trees are the representation used by FastSA and many SA-based macro placers. If you build an SA solution, use B*-trees for efficient perturbations.

---

### Sequence Pair
**Authors:** Sugiyama et al. | **Year:** 1996  
**Venue:** ICCAD

**Core Idea:**  
Encodes macro relative positions as two permutation sequences. Any valid non-overlapping floorplan corresponds to a unique sequence pair. SA perturbation = simple swaps/insertions in the permutations.

**Why it matters:**  
Simpler to implement than B*-tree. O(n²) area evaluation (vs. O(n) for B*-tree) but easier to code correctly. Good starting point for an SA solver.

---

### WireMask-BBO — Wire-Mask-Guided Black-Box Optimization
**Authors:** Yunqi Shi, Ke Xue, Song Lei, Chao Qian (LAMDA, NJU) | **Year:** 2023  
**Venue:** NeurIPS  
**Paper:** https://proceedings.neurips.cc/paper_files/paper/2023/hash/15d6717f8bb33b3a74df26ce1eee0b9a-Abstract-Conference.html  
**Code:** https://github.com/lamda-bbo/WireMask-BBO

**Core Idea:**  
Uses a wire-mask heuristic to evaluate candidate placements efficiently. The wire-mask tracks HPWL increment per grid cell for fast placement scoring. Works with any black-box optimizer (evolutionary algorithms, Bayesian optimization, CMA-ES, etc.). Represents solutions as **continuous macro coordinates** (not grid-based).

**Why it matters:**  
Won or placed highly in multiple ISPD/ICCAD contests. The BBO framework is **optimizer-agnostic** — you can plug in CMA-ES, Bayesian Optimization, or gradient-free evolutionary methods and the wire-mask gives efficient evaluations. Very practical for this competition.

---

## 10. Multilevel / Hierarchical Methods

---

### NTUplace3 — Analytical Placer for Large-Scale Mixed-Size Designs
**Authors:** Tung-Chieh Chen et al. (NTU) | **Year:** 2008  
**Venue:** TCAD  
**Paper:** https://cc.ee.ntu.edu.tw/~ywchang/Papers/tcad08-NTUplace.pdf

**Core Idea:**  
Three-stage flow: global placement → legalization → detailed placement. Global placement uses multilevel quadratic programming with net clustering and bell-shaped density smoothing. Handles large mixed-size designs.

**Why it matters:**  
NTUplace3 pioneered the **bell-shaped density function** for mixed-size placement — used in both APlace3 and DREAMPlace. The three-stage flow is the standard industry template.

---

### hMETIS — Multilevel Hypergraph Partitioning
**Authors:** George Karypis et al. | **Year:** 1999  
**Paper:** Classic, widely cited

**Core Idea:**  
Iterative coarsening of the hypergraph, top-level partitioning, then refinement during uncoarsening using FM local search. Recursive bisection produces hierarchical placements.

**Why it matters:**  
Macro clustering (grouping related macros before placement) reduces search space. IBM benchmarks have 7K–16K nets — hypergraph partitioning identifies which macros should be co-located. Can be used to generate good starting positions before any optimization.

---

## 11. Hybrid & Escape-from-Local-Optima

---

### Hybro — Escaping Local Optima in Global Placement
**Authors:** Yunqi Shi et al. (LAMDA) | **Year:** 2024  
**Venue:** arXiv  
**Paper:** https://arxiv.org/abs/2402.18311

**Core Idea:**  
Iterative escape-and-refine loop:
1. Run DREAMPlace until convergence
2. Apply perturbation to escape local optimum:
   - **Hybro-Shuffle:** random macro shuffling
   - **Hybro-WireMask:** wire-mask-guided perturbation (more targeted)
3. Re-run DREAMPlace
4. Repeat 5+ times

Achieves the best HPWL + timing + congestion on ISPD2005 and ICCAD2015.

**Why it matters:**  
**This is directly applicable to DREAMPlace++ in this competition.** The iterative escape framework is a high-leverage improvement. Hybro-WireMask specifically is smart about which macros to perturb, making each escape more likely to find a better basin.

---

### AutoDMP with MOBO (Bayesian Hyperparameter Tuning)
See [Section 1, AutoDMP](#autodmp--automated-dreamplace-based-macro-placement).  

**Why it matters:**  
With 17 benchmarks, manually tuning density penalty, WA smoothing, and learning rate for each is impractical. MOBO automates this. The same idea applies to any parametric solver.

---

## 12. Benchmarking & Evaluation

---

### TILOS MacroPlacement Repository
**Authors:** TILOS AI Institute | **Year:** 2022–2024  
**Code:** https://github.com/TILOS-AI-Institute/MacroPlacement

Contains:
- All 17 IBM ICCAD04 benchmarks in DEF/LEF format
- Exact proxy cost evaluator (PlacementCost) used in this competition
- SA and RePlAce baseline reproductions
- Circuit Training (Google RL) baseline

**This is the ground truth — the competition evaluator wraps this code.**

---

### ChiPBench — Benchmarking AI Placement End-to-End
**Authors:** MIRA Lab (USTC) | **Year:** 2024  
**Venue:** ICLR  
**Paper:** https://proceedings.iclr.cc/paper_files/paper/2024/file/464917b6103e074e1f9df7a2bf3bf6ba-Paper-Conference.pdf  
**Code:** https://github.com/MIRALab-USTC/ChiPBench

**Core Idea:**  
Evaluates six AI placement methods end-to-end through the full PnR flow (synthesis → placement → routing). Shows that proxy metrics (like the competition's Tier 1 score) frequently **misalign** with real PPA outcomes (WNS, TNS, Area).

**Why it matters:**  
The Grand Prize ($20K) is based on Tier 2 OpenROAD results. A method that scores well on proxy cost might not win the Grand Prize. This benchmark gives insight into the gap — methods that minimize congestion tend to do best on real PnR metrics.

---

## 13. Strategy Recommendations

Based on the literature, here are concrete paths to a competitive solution:

---

### Path A — DREAMPlace-Based (Highest ceiling, hardest to implement)
**Target score: ~1.36–1.38**

1. Install DREAMPlace with CUDA support
2. Run on IBM benchmarks as-is → likely ~1.40–1.45
3. Apply **Hybro-WireMask** perturbation loop: DREAMPlace → perturb → re-run
4. Swap WA wirelength model for **Moreau envelope** (+5% WL improvement)
5. Add **Barzilai-Borwein second-order update** (+6–33% HPWL improvement)
6. Apply **IncreMacro refinement** as post-processing
7. Use **AutoDMP MOBO** to tune hyperparameters across benchmarks

**References:** DREAMPlace, Hybro, Moreau WL, ICCAD2023 2nd-order, IncreMacro

---

### Path B — Force-Directed + SA Refinement (Moderate difficulty)
**Target score: ~1.42–1.48**

1. Build force-directed placer: spring forces along net edges, repulsion between overlapping macros
2. Implement **WA wirelength gradient** for attractive forces
3. Implement **RUDY congestion gradient** for repulsion from congested areas
4. After convergence, apply **SA refinement** with Swap + Shift moves to escape local optima
5. Legalize with **Abacus** at the end

**References:** Kraftwerk2/SimPL, RUDY, FastSA, Abacus

---

### Path C — WireMask-BBO (Easiest to get running quickly)
**Target score: ~1.42–1.50**

1. Implement the **wiremask evaluation** (fast HPWL estimator per grid cell)
2. Plug in **CMA-ES** or another black-box optimizer from the `cmaes` Python package
3. Use the LAMDA group's open-source WireMask-BBO code as reference
4. Add legalization via Tetris for fast inner-loop evaluation

**References:** WireMask-BBO, CMA-ES

---

### Path D — Analytical from Scratch in PyTorch (Good learning, GPU-native)
**Target score: ~1.42–1.46**

1. Implement **WA wirelength loss** in PyTorch (differentiable)
2. Implement **bell-shaped density penalty** via 2D histogram + FFT convolution
3. Run Adam/Nesterov gradient descent on macro positions
4. Project gradients to keep macros inside canvas bounds
5. After convergence, legalize with custom overlap-removal (push apart overlapping pairs)
6. Optionally add **RUDY congestion gradient**

This approach gives you full control over the objective and is fully GPU-acceleratable.

**References:** ePlace, DREAMPlace, NTUplace3 (bell-shaped density)

---

### Key Insight: Congestion is Undertapped

Most entries focus on wirelength. The proxy cost weights **congestion at 0.5×** and **density at 0.5×** — almost as much as wirelength (1.0×). The diffusion model paper showed 35% congestion reduction. Adding an explicit RUDY-based congestion term to your gradient could give significant score improvement that others are leaving on the table.

---

### Proxy Score vs. Grand Prize Alignment

If you're aiming for the **Grand Prize ($20K)** rather than just Proxy score, prioritize:
- **Congestion minimization** (most correlated with routing overflow)
- **Timing-aware net weighting** (larger weights for critical nets — DREAMPlace 4.0)
- Avoid overfitting to proxy cost (ChiPBench shows proxy/PnR misalignment)

---

*Sources: 40+ papers surveyed — see inline links for full references.*
