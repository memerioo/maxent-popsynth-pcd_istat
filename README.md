# maxent-popsynth-pcd — ISTAT / Bologna

> ⚠️ **Work in progress.** This is a fork of the reference implementation of
> Degli Esposti (2026), extended and applied to real ISTAT data.

Synthetic population generation for the **Comune di Bologna** from real,
mutually inconsistent ISTAT aggregate tables, using the `GibbsPCDSolver`
(Persistent Contrastive Divergence MaxEnt), extended with:

1. **geographic reliability weighting** — so that trusted local tables win
   the inevitable compromise between contradictory sources;
2. **sentinel-block Metropolis–Hastings moves** — without which the sampler
   is *not ergodic* once logical impossibilities are enforced exactly;
3. **a rigorous lower bound on the achievable error** — a linear program over
   the local polytope, which separates data inconsistency from solver error.

---

## The problem in one paragraph

ISTAT publishes tables at five geographic levels (Comune di Bologna, Città
metropolitana, Emilia-Romagna, Nord-Est, Italia). They describe different
reference populations and **they do not agree** — conflicts reach 0.45 in
absolute probability. No joint distribution satisfies all published targets
at once, so every solver leaves residual error somewhere; the only question
is *where*. Each constraint is tagged with its geographic source and given a
reliability weight (BO = 1.00 → Italy = 0.15), applied as a per-coordinate
learning-rate multiplier **after** Adam normalisation.

> The placement matters. Scaling the gradient *before* Adam is silently
> cancelled by Adam's own per-coordinate normalisation — a 6.7× difference in
> gradient scale produced numerically identical parameters after 200 steps.

---

## Two things that are easy to get wrong

### Hard structural zeros break the sampler

22 of the 34 attributes carry a *sentinel* category (`NotWorker`,
`NotStudent`, `Under3yo`) meaning "this attribute does not apply". The `h_`
tables force those sentinels to be mutually consistent. Enforcing them
exactly (pinning λ = −30) removes every impossible individual — and
**partitions the state space**.

To turn a full-time worker into an inactive individual, eight attributes must
change *simultaneously*: every single-attribute first move lands on a
forbidden state (weight e⁻³⁰ or worse). Single-site Gibbs cannot do it. The
chain loses irreducibility and the block composition freezes at whatever the
initialisation produced:

| sweep | uniform init | on-target init |
|------:|-------------:|---------------:|
| 0  | 0.3297 | 0.7215 |
| 2  | 0.2087 | 0.7215 |
| 40 | **0.2087** | **0.7215** |

*Non-student share, identical parameters, 20,000 individuals. Target = 0.720.
Zero transitions in 40 sweeps. The answer is decided entirely by the
initialisation.* **No aggregate error metric reveals this** — the run just
plateaus at a value that looks like an ordinary compromise floor.

The fix (`src/istat/block_moves.py`) is a Metropolis–Hastings move that
transposes an **entire sentinel block** in one step, so the forbidden
intermediate states are never visited. Two details are essential:

- the Hastings ratio `q(b)/q(b')` **cannot** be dropped — donors are uniform
  over *individuals*, not over *block patterns*;
- the tempting symmetric alternative (*swapping* blocks between two
  individuals) is useless: a swap conserves the number of workers, which is
  precisely the quantity being fitted.

Per-block acceptance rates are logged every iteration, because a re-freeze
would otherwise be silent.

### The obvious error floor is a heuristic, not a bound

Comparing pairs of sources that state the same quantity misses three whole
families of conflict: multi-way conflicts, infeasibility created by the
structural rules, and the ~480 binary constraints no second source restates.
`src/istat/mre_floor_lp.py` instead minimises the weighted error over the
**local polytope** — all per-scope marginals that are non-negative,
normalised, consistent on overlaps, and zero on forbidden patterns. Since
every genuine joint distribution projects into that set, the LP optimum is a
*rigorous lower bound* on what any solver could achieve. It runs in a
fraction of a second and depends only on the constraint set, not on the run.

It turns worst-fitting constraints into **proven contradictions**. Example:
the three Bologna targets `P(CommuteInward) = 0.27`,
`P(work-commute = Inward) = 0.16`, `P(study-commute = Inward) = 0.11` satisfy
`0.16 + 0.11 = 0.27` exactly, and rule H45 makes the first the *union* of the
other two — so they are jointly attainable only if nobody ever commutes
inward for both work and study, an event MaxEnt necessarily gives positive
probability. Same source, no weighting can arbitrate it.

---

## Layout

```
src/
  constraint_set.py          Core ConstraintSet data structure
  gibbs_pcd_solver.py        GibbsPCDSolver: reliability weights, block moves,
                             cosine/plateau lr schedules, Numba kernel
  solvers.py                 Exact L-BFGS MaxEnt and raking baselines (paper)
  generators.py              Wu / planted-exp-family benchmark generators (paper)
  evaluator.py               Benchmark evaluation helpers (paper)
  istat/
    attr_meta_ISTAT.py       K=34 attributes, marginals, CPTs and h_ tables
                             (geo-source tags live in the inline comments)
    geo_tagging.py           Parses geo tags; defines RELIABILITY weights
    preprocess_istat.py      CPT discovery, reconciliation, constraint builder
    structural_blocks.py     Sentinel-block declarations; ancestral legal
                             pool initialiser; legality audit
    block_moves.py           MH block-toggle kernel (NumPy + Numba)
    mre_floor.py             Pairwise MRE floor + stratified MRE
    mre_floor_lp.py          LP lower bound over the local polytope
    diagnose_istat.py        Raw CPT ↔ marginal inconsistency scan
experiments/
  test_4_ISTAT.py            Main experiment: fit N=500K pool on m=1,698
                             constraints; saves population (.npy) + history (.pkl)
  test_blocks_smoke.py       FAST sanity check — run this first
  plot_diagnostics.py        All thesis figures from the saved run
  see_results.py             Print sample synthetic individuals + legality audit
  repair_legality.py         Repairs the few illegal individuals
```

---

## Usage

```bash
pip install -r requirements.txt          # numba strongly recommended for K=34

# 1. Verify the sampler is ergodic (~1 min without the fit)
python experiments/test_blocks_smoke.py --no-fit

# 2. Full run
export NUMBA_NUM_THREADS=8               # required for bit-reproducibility
python experiments/test_4_ISTAT.py       # several hours at N=500K

# 3. Figures and inspection
python experiments/repair_legality.py    #repair the illegal individuals
python experiments/plot_diagnostics.py   # → experiments/figures/
python experiments/see_results.py 10     # inspect synthetic Bolognesi
```

`test_blocks_smoke.py` prints the freeze table above and then the same
experiment with block moves enabled. If the two initialisations converge to
*different* frozen values, the sampler is not ergodic — which is the point it
is there to demonstrate.

The LP floor needs no fitted model and can be run on its own:

```bash
python src/istat/mre_floor_lp.py --tighten
```

---

## Figures

`plot_diagnostics.py` writes six figures to `experiments/figures/` as both
`.pdf` and `.png`.

### Convergence

![convergence](experiments/figures/convergence.pdf)

Relative error (top: linear, log) and absolute error (bottom: linear, log).
The **log-MAE panel is the diagnostic one**: sampling noise would show as a
band of constant relative width that narrows with pool size, whereas the
observed band widens over a middle stretch and is *wider* at N = 500,000 than
at N = 30,000 — the signature of an Adam limit cycle, not of noise. Ternary
constraints overlap the binary and unary ones heavily, leaving the
coordinates of λ correlated; Adam normalises per coordinate but does nothing
about correlation, so the stiff directions oscillate.

The cosine schedule shrinks the step size at every iteration, so the cycle
never gets a flat high-η stretch in which to grow.

### Where the error lives

![stratified](experiments/figures/stratified_mre.pdf)
![source pie](experiments/figures/source_pie.pdf)

Error stratified by geographic source and by constraint arity, with
contributions that are **exactly additive** (Cg = ng·MREg / ntotal, so Σ Cg =
MRE_global) — "Italy accounts for 61% of the error" is literal arithmetic,
not rhetoric. The pie compares each source's share of the *error* against its
share of the *constraint set*.

### The rigorous floor

![lp floor](experiments/figures/mre_floor_lp.pdf)

Left: observed error against the LP bound, per source, annotated with the
fraction that is *forced by the data*. Right: the constraints carrying the
largest unavoidable error, showing the published target against the closest
value any distribution can reach — the gap cannot be closed by any solver.

### Diversity and fit

![diversity](experiments/figures/diversity.pdf)
![alpha scatter](experiments/figures/alpha_scatter.pdf)

N_eff = N exactly and Gini = 0 by construction, since PCD samples individuals
rather than reweighting them. Under identical conditions raking collapses to
N_eff ≈ 0.01 N with a weight Gini of 0.95.

---

## Results (N = 500,000)

<!-- UPDATE these from the final run -->

| Metric | Value |
|---|---|
| Global MRE (1,060 valid constraints) | 0.26189 |
| Global wMRE (selection metric) | 0.18520 |
| Global MAE | 0.00599 |
| Mean \|diff\| over the 135 unary categories | 0.00664 |
| Logically impossible individuals | 2 / 500,000 |
| Distinct profiles | 472,482  (94.5% of pool) |
| N_eff / N, Gini | 100%, 0.000 |

Error by geographic source — monotone in distance from Bologna:

| Source | n | W | MRE | Contrib | Share |
|---|---:|---:|---:|---:|---:|
| BO | 146 | 1.00 | **0.0514** | 0.0071 | 2.7% |
| PBO | 65 | 0.85 | 0.1837 | 0.0113 | 4.3% |
| EmiliaR | 171 | 0.50 | 0.2321 | 0.0374 | 14.3% |
| NorthEast | 202 | 0.30 | 0.2404 | 0.0458 | 17.5% |
| Italy | 476 | 0.15 | 0.3570 | 0.1603 | 61.2% |

The solver sees the sources only through their weights, so a gradient
organised along the geographic hierarchy is not something solver error would
produce. **≥ 24% of the observed MRE is provably unavoidable.**

---

## Notes on the modelling choices

- **Constraint set**: 34 unary marginals + 162 CPTs (140 binary, 22 ternary),
  of which 79 are empirical and 83 encode logical impossibility. Expanded to
  **m = 1,698** atomic constraints: 1,060 empirical above the 10⁻³ validity
  threshold (135 unary / 544 binary / 381 ternary), 611 structural zeros, 27
  empirical below threshold and excluded from the metrics.
- **Ternary targets** are built as P(c | p₁, p₂) × P(p₁, p₂), with the parent
  joint from a binary CPT linking the parents when one exists, else the
  independence product. The fallback injects an approximation into the
  targets themselves where the parents are in fact correlated.
- **`h_` tables** contribute only structural zeros; their non-zero entries are
  placeholders and are ignored everywhere.
- **Duplicate constraints** (two CPTs implying the same joint) are
  deduplicated first-come-first-kept in table definition order.
- **Structural parameters are pinned, not learned.** A zero target cannot be
  reached by gradient descent: the gradient *is* α̂, so the Adam ratio
  collapses as α̂ falls, and the travel budget would only reach λ ≈ −6 over a
  realistic run — leaving thousands of impossible individuals at N = 500K.
  They are pinned at λ = −30 and frozen. Because the pin is finite, forbidden
  states retain probability ~e⁻³⁰, so the delivered pool is **audited**, not
  assumed clean.
- **Reliability weights are a proxy.** 1/W is meant to be the noise of a table
  *as a measurement of Bologna*, which depends on how much the quantity varies
  geographically — not only on the publishing level. The age structure of
  university enrolment barely varies across Italian regions, so that table is
  weighted above its national provenance.

## Reproducibility

A single seed drives every random stream: pool initialisation, Gibbs sweeps,
the donor draws and acceptance tests of the block move, and the per-thread
RNG states of the Numba kernel. Runs are bit-reproducible **for a fixed
thread count** — set `NUMBA_NUM_THREADS`, which is recorded in the history
file along with λ, the full hyperparameter set, and the constraint patterns
needed to interpret λ.

`plot_diagnostics.py` will not silently mix runs: it matches the pool file
against `N_pool` in the saved history and warns loudly if it has to fall back
to a different one.

## References

- Degli Esposti, M. (2026). *Scalable Maximum Entropy Population Synthesis via
  Persistent Contrastive Divergence.* arXiv:2603.27312
- Pachet, F., Zucker, J.-D. (2026). *Maximum Entropy Relaxation of Multi-Way
  Cardinality Constraints for Synthetic Population Generation.*
- Tieleman, T. (2008). *Training Restricted Boltzmann Machines Using
  Approximations to the Likelihood Gradient.* ICML.
- Wainwright, M. J., Jordan, M. I. (2008). *Graphical Models, Exponential
  Families, and Variational Inference.* FnT ML — for the local polytope
  relaxation used by the LP floor.