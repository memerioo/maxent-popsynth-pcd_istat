
# maxent-popsynth-pcd

> ⚠️ **Work in progress** —This is a FORK for applying the cited paper to ISTAT dataset.

# PCD_ISTAT — Reliability-Weighted MaxEnt Population Synthesis for Bologna

Synthetic population generation for the Comune di Bologna from real,
mutually inconsistent ISTAT aggregate tables, using the GibbsPCDSolver
(Persistent Contrastive Divergence MaxEnt) of Degli Esposti (2026),
extended with a **geographic reliability-weighting** scheme.

ISTAT publishes tables at five geographic levels (Comune di Bologna,
Città metropolitana, Emilia-Romagna, Nord-Est, Italia) that describe
different reference populations and do not agree. Each constraint is
tagged with its geographic source and given a reliability weight
(BO = 1.00 → Italy = 0.15); the weights are applied as per-coordinate
learning-rate multipliers *after* Adam normalisation, so that trusted
local constraints win the inevitable compromise on conflicting targets.

## Layout

```
src/
  constraint_set.py        Core ConstraintSet data structure
  gibbs_pcd_solver.py      GibbsPCDSolver (+ reliability weights, Numba kernel)
  solvers.py               Exact L-BFGS MaxEnt and raking baselines (paper)
  generators.py            Wu / planted-exp-family benchmark generators (paper)
  evaluator.py             Benchmark evaluation helpers (paper)
  istat/
    attr_meta_ISTAT.py     K=34 attribute metadata, marginals and CPT tables
                           (geo-source tags live in the inline comments)
    geo_tagging.py         Parses geo tags; defines RELIABILITY weights
    preprocess_istat.py    CPT discovery, marginal reconciliation,
                           constraint-set builder (geo-tagged)
    mre_floor.py           Data-inconsistency MRE floor + stratified MRE
    diagnose_istat.py      Raw CPT ↔ marginal inconsistency scan
experiments/
  test_4_ISTAT.py          Main experiment: fit N=500K pool on 1,535 constraints;
                           saves population (.npy) and run history (.pkl)
  plot_diagnostics.py      All thesis figures from the saved run
  see_results.py           Print sample synthetic individuals
```

## Usage

```bash
pip install -r requirements.txt          # numba strongly recommended for K=34
python experiments/test_4_ISTAT.py       # several hours at N=500K
python experiments/plot_diagnostics.py   # figures → experiments/figures/
python experiments/see_results.py 10     # inspect synthetic Bolognesi
```

`plot_diagnostics.py` refuses to silently mix runs: it checks that the
saved history matches the constraint set implied by the current
`attr_meta_ISTAT.py` and warns loudly if not.

## Notes on the modelling choices

- Constraints of all arities are fitted: unary marginals, binary CPT
  joints, and the 20 empirical ternary CPTs (targets
  P(c | p1, p2) × P(p1, p2), where the parent joint comes from a binary
  CPT linking the parents when one exists, else the independence product
  of their marginals). `h_`-prefixed tables contribute only structural
  zeros (logically impossible combinations); their non-zero entries are
  placeholders and are ignored.
- Duplicate constraints (two CPTs implying the same joint probability)
  are deduplicated first-come-first-kept in table definition order.
- The reported MRE floor is a conservative lower bound computed from
  directly identified multi-source conflicts; the stratified MRE by
  geographic source is the primary evidence that residual error is
  data inconsistency rather than solver failure.

## References

- Degli Esposti, M. (2026). *Scalable Maximum Entropy Population
  Synthesis via Persistent Contrastive Divergence.* arXiv:2603.27312
- Pachet, F., Zucker, J.-D. (2026). *Maximum Entropy Relaxation of
  Multi-Way Cardinality Constraints for Synthetic Population Generation.*
- Tieleman, T. (2008). *Training Restricted Boltzmann Machines Using
  Approximations to the Likelihood Gradient.* ICML.
