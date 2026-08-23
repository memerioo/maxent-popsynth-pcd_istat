"""
test_4_ISTAT.py
---------------
Professional testing script for the GibbsPCDSolver.
Dynamically imports mathematically reconciled marginals
from the ISTAT data pipeline.

CHANGED (sentinel-block fix)
============================
Two additions, both required by the hard structural zeros:

  * `blocks=`     -- enables the Metropolis-Hastings block-toggle move.
                     Without it, single-site Gibbs cannot move an individual
                     between the active and inactive basin of a sentinel
                     block (it would need up to eight simultaneous flips, and
                     every intermediate state weighs e^-30), so the block
                     composition stays frozen at its initial value forever
                     and P(employment) / P(StudentStat) can never be fitted.

  * `init_pool=`  -- a structurally legal pool whose unary marginals already
                     sit on the published targets, instead of uniform noise.
                     Saves the block moves from walking the whole composition
                     up from random initialisation.

Verified on this constraint set: with hard zeros and NO block moves, the
non-student share freezes at 0.209 within two sweeps (target 0.720) and does
not move again in 40 sweeps, whatever lambda does.
"""

import sys
import os
import numpy as np

# -- Path Setup --------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from gibbs_pcd_solver import GibbsPCDSolver
from istat.preprocess_istat import build_constraint_set, build_constraint_weights
from istat.geo_tagging import parse_source_tags
from istat.mre_floor import print_stratified_mre
from istat.structural_blocks import (resolve_blocks, ancestral_init_pool,
                                     legality_report)

# -- Data Ingestion ----------------------------------------------------------
try:
    from istat.attr_meta_ISTAT import marginals as CLEAN_MARGINALS
except ImportError:
    print("\nERROR: Could not import 'src/istat/attr_meta_ISTAT.py' (marginals).")
    print("Make sure you run this script from the project root:")
    print("    python experiments/test_4_ISTAT.py\n")
    sys.exit(1)

from istat.attr_meta_ISTAT import ATTR_NAMES_SYNTH, ATTR_META


def marginal_check(pool, marginals, title="UNARY MARGINAL CHECK"):
    """Pool frequency vs published target, worst 20 categories."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    rows = []
    for attr, dist in marginals.items():
        if attr not in ATTR_NAMES_SYNTH:
            continue
        col = ATTR_NAMES_SYNTH.index(attr)
        for val, tgt in dist.items():
            vi = ATTR_META[attr]['val_to_int'].get(val)
            if vi is None:
                continue
            f = float((pool[:, col] == vi).mean())
            rows.append((abs(f - tgt), attr, val, f, tgt, f - tgt))
    if not rows:
        print("  (no comparable marginals)")
        return float('nan')
    rows.sort(reverse=True)
    print(f"  {'attribute':<22}{'value':<18}{'pool':>8}{'target':>8}{'diff':>9}")
    print(f"  {'-' * 66}")
    for ad, attr, val, f, tgt, d in rows[:20]:
        flag = "!" if ad > 0.05 else " "
        print(f"  {flag} {attr:<20}{val:<18}{f:>8.4f}{tgt:>8.4f}{d:>+9.4f}")
    mean_abs = float(np.mean([r[0] for r in rows]))
    print(f"  {'-' * 66}")
    print(f"  mean |diff| over {len(rows)} categories: {mean_abs:.5f}"
          f"   (worst {rows[0][0]:.4f})")
    return mean_abs


def run():
    print("=" * 64)
    print("  STEP 1 / 3  --  Building ConstraintSet (geo-tagged)")
    print("=" * 64)

    attr_meta_path = os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py')
    table_source, marginal_source = parse_source_tags(attr_meta_path)

    cs, sources = build_constraint_set(CLEAN_MARGINALS, table_source, marginal_source)
    weights = build_constraint_weights(sources)

    # -- Model Hyperparameters ------------------------------------------------
    SEED            = 42
    N_pool          = 500_000
    n_outer         = 2600
    n_gibbs_sweeps  = 12
    # Learning-rate schedule.
    #   'cosine'  : lr decreases smoothly from `lr` to `lr_min` across the
    #               whole run. The oscillation amplitude scales with the
    #               CURRENT lr, so a monotonically shrinking lr never lets the
    #               Adam limit cycle grow, unlike a plateau schedule that sits
    #               flat at `lr` for hundreds of iterations first.
    #               lr_patience / lr_decay are IGNORED in this mode.
    #   'plateau' : reactive reduce-on-plateau (uses lr_patience / lr_decay).
    lr_schedule     = 'cosine'
    lr              = 0.005      # start of the cosine
    lr_min          = 0.0005     # END of the cosine -- must be set explicitly,
                                 # otherwise it defaults to lr/20 = 0.0002
    lr_patience     = 50         # ignored when lr_schedule == 'cosine'
    lr_decay        = 0.7        # ignored when lr_schedule == 'cosine'

    # Trailing-mean window for the selection metric. Still used in cosine mode
    # for best-snapshot selection (the minimum of a noisy sequence is an
    # upward-biased estimate of the delivered model).
    plateau_smooth  = 60

    # Early stopping is DISABLED for cosine: the schedule is defined against a
    # fixed horizon n_outer, and its low-lr fine-tuning phase only arrives in
    # the last ~20% of the run. Stopping early quits mid-schedule and throws
    # that away. Set tol > 0 only when using 'plateau'.
    tol             = 0.0
    window          = 100

    eps             = 1e-4
    gamma           = 0.001
    verbose_every   = 20

    # Fraction of the pool offered a block toggle per outer iteration.
    # At 0.2 every individual still gets several hundred toggle attempts over
    # the run, which is ample for the composition to equilibrate.
    BLOCK_MOVE_FRAC = 0.2

    print("\n" + "=" * 64)
    print("  STEP 2 / 3  --  Sentinel blocks + legal on-target pool")
    print("=" * 64)

    blocks = resolve_blocks(ATTR_NAMES_SYNTH, ATTR_META, names=("work", "study"))
    for b in blocks:
        print(f"  block {b['name']:<6} driver={b['driver']:<14} "
              f"{len(b['attrs_idx'])} attributes")

    init_pool = ancestral_init_pool(N_pool, CLEAN_MARGINALS,
                                    ATTR_NAMES_SYNTH, ATTR_META, seed=SEED)
    print("\n  Ancestral init pool:")
    print(legality_report(cs, init_pool, ATTR_NAMES_SYNTH, ATTR_META, top=5))
    print("  (residual violations are repaired by the first ~10 Gibbs sweeps;")
    print("   they are single-site fixable and do not move block composition)")

    print("\n" + "=" * 64)
    print("  STEP 3 / 3  --  GibbsPCDSolver (reliability-weighted + block moves)")
    print("=" * 64)
    print(f"Pool size       : {N_pool}")
    print(f"Outer iters     : up to {n_outer}")
    if lr_schedule == 'cosine':
        print(f"Learning rate   : {lr} -> {lr_min}  (COSINE over {n_outer} iters)")
    else:
        print(f"Learning rate   : {lr}  (plateau decay x{lr_decay}, "
              f"patience={lr_patience}, min={lr_min})")
    print(f"Selection smooth: {plateau_smooth} iters (trailing mean)")
    print(f"Early stop tol  : {tol}" + ("   (DISABLED)" if tol == 0 else ""))
    print(f"Gibbs sweeps/it : {n_gibbs_sweeps}")
    print(f"Block move frac : {BLOCK_MOVE_FRAC}")
    print(f"Seed            : {SEED}\n")

    solver = GibbsPCDSolver(cs, use_numba=True, weights=weights,
                            sources=sources, blocks=blocks)
    solver.fit(
        N_pool          = N_pool,
        n_outer         = n_outer,
        n_gibbs_sweeps  = n_gibbs_sweeps,
        lr              = lr,
        lr_schedule     = lr_schedule,
        lr_patience     = lr_patience,
        lr_decay        = lr_decay,
        lr_min          = lr_min,
        plateau_smooth  = plateau_smooth,
        eps             = eps,
        gamma           = gamma,
        seed            = SEED,
        tol             = tol,
        window          = window,
        init_pool       = init_pool,
        block_move_frac = BLOCK_MOVE_FRAC,
        verbose_every   = verbose_every,
    )

    # -- Summary --------------------------------------------------------------
    print(f"\n{'-' * 64}")
    print(f"  BEST SNAPSHOT      : iteration {solver.best_iter} "
          f"(of {solver.n_iters} run)")
    print(f"  MRE  @ best        : {solver.final_mre:.5f}")
    print(f"  wMRE @ best        : {solver.final_weighted_mre:.5f}  "
          f"(selection metric: {solver.selection_metric})")
    print(f"  MAE  @ best        : {solver.final_mae:.5f}")
    print(f"  Early stop         : {solver.stopped_early}")
    print(f"  Fit time           : {solver.fit_time:.1f}s")
    if solver.block_accept_:
        print(f"  Block accept rate  : {np.mean(solver.block_accept_):.3f} "
              f"(mean over {len(solver.block_accept_)} iters)")
        if solver.block_accept_by_name_:
            names = solver.block_accept_by_name_[0].keys()
            for nm in names:
                seq = [d[nm] for d in solver.block_accept_by_name_ if nm in d]
                lo = min(seq[len(seq)//4:]) if len(seq) > 4 else min(seq)
                flag = "  <-- FROZEN?" if lo < 0.05 else ""
                print(f"     {nm:<10} mean={np.mean(seq):.3f}  "
                      f"min(after warm-up)={lo:.3f}{flag}")
    print(f"{'-' * 64}")

    # -- Legality of the delivered population ---------------------------------
    print("\n" + "=" * 64)
    print("  STRUCTURAL LEGALITY OF THE DELIVERED POOL")
    print("=" * 64)
    print(legality_report(cs, solver.pool_, ATTR_NAMES_SYNTH, ATTR_META, top=8))

    # -- Unary marginals: the quantity the freeze used to destroy -------------
    marginal_check(solver.pool_, CLEAN_MARGINALS)

    print_stratified_mre(solver)

    if solver.pool_ is not None:
        out_path = os.path.join(ROOT_DIR, 'experiments',
                                f"test_4_ISTAT_pop{N_pool}.npy")
        np.save(out_path, solver.pool_)
        print(f"\n[+] Population successfully saved to: {out_path}")

    # -- Save run history -----------------------------------------------------
    import pickle
    hist_path = os.path.join(ROOT_DIR, 'experiments', 'test_4_ISTAT_history.pkl')
    run_record = {
        # lambdas are the fitted model. Without them the run cannot be
        # reproduced, resumed, or checked for equilibration (freezing lambda
        # and running extra Gibbs sweeps to confirm the pool is a genuine
        # sample of p_lambda) without a full refit. Required for the
        # reproducibility checklist of Appendix A.8.
        'lambdas':            solver.lambdas,
        'history':            solver.history,
        'sources':            solver.sources,
        'weights':            solver.weights,
        'alphas':             solver.alphas,
        'final_mre':          solver.final_mre,
        'final_weighted_mre': solver.final_weighted_mre,
        'final_mae':          solver.final_mae,
        'fit_time':           solver.fit_time,
        'n_iters':            solver.n_iters,
        'stopped_early':      solver.stopped_early,
        'selection_metric':   solver.selection_metric,
        'best_iter':          solver.best_iter,
        'seed':               SEED,
        'N_pool':             N_pool,
        'block_accept':         solver.block_accept_,
        'block_accept_by_name': solver.block_accept_by_name_,
        'block_move_frac':      BLOCK_MOVE_FRAC,
        'block_names':          [b['name'] for b in blocks],
        'n_gibbs_sweeps':       n_gibbs_sweeps,
        'lr':                   lr,
        'lr_schedule':          lr_schedule,
        'plateau_smooth':       plateau_smooth,
        'lr_patience':          lr_patience,
        'lr_decay':             lr_decay,
        'lr_min':               lr_min,
        'gamma':                gamma,
        'eps':                  eps,
        'tol':                  tol,
        'window':               window,
        'domain_sizes':         cs.domain_sizes,
        'attrs_list':           cs.attrs_list,
        'vals_list':            cs.vals_list,
        'numba_threads':        os.environ.get('NUMBA_NUM_THREADS', 'unset'),
    }
    with open(hist_path, 'wb') as f:
        pickle.dump(run_record, f)
    print(f"[+] Run history saved to: {hist_path}")

    # -- Diagnostics ----------------------------------------------------------
    print("\n" + "=" * 64)
    print("  TOP 15 UNRESOLVED CONSTRAINTS")
    print("=" * 64)

    alpha_hat = solver._estimate_expectations(solver.pool_)
    errors = np.abs(alpha_hat - solver.alphas)
    worst_idx = np.argsort(errors)[::-1][:15]

    for j in worst_idx:
        target = solver.alphas[j]
        actual = alpha_hat[j]
        diff   = errors[j]
        src    = solver.sources[j] if solver.sources else "?"
        attrs  = [ATTR_NAMES_SYNTH[a] for a in solver.cs.attrs_list[j]]
        vals   = [ATTR_META[attrs[i]]['vals'][v]
                  for i, v in enumerate(solver.cs.vals_list[j])]
        cstr   = ", ".join(f"{a}={v}" for a, v in zip(attrs, vals))
        print(f"Diff: {diff:.4f}  |  Target: {target:.4f}  |  Actual: {actual:.4f}"
              f"  |  [{src}]  {cstr}")

    # -- MRE Floor Analysis ---------------------------------------------------
    print("\n" + "=" * 64)
    print("  MRE FLOOR ANALYSIS")
    print("=" * 64)
    try:
        from istat.preprocess_istat import discover_cpts
        import istat.attr_meta_ISTAT as _meta
        from istat.attr_meta_ISTAT import marginals
        from istat.mre_floor import (compute_full_mre_floor,
                                     print_full_mre_floor_summary)

        cpts   = discover_cpts(_meta)
        result = compute_full_mre_floor(
            solver.cs, cpts, marginals, solver.alphas,
            table_source, marginal_source,
            ATTR_NAMES_SYNTH, ATTR_META)
        print_full_mre_floor_summary(solver.final_mre, result,
                                     solver.final_weighted_mre)

    except Exception as e:
        print(f"  [!] Floor analysis failed: {e}")
        import traceback; traceback.print_exc()

    # -- LP MRE floor: rigorous lower bound -----------------------------------
    print("\n" + "=" * 64)
    print("  LP MRE FLOOR  (local-polytope lower bound)")
    print("=" * 64)
    try:
        from istat.mre_floor_lp import (compute_lp_mre_floor,
                                        print_lp_floor_summary)
        lp = compute_lp_mre_floor(cs, weights=weights, tighten='pairwise',
                                  verbose=True)
        print()
        print_lp_floor_summary(lp, solver.final_mre, solver.final_weighted_mre,
                               sources=sources, cs=cs,
                               attr_names=ATTR_NAMES_SYNTH,
                               attr_meta=ATTR_META, top=12)
    except Exception as e:
        print(f"  [!] LP floor failed: {e}")
        import traceback; traceback.print_exc()

    print(f"\n[i] Run  python experiments/plot_diagnostics.py  to generate figures.")
    return solver


if __name__ == "__main__":
    run()