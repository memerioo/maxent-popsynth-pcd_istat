"""
test_blocks_smoke.py
--------------------
FAST sanity check of the sentinel-block fix.  Run this BEFORE committing to a
full 500k / 2100-iteration run.

    python experiments/test_blocks_smoke.py

It does three things, in order of increasing cost:

  1. FREEZE DEMO (no fitting, ~20 s)
     Runs pure Gibbs sweeps with hard structural zeros and NO block moves,
     from two different initialisations, and prints the block composition.
     Expected: both freeze within ~2 sweeps at completely different values,
     proving the composition is decided by initialisation and not by lambda.
     This is the figure/number for Section 4.4 of the thesis.

  2. MOBILITY DEMO (no fitting, ~30 s)
     Same thing WITH block moves.  Expected: the composition now moves, and
     the acceptance rate is a healthy 0.1-0.4.

  3. SHORT FIT (~10-20 min at N=50k)
     A real but small fit, ending in a marginal check.  This is the one that
     tells you whether the fix actually recovers P(StudentStat) and
     P(employment).  Skip it with  --no-fit.
"""

import sys
import os
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from gibbs_pcd_solver import GibbsPCDSolver
from istat.preprocess_istat import build_constraint_set, build_constraint_weights
from istat.geo_tagging import parse_source_tags
from istat.attr_meta_ISTAT import marginals as M, ATTR_NAMES_SYNTH, ATTR_META
from istat.structural_blocks import (resolve_blocks, ancestral_init_pool,
                                     violation_counts)
from istat.block_moves import block_toggle

I = {a: i for i, a in enumerate(ATTR_NAMES_SYNTH)}
V = {a: ATTR_META[a]['val_to_int'] for a in ATTR_NAMES_SYNTH}

TGT_NOTSTUD = M['StudentStat']['NotStudent']
TGT_WORKER  = M['employment']['FullTime'] + M['employment']['PartTime']
TGT_CHILD   = M['age']['0-4'] + M['age']['5-14']


def compo(p):
    return (float((p[:, I['StudentStat']] == V['StudentStat']['NotStudent']).mean()),
            float(np.isin(p[:, I['employment']],
                          [V['employment']['FullTime'],
                           V['employment']['PartTime']]).mean()),
            float(np.isin(p[:, I['age']],
                          [V['age']['0-4'], V['age']['5-14']]).mean()))


def build():
    meta_path = os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py')
    ts, ms = parse_source_tags(meta_path)
    cs, sources = build_constraint_set(M, ts, ms)
    return cs, sources


def demo_freeze(cs, N=20000, n_sweeps=40):
    print("\n" + "=" * 72)
    print("  1. FREEZE DEMO  --  hard zeros, NO block moves")
    print("=" * 72)
    print(f"  targets:  NotStudent={TGT_NOTSTUD:.4f}  worker={TGT_WORKER:.4f}  "
          f"child={TGT_CHILD:.4f}\n")

    for label, mk in (("uniform init", lambda s: s._init_pool(N, seed=42)),
                      ("on-target init",
                       lambda s: ancestral_init_pool(N, M, ATTR_NAMES_SYNTH,
                                                     ATTR_META, seed=42))):
        s = GibbsPCDSolver(cs, use_numba=False)
        s._rng = np.random.default_rng(42)
        lam = np.zeros(s.m)
        lam[s.alphas == 0.0] = -30.0
        pool = mk(s)
        print(f"  --- {label} ---")
        print(f"  {'sweep':>6} {'NotStudent':>11} {'worker':>8} {'child':>8}")
        a, b, c = compo(pool)
        print(f"  {0:>6} {a:>11.4f} {b:>8.4f} {c:>8.4f}")
        for t in range(1, n_sweeps + 1):
            pool = s._gibbs_sweep(pool, lam)
            if t in (1, 2, 5, 10, 20, n_sweeps):
                a, b, c = compo(pool)
                print(f"  {t:>6} {a:>11.4f} {b:>8.4f} {c:>8.4f}")
        print()
    print("  READ: if the two initialisations end at different frozen values,")
    print("  the composition is decided by init, not by lambda. That is the bug.")


def demo_mobility(cs, N=20000, n_iter=15):
    print("\n" + "=" * 72)
    print("  2. MOBILITY DEMO  --  hard zeros, WITH block moves")
    print("=" * 72)
    blocks = resolve_blocks(ATTR_NAMES_SYNTH, ATTR_META, names=("work", "study"))
    s = GibbsPCDSolver(cs, use_numba=False, blocks=blocks)
    s._rng = np.random.default_rng(7)
    lam = np.zeros(s.m)
    lam[s.alphas == 0.0] = -30.0
    pool = s._init_pool(N, seed=42)
    print(f"  {'iter':>5} {'NotStudent':>11} {'worker':>8} {'acc%':>7} {'illegal':>8}")
    a, b, _ = compo(pool)
    print(f"  {0:>5} {a:>11.4f} {b:>8.4f} {'-':>7} "
          f"{int((violation_counts(cs, pool) > 0).sum()):>8}")
    for t in range(1, n_iter + 1):
        accs = []
        for blk in s._blocks_prepared:
            acc, n = block_toggle(pool, lam, blk, s._rng,
                                  kernel=s._block_kernel, frac=1.0)
            accs.append(acc / n)
        pool = s._gibbs_sweep(pool, lam)
        if t % 3 == 0 or t == 1:
            a, b, _ = compo(pool)
            print(f"  {t:>5} {a:>11.4f} {b:>8.4f} {100*np.mean(accs):>6.2f}% "
                  f"{int((violation_counts(cs, pool) > 0).sum()):>8}")
    print("\n  READ: the composition should MOVE (it is now free to), acceptance")
    print("  should sit around 0.1-0.4, and illegal should fall to ~0.")
    print("  It will NOT sit on the targets here: lambda is still all zeros,")
    print("  so this is the free equilibrium, not the fitted one.")


def demo_fit(cs, sources, N=50000, n_outer=150):
    print("\n" + "=" * 72)
    print(f"  3. SHORT FIT  --  N={N:,}, {n_outer} iterations")
    print("=" * 72)
    weights = build_constraint_weights(sources)
    blocks = resolve_blocks(ATTR_NAMES_SYNTH, ATTR_META, names=("work", "study"))
    init = ancestral_init_pool(N, M, ATTR_NAMES_SYNTH, ATTR_META, seed=42)
    s = GibbsPCDSolver(cs, use_numba=True, weights=weights,
                       sources=sources, blocks=blocks)
    s.fit(N_pool=N, n_outer=n_outer, n_gibbs_sweeps=6, lr=0.02,
          lr_patience=0, gamma=1e-3, eps=1e-4, seed=42, tol=0.0,
          init_pool=init, block_move_frac=0.5, verbose_every=25)

    pool = s.pool_
    print(f"\n  illegal individuals: {int((violation_counts(cs, pool) > 0).sum()):,}"
          f" / {N:,}")
    a, b, c = compo(pool)
    print(f"\n  {'quantity':<14}{'pool':>9}{'target':>9}{'diff':>9}")
    for name, got, tgt in (("NotStudent", a, TGT_NOTSTUD),
                           ("worker", b, TGT_WORKER),
                           ("child", c, TGT_CHILD)):
        print(f"  {name:<14}{got:>9.4f}{tgt:>9.4f}{got-tgt:>+9.4f}")

    rows = []
    for attr, dist in M.items():
        if attr not in ATTR_NAMES_SYNTH:
            continue
        col = ATTR_NAMES_SYNTH.index(attr)
        for val, tgt in dist.items():
            vi = ATTR_META[attr]['val_to_int'].get(val)
            if vi is None:
                continue
            f = float((pool[:, col] == vi).mean())
            rows.append((abs(f - tgt), attr, val, f, tgt, f - tgt))
    rows.sort(reverse=True)
    print(f"\n  Worst 12 unary marginals:")
    print(f"  {'attribute':<22}{'value':<18}{'pool':>8}{'target':>8}{'diff':>9}")
    for ad, attr, val, f, tgt, d in rows[:12]:
        print(f"  {attr:<22}{val:<18}{f:>8.4f}{tgt:>8.4f}{d:>+9.4f}")
    print(f"\n  mean |diff| over {len(rows)} categories: "
          f"{np.mean([r[0] for r in rows]):.5f}")


if __name__ == "__main__":
    cs, sources = build()
    demo_freeze(cs)
    demo_mobility(cs)
    if "--no-fit" not in sys.argv:
        demo_fit(cs, sources)
    else:
        print("\n(skipping the short fit; drop --no-fit to run it)")