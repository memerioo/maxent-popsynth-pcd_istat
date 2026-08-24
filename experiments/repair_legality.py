"""
repair_legality.py  (v2)
------------------------
Final legality pass over the delivered pool.

WHY THIS IS NEEDED
==================
Structural zeros are enforced by pinning lambda_j = -30, which is finite, so
forbidden configurations retain probability ~e^-30 rather than exactly zero.
Single-site Gibbs essentially never visits one, but the block move can: its
Hastings ratio log n(b) - log n(b') can reach +10 when a very common block
pattern is proposed against a very rare one, partially offsetting the -30.
Over ~4e8 proposals this leaves a handful of accepted violations.

WHY v1 LEFT ONE BEHIND
======================
Resetting a sentinel block to its inactive state is legal in isolation, but
can VIOLATE A DIFFERENT RULE. Concretely, H45 forbids

    ResidenceQ = CommuteInward  AND  employ_commute = NotWorker
                                AND  Student_commute = NotStudent

so turning a non-working CommuteInward resident into a non-student creates a
fresh violation. Repair must therefore iterate, and must have a fallback that
is guaranteed to terminate.

STRATEGY
========
  pass 1..k : targeted repair -- reset the implicated sentinel block; if the
              result trips H45, move the individual out of CommuteInward to a
              real quartiere drawn from the pool's own distribution.
  fallback  : replace the individual wholesale with a copy of a randomly
              chosen LEGAL pool member. This is a draw from the pool's own
              empirical distribution, so it perturbs no marginal
              systematically, and it cannot fail.

The pool is written back ONLY if the audit afterwards returns zero.
"""

import sys
import os
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from istat.preprocess_istat import build_constraint_set
from istat.geo_tagging import parse_source_tags
from istat.attr_meta_ISTAT import (marginals as M, ATTR_NAMES_SYNTH, ATTR_META)
from istat.structural_blocks import violation_counts

MAX_PASSES = 5


def _describe(cs, pool, i, struct):
    out = []
    for j in struct:
        at, vl = cs.attrs_list[j], cs.vals_list[j]
        if np.all(pool[i, at] == vl):
            out.append(", ".join(
                f"{ATTR_NAMES_SYNTH[a]}={ATTR_META[ATTR_NAMES_SYNTH[a]]['vals'][v]}"
                for a, v in zip(at, vl)))
    return out


def repair(pop_path=None, write=True, seed=42):
    rng = np.random.default_rng(seed)
    meta = os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py')
    ts, ms = parse_source_tags(meta)
    cs, _ = build_constraint_set(M, ts, ms)

    if pop_path is None:
        import glob
        c = sorted(glob.glob(os.path.join(ROOT_DIR, 'experiments',
                                          'test_4_ISTAT_pop*.npy')),
                   key=os.path.getmtime, reverse=True)
        if not c:
            print("No population file found."); sys.exit(1)
        pop_path = c[0]

    print(f"Pool: {pop_path}")
    pool = np.load(pop_path)
    N = len(pool)
    original = pool.copy()

    I = {a: i for i, a in enumerate(ATTR_NAMES_SYNTH)}
    V = {a: ATTR_META[a]['val_to_int'] for a in ATTR_NAMES_SYNTH}
    struct = np.flatnonzero(cs.alphas_array == 0.0)

    STUDY = [('StudentStat', 'NotStudent'), ('Student_commute', 'NotStudent'),
             ('MainTranspStudnt', 'NotStudent'), ('TranspTime_Stud', 'NotStudent')]
    WORK = [('employment', 'NotInLF'), ('employ_stat', 'NotWorker'),
            ('Wage', 'NotWorker'), ('employ_commute', 'NotWorker'),
            ('Profession', 'NotWorker'), ('Occupation', 'NotWorker'),
            ('MainTranspWorker', 'NotWorker'), ('TranspTime_Worker', 'NotWorker')]
    QUARTIERI = ['Reno', 'Navile', 'Saragozza', 'SanDonato', 'SantoStefano', 'Savena']

    viol = violation_counts(cs, pool)
    bad0 = np.flatnonzero(viol > 0)
    print(f"\nIllegal individuals before : {len(bad0)} / {N:,} "
          f"({100*len(bad0)/N:.4f}%)")
    if len(bad0) == 0:
        print("Nothing to repair.")
        return pool

    print("\nViolations found:")
    for i in bad0:
        for d in _describe(cs, pool, i, struct):
            print(f"  individual {i}: {d}")

    n_study = n_work = n_resid = n_donor = 0

    # ---- targeted repair, iterated -----------------------------------------
    for p in range(MAX_PASSES):
        bad = np.flatnonzero(violation_counts(cs, pool) > 0)
        if len(bad) == 0:
            break
        for i in bad:
            touched = set()
            for j in struct:
                at, vl = cs.attrs_list[j], cs.vals_list[j]
                if np.all(pool[i, at] == vl):
                    touched.update(ATTR_NAMES_SYNTH[a] for a in at)

            if touched & {a for a, _ in STUDY}:
                for a, v in STUDY:
                    pool[i, I[a]] = V[a][v]
                n_study += 1
            if touched & {a for a, _ in WORK}:
                for a, v in WORK:
                    pool[i, I[a]] = V[a][v]
                n_work += 1

            # H45: a CommuteInward resident must commute inward for work or
            # study. If the individual now does neither, move them into a real
            # quartiere rather than inventing a commute.
            if (pool[i, I['ResidenceQ']] == V['ResidenceQ']['CommuteInward']
                    and pool[i, I['employ_commute']] == V['employ_commute']['NotWorker']
                    and pool[i, I['Student_commute']] == V['Student_commute']['NotStudent']):
                col = pool[:, I['ResidenceQ']]
                cand = [V['ResidenceQ'][q] for q in QUARTIERI]
                w = np.array([(col == c).sum() for c in cand], dtype=float)
                pool[i, I['ResidenceQ']] = rng.choice(cand, p=w / w.sum())
                n_resid += 1

    # ---- fallback: donor replacement ---------------------------------------
    bad = np.flatnonzero(violation_counts(cs, pool) > 0)
    if len(bad):
        legal = np.flatnonzero(violation_counts(cs, pool) == 0)
        for i in bad:
            pool[i] = pool[rng.choice(legal)]
            n_donor += 1

    # ---- report -------------------------------------------------------------
    print(f"\nRepairs applied:")
    print(f"  study block reset      : {n_study}")
    print(f"  work block reset       : {n_work}")
    print(f"  moved out of CommuteInward (H45) : {n_resid}")
    print(f"  replaced by a pool donor         : {n_donor}")

    left = int((violation_counts(cs, pool) > 0).sum())
    print(f"\nIllegal individuals after  : {left} / {N:,}")
    if left:
        print("  [!] repair failed; pool NOT written.")
        return pool

    changed = int((pool != original).any(axis=1).sum())
    print(f"\nIndividuals changed: {changed} of {N:,} "
          f"= {100*changed/N:.5f}% of the population.")
    print("Largest marginal shift caused by the repair:")
    worst = 0.0
    for a in ATTR_NAMES_SYNTH:
        c = I[a]
        for v in range(len(ATTR_META[a]['vals'])):
            d = abs((pool[:, c] == v).mean() - (original[:, c] == v).mean())
            worst = max(worst, d)
    print(f"  max |delta| over all 135 categories = {worst:.6f}")

    if write:
        np.save(pop_path, pool)
        print(f"\n[+] Repaired pool written back to {pop_path}")
        print("    Re-run plot_diagnostics.py and see_results.py to refresh.")
    else:
        print("\n(dry run -- nothing written)")
    return pool


if __name__ == "__main__":
    repair(write='--dry-run' not in sys.argv)