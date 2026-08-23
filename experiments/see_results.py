"""
see_results.py
--------------
Read and display individual profiles from the saved synthetic population,
plus a legality audit and a unary-marginal check.

Run from the project root:
    python experiments/see_results.py
    python experiments/see_results.py 25          # print 25 profiles

CHANGED: the population file is now auto-discovered instead of hardcoded to
test_4_ISTAT_pop500000.npy. The old behaviour silently read a STALE pool from
a previous run when the current run used a different N, which made the
marginal check and the floor analysis describe two different populations.
"""

import numpy as np
import sys
import os
import glob

# -- Path setup --------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from istat.attr_meta_ISTAT import ATTR_NAMES_SYNTH, ATTR_META, marginals as TARGETS


def find_pool():
    """Newest test_4_ISTAT_pop*.npy in experiments/."""
    pat = os.path.join(ROOT_DIR, 'experiments', 'test_4_ISTAT_pop*.npy')
    files = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
    if not files:
        print(f"No population file matching:\n  {pat}")
        print("Run  python experiments/test_4_ISTAT.py  first.")
        sys.exit(1)
    if len(files) > 1:
        print("Multiple population files found; using the most recent:")
        for f in files:
            mark = "  <-- using" if f == files[0] else ""
            print(f"   {os.path.basename(f)}{mark}")
        print("   (delete the stale ones to avoid confusion)\n")
    return files[0]


def legality_audit(pool):
    """Rebuild the constraint set and count structurally impossible individuals."""
    try:
        from istat.preprocess_istat import build_constraint_set
        from istat.geo_tagging import parse_source_tags
        from istat.structural_blocks import legality_report
    except ImportError as e:
        print(f"  (legality audit unavailable: {e})")
        return
    meta_path = os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py')
    ts, ms = parse_source_tags(meta_path)
    cs, _ = build_constraint_set(TARGETS, ts, ms)
    print(legality_report(cs, pool, ATTR_NAMES_SYNTH, ATTR_META, top=8))


def read_profiles(num_to_print=10):
    pop_path = find_pool()
    print(f"Loading population from:\n  {pop_path}\n")
    pop = np.load(pop_path)
    N, K = pop.shape
    print(f"  Total individuals : {N:,}")
    print(f"  Attributes (K)    : {K}")
    print(f"  Distinct profiles : {len(set(map(tuple, pop))):,}")
    print()

    print("=" * 80)
    print("  STRUCTURAL LEGALITY AUDIT")
    print("=" * 80)
    legality_audit(pop)
    print()

    n = min(num_to_print, N)
    print("=" * 80)
    print(f"  SAMPLE OF {n} SYNTHETIC BOLOGNESI")
    print("=" * 80 + "\n")
    for i in range(n):
        print(f"  Profile {i + 1}")
        print(f"  {'-' * 76}")
        for col_idx, attr_name in enumerate(ATTR_NAMES_SYNTH):
            val_idx = pop[i, col_idx]
            val_str = ATTR_META[attr_name]['vals'][val_idx]
            print(f"    {attr_name:<22} {val_str}")
        print()

    # -- Marginal check -------------------------------------------------------
    print("=" * 80)
    print("  UNARY MARGINAL CHECK  (pool frequency vs. target)")
    print("=" * 80)
    print(f"  {'Attribute':<22} {'Value':<20} {'Pool':>8}  {'Target':>8}  {'Diff':>8}")
    print(f"  {'-' * 70}")
    diffs = []
    for attr_name, dist in TARGETS.items():
        if attr_name not in ATTR_NAMES_SYNTH:
            continue
        col_idx = ATTR_NAMES_SYNTH.index(attr_name)
        for val_str, target in dist.items():
            vi = ATTR_META[attr_name]['val_to_int'].get(val_str)
            if vi is None:
                continue
            f = float((pop[:, col_idx] == vi).mean())
            d = f - target
            diffs.append(abs(d))
            flag = "!" if abs(d) > 0.03 else " "
            print(f"  {flag} {attr_name:<21} {val_str:<20} "
                  f"{f:>8.4f}  {target:>8.4f}  {d:>+8.4f}")
    if diffs:
        print(f"  {'-' * 70}")
        print(f"  mean |diff| = {np.mean(diffs):.5f}   max = {max(diffs):.5f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    read_profiles(num_to_print=num)