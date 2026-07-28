"""
see_results.py
--------------
Read and display individual profiles from the saved synthetic population.

Run from the project root:
    python experiments/see_results.py

Optional: print more profiles
    python experiments/see_results.py 25
"""

import numpy as np
import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
# This file is in experiments/, so ROOT_DIR is one level up
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from istat.attr_meta_ISTAT import ATTR_NAMES_SYNTH, ATTR_META


def read_profiles(num_to_print=10):

    # ── Locate the population file ────────────────────────────────────────────
    pop_path = os.path.join(ROOT_DIR, 'experiments', 'test_4_ISTAT_pop500000.npy')

    if not os.path.exists(pop_path):
        print(f"❌  File not found: {pop_path}")
        print("    Run  python experiments/test_4_ISTAT.py  first to generate the population.")
        sys.exit(1)

    print(f"Loading population from:\n  {pop_path}\n")
    pop = np.load(pop_path)
    N, K = pop.shape
    print(f"  Total individuals : {N:,}")
    print(f"  Attributes (K)    : {K}")
    print(f"  Distinct profiles : {len(set(map(tuple, pop))):,}")
    print()

    # ── Print individual profiles ─────────────────────────────────────────────
    n = min(num_to_print, N)
    print(f"{'━' * 80}")
    print(f"  SAMPLE OF {n} SYNTHETIC BOLOGNESI")
    print(f"{'━' * 80}\n")

    for i in range(n):
        print(f"  Profile {i + 1}")
        print(f"  {'─' * 76}")
        for col_idx, attr_name in enumerate(ATTR_NAMES_SYNTH):
            val_idx = pop[i, col_idx]
            val_str = ATTR_META[attr_name]['vals'][val_idx]
            print(f"    {attr_name:<22} {val_str}")
        print()

    # ── Quick marginal check: compare pool frequencies to target marginals ────
    print(f"{'━' * 80}")
    print("  QUICK MARGINAL CHECK  (pool frequency vs. target)")
    print(f"{'━' * 80}")

    try:
        try:
            from istat.marginals_corrected import marginals as TARGETS
        except ImportError:
            # marginals_corrected.py is only produced by running
            # preprocess_istat.py; fall back to the raw targets actually
            # used by test_4_ISTAT.py.
            from istat.attr_meta_ISTAT import marginals as TARGETS
        print(f"  {'Attribute':<22} {'Value':<20} {'Pool':>8}  {'Target':>8}  {'Diff':>8}")
        print(f"  {'─' * 70}")
        for attr_name, dist in TARGETS.items():
            if attr_name not in ATTR_NAMES_SYNTH:
                continue
            col_idx = ATTR_NAMES_SYNTH.index(attr_name)
            vals, counts = np.unique(pop[:, col_idx], return_counts=True)
            freq = counts / N
            for v_idx, f in zip(vals, freq):
                val_str   = ATTR_META[attr_name]['vals'][v_idx]
                target    = dist.get(val_str, 0.0)
                diff      = f - target
                flag      = "⚠" if abs(diff) > 0.03 else " "
                print(f"  {flag} {attr_name:<21} {val_str:<20} {f:>8.4f}  {target:>8.4f}  {diff:>+8.4f}")
    except ImportError:
        print("  (no marginals module found — skipping marginal check)")

    print(f"\n{'━' * 80}")


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    read_profiles(num_to_print=num)