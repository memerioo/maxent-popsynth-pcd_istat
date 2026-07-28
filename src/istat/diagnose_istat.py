"""
diagnose_istat.py
-----------------
Quick diagnostic: show the biggest CPT ↔ marginal inconsistencies
in your raw attr_meta_ISTAT.py without modifying anything.

Run first to understand what's wrong before applying the fix.

    python diagnose_istat.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from preprocess_istat import (
    discover_cpts, check_row_sums, check_marginal_sums,
    cross_check_cpts, partial_cpt_report,
    _implied_binary, _implied_ternary,
)
from istat.attr_meta_ISTAT import (
    ATTR_NAMES_SYNTH, ATTR_META, marginals,
)
import istat.attr_meta_ISTAT as _meta

SEP = "━" * 70


def implied_vs_stated(cpts, marg, top_n=15):
    """
    For every variable that appears as a child in at least one full-coverage
    CPT, print: stated marginal vs. each CPT-implied marginal.
    Shows the raw disagreement between your data sources.
    """
    from collections import defaultdict
    by_child = defaultdict(list)
    for c in cpts:
        if c['name'].startswith('h_'):
            continue
        by_child[c['child']].append(c)

    rows = []
    for c_attr, cpt_list in by_child.items():
        stated = marg.get(c_attr, {})
        c_vals = ATTR_META[c_attr]['vals']

        for info in cpt_list:
            if info['depth'] == 2:
                impl = _implied_binary(
                    info['norm_cpt'], info['parents'][0], c_attr, marg)
            else:
                impl = _implied_ternary(
                    info['norm_cpt'], info['parents'][0], info['parents'][1],
                    c_attr, marg)
            if impl is None:
                continue
            max_disc = max(
                abs(impl.get(v, 0) - stated.get(v, 0)) for v in c_vals
            ) if stated else 1.0
            rows.append((c_attr, info['name'], max_disc, impl, stated))

    rows.sort(key=lambda x: -x[2])
    return rows[:top_n]


def main():
    print(SEP)
    print("  ISTAT DIAGNOSTICS  —  raw inconsistency scan")
    print(SEP)

    cpts = discover_cpts(_meta)

    # ── 1. Marginal sums ──────────────────────────────────────────
    print("\n1. Marginals that don't sum to 1")
    bad = check_marginal_sums(marginals)
    if bad:
        for a, s in bad:
            print(f"   {a}: sum={s}")
    else:
        print("   (none)")

    # ── 2. CPT row sums ───────────────────────────────────────────
    print("\n2. CPT rows that don't sum to 1  (worst 10)")
    row_bad = check_row_sums(cpts)
    if row_bad:
        for name, path, s in row_bad[:10]:
            print(f"   {name}[{path}]  sum={s:.5f}  Δ={s-1:+.5f}")
    else:
        print("   (none)")

    # ── 3. Partial CPTs ───────────────────────────────────────────
    print("\n3. Partial CPTs (coverage < 1.0)  — sorted by coverage ↑")
    partial = partial_cpt_report(cpts)
    print(f"   {'Name':<35} {'Cov':>6}  {'Parents → Child'}")
    print(f"   {'─'*65}")
    for p in partial:
        print(f"   {p['name']:<35} {p['coverage']:>6.3f}  "
              f"{' × '.join(p['parents'])} → {p['child']}")

    # ── 4. Biggest source inconsistencies ─────────────────────────
    print(f"\n4. Biggest CPT ↔ marginal disagreements (top 15)")
    rows = implied_vs_stated(cpts, marginals, top_n=15)
    print(f"   {'Child attr':<22} {'CPT source':<35} {'MaxΔ':>6}")
    print(f"   {'─'*68}")
    for c_attr, cpt_name, max_disc, impl, stated in rows:
        flag = "⚠⚠" if max_disc > 0.15 else ("⚠" if max_disc > 0.05 else " ")
        print(f"   {flag} {c_attr:<20} {cpt_name:<35} {max_disc:>6.4f}")

    # ── 5. Detailed view of worst variable ───────────────────────
    if rows:
        print(f"\n5. Detailed view: worst variable  ({rows[0][0]})")
        c_attr = rows[0][0]
        c_vals = ATTR_META[c_attr]['vals']
        stated = marginals.get(c_attr, {})

        # Gather all implied marginals for this variable
        child_rows = [r for r in rows if r[0] == c_attr]
        # Also get remaining CPTs for this variable
        by_child_all = {}
        for c in cpts:
            by_child_all.setdefault(c['child'], []).append(c)

        all_for_child = by_child_all.get(c_attr, [])

        print(f"\n   {'Value':<22} {'Stated':>8}", end="")
        for info in all_for_child[:5]:
            print(f"  {info['name'][:12]:>14}", end="")
        print()
        print(f"   {'─'*80}")

        for v in c_vals:
            print(f"   {v:<22} {stated.get(v,0):>8.4f}", end="")
            for info in all_for_child[:5]:
                if info['depth'] == 2:
                    impl = _implied_binary(
                        info['norm_cpt'], info['parents'][0], c_attr, marginals)
                else:
                    impl = _implied_ternary(
                        info['norm_cpt'], info['parents'][0], info['parents'][1],
                        c_attr, marginals)
                val = impl.get(v, 0) if impl else 0
                print(f"  {val:>14.4f}", end="")
            print()

    print(f"\n{SEP}")
    print("  Run  python preprocess_istat.py --diff  to apply the fix.")
    print(SEP)


if __name__ == "__main__":
    main()