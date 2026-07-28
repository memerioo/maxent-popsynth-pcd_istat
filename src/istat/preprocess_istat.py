"""
preprocess_istat.py
-------------------
Preprocessing pipeline: normalise CPT rows, derive self-consistent
marginals for non-anchor variables, and export corrected marginals.
"""

import sys
import os
import argparse
import numpy as np
from copy import deepcopy
from collections import defaultdict

# ─── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import istat.attr_meta_ISTAT as _meta
from istat.attr_meta_ISTAT import (
    ATTR_NAMES_SYNTH, DOMAIN_SIZES_SYNTH, ATTR_META,
    marginals as _RAW_MARGINALS,
)

# ─── anchor variables ─────────────────────────────────────────────────────────
DEFAULT_ANCHORS = {
    'sex', 'age', 'marital', 'citizenship'
}

_SKIP_VARS = {
    'marginals', 'ATTR_META', 'ATTR_NAMES_SYNTH',
    'DOMAIN_SIZES_SYNTH', 'K_SYNTH', '_ATTR_DEFS',
}


# ==============================================================================
# PART 1 — Low-level helpers
# ==============================================================================

def _depth_and_keys(d: dict):
    if not isinstance(d, dict) or not d:
        return 0, []
    keys_here = set(d.keys())
    first_val = next(iter(d.values()))
    if isinstance(first_val, dict):
        sub_depth, sub_keys = _depth_and_keys(first_val)
        return sub_depth + 1, [keys_here] + sub_keys
    return 1, [keys_here]


def _match_attr(key_set: set, hint: str = "",
                exclude: set | None = None) -> str | None:
    """
    Match a set of category labels to an attribute name.

    exclude : attributes already assigned to other roles of the same CPT
        (e.g. the parent, when matching the child). Without this, tables
        whose parent and child share the same value labels — e.g.
        h_MainTranspStudnt_MainTranspWorker, where both levels contain
        {Foot, Bike, PublicTrns, ...} — resolve both roles to the SAME
        attribute and the table is silently dropped from the constraint set.
    """
    exclude = exclude or set()
    candidates = [
        a for a in ATTR_NAMES_SYNTH
        if a not in exclude and key_set.issubset(set(ATTR_META[a]['vals']))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        hint_low = hint.lower().replace("_", "")
        for c in candidates:
            if c.lower().replace("_", "") in hint_low:
                return c
        return candidates[0]
    return None


def _norm_row(row: dict) -> dict:
    total = sum(row.values())
    if total < 1e-12:
        return dict(row)   # ← return all-zeros unchanged
    return {k: v / total for k, v in row.items()}


def _norm_marginal(dist: dict) -> dict:
    total = sum(dist.values())
    if total < 1e-12:
        n = len(dist)
        return {k: 1.0 / n for k in dist}
    return {k: v / total for k, v in dist.items()}


# ==============================================================================
# PART 2 — CPT discovery and normalisation
# ==============================================================================

def discover_cpts(module) -> list[dict]:
    cpts = []
    for var_name, var_val in vars(module).items():
        if (
            not isinstance(var_val, dict)
            or var_name.startswith('_')
            or var_name in _SKIP_VARS
        ):
            continue

        depth, keys_per_level = _depth_and_keys(var_val)

        if depth == 2:
            p_attr = _match_attr(keys_per_level[0], var_name)
            c_attr = _match_attr(keys_per_level[1], var_name,
                                 exclude={p_attr} if p_attr else None)
            if not (p_attr and c_attr and p_attr != c_attr):
                continue

            norm = {pv: _norm_row(row) for pv, row in var_val.items()}
            p_marg = _RAW_MARGINALS.get(p_attr, {})
            coverage = sum(p_marg.get(pv, 0.0) for pv in var_val)
            all_p_vals = set(ATTR_META[p_attr]['vals'])
            is_partial = not set(var_val.keys()).issuperset(all_p_vals)

            cpts.append(dict(
                name=var_name, depth=2,
                parents=[p_attr], child=c_attr,
                raw_cpt=var_val, norm_cpt=norm,
                coverage=float(coverage), is_partial=is_partial,
            ))

        elif depth == 3:
            p1_attr = _match_attr(keys_per_level[0], var_name)
            p2_attr = _match_attr(keys_per_level[1], var_name,
                                  exclude={p1_attr} if p1_attr else None)
            c_attr  = _match_attr(keys_per_level[2], var_name,
                                  exclude={a for a in (p1_attr, p2_attr) if a})
            if not (p1_attr and p2_attr and c_attr):
                continue

            norm = {
                p1v: {p2v: _norm_row(row) for p2v, row in p2d.items()}
                for p1v, p2d in var_val.items()
            }

            p1_marg = _RAW_MARGINALS.get(p1_attr, {})
            p2_marg = _RAW_MARGINALS.get(p2_attr, {})
            coverage = sum(
                p1_marg.get(p1v, 0.0) * p2_marg.get(p2v, 0.0)
                for p1v, p2d in var_val.items()
                for p2v in p2d
            )
            cpts.append(dict(
                name=var_name, depth=3,
                parents=[p1_attr, p2_attr], child=c_attr,
                raw_cpt=var_val, norm_cpt=norm,
                coverage=float(coverage), is_partial=(coverage < 0.98),
            ))

    return cpts


# ==============================================================================
# PART 3 — Implied-marginal computation
# ==============================================================================

def _implied_binary(norm_cpt: dict, p_attr: str, c_attr: str, marg: dict) -> dict | None:
    c_vals  = ATTR_META[c_attr]['vals']
    implied = {v: 0.0 for v in c_vals}
    p_marg  = marg.get(p_attr, {})

    covered_mass = sum(p_marg.get(pv, 0.0) for pv in norm_cpt)
    if covered_mass < 1e-10:
        return None

    for pv, row in norm_cpt.items():
        pp = p_marg.get(pv, 0.0)
        for cv, cond in row.items():
            if cv in implied:
                implied[cv] += cond * pp

    uncovered = max(0.0, 1.0 - covered_mass)
    if uncovered > 1e-6:
        stated = marg.get(c_attr, {})
        stated_total = sum(stated.values())
        for v in c_vals:
            if stated_total > 1e-12:
                implied[v] += uncovered * stated.get(v, 0.0) / stated_total
            else:
                implied[v] += uncovered / len(c_vals)

    total = sum(implied.values())
    if total < 1e-12:
        return None
    return {v: implied[v] / total for v in c_vals}


def _implied_ternary(norm_cpt: dict, p1_attr: str, p2_attr: str, c_attr: str, marg: dict) -> dict | None:
    c_vals  = ATTR_META[c_attr]['vals']
    implied = {v: 0.0 for v in c_vals}
    p1_marg = marg.get(p1_attr, {})
    p2_marg = marg.get(p2_attr, {})

    covered_mass = 0.0
    for p1v, p2d in norm_cpt.items():
        p1p = p1_marg.get(p1v, 0.0)
        for p2v, row in p2d.items():
            p2p  = p2_marg.get(p2v, 0.0)
            joint = p1p * p2p
            covered_mass += joint
            for cv, cond in row.items():
                if cv in implied:
                    implied[cv] += cond * joint

    if covered_mass < 1e-10:
        return None

    uncovered = max(0.0, 1.0 - covered_mass)
    if uncovered > 1e-6:
        stated = marg.get(c_attr, {})
        stated_total = sum(stated.values())
        for v in c_vals:
            if stated_total > 1e-12:
                implied[v] += uncovered * stated.get(v, 0.0) / stated_total
            else:
                implied[v] += uncovered / len(c_vals)

    total = sum(implied.values())
    if total < 1e-12:
        return None
    return {v: implied[v] / total for v in c_vals}


# ==============================================================================
# PART 4 — Marginal reconciliation
# ==============================================================================

def reconcile_marginals(cpts: list, anchor_vars: set, marg: dict) -> tuple[dict, dict]:
    by_child: dict[str, list] = defaultdict(list)
    for cpt in cpts:
        by_child[cpt['child']].append(cpt)

    updated = deepcopy(marg)
    for a in anchor_vars:
        if a in updated:
            updated[a] = _norm_marginal(updated[a])

    report = {}

    for c_attr, cpt_list in by_child.items():
        if c_attr in anchor_vars:
            continue

        c_vals   = ATTR_META[c_attr]['vals']
        old_marg = marg.get(c_attr, {})

        sources = []
        for cpt in cpt_list:
            if cpt['name'].startswith('h_'):
                continue

            if cpt['depth'] == 2:
                impl = _implied_binary(cpt['norm_cpt'], cpt['parents'][0], c_attr, updated)
                arity_weight = 1.0
            else:
                impl = _implied_ternary(cpt['norm_cpt'], cpt['parents'][0], cpt['parents'][1], c_attr, updated)
                arity_weight = 0.5

            if impl is not None:
                sources.append({
                    'source'       : cpt['name'],
                    'coverage'     : cpt['coverage'],
                    'is_partial'   : cpt['is_partial'],
                    'depth'        : cpt['depth'],
                    'marginal'     : impl,
                    'weight'       : cpt['coverage'] * arity_weight,
                })

        if not sources:
            continue

        total_weight = sum(s['weight'] for s in sources)
        avg = {v: 0.0 for v in c_vals}

        if total_weight > 1e-12:
            for s in sources:
                w = s['weight'] / total_weight
                for v in c_vals:
                    avg[v] += w * s['marginal'].get(v, 0.0)
        else:
            n = len(sources)
            for s in sources:
                for v in c_vals:
                    avg[v] += s['marginal'].get(v, 0.0) / n

        avg = _norm_marginal(avg)
        max_disc = max(abs(avg.get(v, 0.0) - old_marg.get(v, 0.0)) for v in c_vals) if old_marg else 1.0

        report[c_attr] = {
            'n_sources'       : len(sources),
            'sources'         : [s['source'] for s in sources],
            'coverages'       : [round(s['coverage'], 3) for s in sources],
            'depths'          : [s['depth'] for s in sources],
            'any_partial'     : any(s['is_partial'] for s in sources),
            'old_marginal'    : dict(old_marg),
            'new_marginal'    : avg,
            'max_discrepancy' : max_disc,
        }
        updated[c_attr] = avg

    # ─── FORCE PHYSICAL IDENTITIES ───
    GT_NOT_STUDENT = marg.get("StudentStat", {}).get("NotStudent", 0.67)
    GT_NOT_WORKER  = marg.get("employ_stat", {}).get("NotWorker", 0.54)

    STUDENT_VARS = ["StudentStat", "Student_commute", "MainTranspStudnt", "TranspTime_Stud"]
    WORKER_VARS  = ["employ_stat", "Wage", "employ_commute", "Profession", "Occupation", "MainTranspWorker", "TranspTime_Worker", "employment"]

    for attr in STUDENT_VARS:
        if attr in updated and "NotStudent" in updated[attr]:
            current_dist = updated[attr]
            current_dist["NotStudent"] = GT_NOT_STUDENT
            remaining_mass = sum(v for k, v in current_dist.items() if k != "NotStudent")
            target_remaining = 1.0 - GT_NOT_STUDENT
            if remaining_mass > 0:
                for k in current_dist:
                    if k != "NotStudent":
                        current_dist[k] = current_dist[k] * (target_remaining / remaining_mass)
            updated[attr] = _norm_marginal(current_dist)

    if "employment" in updated:
        emp_dist = updated["employment"]
        worker_mass = emp_dist.get("FullTime", 0.0) + emp_dist.get("PartTime", 0.0)
        non_worker_mass = emp_dist.get("Unemployed", 0.0) + emp_dist.get("NotInLF", 0.0)
        
        target_worker = 1.0 - GT_NOT_WORKER      # 0.46
        target_non_worker = GT_NOT_WORKER        # 0.54
        
        if worker_mass > 0:
            emp_dist["FullTime"] *= (target_worker / worker_mass)
            emp_dist["PartTime"] *= (target_worker / worker_mass)
        if non_worker_mass > 0:
            emp_dist["Unemployed"] *= (target_non_worker / non_worker_mass)
            emp_dist["NotInLF"] *= (target_non_worker / non_worker_mass)
            
        updated["employment"] = _norm_marginal(emp_dist)

    for attr in WORKER_VARS:
        if attr in updated and "NotWorker" in updated[attr]:
            current_dist = updated[attr]
            current_dist["NotWorker"] = GT_NOT_WORKER
            remaining_mass = sum(v for k, v in current_dist.items() if k != "NotWorker")
            target_remaining = 1.0 - GT_NOT_WORKER
            if remaining_mass > 0:
                for k in current_dist:
                    if k != "NotWorker":
                        current_dist[k] = current_dist[k] * (target_remaining / remaining_mass)
            updated[attr] = _norm_marginal(current_dist)

    return updated, report


def check_row_sums(cpts: list) -> list[tuple]:
    bad = []
    for info in cpts:
        cpt  = info['raw_cpt']
        name = info['name']
        if info['depth'] == 2:
            for pv, row in cpt.items():
                s = sum(row.values())
                if abs(s - 1.0) > 0.005:
                    bad.append((name, str(pv), s))
        elif info['depth'] == 3:
            for p1v, p2d in cpt.items():
                for p2v, row in p2d.items():
                    s = sum(row.values())
                    if abs(s - 1.0) > 0.005:
                        bad.append((name, f"{p1v}/{p2v}", s))
    return bad

    
def check_marginal_sums(marginals: dict, tol: float = 1e-6) -> list:
    """Return (attr_name, sum) for every marginal that doesn't sum to 1."""
    bad = []
    for attr, dist in marginals.items():
        s = sum(dist.values())
        if abs(s - 1.0) > tol:
            bad.append((attr, round(s, 8)))
    return bad


def cross_check_cpts(cpts: list) -> list:
    """
    Check that for every parent value present in a CPT, the row sums to 1.
    Returns list of (cpt_name, parent_path, row_sum) for problematic rows.
    Thin wrapper around check_row_sums kept for API compatibility.
    """
    return check_row_sums(cpts)


def partial_cpt_report(cpts: list) -> list:
    """
    Return a list of dicts for CPTs whose parent-value coverage is < 1.0,
    i.e. some parent values are absent so the implied marginal is incomplete.
    Sorted by coverage ascending (worst first).
    """
    try:
        from istat.attr_meta_ISTAT import marginals as _marg
    except ImportError:
        return []

    results = []
    for info in cpts:
        depth   = info['depth']
        parents = info['parents']
        cpt     = info['norm_cpt']

        if depth == 2:
            p_marg   = _marg.get(parents[0], {})
            coverage = sum(p for pv, p in p_marg.items() if pv in cpt)

        elif depth == 3:
            p1_marg  = _marg.get(parents[0], {})
            p2_marg  = _marg.get(parents[1], {})
            coverage = 0.0
            for p1v, p1p in p1_marg.items():
                if p1v in cpt:
                    for p2v, p2p in p2_marg.items():
                        if p2v in cpt[p1v]:
                            coverage += p1p * p2p
        else:
            coverage = 1.0

        if coverage < 1.0 - 1e-6:
            results.append({
                'name':     info['name'],
                'coverage': round(coverage, 6),
                'parents':  info['parents'],
                'child':    info['child'],
            })

    results.sort(key=lambda x: x['coverage'])
    return results

    
def run_preprocessing(anchor_vars: set | None = None, verbose: bool = True) -> dict:
    if anchor_vars is None:
        anchor_vars = DEFAULT_ANCHORS

    SEP = "═" * 64
    cpts = discover_cpts(_meta)
    n2   = sum(1 for c in cpts if c['depth'] == 2)
    n3   = sum(1 for c in cpts if c['depth'] == 3)

    if verbose:
        print(SEP)
        print("  ISTAT PREPROCESSING  —  CPT consistency & marginal fix")
        print(SEP)
        print(f"\n[1] CPT discovery")
        print(f"     Found {len(cpts)} tables  ({n2} binary, {n3} ternary)")

    row_bad = check_row_sums(cpts)
    if verbose and row_bad:
        print(f"\n[2] CPT row-sum issues  ←  {len(row_bad)} rows ≠ 1  (fixed by norm)")

    updated, _ = reconcile_marginals(cpts, anchor_vars, _RAW_MARGINALS)
    return updated


def export_marginals(updated: dict, path: str):
    lines = [
        "# ─────────────────────────────────────────────────────────────\n",
        "# Auto-generated by preprocess_istat.py\n",
        "# ─────────────────────────────────────────────────────────────\n\n",
        "marginals = {\n",
    ]
    for attr in ATTR_NAMES_SYNTH:
        if attr not in updated:
            continue
        dist  = updated[attr]
        inner = ", ".join(f'"{v}": {dist[v]:.5f}' for v in ATTR_META[attr]['vals'] if v in dist)
        lines.append(f'    "{attr}": {{{inner}}},\n')
    lines.append("}\n")

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"\nCorrected marginals written → {path}")


# ==============================================================================
# PART 6 — Constraint Set Builder (Pure MaxEnt Principle — No Matrix Overwrites)
# ==============================================================================

def parent_joint_table(p1_attr: str, p2_attr: str,
                       clean_marginals: dict, cpts: list):
    """
    Joint distribution of a ternary CPT's two parents:
        {(p1v, p2v): P(p1v, p2v)}

    If a discovered (non-h_) binary CPT links the two parents, the joint is
    computed from it (first table in definition order, consistent with the
    dedup policy of build_constraint_set); otherwise the parents are assumed
    independent and the product of their marginals is used.

    Returns (joint_dict, source_table_name_or_None). Used by BOTH
    build_constraint_set and mre_floor.compute_full_mre_floor so that the
    ternary targets and the floor's implied values are computed identically.
    """
    p1_marg = clean_marginals.get(p1_attr, {})
    p2_marg = clean_marginals.get(p2_attr, {})

    for info in cpts:
        if info['depth'] != 2 or info['name'].startswith('h_'):
            continue
        pa, ch = info['parents'][0], info['child']
        if {pa, ch} == {p1_attr, p2_attr}:
            cpt = info['norm_cpt']
            joint = {}
            if pa == p1_attr:                       # P(p2 | p1) * P(p1)
                for pv, row in cpt.items():
                    pp = p1_marg.get(pv, 0.0)
                    for cv, cond in row.items():
                        joint[(pv, cv)] = float(cond) * float(pp)
            else:                                   # P(p1 | p2) * P(p2)
                for pv, row in cpt.items():
                    pp = p2_marg.get(pv, 0.0)
                    for cv, cond in row.items():
                        joint[(cv, pv)] = float(cond) * float(pp)
            return joint, info['name']

    # Independence fallback
    return ({(a, b): float(p1_marg.get(a, 0.0)) * float(p2_marg.get(b, 0.0))
             for a in p1_marg for b in p2_marg}, None)


def build_constraint_set(clean_marginals: dict,
                          table_source: dict | None = None,
                          marginal_source: dict | None = None,
                          default_source: str = "Italy"):
    """
    Parameters
    ----------
    table_source, marginal_source : dict or None
        Output of geo_tagging.parse_source_tags(...). If given, every
        atomic constraint added to `cs` is tagged with a geo-source string
        (falling back to `default_source` when the table/marginal wasn't
        recognised by the tagger). If both are None, `sources` is returned
        as None and nothing about weighting changes (fully backward
        compatible with the original single-return-value signature -- but
        note the return type is now always a tuple, see below).

    Returns
    -------
    (cs, sources) : (ConstraintSet, list[str] | None)
        `sources[j]` is the geo tag of the j-th atomic constraint in `cs`,
        in the same order as cs.alphas_array. `sources` is None if neither
        table_source nor marginal_source was supplied.
    """
    try:
        from constraint_set import ConstraintSet
    except ImportError:
        raise ImportError("Could not import ConstraintSet. Make sure src/ is on sys.path.")

    track_sources = (table_source is not None) or (marginal_source is not None)
    table_source = table_source or {}
    marginal_source = marginal_source or {}

    cs   = ConstraintSet(domain_sizes=DOMAIN_SIZES_SYNTH)
    cpts = discover_cpts(_meta)
    sources: list[str] = []

    # 1. Add Single-Attribute Marginal Constraints
    for attr_name, dist in clean_marginals.items():
        if attr_name not in ATTR_NAMES_SYNTH:
            continue
        attr_idx = ATTR_NAMES_SYNTH.index(attr_name)
        tag = marginal_source.get(attr_name, default_source)
        for val_name, prob in dist.items():
            val_idx = ATTR_META[attr_name]['val_to_int'].get(val_name)
            if val_idx is not None:
                cs.add([attr_idx], [val_idx], prob)
                if track_sources:
                    sources.append(tag)

    seen_constraints = set()   
    tables_added = 0
    ternary_added = 0

    for info in cpts:
        name    = info['name']
        depth   = info['depth']
        parents = info['parents']
        child   = info['child']
        cpt     = info['norm_cpt']
        tag     = table_source.get(name, default_source)

        if depth == 2:
            p_attr  = parents[0]
            p_marg  = clean_marginals.get(p_attr, {})
            p_idx   = ATTR_NAMES_SYNTH.index(p_attr)
            c_idx   = ATTR_NAMES_SYNTH.index(child)

            for pv in cpt:  
                p_prob = p_marg.get(pv, 0.0)
                pv_idx = ATTR_META[p_attr]['val_to_int'][pv]

                for cv, cond_prob in cpt[pv].items():
                    cv_idx = ATTR_META[child]['val_to_int'][cv]
                    
                    # FIX: Canonicalize key based on attribute index order
                    if p_idx < c_idx:
                        key = ((p_idx, c_idx), (pv_idx, cv_idx))
                    else:
                        key = ((c_idx, p_idx), (cv_idx, pv_idx))

                    if name.startswith('h_'):
                        if cond_prob == 0.0:
                            if key not in seen_constraints:
                                seen_constraints.add(key)
                                cs.add([p_idx, c_idx], [pv_idx, cv_idx], 0.0)
                                if track_sources:
                                    sources.append(tag)
                        continue

                    joint_prob = cond_prob * p_prob
                    if key not in seen_constraints:
                        seen_constraints.add(key)
                        cs.add([p_idx, c_idx], [pv_idx, cv_idx], joint_prob)
                        if track_sources:
                            sources.append(tag)

        elif depth == 3:
            p1_attr = parents[0]
            p2_attr = parents[1]
            p1_idx  = ATTR_NAMES_SYNTH.index(p1_attr)
            p2_idx  = ATTR_NAMES_SYNTH.index(p2_attr)
            c_idx   = ATTR_NAMES_SYNTH.index(child)

            if name.startswith('h_'):
                # Structural logic table: only the zeros matter (logically
                # impossible combinations); non-zero entries are placeholders.
                ternary_added += 1
                for p1v in cpt:
                    p1v_idx = ATTR_META[p1_attr]['val_to_int'][p1v]
                    for p2v in cpt[p1v]:
                        p2v_idx = ATTR_META[p2_attr]['val_to_int'][p2v]
                        for cv, cond_prob in cpt[p1v][p2v].items():
                            cv_idx = ATTR_META[child]['val_to_int'][cv]
                            if cond_prob == 0.0:
                                key = tuple(sorted(((p1_idx, p1v_idx),
                                                    (p2_idx, p2v_idx),
                                                    (c_idx,  cv_idx))))
                                if key not in seen_constraints:
                                    seen_constraints.add(key)
                                    cs.add([p1_idx, p2_idx, c_idx],
                                           [p1v_idx, p2v_idx, cv_idx], 0.0)
                                    if track_sources:
                                        sources.append(tag)
            else:
                # Empirical ternary CPT: P(c | p1, p2) × P(p1, p2).
                # The parent joint comes from a discovered binary CPT linking
                # the two parents when available, else the independence
                # product of their marginals (see parent_joint_table).
                pjoint, _pj_src = parent_joint_table(
                    p1_attr, p2_attr, clean_marginals, cpts)
                for p1v in cpt:
                    p1v_idx = ATTR_META[p1_attr]['val_to_int'][p1v]
                    for p2v in cpt[p1v]:
                        p2v_idx = ATTR_META[p2_attr]['val_to_int'][p2v]
                        pj = pjoint.get((p1v, p2v), 0.0)
                        for cv, cond_prob in cpt[p1v][p2v].items():
                            cv_idx = ATTR_META[child]['val_to_int'][cv]
                            key = tuple(sorted(((p1_idx, p1v_idx),
                                                (p2_idx, p2v_idx),
                                                (c_idx,  cv_idx))))
                            if key not in seen_constraints:
                                seen_constraints.add(key)
                                cs.add([p1_idx, p2_idx, c_idx],
                                       [p1v_idx, p2v_idx, cv_idx],
                                       float(cond_prob) * pj)
                                if track_sources:
                                    sources.append(tag)
                ternary_added += 1

        tables_added += 1

    assert not track_sources or len(sources) == cs.m, \
        f"source tracking desynced from cs.add() calls: {len(sources)} vs {cs.m}"

    print(f"ConstraintSet built from {tables_added} CPT tables "
          f"({tables_added - ternary_added} binary, {ternary_added} ternary)"
          f"  |  m = {cs.m} atomic constraints")
    return cs, (sources if track_sources else None)


def build_constraint_weights(sources: list, reliability: dict | None = None,
                              default_tag: str = "Italy") -> "np.ndarray":
    """
    Turn a `sources` list (as returned by build_constraint_set) into a
    (m,) weights array ready for GibbsPCDSolver(..., weights=...).
    """
    import numpy as np
    if reliability is None:
        from istat.geo_tagging import RELIABILITY as reliability
    return np.array([reliability.get(tag, reliability[default_tag]) for tag in sources],
                     dtype=np.float64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ISTAT CPTs and marginals for GibbsPCDSolver.")
    parser.add_argument("--quiet", action="store_true", help="suppress report")
    
    # FIX: Dynamically compute the absolute path to default output target 'src/istat/marginals_corrected.py'
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_OUT = os.path.join(CURRENT_DIR, "marginals_corrected.py")
    
    parser.add_argument("--out", default=DEFAULT_OUT, help="output path")
    args = parser.parse_args()

    updated = run_preprocessing(verbose=not args.quiet)
    export_marginals(updated, path=args.out)