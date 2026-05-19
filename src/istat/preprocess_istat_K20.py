"""
preprocess_istat_K20.py
-----------------------
Preprocessing pipeline for the streamlined K=20 setup.
"""

import sys
import os
import argparse
import numpy as np
from copy import deepcopy
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import istat.attr_meta_ISTAT_K20 as _meta
from istat.attr_meta_ISTAT_K20 import (
    ATTR_NAMES_SYNTH, DOMAIN_SIZES_SYNTH, ATTR_META,
    marginals as _RAW_MARGINALS,
)

DEFAULT_ANCHORS = {'sex', 'age', 'marital'}
_SKIP_VARS = {'marginals', 'ATTR_META', 'ATTR_NAMES_SYNTH', 'DOMAIN_SIZES_SYNTH', 'K_SYNTH', '_ATTR_DEFS'}

def _depth_and_keys(d: dict):
    if not isinstance(d, dict) or not d:
        return 0, []
    keys_here = set(d.keys())
    first_val = next(iter(d.values()))
    if isinstance(first_val, dict):
        sub_depth, sub_keys = _depth_and_keys(first_val)
        return sub_depth + 1, [keys_here] + sub_keys
    return 1, [keys_here]

def _match_attr(key_set: set, hint: str = "") -> str | None:
    candidates = [a for a in ATTR_NAMES_SYNTH if key_set.issubset(set(ATTR_META[a]['vals']))]
    if len(candidates) == 1: return candidates[0]
    if len(candidates) > 1:
        hint_low = hint.lower().replace("_", "")
        for c in candidates:
            if c.lower().replace("_", "") in hint_low: return c
        return candidates[0]
    return None

def _norm_row(row: dict) -> dict:
    total = sum(row.values())
    if total < 1e-12:
        return {k: 1.0 / len(row) for k in row}
    return {k: v / total for k, v in row.items()}

def _norm_marginal(dist: dict) -> dict:
    total = sum(dist.values())
    if total < 1e-12:
        return {k: 1.0 / len(dist) for k in dist}
    return {k: v / total for k, v in dist.items()}

def discover_cpts(module) -> list[dict]:
    cpts = []
    for var_name, var_val in vars(module).items():
        if not isinstance(var_val, dict) or var_name.startswith('_') or var_name in _SKIP_VARS:
            continue
        depth, keys_per_level = _depth_and_keys(var_val)

        if depth == 2:
            p_attr = _match_attr(keys_per_level[0], var_name)
            c_attr = _match_attr(keys_per_level[1], var_name)
            if not (p_attr and c_attr and p_attr != c_attr): continue
            norm = {pv: _norm_row(row) for pv, row in var_val.items()}
            p_marg = _RAW_MARGINALS.get(p_attr, {})
            coverage = sum(p_marg.get(pv, 0.0) for pv in var_val)
            cpts.append(dict(name=var_name, depth=2, parents=[p_attr], child=c_attr, raw_cpt=var_val, norm_cpt=norm, coverage=float(coverage), is_partial=False))

        elif depth == 3:
            p1_attr = _match_attr(keys_per_level[0], var_name)
            p2_attr = _match_attr(keys_per_level[1], var_name)
            c_attr  = _match_attr(keys_per_level[2], var_name)
            if not (p1_attr and p2_attr and c_attr): continue
            norm = {p1v: {p2v: _norm_row(row) for p2v, row in p2d.items()} for p1v, p2d in var_val.items()}
            p1_marg = _RAW_MARGINALS.get(p1_attr, {})
            p2_marg = _RAW_MARGINALS.get(p2_attr, {})
            coverage = sum(p1_marg.get(p1v, 0.0) * p2_marg.get(p2v, 0.0) for p1v, p2d in var_val.items() for p2v in p2d)
            cpts.append(dict(name=var_name, depth=3, parents=[p1_attr, p2_attr], child=c_attr, raw_cpt=var_val, norm_cpt=norm, coverage=float(coverage), is_partial=False))
    return cpts

def _implied_binary(norm_cpt: dict, p_attr: str, c_attr: str, marg: dict) -> dict | None:
    c_vals = ATTR_META[c_attr]['vals']
    implied = {v: 0.0 for v in c_vals}
    p_marg = marg.get(p_attr, {})
    for pv, row in norm_cpt.items():
        pp = p_marg.get(pv, 0.0)
        for cv, cond in row.items():
            if cv in implied: implied[cv] += cond * pp
    total = sum(implied.values())
    return {v: implied[v] / total for v in c_vals} if total > 1e-12 else None

def _implied_ternary(norm_cpt: dict, p1_attr: str, p2_attr: str, c_attr: str, marg: dict) -> dict | None:
    c_vals = ATTR_META[c_attr]['vals']
    implied = {v: 0.0 for v in c_vals}
    p1_marg = marg.get(p1_attr, {})
    p2_marg = marg.get(p2_attr, {})
    for p1v, p2d in norm_cpt.items():
        p1p = p1_marg.get(p1v, 0.0)
        for p2v, row in p2d.items():
            joint = p1p * p2_marg.get(p2v, 0.0)
            for cv, cond in row.items():
                if cv in implied: implied[cv] += cond * joint
    total = sum(implied.values())
    return {v: implied[v] / total for v in c_vals} if total > 1e-12 else None

def reconcile_marginals(cpts: list, anchor_vars: set, marg: dict) -> dict:
    by_child = defaultdict(list)
    for cpt in cpts: by_child[cpt['child']].append(cpt)
    updated = deepcopy(marg)
    
    for c_attr, cpt_list in by_child.items():
        if c_attr in anchor_vars: continue
        c_vals = ATTR_META[c_attr]['vals']
        sources = []
        for cpt in cpt_list:
            if cpt['name'].startswith('h_'): continue
            impl = _implied_binary(cpt['norm_cpt'], cpt['parents'][0], c_attr, updated) if cpt['depth'] == 2 else _implied_ternary(cpt['norm_cpt'], cpt['parents'][0], cpt['parents'][1], c_attr, updated)
            if impl is not None: sources.append(impl)
        if not sources: continue
        avg = {v: np.mean([s.get(v, 0.0) for s in sources]) for v in c_vals}
        updated[c_attr] = _norm_marginal(avg)

    GT_NOT_STUDENT = marg.get("StudentStat", {}).get("NotStudent", 0.67)
    GT_NOT_WORKER  = marg.get("employ_stat", {}).get("NotWorker", 0.54)

    # Reconcile employment category baseline alignment
    if "employment" in updated:
        emp_dist = updated["employment"]
        worker_mass = emp_dist.get("FullTime", 0.0) + emp_dist.get("PartTime", 0.0)
        non_worker_mass = emp_dist.get("Unemployed", 0.0) + emp_dist.get("NotInLF", 0.0)
        if worker_mass > 0:
            emp_dist["FullTime"] *= ((1.0 - GT_NOT_WORKER) / worker_mass)
            emp_dist["PartTime"] *= ((1.0 - GT_NOT_WORKER) / worker_mass)
        if non_worker_mass > 0:
            emp_dist["Unemployed"] *= (GT_NOT_WORKER / non_worker_mass)
            emp_dist["NotInLF"] *= (GT_NOT_WORKER / non_worker_mass)
        updated["employment"] = _norm_marginal(emp_dist)

    STUDENT_VARS = ["StudentStat", "Student_commute", "MainTranspStudnt", "TranspTime_Stud"]
    WORKER_VARS  = ["employ_stat", "Wage", "employ_commute", "Occupation", "MainTranspWorker", "TranspTime_Worker"]

    for attr in STUDENT_VARS:
        if attr in updated and "NotStudent" in updated[attr]:
            current_dist = updated[attr]
            current_dist["NotStudent"] = GT_NOT_STUDENT
            rem = sum(v for k, v in current_dist.items() if k != "NotStudent")
            if rem > 0:
                for k in current_dist:
                    if k != "NotStudent": current_dist[k] *= ((1.0 - GT_NOT_STUDENT) / rem)
            updated[attr] = _norm_marginal(current_dist)

    for attr in WORKER_VARS:
        if attr in updated and "NotWorker" in updated[attr]:
            current_dist = updated[attr]
            current_dist["NotWorker"] = GT_NOT_WORKER
            rem = sum(v for k, v in current_dist.items() if k != "NotWorker")
            if rem > 0:
                for k in current_dist:
                    if k != "NotWorker": current_dist[k] *= ((1.0 - GT_NOT_WORKER) / rem)
            updated[attr] = _norm_marginal(current_dist)
    return updated

def build_constraint_set(clean_marginals: dict):
    from constraint_set import ConstraintSet
    cs = ConstraintSet(domain_sizes=DOMAIN_SIZES_SYNTH)
    cpts = discover_cpts(_meta)

    for attr_name, dist in clean_marginals.items():
        if attr_name not in ATTR_NAMES_SYNTH: continue
        attr_idx = ATTR_NAMES_SYNTH.index(attr_name)
        for val_name, prob in dist.items():
            val_idx = ATTR_META[attr_name]['val_to_int'].get(val_name)
            if val_idx is not None: cs.add([attr_idx], [val_idx], prob)

    seen_constraints = set()
    tables_added = 0

    for info in cpts:
        name, depth, parents, child, cpt = info['name'], info['depth'], info['parents'], info['child'], info['norm_cpt']
        if depth == 2:
            p_idx, c_idx = ATTR_NAMES_SYNTH.index(parents[0]), ATTR_NAMES_SYNTH.index(child)
            p_marg = clean_marginals.get(parents[0], {})
            for pv in cpt:
                p_prob = p_marg.get(pv, 0.0)
                pv_idx = ATTR_META[parents[0]]['val_to_int'][pv]
                for cv, cond_prob in cpt[pv].items():
                    cv_idx = ATTR_META[child]['val_to_int'][cv]
                    
                    # Fix: Order matching tracking metrics canonical form
                    if p_idx < c_idx: key = ((p_idx, c_idx), (pv_idx, cv_idx))
                    else: key = ((c_idx, p_idx), (cv_idx, pv_idx))

                    if name.startswith('h_'):
                        if cond_prob == 0.0 and key not in seen_constraints:
                            seen_constraints.add(key)
                            cs.add([p_idx, c_idx], [pv_idx, cv_idx], 0.0)
                        continue
                    if key not in seen_constraints:
                        seen_constraints.add(key)
                        cs.add([p_idx, c_idx], [pv_idx, cv_idx], cond_prob * p_prob)

        elif depth == 3:
            # Enforce structural zeros for ternary entries, skipping conflicting empirical joint estimations
            if name.startswith('h_'):
                p1_idx, p2_idx, c_idx = ATTR_NAMES_SYNTH.index(parents[0]), ATTR_NAMES_SYNTH.index(parents[1]), ATTR_NAMES_SYNTH.index(child)
                for p1v in cpt:
                    p1v_idx = ATTR_META[parents[0]]['val_to_int'][p1v]
                    for p2v in cpt[p1v]:
                        p2v_idx = ATTR_META[parents[1]]['val_to_int'][p2v]
                        for cv, cond_prob in cpt[p1v][p2v].items():
                            cv_idx = ATTR_META[child]['val_to_int'][cv]
                            if cond_prob == 0.0:
                                cs.add([p1_idx, p2_idx, c_idx], [p1v_idx, p2v_idx, cv_idx], 0.0)
        tables_added += 1
    print(f"ConstraintSet built from {tables_added} CPT tables | m = {cs.m} atomic constraints")
    return cs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="src/istat/marginals_corrected_K20.py")
    args = parser.parse_args()
    cpts = discover_cpts(_meta)
    updated = reconcile_marginals(cpts, DEFAULT_ANCHORS, _RAW_MARGINALS)
    
    lines = ["# Auto-generated K=20 marginals\n", "marginals = {\n"]
    for attr in ATTR_NAMES_SYNTH:
        dist = updated[attr]
        inner = ", ".join(f'"{v}": {dist[v]:.5f}' for v in ATTR_META[attr]['vals'])
        lines.append(f'    "{attr}": {{{inner}}},\n')
    lines.append("}\n")
    with open(args.out, "w") as f: f.writelines(lines)
    print(f"Corrected marginals written → {args.out}")