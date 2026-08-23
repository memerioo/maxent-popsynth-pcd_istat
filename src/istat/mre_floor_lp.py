"""
mre_floor_lp.py
---------------
A RIGOROUS lower bound on the MRE that no solver can avoid, obtained by
linear programming over the local polytope.

WHY A SECOND FLOOR
==================
`mre_floor.py` estimates the unavoidable error by looking for quantities
that MORE THAN ONE source states directly, averaging the stated values
with the reliability weights, and charging every constraint its distance
to that average.  That is informative but it is a heuristic, not a bound,
and it misses three whole families of conflict:

  1. Multi-way conflicts.  A marginal can contradict the PRODUCT of a
     conditional and a different marginal without any two tables ever
     stating the same quantity.  Example from the Bologna data:
     P(employ_commute=Inward)=0.16 forces a large share of long
     commutes through P(TranspTime_Worker | employ_commute), which
     contradicts the marginal P(TranspTime_Worker=30m+)=0.06.  Three
     tables, no two of them comparable, jointly infeasible.

  2. Structural infeasibility.  The h_ tables carve regions out of the
     distribution space, which can make otherwise reasonable targets
     unreachable.  Example: H45 makes ResidenceQ=CommuteInward
     equivalent to "inward for work OR inward for study", and the three
     published targets are 0.27, 0.16 and 0.11 -- attainable only if the
     two kinds of inward commuting never co-occur.  The pairwise floor
     sees nothing here; all three tables are BO-tagged and none restates
     another.

  3. Constraints that are simply never stated twice.  481 of the 544
     binary constraints receive no pairwise floor at all and are scored
     as conflict-free, which drags the global figure down.

THE RELAXATION
==============
The quantity we actually want is

    floor* = min over joint distributions p of
             (1/n) sum_j W_j |alpha_hat_j(p) - alpha_j| / alpha_j

which is the error the best conceivable solver would still incur.  p has
2.6e19 entries, so it cannot be optimised directly.

But the objective only ever looks at p through its marginals on the
constraint scopes.  Every genuine joint distribution induces a family of
small distributions {mu_S}, one per scope S, and those necessarily
satisfy three local conditions:

    (a) mu_S >= 0  and  sum_v mu_S(v) = 1
    (b) overlapping scopes agree:  marginalising mu_S and mu_T onto a
        shared attribute k gives the same distribution over D_k
    (c) mu_S(v) = 0 for every pattern forbidden by an h_ table

The set of families satisfying (a)-(c) is the LOCAL POLYTOPE L; the set
actually realisable by some joint distribution is the MARGINAL POLYTOPE
M, and M is contained in L (Wainwright & Jordan 2008, Sec. 4).  Since we
minimise over a LARGER set,

    min over L   <=   min over M   =   floor*

so the LP optimum is a valid lower bound on the true floor.  That is
exactly the direction a floor needs, and it is a guarantee the pairwise
heuristic does not have.

The relaxation is not tight -- L contains "pseudo-marginals" that no real
distribution realises -- so the bound may be loose.  `tighten='pairwise'`
adds agreement on shared PAIRS between scopes that overlap in two
attributes, which is one level of the Sherali-Adams hierarchy and
usually closes much of the gap.

WHAT YOU GET
============
    result['floor']          scalar lower bound on unavoidable wMRE
    result['floor_unw']      same, unweighted
    result['per_constraint'] (m,) array: |mu*_j - alpha_j| / alpha_j
    result['mu']             the optimal pseudo-marginals
    result['status']         solver status string

Since the bound is on the WEIGHTED error by default, compare it against
`solver.final_weighted_mre`.  The solver's achieved error must be >= the
bound; if it is close, the solver is near-optimal given the data.

Requires scipy (already a dependency).  cvxpy is used if available and
`backend='cvxpy'`, which is slower but easier to modify.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack, identity, csr_matrix


# ------------------------------------------------------------------ #
#  Scope construction                                                  #
# ------------------------------------------------------------------ #

def _build_scopes(cs, max_scope_cells: int = 20000):
    """
    Collect the distinct attribute scopes appearing in the constraint set,
    plus every singleton (needed to state the agreement conditions).

    Returns
    -------
    scopes    : list of tuples of attribute indices
    scope_id  : dict scope -> position in `scopes`
    offsets   : (n_scopes+1,) int -- start of each scope's block of
                variables in the flat variable vector
    shapes    : list of tuples -- domain sizes of each scope
    """
    seen = {}
    scopes = []

    def _add(sc):
        sc = tuple(sorted(int(a) for a in sc))
        if sc in seen:
            return
        n_cells = int(np.prod([cs.domain_sizes[a] for a in sc]))
        if n_cells > max_scope_cells:
            raise ValueError(
                f"scope {sc} has {n_cells} cells (> max_scope_cells="
                f"{max_scope_cells}); raise the limit or drop the scope")
        seen[sc] = len(scopes)
        scopes.append(sc)

    for k in range(cs.K):
        _add((k,))
    for j in range(cs.m):
        _add(tuple(cs.attrs_list[j]))

    shapes = [tuple(int(cs.domain_sizes[a]) for a in sc) for sc in scopes]
    sizes = [int(np.prod(s)) for s in shapes]
    offsets = np.zeros(len(scopes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    return scopes, seen, offsets, shapes


def _cell_index(scope, shape, attrs, vals):
    """Flat index of the cell (attrs=vals) inside `scope`'s block."""
    pos = {a: i for i, a in enumerate(scope)}
    idx = np.zeros(len(scope), dtype=np.int64)
    for a, v in zip(attrs, vals):
        idx[pos[int(a)]] = int(v)
    return int(np.ravel_multi_index(tuple(idx), shape))


# ------------------------------------------------------------------ #
#  Main entry point                                                    #
# ------------------------------------------------------------------ #

def compute_lp_mre_floor(cs,
                         weights: np.ndarray | None = None,
                         min_prob_threshold: float = 1e-3,
                         weighted: bool = True,
                         tighten: str | None = None,
                         verbose: bool = True,
                         max_scope_cells: int = 20000) -> dict:
    """
    Lower-bound the achievable (weighted) MRE by LP over the local polytope.

    Parameters
    ----------
    cs      : ConstraintSet -- structural zeros (alpha == 0) are enforced as
              hard zeros, exactly as the solver treats them.
    weights : (m,) reliability weights, or None for unweighted.
    weighted: if True the objective is the reliability-weighted MRE, so the
              result is comparable with solver.final_weighted_mre; if False
              it is comparable with solver.final_mre.
    tighten : None or 'pairwise'.  'pairwise' adds agreement on shared PAIRS
              between scopes overlapping in two attributes (one level of
              Sherali-Adams); tighter but larger.
    """
    t0 = time.time()
    alphas = cs.alphas_array
    m = cs.m
    W = (np.ones(m) if weights is None else np.asarray(weights, float))

    valid = alphas > min_prob_threshold
    struct = alphas == 0.0
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("no valid constraints above min_prob_threshold")

    scopes, scope_id, offsets, shapes = _build_scopes(cs, max_scope_cells)
    n_mu = int(offsets[-1])
    n_e = n_valid
    n_var = n_mu + n_e
    if verbose:
        print(f"  [LP floor] {len(scopes)} scopes, {n_mu:,} pseudo-marginal "
              f"vars, {n_e:,} slacks")

    valid_idx = np.flatnonzero(valid)
    e_of = {int(j): n_mu + i for i, j in enumerate(valid_idx)}

    rows_eq, cols_eq, vals_eq, b_eq = [], [], [], []
    rows_ub, cols_ub, vals_ub, b_ub = [], [], [], []
    r_eq = r_ub = 0

    # --- (a) each scope normalises to 1 -------------------------------
    for s, sc in enumerate(scopes):
        lo, hi = int(offsets[s]), int(offsets[s + 1])
        for c in range(lo, hi):
            rows_eq.append(r_eq); cols_eq.append(c); vals_eq.append(1.0)
        b_eq.append(1.0); r_eq += 1

    # --- (b) agreement between each scope and its singletons ----------
    for s, sc in enumerate(scopes):
        if len(sc) == 1:
            continue
        lo = int(offsets[s]); shape = shapes[s]
        grids = np.indices(shape).reshape(len(sc), -1)
        for p, a in enumerate(sc):
            s1 = scope_id[(int(a),)]
            lo1 = int(offsets[s1])
            for v in range(shape[p]):
                sel = np.flatnonzero(grids[p] == v)
                for c in sel:
                    rows_eq.append(r_eq); cols_eq.append(lo + int(c))
                    vals_eq.append(1.0)
                rows_eq.append(r_eq); cols_eq.append(lo1 + v); vals_eq.append(-1.0)
                b_eq.append(0.0); r_eq += 1

    # --- (b') optional pairwise agreement (Sherali-Adams level 1) ------
    if tighten == 'pairwise':
        n_pair = 0
        for s, sc in enumerate(scopes):
            if len(sc) < 2:
                continue
            for t, tc in enumerate(scopes):
                if t <= s or len(tc) < 2:
                    continue
                shared = tuple(sorted(set(sc) & set(tc)))
                if len(shared) != 2:
                    continue
                ps = scope_id.get(shared)
                if ps is None:
                    continue
                for (sc_i, other) in ((s, ps), (t, ps)):
                    lo = int(offsets[sc_i]); shp = shapes[sc_i]
                    g = np.indices(shp).reshape(len(scopes[sc_i]), -1)
                    posn = [scopes[sc_i].index(a) for a in shared]
                    lo_p = int(offsets[other]); shp_p = shapes[other]
                    for cell in range(int(np.prod(shp_p))):
                        vv = np.unravel_index(cell, shp_p)
                        sel = np.flatnonzero(
                            np.all([g[posn[q]] == vv[q]
                                    for q in range(2)], axis=0))
                        for c in sel:
                            rows_eq.append(r_eq); cols_eq.append(lo + int(c))
                            vals_eq.append(1.0)
                        rows_eq.append(r_eq); cols_eq.append(lo_p + cell)
                        vals_eq.append(-1.0)
                        b_eq.append(0.0); r_eq += 1
                    n_pair += 1
        if verbose:
            print(f"  [LP floor] pairwise tightening: {n_pair} scope/pair links")

    # --- (c) structural zeros ------------------------------------------
    n_struct = 0
    for j in np.flatnonzero(struct):
        sc = tuple(sorted(int(a) for a in cs.attrs_list[j]))
        s = scope_id[sc]
        c = _cell_index(sc, shapes[s], cs.attrs_list[j], cs.vals_list[j])
        rows_eq.append(r_eq); cols_eq.append(int(offsets[s]) + c)
        vals_eq.append(1.0); b_eq.append(0.0); r_eq += 1
        n_struct += 1
    if verbose:
        print(f"  [LP floor] {n_struct} structural zeros pinned")

    # --- objective slacks:  e_j >= |mu_j - alpha_j| ---------------------
    for j in valid_idx:
        j = int(j)
        sc = tuple(sorted(int(a) for a in cs.attrs_list[j]))
        s = scope_id[sc]
        c = int(offsets[s]) + _cell_index(sc, shapes[s],
                                          cs.attrs_list[j], cs.vals_list[j])
        e = e_of[j]
        #  mu - e <= alpha
        rows_ub += [r_ub, r_ub]; cols_ub += [c, e]; vals_ub += [1.0, -1.0]
        b_ub.append(alphas[j]); r_ub += 1
        # -mu - e <= -alpha
        rows_ub += [r_ub, r_ub]; cols_ub += [c, e]; vals_ub += [-1.0, -1.0]
        b_ub.append(-alphas[j]); r_ub += 1

    A_eq = coo_matrix((vals_eq, (rows_eq, cols_eq)), shape=(r_eq, n_var)).tocsr()
    A_ub = coo_matrix((vals_ub, (rows_ub, cols_ub)), shape=(r_ub, n_var)).tocsr()

    # --- cost:  (1/n) sum_j  w_j * e_j / alpha_j ------------------------
    cost = np.zeros(n_var)
    wj = W[valid_idx] if weighted else np.ones(n_valid)
    denom = wj.sum() if weighted else float(n_valid)
    for i, j in enumerate(valid_idx):
        cost[e_of[int(j)]] = wj[i] / (alphas[int(j)] * denom)

    bounds = [(0.0, 1.0)] * n_mu + [(0.0, None)] * n_e

    if verbose:
        print(f"  [LP floor] solving: {n_var:,} vars, {r_eq:,} eq, {r_ub:,} ineq ...")
    res = linprog(cost, A_ub=A_ub, b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=np.array(b_eq),
                  bounds=bounds, method='highs')

    out = {'status': res.message, 'success': bool(res.success),
           'n_scopes': len(scopes), 'n_vars': n_var,
           'n_valid': n_valid, 'weighted': weighted,
           'tighten': tighten, 'time': time.time() - t0}

    if not res.success:
        out['floor'] = float('nan')
        return out

    x = res.x
    per = np.zeros(m)
    mu_hat = np.zeros(m)
    for j in range(m):
        sc = tuple(sorted(int(a) for a in cs.attrs_list[j]))
        s = scope_id[sc]
        c = int(offsets[s]) + _cell_index(sc, shapes[s],
                                          cs.attrs_list[j], cs.vals_list[j])
        mu_hat[j] = x[c]
        if alphas[j] > min_prob_threshold:
            per[j] = abs(x[c] - alphas[j]) / alphas[j]

    out['floor'] = float(res.fun)
    out['per_constraint'] = per
    out['mu_hat'] = mu_hat
    out['floor_unweighted'] = float(np.mean(per[valid]))
    out['floor_weighted'] = float(np.average(per[valid], weights=W[valid]))
    out['n_floored'] = int((per[valid] > 1e-9).sum())
    out['mu'] = x[:n_mu]
    return out


# ------------------------------------------------------------------ #
#  Reporting                                                           #
# ------------------------------------------------------------------ #

def print_lp_floor_summary(result: dict, observed_mre: float,
                           observed_wmre: float,
                           sources: list | None = None,
                           cs=None, attr_names=None, attr_meta=None,
                           top: int = 12):
    """Human-readable summary, mirroring print_full_mre_floor_summary."""
    line = "\u2500" * 78
    print(line)
    print("  LP MRE FLOOR  (local-polytope relaxation -- rigorous lower bound)")
    print(line)
    if not result.get('success'):
        print(f"  LP FAILED: {result['status']}")
        print(line)
        return

    fw = result['floor_weighted']
    fu = result['floor_unweighted']
    print(f"  scopes / variables      : {result['n_scopes']} / {result['n_vars']:,}")
    print(f"  tightening              : {result['tighten'] or 'none (local polytope)'}")
    print(f"  solve time              : {result['time']:.1f}s")
    print()
    print(f"  MRE observed            : {observed_mre:.5f}")
    print(f"  wMRE observed           : {observed_wmre:.5f}")
    print()
    print(f"  LP floor  (unweighted)  : {fu:.5f}")
    print(f"  LP floor  (weighted)    : {fw:.5f}")
    print(f"  constraints with floor  : {result['n_floored']} of {result['n_valid']}")
    print()
    if observed_mre > 0:
        print(f"  Data inconsistency      : >= {100*fu/observed_mre:.1f}% of observed MRE")
        print(f"  Solver approximation    : <= {100*(1-fu/observed_mre):.1f}% of observed MRE")
    print(f"  MRE adjusted            : {max(0.0, observed_mre-fu):.5f}")
    print()
    if observed_wmre + 1e-12 < fw:
        print("  [!] observed wMRE is BELOW the LP bound. That is impossible for a")
        print("      genuine joint distribution and means the pool is not actually")
        print("      satisfying the structural zeros, or cs/weights differ from the run.")
    else:
        print(f"  Consistency check       : observed wMRE >= LP floor  OK "
              f"(slack {observed_wmre-fw:+.5f})")

    per = result.get('per_constraint')
    if per is not None and cs is not None and attr_names is not None:
        order = np.argsort(per)[::-1][:top]
        print()
        print(f"  Largest per-constraint floors (unavoidable relative error):")
        for j in order:
            if per[j] <= 1e-9:
                break
            src = f"[{sources[j]}]" if sources else ""
            attrs = [attr_names[a] for a in cs.attrs_list[j]]
            vals = [attr_meta[attrs[i]]['vals'][v]
                    for i, v in enumerate(cs.vals_list[j])]
            desc = ", ".join(f"{a}={v}" for a, v in zip(attrs, vals))
            print(f"    floor={per[j]:.4f}  target={cs.alphas[j]:.4f}  "
                  f"best={result['mu_hat'][j]:.4f}  {src} {desc}")
    print(line)


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys, os, pickle
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.join(ROOT, 'src'))

    from istat.preprocess_istat import build_constraint_set, build_constraint_weights
    from istat.geo_tagging import parse_source_tags
    from istat.attr_meta_ISTAT import (marginals as M, ATTR_NAMES_SYNTH, ATTR_META)

    meta_path = os.path.join(ROOT, 'src', 'istat', 'attr_meta_ISTAT.py')
    ts, ms = parse_source_tags(meta_path)
    cs, sources = build_constraint_set(M, ts, ms)
    W = build_constraint_weights(sources)

    obs_mre = obs_wmre = float('nan')
    hist = os.path.join(ROOT, 'experiments', 'test_4_ISTAT_history.pkl')
    if os.path.exists(hist):
        with open(hist, 'rb') as f:
            rec = pickle.load(f)
        obs_mre = rec.get('final_mre', float('nan'))
        obs_wmre = rec.get('final_weighted_mre', float('nan'))

    tighten = 'pairwise' if '--tighten' in sys.argv else None
    r = compute_lp_mre_floor(cs, weights=W, tighten=tighten, verbose=True)
    print()
    print_lp_floor_summary(r, obs_mre, obs_wmre, sources=sources,
                           cs=cs, attr_names=ATTR_NAMES_SYNTH,
                           attr_meta=ATTR_META)