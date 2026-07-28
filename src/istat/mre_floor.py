"""
mre_floor.py
------------
Estimates the "MRE floor" implied by data-source inconsistency (as opposed
to solver error), and produces stratified / reliability-weighted MRE
reporting broken down by geographic source.

WHAT THIS DOES AND DOES NOT CLAIM
----------------------------------
There is no closed-form exact floor for a general log-linear (MaxEnt) model
with an inconsistent constraint set -- the true floor would require solving
the (nonconvex-in-general) minimum-weighted-residual projection of alpha
onto the achievable-moment manifold of the exponential family, which is
exactly as hard as fitting the model itself. What we CAN compute cheaply
and defensibly is a *local* floor per conflict: when two or more sources
give different targets for essentially the same quantity (e.g. two CPTs
imply different marginals for `employment`, or a Bologna-specific binary
constraint disagrees with a national CPT-implied version of the same
edge), no single shared parameter can satisfy all of them at once, and the
best any weighted-gradient optimiser can do -- given how GibbsPCDSolver's
Adam-with-per-coordinate-learning-rate weighting behaves -- is converge
close to the *reliability-weighted average* of the conflicting targets.

    floor_point   = sum_s W_s * alpha_s / sum_s W_s
    floor_residual_s = | floor_point - alpha_s |

This is an approximation (empirically it tracks the actual PCD equilibrium
reasonably well when the conflicting constraints share a single coupled
degree of freedom -- see the discussion in
`GibbsPCDSolver.fit.__doc__`), not a guarantee.
Treat `floor_residual` as a defensible LOWER BOUND on what you should
expect the solver to achieve on that constraint, not an exact prediction.
Report it as such in the thesis: "the model achieves X, the estimated
data-inconsistency floor is Y <= X, therefore at most X-Y is attributable
to solver approximation error."

USAGE
-----
    from diagnose_istat import implied_vs_stated, discover_cpts
    from geo_tagging import parse_source_tags, RELIABILITY
    import istat.attr_meta_ISTAT as _meta
    from istat.attr_meta_ISTAT import marginals, ATTR_META

    table_source, marginal_source = parse_source_tags("attr_meta_ISTAT.py")
    cpts = discover_cpts(_meta)
    rows = implied_vs_stated(cpts, marginals, top_n=30)   # from diagnose_istat.py
    report = variable_floor_report(rows, marginals, table_source, marginal_source)
    print_floor_report(report)
"""

from __future__ import annotations
import numpy as np


def weighted_compromise(targets, weights):
    """
    sum_s W_s * alpha_s / sum_s W_s  -- the estimated equilibrium point
    of a set of mutually-conflicting targets on the same coupled degree
    of freedom, under reliability weights `weights`.
    """
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        return float(np.mean(targets))
    return float(np.sum(weights * targets) / np.sum(weights))


def variable_floor_report(implied_rows, marginals, table_source, marginal_source,
                           reliability=None, default_tag="Italy"):
    """
    implied_rows : output of diagnose_istat.implied_vs_stated(cpts, marginals)
        list of (child_attr, cpt_name, max_disc, implied_dict, stated_dict)
    Returns a dict: {child_attr: {...}} with the weighted-compromise floor
    for every attribute that has at least one CPT-implied vs. stated
    marginal disagreement.
    """
    if reliability is None:
        from istat.geo_tagging import RELIABILITY as reliability

    from collections import defaultdict
    by_attr = defaultdict(list)
    for c_attr, cpt_name, max_disc, implied, stated in implied_rows:
        by_attr[c_attr].append((cpt_name, implied))

    report = {}
    for c_attr, sources in by_attr.items():
        stated = marginals.get(c_attr, {})
        stated_tag = marginal_source.get(c_attr, default_tag)
        stated_w = reliability.get(stated_tag, reliability[default_tag])

        # collect all values (stated + each CPT-implied) with their weights
        all_vals = {v: [] for v in stated}   # value -> list of (source_label, prob, weight)
        for v, p in stated.items():
            all_vals.setdefault(v, []).append((f"stated[{stated_tag}]", p, stated_w))
        for cpt_name, implied in sources:
            tag = table_source.get(cpt_name, default_tag)
            w = reliability.get(tag, reliability[default_tag])
            for v, p in implied.items():
                all_vals.setdefault(v, []).append((f"{cpt_name}[{tag}]", p, w))

        per_value = {}
        for v, entries in all_vals.items():
            probs = [e[1] for e in entries]
            weights = [e[2] for e in entries]
            floor_pt = weighted_compromise(probs, weights)
            residuals = {e[0]: abs(floor_pt - e[1]) for e in entries}
            per_value[v] = {
                'floor_point': floor_pt,
                'sources': entries,
                'residuals': residuals,
                'max_residual': max(residuals.values()),
            }

        report[c_attr] = per_value

    return report


def print_floor_report(report, top_n=15):
    rows = []
    for attr, per_value in report.items():
        for v, info in per_value.items():
            rows.append((attr, v, info['max_residual'], info['floor_point'], info['sources']))
    rows.sort(key=lambda r: -r[2])

    print("─" * 78)
    print(f"  ESTIMATED MRE FLOOR FROM SOURCE CONFLICTS (top {top_n})")
    print("─" * 78)
    for attr, v, max_res, floor_pt, sources in rows[:top_n]:
        print(f"\n  {attr} = {v}   estimated floor point = {floor_pt:.4f}")
        for label, p, w in sources:
            print(f"     {label:<28} target={p:.4f}   W={w:.2f}   "
                  f"|floor-target|={abs(floor_pt-p):.4f}")
    print()


# ------------------------------------------------------------------ #
#  Weighted / stratified MRE reporting from an already-fit solver      #
#  (thin convenience wrapper around GibbsPCDSolver.stratified_mre)     #
# ------------------------------------------------------------------ #

def print_stratified_mre(solver, title="STRATIFIED / WEIGHTED MRE"):
    strat = solver.stratified_mre()
    print("─" * 78)
    print(f"  {title}")
    print("─" * 78)
    print(f"  {'Source':<12} {'n':>5} {'MRE':>9} {'wMRE':>9} {'MAE':>9} "
          f"{'Contrib':>9} {'Share':>7}")
    order = [t for t in strat if t != 'ALL'] + ['ALL']
    for tag in order:
        s = strat[tag]
        print(f"  {tag:<12} {s['n_constraints']:>5} {s['mre']:>9.4f} "
              f"{s['weighted_mre']:>9.4f} {s['mae']:>9.4f} "
              f"{s['contribution']:>9.4f} {s['contribution_pct']:>6.1f}%")
    print()

    # ── by constraint arity ─────────────────────────────────────────────────
    try:
        arity = solver.stratified_mre_arity()
        print(f"  {'Arity':<12} {'n':>5} {'MRE':>9} {'wMRE':>9} {'MAE':>9} "
              f"{'Contrib':>9} {'Share':>7}")
        for lbl in ('unary', 'binary', 'ternary', 'ALL'):
            s = arity[lbl]
            print(f"  {lbl:<12} {s['n_constraints']:>5} {s['mre']:>9.4f} "
                  f"{s['weighted_mre']:>9.4f} {s['mae']:>9.4f} "
                  f"{s['contribution']:>9.4f} {s['contribution_pct']:>6.1f}%")
        print()

        # ── source × arity cross-tab (contribution shares of global MRE) ──
        cross = solver.stratified_mre_source_arity()
        print("  Source × arity — MRE (n) and share of global MRE:")
        hdr = f"  {'':<12}" + "".join(f"{a:>21}" for a in ('unary', 'binary', 'ternary'))
        print(hdr)
        for tag in sorted(cross):
            row = f"  {tag:<12}"
            for a in ('unary', 'binary', 'ternary'):
                c = cross[tag][a]
                if c['n_constraints'] == 0:
                    row += f"{'—':>21}"
                else:
                    row += f"{c['mre']:>7.3f} ({c['n_constraints']:>3}) {c['contribution_pct']:>5.1f}%"
            print(row)
        print()
    except (ValueError, RuntimeError):
        pass
    return strat


def mre_adjusted(observed_mre: float, floor_mre: float) -> float:
    """
    MRE_adjusted = max(0, MRE_observed - MRE_floor)
    The clip at 0 matters: with a stochastic solver, observed can dip
    below the (approximate) floor on some strata just from noise.
    """
    return max(0.0, observed_mre - floor_mre)


def compute_scalar_mre_floor(floor_report, cs, attr_names, attr_meta,
                              alphas, min_prob=1e-3):
    """
    Aggregate the per-variable floor report into a scalar MRE floor
    that is directly comparable to the observed MRE.

    Strategy
    --------
    The observed MRE runs over all m_valid atomic constraints
    (unary + binary + ternary) with alpha_j > min_prob.

    For UNARY constraints whose variable appears in floor_report:
        floor_j = |floor_point - alpha_j| / alpha_j
        (the compromise point directly predicts where the solver will land)

    For BINARY / TERNARY constraints and unary constraints with no
    identified source conflict:
        floor_j = 0   (conservative lower bound — we cannot decompose
                       these without coupling analysis)

    The scalar MRE floor is then:
        MRE_floor = (1 / m_valid) * sum_j floor_j

    This is a LOWER BOUND on the true floor, not an exact value.
    It understates the true floor because binary/ternary conflicts
    also contribute irreducible error (e.g. sex × employ_stat conflicts
    that appear in the top-15 unresolved list), but those are harder to
    decompose analytically.

    Also returns a UNARY-ONLY version for transparency: the floor
    computed only over unary constraints, which is the subset where
    the approximation is exact (not just a lower bound).

    Parameters
    ----------
    floor_report : dict
        Output of variable_floor_report().
    cs : ConstraintSet
        The fitted constraint set (needs attrs_list, vals_list).
    attr_names : list[str]
        ATTR_NAMES_SYNTH — maps attribute index to name.
    attr_meta : dict
        ATTR_META — maps attribute name to {'vals': [...], ...}.
    alphas : np.ndarray
        The target probabilities, shape (m,).
    min_prob : float
        Threshold below which constraints are excluded (same as MRE).

    Returns
    -------
    dict with keys:
        mre_floor          : scalar (lower bound, over all m_valid constraints)
        mre_floor_unary    : scalar (exact for identified conflicts, unary only)
        mre_adjusted       : max(0, mre_floor is subtracted from observed externally)
        n_floored          : number of constraints with a non-zero floor estimate
        n_valid            : total number of valid constraints in denominator
        per_constraint     : np.ndarray of per-constraint floor values (length m)
        conflicted_attrs   : set of attribute names with identified source conflicts
    """
    m = len(alphas)
    valid = alphas > min_prob
    floor_vals = np.zeros(m, dtype=np.float64)

    conflicted_attrs = set(floor_report.keys())

    for j in range(m):
        if not valid[j]:
            continue
        attrs = cs.attrs_list[j]
        vals  = cs.vals_list[j]

        # Only handle unary constraints directly
        if len(attrs) != 1:
            continue

        attr_idx  = int(attrs[0])
        val_idx   = int(vals[0])
        attr_name = attr_names[attr_idx]
        val_name  = attr_meta[attr_name]['vals'][val_idx]

        if attr_name in floor_report and val_name in floor_report[attr_name]:
            floor_pt  = floor_report[attr_name][val_name]['floor_point']
            alpha_j   = float(alphas[j])
            if alpha_j > min_prob:
                floor_vals[j] = abs(floor_pt - alpha_j) / alpha_j

    n_valid   = int(valid.sum())
    n_floored = int((floor_vals > 1e-9).sum())

    # Over all valid constraints (binary/ternary contribute 0 — lower bound)
    mre_floor = float(floor_vals[valid].mean()) if n_valid > 0 else 0.0

    # Over valid UNARY constraints only (exact for identified conflicts)
    unary_mask = np.array([len(cs.attrs_list[j]) == 1 for j in range(m)])
    unary_valid = unary_mask & valid
    mre_floor_unary = float(
        floor_vals[unary_valid].mean()
    ) if unary_valid.sum() > 0 else 0.0

    return {
        'mre_floor':        mre_floor,
        'mre_floor_unary':  mre_floor_unary,
        'n_floored':        n_floored,
        'n_valid':          n_valid,
        'per_constraint':   floor_vals,
        'conflicted_attrs': conflicted_attrs,
    }


def print_mre_floor_summary(observed_mre, floor_result, observed_wmre=None):
    """
    Print the three-line MRE summary for the thesis:
        MRE observed, MRE floor (estimated), MRE adjusted.

    Parameters
    ----------
    observed_mre  : float  — solver's final MRE (from solver.final_mre)
    floor_result  : dict   — output of compute_scalar_mre_floor()
    observed_wmre : float or None — solver's final weighted MRE (optional)
    """
    mre_floor    = floor_result['mre_floor']
    mre_adj      = mre_adjusted(observed_mre, mre_floor)
    n_floored    = floor_result['n_floored']
    n_valid      = floor_result['n_valid']
    mre_floor_u  = floor_result['mre_floor_unary']
    pct_solver   = 100.0 * mre_adj / observed_mre if observed_mre > 0 else 0.0

    SEP = "─" * 78
    print(SEP)
    print("  MRE FLOOR ANALYSIS")
    print(SEP)
    print(f"  MRE observed            : {observed_mre:.5f}")
    if observed_wmre is not None:
        print(f"  Weighted MRE observed   : {observed_wmre:.5f}")
    print()
    print(f"  MRE floor (lower bound) : {mre_floor:.5f}  "
          f"({n_floored} of {n_valid} valid constraints have non-zero floor)")
    print(f"  MRE floor (unary only)  : {mre_floor_u:.5f}  "
          f"(exact for identified source conflicts on unary constraints)")
    print()
    print(f"  MRE adjusted            : {mre_adj:.5f}  "
          f"= max(0, {observed_mre:.5f} − {mre_floor:.5f})")
    print(f"  → At most {pct_solver:.1f}% of the observed MRE is attributable")
    print(f"    to solver approximation error; the remainder ({100-pct_solver:.1f}%)")
    print(f"    is the estimated data-inconsistency floor.")
    print(SEP)
    print()

# ================================================================== #
#  FULL MRE FLOOR — unary + binary, replaces compute_scalar_mre_floor #
# ================================================================== #

def compute_full_mre_floor(cs, cpts, marginals, alphas,
                            table_source, marginal_source,
                            attr_names, attr_meta,
                            reliability=None,
                            default_tag="Italy",
                            min_prob=1e-3):
    """
    Compute the MRE floor for ALL atomic constraints (unary + binary).

    For each constraint j we collect every independent estimate of the
    same joint probability P(A=a) or P(A=a, B=b):

      • Stated unary marginals from the marginals dict
      • CPT-implied unary marginals  (marginalise each binary CPT over parent)
      • CPT-implied binary probabilities  P(B=b|A=a) × P(A=a) for each CPT

    When multiple sources give different values for the same quantity,
    no algorithm can satisfy all simultaneously. The weighted compromise

        floor_point = Σ W_s α_s / Σ W_s

    is the estimated equilibrium, and

        floor_j = |floor_point − α_j| / α_j

    is the irreducible floor for that constraint.
    """
    from collections import defaultdict

    if reliability is None:
        from istat.geo_tagging import RELIABILITY as reliability

    attr_to_idx = {name: i for i, name in enumerate(attr_names)}

    # conflict_map  canonical_key → [(prob, weight, source_name, geo_tag)]
    # canonical_key for unary:  ((attr_idx, val_idx),)
    # canonical_key for binary: ((smaller_attr_idx, val_idx), (larger_attr_idx, val_idx))
    conflict_map = defaultdict(list)

    # ── 1a. stated unary marginals ─────────────────────────────────────────
    for attr_name, dist in marginals.items():
        a_idx = attr_to_idx.get(attr_name)
        if a_idx is None:
            continue
        tag = marginal_source.get(attr_name, default_tag)
        w   = reliability.get(tag, reliability[default_tag])
        for val_name, prob in dist.items():
            v_idx = attr_meta[attr_name].get('val_to_int', {}).get(val_name)
            if v_idx is None:
                continue
            conflict_map[((a_idx, v_idx),)].append(
                (float(prob), w, f"stated[{attr_name}]", tag))

    # ── 1b. CPT-implied probabilities ──────────────────────────────────────
    # projections of each ternary table, kept for pass-2 attribution of
    # conflict floors back to the ternary constraints themselves
    ternary_proj = {}

    for info in cpts:
        if info['name'].startswith('h_'):
            continue

        depth   = info['depth']
        parents = info['parents']
        child   = info['child']
        cpt     = info['norm_cpt']
        tag     = table_source.get(info['name'], default_tag)
        w       = reliability.get(tag, reliability[default_tag])
        c_idx   = attr_to_idx.get(child)
        if c_idx is None:
            continue

        if depth == 2:
            p_attr = parents[0]
            p_idx  = attr_to_idx.get(p_attr)
            p_marg = marginals.get(p_attr, {})
            if p_idx is None:
                continue

            # binary: P(A=pv, B=cv) = P(cv | pv) × P(pv)
            for pv, p_prob in p_marg.items():
                if p_prob < min_prob:
                    continue
                pv_idx = attr_meta[p_attr].get('val_to_int', {}).get(pv)
                if pv_idx is None or pv not in cpt:
                    continue
                for cv, cond_prob in cpt[pv].items():
                    cv_idx = attr_meta[child].get('val_to_int', {}).get(cv)
                    if cv_idx is None:
                        continue
                    joint = float(cond_prob) * float(p_prob)
                    # canonical key (smaller attr_idx first, matching build_constraint_set)
                    if p_idx < c_idx:
                        key = ((p_idx, pv_idx), (c_idx, cv_idx))
                    else:
                        key = ((c_idx, cv_idx), (p_idx, pv_idx))
                    conflict_map[key].append((joint, w, info['name'], tag))

            # unary: CPT-implied P(child=cv) by marginalising over parent
            c_vals = attr_meta[child].get('vals', [])
            for cv in c_vals:
                cv_idx = attr_meta[child].get('val_to_int', {}).get(cv)
                if cv_idx is None:
                    continue
                implied = sum(
                    float(cpt.get(pv, {}).get(cv, 0.0)) * float(pp)
                    for pv, pp in p_marg.items()
                )
                conflict_map[((c_idx, cv_idx),)].append(
                    (implied, w, f"{info['name']}→unary", tag))

        elif depth == 3:
            from istat.preprocess_istat import parent_joint_table
            p1_attr, p2_attr = parents[0], parents[1]
            p1_idx = attr_to_idx.get(p1_attr)
            p2_idx = attr_to_idx.get(p2_attr)
            if p1_idx is None or p2_idx is None:
                continue
            pjoint, _ = parent_joint_table(p1_attr, p2_attr, marginals, cpts)

            v2i_p1 = attr_meta[p1_attr].get('val_to_int', {})
            v2i_p2 = attr_meta[p2_attr].get('val_to_int', {})
            v2i_c  = attr_meta[child].get('val_to_int', {})

            # ternary joints: P(p1v, p2v, cv) = P(cv | p1v, p2v) × P(p1v, p2v)
            # (same construction as build_constraint_set, so a ternary target
            #  compares against itself with zero residual — the conflicts come
            #  from other sources implying the same binary/unary quantities)
            implied_b1 = defaultdict(float)   # (p1v, cv) → P(p1v, cv)
            implied_b2 = defaultdict(float)   # (p2v, cv) → P(p2v, cv)
            implied_u  = defaultdict(float)   # cv → P(cv)
            for p1v in cpt:
                p1v_idx = v2i_p1.get(p1v)
                if p1v_idx is None:
                    continue
                for p2v, row in cpt[p1v].items():
                    p2v_idx = v2i_p2.get(p2v)
                    if p2v_idx is None:
                        continue
                    pj = float(pjoint.get((p1v, p2v), 0.0))
                    for cv, cond in row.items():
                        cv_idx = v2i_c.get(cv)
                        if cv_idx is None:
                            continue
                        joint = float(cond) * pj
                        key = tuple(sorted(((p1_idx, p1v_idx),
                                            (p2_idx, p2v_idx),
                                            (c_idx,  cv_idx))))
                        conflict_map[key].append((joint, w, info['name'], tag))
                        implied_b1[(p1v_idx, cv_idx)] += joint
                        implied_b2[(p2v_idx, cv_idx)] += joint
                        implied_u[cv_idx]             += joint

            # Marginalised projections of the ternary CPT — these are what
            # conflict with the binary CPTs on the same variable pairs
            # (e.g. employment_age_sex marginalised over age vs the
            # sex→employment binary table). Skipped for partial-coverage
            # tables, where the projections would understate the true mass
            # and create spurious conflicts.
            if not info.get('is_partial', False):
                ternary_proj[info['name']] = {
                    'b1': dict(implied_b1), 'b2': dict(implied_b2),
                    'u': dict(implied_u),
                    'p1_idx': p1_idx, 'p2_idx': p2_idx, 'c_idx': c_idx,
                }
                for (p1v_idx, cv_idx), prob in implied_b1.items():
                    key = tuple(sorted(((p1_idx, p1v_idx), (c_idx, cv_idx))))
                    conflict_map[key].append(
                        (prob, w, f"{info['name']}→{p1_attr}×{child}", tag))
                for (p2v_idx, cv_idx), prob in implied_b2.items():
                    key = tuple(sorted(((p2_idx, p2v_idx), (c_idx, cv_idx))))
                    conflict_map[key].append(
                        (prob, w, f"{info['name']}→{p2_attr}×{child}", tag))
                for cv_idx, prob in implied_u.items():
                    conflict_map[((c_idx, cv_idx),)].append(
                        (prob, w, f"{info['name']}→unary", tag))

    # ── 2. per-constraint floor ─────────────────────────────────────────────
    m          = len(alphas)
    floor_vals = np.zeros(m, dtype=np.float64)
    valid      = alphas > min_prob

    for j in range(m):
        if not valid[j]:
            continue
        attrs = cs.attrs_list[j]
        vals  = cs.vals_list[j]
        # canonical key matching build_constraint_set
        key   = tuple(sorted(zip([int(a) for a in attrs],
                                  [int(v) for v in vals])))
        sources = conflict_map.get(key, [])
        if len(sources) > 1:
            probs   = np.array([s[0] for s in sources])
            weights = np.array([s[1] for s in sources])
            if weights.sum() <= 0:
                continue
            floor_pt = float(np.dot(probs, weights) / weights.sum())
            alpha_j  = float(alphas[j])
            if alpha_j > min_prob:
                floor_vals[j] = abs(floor_pt - alpha_j) / alpha_j
            continue

        # ── Ternary attribution ────────────────────────────────────────────
        # A ternary target has a single direct source (itself), so its
        # pairwise floor is zero. But its marginalised projections DO
        # conflict with binary CPTs implying the same quantity. If the
        # weighted compromise moves a projected mass P(pv, cv) by a
        # relative amount r, the ternary cells in that slice must shift
        # by ~the same relative amount (proportional allocation), so we
        # attribute floor_j = max over the constraint's three projections
        # of |compromise − table's own projection| / own projection.
        if len(sources) == 1 and len(attrs) == 3:
            tname = sources[0][2]
            proj  = ternary_proj.get(tname)
            if proj is None:
                continue
            val_of = {int(a): int(v) for a, v in zip(attrs, vals)}
            p1i, p2i, ci = proj['p1_idx'], proj['p2_idx'], proj['c_idx']
            if not all(ix in val_of for ix in (p1i, p2i, ci)):
                continue
            p1v, p2v, cv = val_of[p1i], val_of[p2i], val_of[ci]
            candidates = [
                (tuple(sorted(((p1i, p1v), (ci, cv)))), proj['b1'].get((p1v, cv))),
                (tuple(sorted(((p2i, p2v), (ci, cv)))), proj['b2'].get((p2v, cv))),
                (((ci, cv),),                            proj['u'].get(cv)),
            ]
            best = 0.0
            for pkey, own in candidates:
                if own is None or own <= min_prob:
                    continue
                entries = conflict_map.get(pkey, [])
                if len(entries) <= 1:
                    continue
                pw = np.array([e[1] for e in entries])
                if pw.sum() <= 0:
                    continue
                compromise = float(np.dot([e[0] for e in entries], pw) / pw.sum())
                best = max(best, abs(compromise - own) / own)
            floor_vals[j] = best

    # ── 3. aggregate ────────────────────────────────────────────────────────
    n_valid     = int(valid.sum())
    n_floored   = int((floor_vals > 1e-9).sum())
    n_conflicts = sum(1 for v in conflict_map.values() if len(v) > 1)
    mre_floor   = float(floor_vals[valid].mean()) if n_valid > 0 else 0.0

    arity = np.array([len(a) for a in cs.attrs_list])
    floor_by_arity = {}
    for ar in (1, 2, 3):
        msk = valid & (arity == ar)
        floor_by_arity[ar] = {
            'mean_floor': float(floor_vals[msk].mean()) if msk.any() else 0.0,
            'n_valid':    int(msk.sum()),
            'n_floored':  int((floor_vals[msk] > 1e-9).sum()),
        }
    conflicted = floor_vals > 1e-9
    floor_over_conflicted = (float(floor_vals[conflicted].mean())
                             if conflicted.any() else 0.0)

    return {
        'mre_floor':        mre_floor,
        'n_floored':        n_floored,
        'n_conflicts':      n_conflicts,
        'n_valid':          n_valid,
        'per_constraint':   floor_vals,
        'conflict_details': dict(conflict_map),
        'floor_by_arity':   floor_by_arity,
        'floor_over_conflicted': floor_over_conflicted,
    }


def print_full_mre_floor_summary(observed_mre, floor_result,
                                  observed_wmre=None):
    """Print the complete floor summary (unary + binary)."""
    mre_floor  = floor_result['mre_floor']
    mre_adj    = mre_adjusted(observed_mre, mre_floor)
    n_floored  = floor_result['n_floored']
    n_valid    = floor_result['n_valid']
    n_conf     = floor_result['n_conflicts']
    pct_data   = 100.0 * mre_floor / observed_mre if observed_mre > 0 else 0.0
    pct_solver = 100.0 * mre_adj  / observed_mre if observed_mre > 0 else 0.0

    SEP = "─" * 78
    print(SEP)
    print("  MRE FLOOR ANALYSIS  (unary + binary + ternary constraints)")
    print(SEP)
    print(f"  MRE observed            : {observed_mre:.5f}")
    if observed_wmre is not None:
        print(f"  Weighted MRE observed   : {observed_wmre:.5f}")
    print()
    print(f"  Conflict pairs detected : {n_conf}  "
          f"(joint probs implied by >1 independent source)")
    print(f"  Constraints with floor  : {n_floored} of {n_valid} valid")
    print()
    print(f"  MRE floor (estimated)   : {mre_floor:.5f}")
    fba = floor_result.get('floor_by_arity')
    if fba:
        for ar, lbl in ((1, 'unary  '), (2, 'binary '), (3, 'ternary')):
            s = fba.get(ar)
            if s and s['n_valid'] > 0:
                print(f"    floor {lbl}           : {s['mean_floor']:.5f}  "
                      f"({s['n_floored']} of {s['n_valid']} valid floored)")
    foc = floor_result.get('floor_over_conflicted')
    if foc:
        print(f"    mean over floored     : {foc:.5f}  "
              f"(conditional on having an identified conflict)")
    print(f"  MRE adjusted            : {mre_adj:.5f}  "
          f"= max(0, {observed_mre:.5f} − {mre_floor:.5f})")
    print()
    print(f"  Data inconsistency      : ~{pct_data:.1f}% of observed MRE")
    print(f"  Solver approximation    : at most ~{pct_solver:.1f}% of observed MRE")
    print()
    if pct_data < 30:
        print("  NOTE: the floor is a conservative lower bound — it only covers")
        print("  quantities implied by >1 independent source. Ternary floors are")
        print("  attributed via their marginalised projections (proportional")
        print("  allocation of the projected conflict). Systemic conflicts —")
        print("  one table vs the whole constraint system — remain invisible")
        print("  here; the stratified MRE by geographic source is the stronger")
        print("  evidence of data-inconsistency dominance.")
    print(SEP)
    print()