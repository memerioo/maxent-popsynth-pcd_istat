"""
plot_diagnostics.py
-------------------
Generates all figures and diversity metrics for the thesis, loading the
saved run history from test_4_ISTAT_history.pkl and the saved population
from test_4_ISTAT_pop500000.npy.

Run from the project root after a completed experiment:
    python experiments/plot_diagnostics.py

Outputs (written to experiments/figures/):
    convergence.pdf       — MRE / wMRE / MAE vs iteration (3 curves)
    stratified_mre.pdf    — bar chart: MRE per geographic source
    diversity.pdf         — diversity metrics vs raking (bar chart)
    mre_floor.pdf         — floor analysis for top conflicting variables

All figures are also saved as .png for easy inclusion in the thesis.
"""

import sys
import os
import pickle
import glob
import numpy as np

# ── Path Setup ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

EXP_DIR = os.path.join(ROOT_DIR, 'experiments')
FIG_DIR = os.path.join(EXP_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')        # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'figure.dpi':       150,
})

GEO_COLORS = {
    'BO':        '#1a6faf',   # deep blue  — most trusted
    'PBO':       '#3fa0d0',   # sky blue
    'EmiliaR':   '#f0a500',   # amber
    'NorthEast': '#d45f00',   # burnt orange
    'Italy':     '#b02020',   # red  — least trusted
    'ALL':       '#444444',
}


# ─────────────────────────────────────────────────────────────────────────────
#  Load saved data
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_n(n):
    """Format the pool size for figure titles ('500K', '50K', or '?')."""
    if n is None:
        return "?"
    n = int(n)
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}K"
    return f"{n:,}"


def load_run(exp_dir=EXP_DIR):
    """
    Load the saved history and its population.

    The pool filename is DISCOVERED rather than hardcoded. The previous
    version looked only for test_4_ISTAT_pop500000.npy, so a run at any
    other pool size silently produced "pool not found", skipping Figures 3
    and 5 and falling back to the last-iteration alpha_hat for the
    stratified table (which is why its numbers differed slightly from the
    run's own best-snapshot figures). Worse, when a stale 500k pool from an
    older run WAS present, it was loaded next to a fresh history and the two
    described different experiments.

    Preference order:
      1. the pool whose N matches run['N_pool']  (the correct one)
      2. the most recently modified test_4_ISTAT_pop*.npy, with a warning
    """
    hist_path = os.path.join(exp_dir, 'test_4_ISTAT_history.pkl')

    if not os.path.exists(hist_path):
        raise FileNotFoundError(
            f"No history file found at {hist_path}.\n"
            "Run  python experiments/test_4_ISTAT.py  first.")

    with open(hist_path, 'rb') as f:
        run = pickle.load(f)

    pool, pop_path = None, None
    n_pool = run.get('N_pool')
    if n_pool is not None:
        exact = os.path.join(exp_dir, f'test_4_ISTAT_pop{n_pool}.npy')
        if os.path.exists(exact):
            pop_path = exact

    if pop_path is None:
        cands = sorted(glob.glob(os.path.join(exp_dir, 'test_4_ISTAT_pop*.npy')),
                       key=os.path.getmtime, reverse=True)
        if cands:
            pop_path = cands[0]
            if n_pool is not None:
                print(f"  [!] No pool file for N={n_pool:,} (the size recorded in the")
                print(f"      history). Falling back to {os.path.basename(pop_path)},")
                print(f"      which may come from a DIFFERENT run.")

    if pop_path is not None:
        pool = np.load(pop_path)
        run['_pop_path'] = pop_path
        if n_pool is not None and len(pool) != n_pool:
            print(f"  [!] WARNING: pool has {len(pool):,} rows but the history says")
            print(f"      N_pool={n_pool:,}. These are different runs; pool-based")
            print(f"      figures and history-based figures will NOT agree.")

    return run, pool


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1: Convergence curves (MRE / wMRE / MAE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(run, fig_dir=FIG_DIR):
    hist  = run['history']
    iters = [h['t']            for h in hist]
    mre   = [h['mre']          for h in hist]
    wmre  = [h['weighted_mre'] for h in hist]
    mae   = [h['mae']          for h in hist]

    # best snapshot iteration: stored explicitly by the solver; fall back to
    # matching the selection metric for pkls saved by older versions.
    best_t = run.get('best_iter')
    if not best_t:
        sel = run.get('selection_metric', 'mre')
        key = 'weighted_mre' if sel == 'weighted_mre' else 'mre'
        tgt = run.get('final_weighted_mre' if key == 'weighted_mre' else 'final_mre')
        best_t = next((h['t'] for h in hist if h[key] == tgt), hist[-1]['t'])

    # 2x2 layout: relative error on top (linear + log), absolute error on the
    # bottom (linear + log).  The log-MAE panel is the informative one when the
    # iterate is oscillating: a limit cycle shows as a band of constant width
    # on the log axis, whereas genuine convergence narrows it.
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # lr-decay events (if the run recorded per-iteration lr)
    # lr-decay markers.
    #
    # These are only meaningful for a REACTIVE (plateau) schedule, where lr is
    # piecewise constant and drops a handful of times. Under a CONTINUOUS
    # schedule (cosine / exp) lr changes at EVERY iteration, so marking every
    # change would draw one vertical line per iteration and wash the whole
    # panel out. Detect that case and draw no markers instead; the schedule is
    # already described in the figure title.
    lr_hist   = [h.get('lr') for h in hist]
    lr_events = []
    if lr_hist[0] is not None:
        raw = [iters[i] for i in range(1, len(lr_hist))
               if lr_hist[i] != lr_hist[i - 1]]
        # a plateau schedule has few, large, discrete drops
        if len(raw) <= 20:
            lr_events = raw

    # NOTE: no smoothing anywhere — every history entry (one per outer
    # iteration) is plotted as its own dot, connected by straight segments.
    _line = dict(lw=1.0, marker='.', markersize=2.5)

    def _decorate(ax, show_best_label=False):
        ax.axvline(best_t, color='grey', lw=0.8, ls=':', alpha=0.7,
                   label=f'Best snapshot (iter {best_t})' if show_best_label else None)
        for i, ev in enumerate(lr_events):
            ax.axvline(ev, color='#8a4fbf', lw=0.7, ls='--', alpha=0.5,
                       label='lr decay' if (i == 0 and show_best_label) else None)

    # ── Top-left: linear-scale MRE and wMRE ─────────────────────────────────
    ax1.plot(iters, mre,  color='#d45f00',
             label='MRE (unweighted, all constraints)', **_line)
    ax1.plot(iters, wmre, color='#1a6faf',
             label='wMRE (reliability-weighted)', **_line)
    ax1.axhline(run['final_mre'],          color='#d45f00', lw=0.8, ls='--', alpha=0.6)
    ax1.axhline(run['final_weighted_mre'], color='#1a6faf', lw=0.8, ls='--', alpha=0.6)
    _decorate(ax1, show_best_label=True)
    ax1.set_xlabel('Outer iteration')
    ax1.set_ylabel('Mean Relative Error')
    ax1.set_title('Relative error (linear scale)')
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.set_ylim(bottom=0)

    # ── Top-right: log-scale MRE ────────────────────────────────────────────
    ax2.semilogy(iters, mre,  color='#d45f00', label='MRE (unweighted)', **_line)
    ax2.semilogy(iters, wmre, color='#1a6faf',
                 label='wMRE (reliability-weighted)', **_line)
    ax2.axhline(run['final_mre'],          color='#d45f00', lw=0.8, ls='--', alpha=0.6)
    ax2.axhline(run['final_weighted_mre'], color='#1a6faf', lw=0.8, ls='--', alpha=0.6)
    _decorate(ax2)
    bo_mre = run.get('bo_mre', None)
    if bo_mre:
        ax2.axhline(bo_mre, color='#1a6faf', lw=1.0, ls='-.', alpha=0.5,
                    label=f'BO MRE = {bo_mre:.4f}')
    ax2.set_xlabel('Outer iteration')
    ax2.set_ylabel('Mean Relative Error  (log scale)')
    ax2.set_title('Relative error (log scale)')
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda y, _: f'{y:.2f}' if y >= 0.01 else f'{y:.3f}'))
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, which='both', alpha=0.25)

    # ── Bottom-left: linear-scale MAE ───────────────────────────────────────
    ax3.plot(iters, mae, color='#2aa44e', label='MAE (all constraints)', **_line)
    ax3.axhline(run['final_mae'], color='#2aa44e', lw=0.8, ls='--', alpha=0.6,
                label=f"Best MAE = {run['final_mae']:.5f}")
    _decorate(ax3)
    ax3.set_xlabel('Outer iteration')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.set_title('Absolute error (linear scale)')
    ax3.legend(fontsize=8, framealpha=0.9)
    ax3.set_ylim(bottom=0)

    # ── Bottom-right: log-scale MAE ─────────────────────────────────────────
    ax4.semilogy(iters, mae, color='#2aa44e', label='MAE (all constraints)', **_line)
    ax4.axhline(run['final_mae'], color='#2aa44e', lw=0.8, ls='--', alpha=0.6,
                label=f"Best MAE = {run['final_mae']:.5f}")
    _decorate(ax4)
    ax4.set_xlabel('Outer iteration')
    ax4.set_ylabel('Mean Absolute Error  (log scale)')
    ax4.set_title('Absolute error (log scale)')
    ax4.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda y, _: f'{y:.3f}' if y >= 0.001 else f'{y:.4f}'))
    ax4.legend(fontsize=8, framealpha=0.9)
    ax4.grid(True, which='both', alpha=0.25)

    sched = run.get('lr_schedule', 'plateau')
    if sched == 'plateau':
        sched_txt = f"lr {run.get('lr', '?')} plateau"
    else:
        sched_txt = f"lr {run.get('lr', '?')}\u2192{run.get('lr_min', '?')} {sched}"
    fig.suptitle(
        f"GibbsPCDSolver — Bologna ISTAT  "
        f"(N={_fmt_n(run.get('N_pool'))}, {run['n_iters']} iters, "
        f"{run['fit_time']/3600:.1f}h, {sched_txt})",
        fontsize=13, y=0.995)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'convergence.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] convergence.pdf / .png  (2x2: MRE linear/log, MAE linear/log)")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2: Stratified MRE by geographic source
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3: Diversity metrics  (Section 5.3 of Degli Esposti 2026)
# ─────────────────────────────────────────────────────────────────────────────
#
#  The paper defines and uses:
#    Neff     = (sum_i w_i^2)^{-1}       [eq. 1]
#    Neff/N                              effective fraction of pool
#    H        = -sum_x p_hat(x) log(p_hat(x))   Shannon entropy [nats]
#    H_max    = log|X|                   theoretical maximum entropy
#    H/H_max  = fractional entropy
#    Gini     = weight concentration (0=perfect equality, 1=fully concentrated)
#    Distinct profiles = |{unique rows in pool}|
#
#  For GibbsPCDSolver all pool weights are uniform (w_i = 1/N by construction),
#  so Neff = N and Gini = 0. For raking, Neff ≈ 0.012 N and Gini ≈ 0.951.
#  We still compute them formally so the thesis table is consistent.
#
#  Added for your Bologna project (not in the paper):
#    Entropy per attribute  H_k = -sum_v p_k(v) log p_k(v)  — useful for
#    spotting which attributes are being collapsed by the solver.

def compute_diversity(pool, K_attr_names=None):
    """
    pool : (N, K) int32 array — persistent Gibbs pool
    Returns a dict of diversity metrics.
    """
    N, K = pool.shape

    # ── Neff (trivially N for PCD, compute formally) ──────────────────────────
    weights = np.ones(N) / N          # uniform pool weights
    neff    = float(1.0 / np.sum(weights ** 2))   # = N
    neff_ratio = neff / N

    # ── Distinct profiles ─────────────────────────────────────────────────────
    # Convert each row to a tuple and count unique ones
    tuples = [tuple(row) for row in pool]
    n_distinct = len(set(tuples))

    # ── Shannon entropy from empirical profile distribution ───────────────────
    from collections import Counter
    counts = Counter(tuples)
    probs  = np.array(list(counts.values()), dtype=np.float64) / N
    H      = float(-np.sum(probs * np.log(probs + 1e-300)))   # nats

    # Theoretical H_max = log|X|
    # Can't enumerate X, so estimate from pool's marginals:
    #   H_max = sum_k H(marginal_k)  [upper bound assuming independence]
    H_max_indep = 0.0
    for k in range(K):
        vals, cnts = np.unique(pool[:, k], return_counts=True)
        pk = cnts / N
        H_max_indep -= float(np.sum(pk * np.log(pk + 1e-300)))

    # ── Gini coefficient on weights ───────────────────────────────────────────
    # For uniform weights Gini=0; include formula for completeness / raking comparison
    sorted_w = np.sort(weights)
    n_w      = len(sorted_w)
    cum_w    = np.cumsum(sorted_w)
    gini     = float((2 * np.sum((np.arange(1, n_w+1)) * sorted_w) / (n_w * cum_w[-1])) - (n_w + 1) / n_w)

    # ── Per-attribute entropy ─────────────────────────────────────────────────
    per_attr_H = {}
    for k in range(K):
        vals, cnts = np.unique(pool[:, k], return_counts=True)
        pk = cnts / N
        attr_name = K_attr_names[k] if K_attr_names else str(k)
        per_attr_H[attr_name] = float(-np.sum(pk * np.log(pk + 1e-300)))

    return {
        'N':              N,
        'neff':           neff,
        'neff_ratio':     neff_ratio,
        'n_distinct':     n_distinct,
        'distinct_ratio': n_distinct / N,
        'H':              H,
        'H_max_indep':    H_max_indep,
        'H_frac':         H / H_max_indep if H_max_indep > 0 else float('nan'),
        'gini':           gini,
        'per_attr_H':     per_attr_H,
    }


def plot_diversity(pool, fig_dir=FIG_DIR):
    """
    Produces two panels:
    Left  — bar chart comparing GibbsPCD vs raking on the three main metrics
             (Neff/N, H/H_max, distinct profiles/N), using raking reference
             values from Table 6 of Degli Esposti 2026.
    Right — per-attribute Shannon entropy (shows which attributes are
             well-spread vs collapsed).
    """
    try:
        from istat.attr_meta_ISTAT import ATTR_NAMES_SYNTH
    except ImportError:
        ATTR_NAMES_SYNTH = None

    print("  Computing diversity metrics from pool...")
    div = compute_diversity(pool, K_attr_names=ATTR_NAMES_SYNTH)

    # ── Print to console ──────────────────────────────────────────────────────
    print(f"\n  ── Diversity Metrics ──────────────────────────────")
    print(f"  Pool size N              : {div['N']:,}")
    print(f"  Neff                     : {div['neff']:,.0f}  (= N for PCD by construction)")
    print(f"  Neff / N                 : {div['neff_ratio']:.1%}")
    print(f"  Distinct profiles        : {div['n_distinct']:,}  ({div['distinct_ratio']:.1%} of pool)")
    print(f"  Shannon entropy H        : {div['H']:.4f} nats")
    print(f"  H_max (indep. bound)     : {div['H_max_indep']:.4f} nats")
    print(f"  H / H_max                : {div['H_frac']:.3f}")
    print(f"  Gini (weight distrib.)   : {div['gini']:.4f}  (0 = perfect equality)")
    print()

    # Reference values from Table 6, Degli Esposti 2026 (Syn-ISTAT K=15 N=100K)
    # We don't have raking output for YOUR specific run, so we show the paper
    # ratios as a reference band. Label clearly in the thesis.
    RAKING_NEFF_RATIO   = 0.012      # Table 6: 1,152 / 100,000
    RAKING_DISTINCT_RATIO = 0.258    # Table 6: 25,786 / 100,000

    fig, (ax_cmp, ax_attr) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: comparison bars
    metrics = ['Neff/N', 'Distinct/N']
    gibbs_vals  = [div['neff_ratio'], div['distinct_ratio']]
    raking_vals = [RAKING_NEFF_RATIO, RAKING_DISTINCT_RATIO]
    x = np.arange(len(metrics))
    w = 0.3

    ax_cmp.bar(x - w/2, gibbs_vals,  w, label='GibbsPCDSolver (this run)', color='#1a6faf', alpha=0.85)
    ax_cmp.bar(x + w/2, raking_vals, w, label='Raking (Degli Esposti 2026, K=15 reference)', color='#b02020', alpha=0.70)

    for i, (gv, rv) in enumerate(zip(gibbs_vals, raking_vals)):
        ratio = gv / rv if rv > 0 else float('inf')
        ax_cmp.text(x[i], max(gv, rv) + 0.02, f'{ratio:.0f}×', ha='center', fontsize=10, fontweight='bold')

    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(metrics, fontsize=11)
    ax_cmp.set_ylabel('Fraction of N')
    ax_cmp.set_ylim(0, 1.15)
    ax_cmp.set_title('Population Diversity\n(ratio above bars = GibbsPCD advantage)')
    ax_cmp.legend(fontsize=9)

    # Add entropy as text annotation since it's a different scale
    ax_cmp.text(0.97, 0.97,
                f"Shannon entropy\nGibbsPCD : {div['H']:.2f} nats\n"
                f"H_max (indep): {div['H_max_indep']:.2f} nats\n"
                f"H / H_max : {div['H_frac']:.3f}",
                transform=ax_cmp.transAxes, va='top', ha='right',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='#e8f0fe', alpha=0.8))

    # Right: per-attribute entropy
    attr_names = list(div['per_attr_H'].keys())
    attr_H     = list(div['per_attr_H'].values())
    # theoretical max per attribute = log(d_k)
    # We can't easily get d_k here without ConstraintSet, so just show raw H
    y_pos = np.arange(len(attr_names))
    ax_attr.barh(y_pos, attr_H, color='#3fa0d0', alpha=0.80)
    ax_attr.set_yticks(y_pos)
    ax_attr.set_yticklabels(attr_names, fontsize=8)
    ax_attr.set_xlabel('Shannon entropy H_k [nats]')
    ax_attr.set_title('Per-attribute entropy in synthesised population\n'
                      '(low = that attribute is poorly diversified)')
    ax_attr.invert_yaxis()

    fig.suptitle('GibbsPCDSolver — Population Diversity Diagnostics', fontsize=12)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'diversity.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] diversity.pdf / .png")
    return div


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4: MRE floor / source-conflict analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_mre_floor(fig_dir=FIG_DIR):
    """
    Runs the mre_floor report and plots the top conflicting variables with
    their weighted-compromise floor points vs. each source's stated target.
    """
    from istat.preprocess_istat import discover_cpts, _implied_binary, _implied_ternary
    import istat.attr_meta_ISTAT as _meta
    from istat.attr_meta_ISTAT import marginals, ATTR_META
    from istat.geo_tagging import parse_source_tags, RELIABILITY
    from istat.mre_floor import variable_floor_report

    attr_meta_path = os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py')
    table_source, marginal_source = parse_source_tags(attr_meta_path)
    cpts = discover_cpts(_meta)

    # Build implied_vs_stated inline (avoids the broken import in diagnose_istat.py)
    from collections import defaultdict
    def implied_vs_stated(cpts, marg, top_n=20):
        by_child = defaultdict(list)
        for c in cpts:
            if c['name'].startswith('h_'):   # ← add this line
                continue                     # h_ tables are logic, not data
            by_child[c['child']].append(c)
        rows = []
        for c_attr, cpt_list in by_child.items():
            stated = marg.get(c_attr, {})
            c_vals = ATTR_META[c_attr]['vals']
            for info in cpt_list:
                if info['depth'] == 2:
                    impl = _implied_binary(info['norm_cpt'], info['parents'][0], c_attr, marg)
                else:
                    impl = _implied_ternary(info['norm_cpt'], info['parents'][0],
                                            info['parents'][1], c_attr, marg)
                if impl is None:
                    continue
                max_disc = max(abs(impl.get(v,0) - stated.get(v,0)) for v in c_vals) if stated else 1.0
                rows.append((c_attr, info['name'], max_disc, impl, stated))
        rows.sort(key=lambda x: -x[2])
        return rows[:top_n]

    rows  = implied_vs_stated(cpts, marginals, top_n=30)
    report = variable_floor_report(rows, marginals, table_source, marginal_source)

    # Extract top N conflicts with largest max_residual
    all_conflicts = []
    for attr, per_value in report.items():
        for v, info in per_value.items():
            if info['max_residual'] > 0.01:
                all_conflicts.append((attr, v, info['max_residual'],
                                      info['floor_point'], info['sources']))
    all_conflicts.sort(key=lambda x: -x[2])
    top = all_conflicts[:12]

    if not top:
        print("  [i] No conflicts above threshold — skipping mre_floor figure.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    y_labels = [f"{a}={v}" for a, v, *_ in top]
    y_pos    = np.arange(len(y_labels))

    # For each conflict draw: floor point (diamond) and each source target (dot)
    MARKER = {'BO': 'D', 'PBO': 's', 'EmiliaR': '^', 'NorthEast': 'v', 'Italy': 'o'}

    for i, (attr, v, max_res, floor_pt, sources_list) in enumerate(top):
        # floor point
        ax.plot(floor_pt, i, 'k|', ms=16, mew=2.5, zorder=5)
        seen_labels = set()
        for label, p, w in sources_list:
            # extract tag from label like "stated[BO]" or "sex_MainTranspWorker[Italy]"
            tag = label.split('[')[-1].rstrip(']') if '[' in label else 'Italy'
            color = GEO_COLORS.get(tag, '#888888')
            mk    = MARKER.get(tag, 'o')
            lbl   = tag if tag not in seen_labels else ''
            seen_labels.add(tag)
            ax.plot(p, i + (0 if lbl else 0), mk, color=color, ms=8, alpha=0.85,
                    label=lbl, zorder=4)
            # thin line from target to floor
            ax.plot([p, floor_pt], [i, i], '-', color=color, lw=0.8, alpha=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Target probability')
    ax.set_title('MRE floor analysis: source conflicts\n'
                 'Black bar = reliability-weighted compromise point; '
                 'coloured markers = each source\'s stated target\n'
                 '(distance between marker and black bar = irreducible solver floor for that constraint)')

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=9, title='Source', loc='lower right')
    ax.axvline(0, color='black', lw=0.4, alpha=0.3)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'mre_floor.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] mre_floor.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: re-estimate alpha_hat from pool without loading the full solver
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_alpha_hat(pool, run):
    """
    Vectorised: for each constraint j defined by (attrs, vals), count the
    fraction of pool rows that match. Uses the last history entry's alpha_hat
    as fallback if pool is None.
    """
    # We need attrs_list and vals_list — they weren't saved in the history pkl.
    # Load them fresh from the constraint set.
    try:
        from istat.attr_meta_ISTAT import marginals as CLEAN_MARGINALS
        from istat.preprocess_istat import build_constraint_set, build_constraint_weights
        from istat.geo_tagging import parse_source_tags
        attr_meta_path = os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py')
        table_source, marginal_source = parse_source_tags(attr_meta_path)
        cs, _ = build_constraint_set(CLEAN_MARGINALS, table_source, marginal_source)
    except Exception as e:
        print(f"  [!] Could not rebuild ConstraintSet for alpha_hat estimation: {e}")
        print(f"  [!] Falling back to last-iteration alpha_hat from history.")
        return run['history'][-1]['alpha_hat']

    m = cs.m
    alpha_hat = np.zeros(m)
    for j in range(m):
        attrs = cs.attrs_list[j]
        vals  = cs.vals_list[j]
        alpha_hat[j] = np.all(pool[:, attrs] == vals[np.newaxis, :], axis=1).mean()
    return alpha_hat


# patch the stratified plot to use this helper
def plot_stratified_mre_fixed(run, pool, fig_dir=FIG_DIR):
    # Always rebuild cs from the current attr_meta_ISTAT.py so that the
    # constraint count matches the pool, regardless of which pkl is loaded.
    # This handles the common case where the pkl is from an older run.
    try:
        from istat.attr_meta_ISTAT import marginals as CLEAN_MARGINALS
        from istat.preprocess_istat import build_constraint_set, build_constraint_weights
        from istat.geo_tagging import parse_source_tags
        attr_meta_path = os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py')
        table_source, marginal_source = parse_source_tags(attr_meta_path)
        cs, sources = build_constraint_set(CLEAN_MARGINALS, table_source, marginal_source)
        alphas  = cs.alphas_array
        weights = build_constraint_weights(sources)
    except Exception as e:
        print(f"  [!] Could not rebuild ConstraintSet: {e} — falling back to saved pkl values")
        cs      = None
        sources = run['sources']
        alphas  = run['alphas']
        weights = run['weights']

    # Compute alpha_hat from pool using the rebuilt cs
    if pool is not None:
        m = len(alphas)
        alpha_hat = np.zeros(m)
        for j in range(m):
            attrs = cs.attrs_list[j]
            vals  = cs.vals_list[j]
            alpha_hat[j] = np.all(pool[:, attrs] == vals[np.newaxis, :], axis=1).mean()
    else:
        alpha_hat = run['history'][-1]['alpha_hat']
        if len(alpha_hat) != len(alphas):
            print("  [!] Shape mismatch and no pool — cannot compute stratified MRE")
            return

    valid   = alphas > 1e-3
    abs_err = np.abs(alpha_hat - alphas)
    rel_err = np.where(valid, abs_err / np.where(valid, alphas, 1.0), np.nan)
    src_arr = np.array(sources)
    arity   = (np.array([len(a) for a in cs.attrs_list])
               if cs is not None else np.full(len(alphas), 2))
    n_valid_total = int(valid.sum())
    global_mre    = float(np.nanmean(rel_err[valid]))

    from istat.geo_tagging import RELIABILITY
    order   = ['BO', 'PBO', 'EmiliaR', 'NorthEast', 'Italy']
    tags_plot = [t for t in order if t in set(sources)]

    def _row(mask):
        mask = mask & valid
        if not np.any(mask):
            return None
        w = weights[mask]
        mre_g = float(np.nanmean(rel_err[mask]))
        return {
            'mre':  mre_g,
            'wmre': float(np.average(rel_err[mask], weights=w)) if w.sum() > 0 else mre_g,
            'n':    int(mask.sum()),
            'share': 100.0 * mre_g * mask.sum() / (n_valid_total * global_mre),
        }

    rows = {}
    for tag in tags_plot:
        r = _row(src_arr == tag)
        if r:
            rows[tag] = r

    arity_labels = {1: 'unary', 2: 'binary', 3: 'ternary'}
    arity_rows = {}
    for a in (1, 2, 3):
        r = _row(arity == a)
        if r:
            arity_rows[arity_labels[a]] = r

    fig, (ax, axA) = plt.subplots(1, 2, figsize=(15, 4.8),
                                  gridspec_kw={'width_ratios': [1.5, 1]})
    x = np.arange(len(rows))
    bw = 0.35
    keys   = list(rows.keys())
    mres   = [rows[t]['mre']  for t in keys]
    wmres  = [rows[t]['wmre'] for t in keys]
    ns     = [rows[t]['n']    for t in keys]
    colors = [GEO_COLORS.get(t, '#888') for t in keys]

    bars1 = ax.bar(x - bw/2, mres,  bw, label='MRE (plain)',   alpha=0.85, color=colors)
    ax.bar(       x + bw/2, wmres, bw, label='MRE (weighted)', alpha=0.55, color=colors, hatch='//')

    for bar, n, t in zip(bars1, ns, keys):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'n={n}\n{rows[t]["share"]:.0f}% of MRE',
                ha='center', va='bottom', fontsize=7.5)

    for i, tag in enumerate(keys):
        w_val = RELIABILITY.get(tag, 0.15)
        ax.text(x[i], -0.02, f'W={w_val:.2f}', ha='center', va='top', fontsize=8,
                color=GEO_COLORS.get(tag, '#888'))

    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel('Mean Relative Error')
    ax.set_title('Stratified MRE by geographic source\n'
                 'annotation = additive share of the global MRE')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=-0.05)
    ax.axhline(0, color='black', lw=0.5)

    # ── right panel: by constraint arity ────────────────────────────────────
    ARITY_COLORS = {'unary': '#4c78a8', 'binary': '#f58518', 'ternary': '#54a24b'}
    xa    = np.arange(len(arity_rows))
    akeys = list(arity_rows.keys())
    amre  = [arity_rows[k]['mre']  for k in akeys]
    awmre = [arity_rows[k]['wmre'] for k in akeys]
    acol  = [ARITY_COLORS[k] for k in akeys]
    barsA = axA.bar(xa - bw/2, amre,  bw, alpha=0.85, color=acol, label='MRE (plain)')
    axA.bar(       xa + bw/2, awmre, bw, alpha=0.55, color=acol, hatch='//',
                   label='MRE (weighted)')
    for bar, k in zip(barsA, akeys):
        axA.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"n={arity_rows[k]['n']}\n{arity_rows[k]['share']:.0f}% of MRE",
                 ha='center', va='bottom', fontsize=7.5)
    axA.set_xticks(xa)
    axA.set_xticklabels(akeys)
    axA.set_title('Stratified MRE by constraint arity\n'
                  'annotation = additive share of the global MRE')
    axA.legend(fontsize=9)
    axA.set_ylim(bottom=-0.05)
    axA.axhline(0, color='black', lw=0.5)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'stratified_mre.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] stratified_mre.pdf / .png")

    print(f"\n  {'Source':<12} {'n':>5} {'W':>5} {'MRE':>9} {'wMRE':>9}")
    print(f"  {'─'*46}")
    for tag in keys:
        w_val = RELIABILITY.get(tag, 0.15)
        print(f"  {tag:<12} {rows[tag]['n']:>5} {w_val:>5.2f} "
              f"{rows[tag]['mre']:>9.4f} {rows[tag]['wmre']:>9.4f}")




# ─────────────────────────────────────────────────────────────────────────────
#  Figure 5: α̂ vs α scatter  (Degli Esposti 2026 Fig. 1 right panel style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_scatter(pool, fig_dir=FIG_DIR):
    """
    Scatter plot of estimated constraint frequencies α̂_j (pool) vs.
    targets α_j, coloured by geographic source and shaped by arity.
    Points on the diagonal = perfect fit.
    Distance from diagonal = residual error for that constraint.

    Matches the right panel of Figure 1 in Degli Esposti (2026).
    """
    from istat.attr_meta_ISTAT import marginals as CLEAN_MARGINALS
    from istat.preprocess_istat import build_constraint_set, build_constraint_weights
    from istat.geo_tagging import parse_source_tags, RELIABILITY

    attr_meta_path = os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py')
    table_source, marginal_source = parse_source_tags(attr_meta_path)
    cs, sources = build_constraint_set(CLEAN_MARGINALS, table_source, marginal_source)
    alphas  = cs.alphas_array
    weights = build_constraint_weights(sources)
    N = pool.shape[0]

    # Estimate α̂ from pool
    alpha_hat = np.zeros(cs.m)
    for j in range(cs.m):
        attrs = cs.attrs_list[j]
        vals  = cs.vals_list[j]
        alpha_hat[j] = np.all(pool[:, attrs] == vals[np.newaxis, :], axis=1).mean()

    valid = alphas > 1e-3
    src_arr  = np.array(sources)
    arity    = np.array([len(cs.attrs_list[j]) for j in range(cs.m)])

    # ── Two panels: full range and zoomed ────────────────────────────────────
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5.5))

    order  = ['BO', 'PBO', 'EmiliaR', 'NorthEast', 'Italy']
    marker = {1: 'o', 2: 's', 3: '^'}   # arity → marker shape

    for panel, ax, title_suffix in [
            (0, ax_full, 'full range'),
            (1, ax_zoom, 'zoomed  α < 0.15')]:

        zoom_mask = (alphas < 0.15) if panel == 1 else np.ones(cs.m, dtype=bool)

        for tag in order:
            tag_mask = (src_arr == tag) & valid & zoom_mask
            if not np.any(tag_mask):
                continue
            for ar in [1, 2, 3]:
                mask = tag_mask & (arity == ar)
                if not np.any(mask):
                    continue
                label = f'{tag} ({"unary" if ar==1 else "binary" if ar==2 else "ternary"})'
                ax.scatter(
                    alphas[mask], alpha_hat[mask],
                    color=GEO_COLORS.get(tag, '#888'),
                    marker=marker[ar],
                    s=18 if ar == 1 else 12,
                    alpha=0.65,
                    linewidths=0,
                    label=label,
                )

        # Diagonal y = x (perfect fit)
        lim = 0.15 if panel == 1 else max(alphas[valid].max(), alpha_hat[valid].max()) * 1.05
        ax.plot([0, lim], [0, lim], 'k-', lw=1.0, alpha=0.5, label='y = x  (perfect fit)')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel('Target  α_j', fontsize=11)
        ax.set_ylabel('Estimated  α̂_j  (pool)', fontsize=11)
        ax.set_title(f'Constraint frequencies: target vs. estimate\n({title_suffix})')
        ax.set_aspect('equal')

        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            key = l.split('(')[0].strip()   # deduplicate by source tag only
            if key not in seen:
                seen.add(key); h2.append(h); l2.append(l)
        ax.legend(h2, l2, fontsize=7, ncol=2, framealpha=0.9)

    # Compute R² on valid constraints for annotation
    ss_res = np.sum((alpha_hat[valid] - alphas[valid])**2)
    ss_tot = np.sum((alphas[valid]    - alphas[valid].mean())**2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    mae    = float(np.mean(np.abs(alpha_hat[valid] - alphas[valid])))
    ax_full.text(0.03, 0.97,
                 f'R² = {r2:.4f}\nMAE = {mae:.5f}\nn = {valid.sum()}',
                 transform=ax_full.transAxes, va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(
        'GibbsPCDSolver — Bologna ISTAT: estimated vs. target constraint frequencies\n'
        'Marker shape: ○ unary  □ binary  △ ternary  |  Colour: geographic source',
        fontsize=11)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'alpha_scatter.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] alpha_scatter.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4 (replacement): LP MRE floor
# ─────────────────────────────────────────────────────────────────────────────
#
#  Replaces the old pairwise source-conflict dot plot. The pairwise floor only
#  saw quantities that two sources state DIRECTLY, so it was blind to
#  multi-way conflicts (a marginal contradicting a conditional times another
#  marginal) and to infeasibility created by the structural zeros. It was also
#  a heuristic rather than a bound.
#
#  The LP floor minimises the weighted error over the LOCAL POLYTOPE: all
#  families of per-scope marginals that are non-negative, normalised,
#  consistent on overlaps, and zero on structurally forbidden patterns. Every
#  genuine joint distribution projects into that set, so the LP optimum is a
#  rigorous LOWER BOUND on the error any solver could achieve.
#
#  Left panel : observed error vs the bound, per geographic source. The gap is
#               the part that is NOT forced by the data.
#  Right panel: the individual constraints with the largest unavoidable
#               relative error, with the target and the best value any
#               distribution can give it.

def plot_lp_floor(run, pool, fig_dir=FIG_DIR, top=12):
    try:
        from istat.mre_floor_lp import compute_lp_mre_floor
        from istat.preprocess_istat import build_constraint_weights
        from istat.preprocess_istat import build_constraint_set as _bcs
        from istat.geo_tagging import parse_source_tags as _pst
        from istat.attr_meta_ISTAT import (marginals as _marg,
                                           ATTR_NAMES_SYNTH as _AN,
                                           ATTR_META as _AM)
    except ImportError as e:
        print(f"  [!] LP floor figure skipped: {e}")
        return None

    ts, ms = _pst(os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py'))
    cs, sources = _bcs(_marg, ts, ms)
    W = build_constraint_weights(sources)
    res = compute_lp_mre_floor(cs, weights=W, tighten='pairwise', verbose=False)
    if not res.get('success'):
        print(f"  [!] LP floor figure skipped: {res.get('status')}")
        return None

    per     = res['per_constraint']
    alphas  = cs.alphas_array
    valid   = alphas > 1e-3
    src_arr = np.array(sources)

    # observed per-constraint relative error from the delivered pool
    obs_per = None
    if pool is not None:
        alpha_hat = _estimate_alpha_hat(pool, run)
        if alpha_hat is not None:
            obs_per = np.where(valid,
                               np.abs(alpha_hat - alphas) / np.where(valid, alphas, 1.0),
                               np.nan)

    order = [t for t in ('BO', 'PBO', 'EmiliaR', 'NorthEast', 'Italy')
             if t in set(sources)]
    floor_by = [float(np.mean(per[(src_arr == t) & valid])) for t in order]
    obs_by   = ([float(np.nanmean(obs_per[(src_arr == t) & valid])) for t in order]
                if obs_per is not None else None)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={'width_ratios': [1, 1.35]})

    x = np.arange(len(order))
    if obs_by is not None:
        axL.bar(x - 0.2, obs_by, 0.4, color='#d45f00', label='Observed MRE')
    axL.bar(x + 0.2, floor_by, 0.4, color='#3c7ab5',
            label='LP floor (unavoidable)')
    if obs_by is not None:
        for xi, (o, f) in enumerate(zip(obs_by, floor_by)):
            if o > 0:
                axL.annotate(f'{100*f/o:.0f}%\nforced',
                             xy=(xi, max(o, f)), xytext=(0, 6),
                             textcoords='offset points', ha='center', fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels(order)
    axL.set_ylabel('Mean Relative Error')
    axL.set_title('Observed error vs. rigorous lower bound\n'
                  'by geographic source', fontsize=11)
    axL.legend(fontsize=9)
    axL.grid(axis='y', alpha=0.25)

    idx = np.argsort(per)[::-1][:top]
    idx = [j for j in idx if per[j] > 1e-9][::-1]
    labels, tg, bs = [], [], []
    for j in idx:
        attrs = [_AN[a] for a in cs.attrs_list[j]]
        vals  = [_AM[attrs[i]]['vals'][v] for i, v in enumerate(cs.vals_list[j])]
        lab = ", ".join(f"{a}={v}" for a, v in zip(attrs, vals))
        labels.append(f"[{sources[j]}] {lab[:46]}")
        tg.append(alphas[j]); bs.append(res['mu_hat'][j])

    y = np.arange(len(idx))
    for yi, (t, b) in enumerate(zip(tg, bs)):
        axR.plot([t, b], [yi, yi], color='#bbbbbb', lw=1.4, zorder=1)
    axR.scatter(tg, y, s=52, color='#d45f00', zorder=3,
                label='Published target', marker='o')
    axR.scatter(bs, y, s=52, color='#3c7ab5', zorder=3,
                label='Best achievable (LP)', marker='D')
    axR.set_yticks(y); axR.set_yticklabels(labels, fontsize=7.5)
    axR.set_xlabel('Probability')
    axR.set_title('Constraints with the largest unavoidable error\n'
                  '(gap = irreducible, no distribution can close it)', fontsize=11)
    axR.legend(fontsize=9, loc='lower right')
    axR.grid(axis='x', alpha=0.25)

    fig.suptitle(
        f"LP MRE floor — local-polytope lower bound   "
        f"(floor={res['floor_unweighted']:.4f} unweighted, "
        f"{res['floor_weighted']:.4f} weighted; "
        f"{res['n_floored']} of {res['n_valid']} constraints floored)",
        fontsize=12, y=1.0)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'mre_floor_lp.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print("  [✓] mre_floor_lp.pdf / .png")
    return res


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 6: pie chart — share of the global MRE by geographic source
# ─────────────────────────────────────────────────────────────────────────────
#
#  Uses the additive contribution of Equation (19):
#      C_g = n_g * MRE_g / n_total ,   sum_g C_g = MRE_global
#  so the wedges are literal shares of the headline number, not a rescaling.
#  The companion wedge set shows how many CONSTRAINTS each source holds, which
#  makes the point that Italy dominates the error partly because it is the
#  largest group and partly because it fits worst.

def plot_source_pie(run, pool, fig_dir=FIG_DIR):
    if pool is None:
        print("  [!] source pie skipped (no pool)")
        return
    from istat.preprocess_istat import build_constraint_set as _bcs
    from istat.geo_tagging import parse_source_tags as _pst
    from istat.attr_meta_ISTAT import (marginals as _marg,
                                       ATTR_NAMES_SYNTH as _AN,
                                       ATTR_META as _AM)
    ts, ms = _pst(os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py'))
    cs, sources = _bcs(_marg, ts, ms)
    alphas = cs.alphas_array
    valid  = alphas > 1e-3
    alpha_hat = _estimate_alpha_hat(pool, run)
    if alpha_hat is None:
        print("  [!] source pie skipped (cannot estimate alpha_hat)")
        return
    rel = np.where(valid, np.abs(alpha_hat - alphas) / np.where(valid, alphas, 1.0),
                   np.nan)
    src_arr = np.array(sources)
    n_valid = int(valid.sum())
    global_mre = float(np.nanmean(rel[valid]))

    order = [t for t in ('BO', 'PBO', 'EmiliaR', 'NorthEast', 'Italy')
             if t in set(sources)]
    colors = {'BO': '#2c6fbb', 'PBO': '#63a2d8', 'EmiliaR': '#f0c419',
              'NorthEast': '#e8843c', 'Italy': '#c0392b'}
    contrib, counts = [], []
    for t in order:
        m = (src_arr == t) & valid
        contrib.append(float(np.nanmean(rel[m])) * m.sum() / n_valid)
        counts.append(int(m.sum()))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6))
    W_of = {'BO': 1.00, 'PBO': 0.85, 'EmiliaR': 0.50,
            'NorthEast': 0.30, 'Italy': 0.15}

    wedges, _, _ = axA.pie(
        contrib, labels=[f"{t}\nW={W_of.get(t, float('nan')):.2f}" for t in order],
        autopct=lambda p: f'{p:.1f}%', startangle=90, counterclock=False,
        colors=[colors.get(t, '#999999') for t in order],
        textprops={'fontsize': 9},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axA.set_title(f'Share of the global MRE by source\n'
                  f'(additive contributions, total MRE = {global_mre:.4f})',
                  fontsize=11)

    axB.pie(counts, labels=[f"{t}\nn={c}" for t, c in zip(order, counts)],
            autopct=lambda p: f'{p:.1f}%', startangle=90, counterclock=False,
            colors=[colors.get(t, '#999999') for t in order],
            textprops={'fontsize': 9},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axB.set_title('Share of the constraint set by source\n'
                  '(for comparison: error share vs. group size)', fontsize=11)

    fig.suptitle('Where the error lives — geographic decomposition',
                 fontsize=13, y=1.0)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'source_pie.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print("  [✓] source_pie.pdf / .png")


def main():
    print("━" * 64)
    print("  ISTAT Diagnostics — loading saved run")
    print("━" * 64)

    run, pool = load_run()
    print(f"  Loaded: {run['n_iters']} iterations, "
          f"final MRE={run['final_mre']:.5f}, "
          f"wMRE={run['final_weighted_mre']:.5f}")
    print(f"  Pool: {'loaded' if pool is not None else 'not found — using last history alpha_hat'}\n")

    # ── Consistency guard: does the saved history match the saved pool? ─────
    # If test_4_ISTAT_history.pkl comes from an OLDER run than the .npy pool
    # (different attr_meta version, different constraint set), every number
    # derived from the pkl (convergence curves, floor analysis) describes a
    # different experiment than the pool-based figures. Refuse to mix silently.
    stale_pkl = False
    try:
        from istat.preprocess_istat import build_constraint_set as _bcs
        from istat.geo_tagging import parse_source_tags as _pst
        from istat.attr_meta_ISTAT import marginals as _marg
        _ts, _ms = _pst(os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py'))
        _cs, _ = _bcs(_marg, _ts, _ms)
        _a_run = np.asarray(run['alphas'])
        if len(_a_run) != _cs.m or not np.allclose(_a_run, _cs.alphas_array, atol=1e-9):
            stale_pkl = True
            print("  " + "!" * 62)
            print("  [!] WARNING: test_4_ISTAT_history.pkl does NOT match the current")
            print(f"      constraint set (pkl m={len(_a_run)}, current m={_cs.m}).")
            print("      The history pkl is from an OLDER run. Convergence curves and")
            print("      any pkl-derived numbers describe a DIFFERENT experiment than")
            print("      the pool-based figures. Re-run experiments/test_4_ISTAT.py")
            print("      (it now saves the history) before using these figures.")
            print("  " + "!" * 62 + "\n")
    except Exception as _e:
        print(f"  [!] Consistency guard failed: {_e}\n")

    print(f"  Writing figures to {FIG_DIR}/\n")

    print("  Figure 1: Convergence curves")
    plot_convergence(run)

    print("  Figure 2: Stratified MRE")
    plot_stratified_mre_fixed(run, pool)

    if pool is not None:
        print("  Figure 3: Diversity metrics")
        plot_diversity(pool)
    else:
        print("  Figure 3: skipped (pool .npy not found)")

    print("  Figure 4: LP MRE floor (rigorous lower bound)")

    plot_lp_floor(run, pool)

    if pool is not None:
        print("  Figure 6: MRE share by source (pie)")
        plot_source_pie(run, pool)

        print("  Figure 5: α̂ vs α scatter")
        plot_alpha_scatter(pool)
    else:
        print("  Figure 5: skipped (pool .npy not found)")

    # ── Scalar MRE floor summary (unary + binary) ───────────────────────────
    print("  MRE floor scalar summary")
    try:
        from istat.preprocess_istat import discover_cpts, build_constraint_set
        import istat.attr_meta_ISTAT as _meta
        from istat.attr_meta_ISTAT import marginals, ATTR_META, ATTR_NAMES_SYNTH
        from istat.geo_tagging import parse_source_tags
        from istat.mre_floor import (compute_full_mre_floor,
                                     print_full_mre_floor_summary)

        attr_meta_path = os.path.join(ROOT_DIR, 'src', 'istat', 'attr_meta_ISTAT.py')
        table_source, marginal_source = parse_source_tags(attr_meta_path)

        cs, cs_sources = build_constraint_set(marginals, table_source, marginal_source)
        cpts   = discover_cpts(_meta)

        # The floor is a property of the DATA (current constraint set), so use
        # the current targets — never run['alphas'], which may be from an older
        # run with a different constraint set and would silently misalign every
        # per-constraint comparison.
        alphas = cs.alphas_array

        # Observed MRE: prefer recomputing from the saved pool against the
        # current targets, so observed and floor refer to the same experiment.
        if pool is not None:
            from istat.preprocess_istat import build_constraint_weights as _bcw
            alpha_hat = np.zeros(cs.m)
            for j in range(cs.m):
                _at, _vl = cs.attrs_list[j], cs.vals_list[j]
                alpha_hat[j] = np.all(pool[:, _at] == _vl[np.newaxis, :], axis=1).mean()
            _valid  = alphas > 1e-3
            _relerr = np.abs(alpha_hat[_valid] - alphas[_valid]) / alphas[_valid]
            observed_mre  = float(np.mean(_relerr))
            _w = _bcw(cs_sources)[_valid]
            observed_wmre = float(np.average(_relerr, weights=_w))
            print(f"  (observed MRE recomputed from pool vs current targets)")
        else:
            observed_mre  = run['final_mre']
            observed_wmre = run['final_weighted_mre']
            if stale_pkl:
                print("  [!] No pool available and pkl is stale — floor summary skipped.")
                raise RuntimeError("stale history pkl without pool")

        result = compute_full_mre_floor(
            cs, cpts, marginals, alphas,
            table_source, marginal_source,
            ATTR_NAMES_SYNTH, ATTR_META)
        print_full_mre_floor_summary(observed_mre, result, observed_wmre)

    except Exception as e:
        print(f"  [!] Floor scalar failed: {e}")
        import traceback; traceback.print_exc()

    # ── LP MRE floor: rigorous lower bound over the local polytope ──────────
    # Complements the pairwise floor above. The pairwise version only sees
    # quantities that two sources state directly; this one also captures
    # multi-way conflicts and infeasibility created by the structural zeros,
    # and it is a genuine lower bound rather than a heuristic.
    try:
        from istat.mre_floor_lp import (compute_lp_mre_floor,
                                        print_lp_floor_summary)
        from istat.preprocess_istat import build_constraint_weights
        from istat.attr_meta_ISTAT import (ATTR_NAMES_SYNTH as _AN,
                                           ATTR_META as _AM)
        print("\n  LP MRE floor (local-polytope lower bound)")
        _ts2, _ms2 = _pst(os.path.join(ROOT_DIR, 'src', 'istat',
                                       'attr_meta_ISTAT.py'))
        _cs2, _src2 = _bcs(_marg, _ts2, _ms2)
        _W2 = build_constraint_weights(_src2)
        _lp = compute_lp_mre_floor(_cs2, weights=_W2, tighten='pairwise',
                                   verbose=True)
        print()
        print_lp_floor_summary(_lp, observed_mre, observed_wmre,
                               sources=_src2, cs=_cs2,
                               attr_names=_AN, attr_meta=_AM, top=12)
    except Exception as e:
        print(f"  [!] LP floor failed: {e}")
        import traceback; traceback.print_exc()

    # ── Manifest: what was actually written, and what is stale ──────────────
    # Figures are written by several independent code paths, some of which are
    # skipped when the pool is missing, and one figure (mre_floor) is no longer
    # produced at all since the LP bound replaced it. Without a manifest a
    # stale file from an earlier run is indistinguishable from a fresh one,
    # and can end up in the thesis.
    import time
    expected = ['convergence', 'stratified_mre', 'diversity',
                'mre_floor_lp', 'source_pie', 'alpha_scatter']
    now = time.time()
    print(f"\n{'━'*64}")
    print(f"  FIGURE MANIFEST  ({FIG_DIR}/)")
    print(f"{'━'*64}")
    print(f"  {'file':<24}{'age':>12}   status")
    print(f"  {'-'*54}")
    for name in expected:
        f = os.path.join(FIG_DIR, f'{name}.png')
        if not os.path.exists(f):
            print(f"  {name+'.png':<24}{'--':>12}   NOT WRITTEN  <-- check the log above")
            continue
        age = now - os.path.getmtime(f)
        age_s = f"{age:.0f}s" if age < 90 else f"{age/60:.0f}m" if age < 5400 else f"{age/3600:.1f}h"
        flag = "ok" if age < 300 else "STALE  <-- from an earlier run"
        print(f"  {name+'.png':<24}{age_s:>12}   {flag}")
    # anything present but no longer produced
    if os.path.isdir(FIG_DIR):
        extra = sorted(x[:-4] for x in os.listdir(FIG_DIR)
                       if x.endswith('.png') and x[:-4] not in expected)
        for name in extra:
            print(f"  {name+'.png':<24}{'':>12}   ORPHAN — no longer generated, safe to delete")
    print(f"\n{'━'*64}")
    print(f"  Done. All figures saved to {FIG_DIR}/")
    print(f"{'━'*64}")


if __name__ == '__main__':
    main()