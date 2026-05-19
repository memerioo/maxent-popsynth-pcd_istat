"""
test_5_ISTAT_K20.py
-------------------
Execution testing script for the scaled K=20 model.
Saves the generated population matrix directly to the experiments folder
and prints the top 15 worst unresolved constraints at the end.
"""

import sys
import os
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from gibbs_pcd_solver import GibbsPCDSolver
from istat.preprocess_istat_K20 import build_constraint_set

try:
    from istat.marginals_corrected_K20 import marginals as CLEAN_MARGINALS
except ImportError:
    print("\n❌ ERROR: Could not find 'istat/marginals_corrected_K20.py'. Please run pipeline first:")
    print("    python src/istat/preprocess_istat_K20.py\n")
    sys.exit(1)

def run():
    print("━" * 64)
    print("  STEP 1 / 2  —  Building ConstraintSet (K=20)")
    print("━" * 64)
    cs = build_constraint_set(CLEAN_MARGINALS)

    print("\n" + "━" * 64)
    print("  STEP 2 / 2  —  GibbsPCDSolver")
    print("━" * 64)

    # Stabilized hyperparameters for deep structural space
    N_pool         = 50_000      
    n_outer        = 1000        
    n_gibbs_sweeps = 15      
    lr             = 0.003   
    eps            = 1e-4   
    tol            = 0.001       
    window         = 100         
    verbose_every  = 20

    print(f"Pool size       : {N_pool}")
    print(f"Outer iters     : up to {n_outer}")
    print(f"Learning rate   : {lr}")
    print(f"Gibbs sweeps/it : {n_gibbs_sweeps}\n")

    solver = GibbsPCDSolver(cs, use_numba=True)
    solver.fit(
        N_pool         = N_pool,
        n_outer        = n_outer,        
        n_gibbs_sweeps = n_gibbs_sweeps, 
        lr             = lr, 
        eps            = eps,
        tol            = tol,     
        window         = window,         
        verbose_every  = verbose_every,
    )

    print(f"\n{'─'*64}")
    print(f"  Final MRE : {solver.final_mre:.5f}")
    print(f"  Final MAE : {solver.final_mae:.5f}")
    print(f"  Iterations: {solver.n_iters}")
    print(f"  Early stop: {solver.stopped_early}")
    print(f"  Fit time  : {solver.fit_time:.1f}s")
    print(f"{'─'*64}")

    if solver.pool_ is not None:
        out_path = os.path.join(ROOT_DIR, 'experiments', f"test_5_ISTAT_pop{N_pool}.npy")
        np.save(out_path, solver.pool_)
        print(f"\n[+] Population successfully saved to: {out_path}")

    # ── DIAGNOSTICS: TOP 15 UNRESOLVED CONSTRAINTS ───────────────────────────
    print("\n" + "━" * 64)
    print("  TOP 15 UNRESOLVED CONSTRAINTS")
    print("━" * 64)
    
    from istat.attr_meta_ISTAT_K20 import ATTR_NAMES_SYNTH, ATTR_META
    alpha_hat = solver._estimate_expectations(solver.pool_)
    errors = np.abs(alpha_hat - solver.alphas)
    worst_idx = np.argsort(errors)[::-1][:15]
    
    for j in worst_idx:
        target = solver.alphas[j]
        actual = alpha_hat[j]
        diff   = errors[j]
        attrs  = cs.attrs_list[j]
        vals   = cs.vals_list[j]
        
        str_list = []
        for a, v in zip(attrs, vals):
            attr_name = ATTR_NAMES_SYNTH[a]
            val_name = ATTR_META[attr_name]['vals'][v]
            str_list.append(f"{attr_name}={val_name}")
        attr_vals_str = ", ".join(str_list)
        print(f"Diff: {diff:.4f}  |  Target: {target:.4f}  |  Actual: {actual:.4f}  |  {attr_vals_str}")

    return solver

if __name__ == "__main__":
    run()