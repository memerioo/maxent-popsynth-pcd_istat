"""
gibbs_pcd_solver.py
-------------------
GibbsPCDSolver: scalable Maximum Entropy population synthesis
via Persistent Contrastive Divergence.

Replaces the exact expectation step of Pachet & Zucker (2026)
with a stochastic approximation from a persistent Gibbs pool,
removing the |X| barrier that limits exact MaxEnt to K ~< 20.

Reference:
    Degli Esposti, M. (2026). Scalable Maximum Entropy Population
    Synthesis via Persistent Contrastive Divergence. arXiv:2503.XXXXX
    Pachet & Zucker (2026). Maximum Entropy Relaxation of Multi-Way
    Cardinality Constraints for Synthetic Population Generation.
    Tieleman, T. (2008). Training Restricted Boltzmann Machines Using
    Approximations to the Likelihood Gradient. ICML.
"""

import time
import numpy as np
from constraint_set import ConstraintSet


class GibbsPCDSolver:
    """
    Maximum Entropy solver via Persistent Contrastive Divergence.

    Maintains a persistent pool of N synthetic individuals updated
    by Gibbs (Glauber) sweeps at each gradient step. The pool's
    empirical frequencies provide a stochastic approximation of
    the model expectations E_{p_lambda}[f_j], without ever
    materialising X or computing Z(lambda).

    Parameters
    ----------
    cs : ConstraintSet
        Constraint set defining the MaxEnt problem.
    use_numba : bool, optional
        If True, use the Numba-accelerated Gibbs kernel (recommended
        for K >= 20). Falls back to pure NumPy if Numba is unavailable.

    Notes
    -----
    The Gibbs conditional update for attribute k is:

        p(A_k = v | x_{-k}) ∝ exp( sum_{j in J(k,v,x_{-k})} lambda_j )

    where J(k, v, x_{-k}) is the set of constraints j with k in S_j,
    v_j^(k) = v, and x_{S_j \\ {k}} = v_j^(-k).

    This leaves p_lambda invariant by detailed balance, and the chain
    is ergodic because p_lambda > 0 on all of X.

    Attributes updated by fit()
    ---------------------------
    lambdas : (m,) float64 array — learned Lagrange multipliers
    history : list of dicts — per-iteration diagnostics
    final_mre : float — MRE at convergence
    fit_time : float — wall-clock seconds
    n_iters : int — actual iterations run
    stopped_early : bool
    """

    def __init__(self, cs: ConstraintSet, use_numba: bool = False):
        self.cs        = cs
        self.K         = cs.K
        self.m         = cs.m
        self.alphas    = cs.alphas_array
        self.use_numba = use_numba
        self.pool_: np.ndarray | None = None

        # Precompute attr_lookup for vectorised energy accumulation
        self.lookup = cs.build_attr_lookup()

        # Numba kernel (compiled on first call if available)
        self._numba_kernel = None
        self._numba_args = None
        if use_numba:
            self._numba_kernel = _make_gibbs_numba_kernel()
            if self._numba_kernel is not None:
                self._numba_args = self._prepare_numba_lookup()

        # Results (populated by fit)
        self.lambdas:      np.ndarray | None = None
        self.history:      list[dict]        = []
        self.final_mre:    float             = float('nan')
        self.final_mae:    float             = float('nan')
        self.fit_time:     float             = 0.0
        self.n_iters:      int               = 0
        self.stopped_early: bool             = False

    def _prepare_numba_lookup(self):
        """Flattens the nested Python dictionary into strictly typed 1D C-arrays for Numba."""
        num_entries = 0
        num_others = 0
        for k in range(self.K):
            for entry in self.lookup[k]:
                num_entries += 1
                num_others += len(entry[2])

        entry_j = np.zeros(num_entries, dtype=np.int32)
        entry_v_k = np.zeros(num_entries, dtype=np.int32)
        entry_offset = np.zeros(num_entries + 1, dtype=np.int32)
        flat_other_attrs = np.zeros(num_others, dtype=np.int32)
        flat_other_vals = np.zeros(num_others, dtype=np.int32)
        k_offsets = np.zeros(self.K + 1, dtype=np.int32)

        idx = 0
        offset = 0
        for k in range(self.K):
            k_offsets[k] = idx
            for (j, v_k, o_attrs, o_vals) in self.lookup[k]:
                entry_j[idx] = j
                entry_v_k[idx] = v_k
                entry_offset[idx] = offset
                n_o = len(o_attrs)
                flat_other_attrs[offset:offset+n_o] = o_attrs
                flat_other_vals[offset:offset+n_o] = o_vals
                offset += n_o
                idx += 1
        k_offsets[self.K] = idx
        entry_offset[idx] = offset

        return k_offsets, entry_j, entry_v_k, entry_offset, flat_other_attrs, flat_other_vals

    # ------------------------------------------------------------------ #
    #  Pool initialisation                                                  #
    # ------------------------------------------------------------------ #

    def _init_pool(self, N: int, seed: int = 1) -> np.ndarray:
        """Initialise pool uniformly over attribute domains."""
        rng  = np.random.default_rng(seed)
        pool = np.zeros((N, self.K), dtype=np.int32)
        for k in range(self.K):
            pool[:, k] = rng.integers(0, self.cs.domain_sizes[k], size=N)
        return pool

    # ------------------------------------------------------------------ #
    #  Gibbs sweep (pure NumPy)                                            #
    # ------------------------------------------------------------------ #

    def _gibbs_sweep(self, pool: np.ndarray,
                     lam: np.ndarray) -> np.ndarray:
        """
        One full Gibbs sweep over all K attributes (random permuted order).

        For each attribute k, computes log-energies of shape (N, d_k)
        via the precomputed lookup table, applies numerically stable
        softmax, and samples new values for all N individuals.

        Cost: O(N * K * d_max * mean_J) per sweep.
        """
        N   = pool.shape[0]
        rng = np.random.default_rng()

        for k in rng.permutation(self.K):
            d_k     = int(self.cs.domain_sizes[k])
            log_e   = np.zeros((N, d_k), dtype=np.float64)

            for (j, v_k, other_attrs, other_vals) in self.lookup[k]:
                # Identify individuals whose context matches this constraint
                if len(other_attrs) > 0:
                    ctx_match = np.all(
                        pool[:, other_attrs] == other_vals[np.newaxis, :],
                        axis=1
                    )
                else:
                    ctx_match = np.ones(N, dtype=bool)
                log_e[ctx_match, v_k] += lam[j]

            # Numerically stable softmax -> categorical sample
            log_e -= log_e.max(axis=1, keepdims=True)
            probs  = np.exp(log_e)
            probs /= probs.sum(axis=1, keepdims=True)

            # Vectorised categorical sampling for all N individuals
            cdf  = probs.cumsum(axis=1)
            u    = np.random.rand(N, 1)
            pool[:, k] = (u > cdf).sum(axis=1).clip(0, d_k - 1)

        return pool

    # ------------------------------------------------------------------ #
    #  Expectation estimation                                               #
    # ------------------------------------------------------------------ #

    def _estimate_expectations(self, pool: np.ndarray) -> np.ndarray:
        """
        Estimate alpha_hat_j = (1/N) sum_i f_j(pool[i]) for all j.
        """
        alpha_hat = np.zeros(self.m, dtype=np.float64)
        for j in range(self.m):
            attrs = self.cs.attrs_list[j]
            vals  = self.cs.vals_list[j]
            alpha_hat[j] = np.all(
                pool[:, attrs] == vals[np.newaxis, :], axis=1
            ).mean()
        return alpha_hat

    # ------------------------------------------------------------------ #
    #  Main fitting loop                                                    #
    # ------------------------------------------------------------------ #

    def fit(self,
            N_pool:         int   = 10_000,
            n_outer:        int   = 500,
            n_gibbs_sweeps: int   = 5,
            lr:             float = 0.01,
            gamma:          float = 0.001,  # ──> NEW: L2 Regularization parameter (Weight Decay)
            beta1:          float = 0.9,
            beta2:          float = 0.999,
            eps:            float = 1e-8,
            seed:           int   = 1,
            tol:            float = 0.02,
            window:         int   = 50,
            verbose_every:  int   = 50) -> 'GibbsPCDSolver':
        """
        Fit GibbsPCDSolver using Adam optimiser with adaptive stopping and L2 Regularization.
        """
        lam  = np.zeros(self.m, dtype=np.float64)
        pool = self._init_pool(N_pool, seed=seed)

        sweep_fn = (self._gibbs_sweep_numba
                    if (self.use_numba and self._numba_kernel is not None)
                    else self._gibbs_sweep)

        # Adam state
        m1 = np.zeros(self.m, dtype=np.float64)
        m2 = np.zeros(self.m, dtype=np.float64)

        self.history       = []
        self.stopped_early = False
        t_start            = time.time()

        # --- BEST TRACKER ---
        best_mre  = float('inf')
        best_lam  = None
        best_pool = None

        # Precompute mask for empirical constraints to protect structural zeros
        empirical_mask = self.alphas > 0.0

        for t in range(1, n_outer + 1):

            # Inner loop: Gibbs sweeps on persistent pool
            for _ in range(n_gibbs_sweeps):
                pool = sweep_fn(pool, lam)

            # Stochastic gradient estimate: grad = alpha_hat - alpha
            alpha_hat = self._estimate_expectations(pool)
            grad      = alpha_hat - self.alphas

            #  FIX: Apply L2 penalty to empirical constraints to balance contradictions
            if gamma > 0.0:
                grad[empirical_mask] += gamma * lam[empirical_mask]

            # Adam update
            m1  = beta1 * m1 + (1.0 - beta1) * grad
            m2  = beta2 * m2 + (1.0 - beta2) * grad ** 2
            m1h = m1 / (1.0 - beta1 ** t)
            m2h = m2 / (1.0 - beta2 ** t)
            
            lam -= lr * m1h / (np.sqrt(m2h) + eps)
            lam = np.clip(lam, -30.0, 30.0)  

            # Create a mask for valid targets (ignoring the exact zeros)
            min_prob_threshold = 1e-3
            valid_mask = self.alphas > min_prob_threshold

            if np.any(valid_mask):
                mre = float(np.mean(
                    np.abs(alpha_hat[valid_mask] - self.alphas[valid_mask]) / self.alphas[valid_mask]
                ))
                mae = float(np.mean(
                    np.abs(alpha_hat[valid_mask] - self.alphas[valid_mask])
                ))
            else:
                mre = 0.0
                mae = 0.0

            ## ------ BEST POOL TRACKER ------
            if mre < best_mre:
                best_mre  = mre
                best_lam  = lam.copy()
                best_pool = pool.copy()

            self.history.append({
                't':         t,
                'mre':       mre,
                'mae':       mae,
                'alpha_hat': alpha_hat.copy(),
                'elapsed':   time.time() - t_start,
            })

            if verbose_every and t % verbose_every == 0:
                print(f"  [Gibbs] iter {t:4d}  MRE={mre:.5f}  MAE={mae:.5f}  "
                      f"N={N_pool}  t={time.time()-t_start:.1f}s")

            if tol > 0.0 and t >= 2 * window:
                recent_min  = min(h['mre'] for h in self.history[-window:])
                earlier_min = min(h['mre'] for h in
                                  self.history[-2*window:-window])
                if earlier_min > 0:
                    rel_improv = (earlier_min - recent_min) / earlier_min
                else:
                    rel_improv = 0.0
                if rel_improv < tol:
                    self.stopped_early = True
                    if verbose_every:
                        print(f"  [Gibbs] Early stop at iter {t}  "
                              f"rel_improv={rel_improv:.4f} < tol={tol}  "
                              f"MRE={mre:.5f}")
                    break

        self.lambdas   = best_lam
        self.pool_     = best_pool         
        self.fit_time  = time.time() - t_start
        self.final_mre = best_mre
        self.final_mae = next(h['mae'] for h in self.history if h['mre'] == best_mre)
        self.n_iters   = len(self.history)
        return self
        

    def _gibbs_sweep_numba(self, pool: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Numba-accelerated Gibbs sweep."""
        if self._numba_kernel is None or self._numba_args is None:
            return self._gibbs_sweep(pool, lam)
        return self._numba_kernel(pool, lam, self.cs.domain_sizes, 
                                  *self._numba_args, self.K)

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def mre_curve(self) -> np.ndarray:
        """Return MRE values across iterations as a 1D array."""
        return np.array([h['mre'] for h in self.history])

    def __repr__(self):
        status = (f"fitted, MRE={self.final_mre:.4f}, "
                  f"{self.n_iters} iters"
                  if self.lambdas is not None else "not fitted")
        return f"GibbsPCDSolver(K={self.K}, m={self.m}, {status})"


# ------------------------------------------------------------------ #
#  Optional Numba kernel (standalone, outside class)                   #
# ------------------------------------------------------------------ #

def _make_gibbs_numba_kernel():
    try:
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _gibbs_sweep_numba_kernel(pool, lam, domain_sizes,
                                      k_offsets, entry_j, entry_v_k, entry_offset,
                                      flat_other_attrs, flat_other_vals, K):
            N = pool.shape[0]
            attr_order = np.arange(K)
            np.random.shuffle(attr_order)
            
            # Find max domain size to pre-allocate memory outside the parallel loop
            max_d_k = 0
            for k in range(K):
                if domain_sizes[k] > max_d_k:
                    max_d_k = domain_sizes[k]
                    
            log_e = np.zeros((N, max_d_k))

            for ki in range(K):
                k = attr_order[ki]
                d_k = domain_sizes[k]

                start_idx = k_offsets[k]
                end_idx = k_offsets[k+1]

                for i in prange(N):
                    # Zero out the energy array for this specific individual
                    for v in range(d_k):
                        log_e[i, v] = 0.0
                    
                    # Accumulate energies using flat arrays
                    for entry_idx in range(start_idx, end_idx):
                        match = True
                        o_start = entry_offset[entry_idx]
                        o_end = entry_offset[entry_idx+1]
                        
                        for p in range(o_start, o_end):
                            if pool[i, flat_other_attrs[p]] != flat_other_vals[p]:
                                match = False
                                break
                                
                        if match:
                            log_e[i, entry_v_k[entry_idx]] += lam[entry_j[entry_idx]]

                    # Softmax + categorical sample (vector-safe manual loops)
                    row_max = log_e[i, 0]
                    for v in range(1, d_k):
                        if log_e[i, v] > row_max:
                            row_max = log_e[i, v]
                            
                    row_sum = 0.0
                    for v in range(d_k):
                        log_e[i, v] = np.exp(log_e[i, v] - row_max)
                        row_sum += log_e[i, v]
                        
                    u = np.random.random()
                    cumsum = 0.0
                    chosen = d_k - 1
                    for v in range(d_k):
                        cumsum += log_e[i, v] / row_sum
                        if u <= cumsum:
                            chosen = v
                            break
                            
                    pool[i, k] = chosen

            return pool

        return _gibbs_sweep_numba_kernel

    except ImportError:
        return None