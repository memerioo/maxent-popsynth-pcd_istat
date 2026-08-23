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
    Synthesis via Persistent Contrastive Divergence. arXiv:2603.27312
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

    def __init__(self, cs: ConstraintSet, use_numba: bool = False,
                 weights: np.ndarray | None = None,
                 sources: list | None = None,
                 blocks: list | None = None):
        """
        blocks : list of dicts or None
            Sentinel blocks from `istat.structural_blocks.resolve_blocks`.
            When supplied, each outer iteration runs a Metropolis-Hastings
            block-toggle sweep (block_moves.py) in addition to the Gibbs
            sweeps.  This is REQUIRED whenever structural zeros are pinned
            hard: single-site Gibbs cannot move an individual between the
            active and inactive basin of a sentinel block (it would have to
            flip up to eight attributes at once, and every intermediate
            state has weight e^-30), so without block moves the block
            composition is frozen at its initial value and quantities such
            as P(employment) and P(StudentStat) can never be fitted.
        weights : (m,) array or None
            Per-constraint reliability weights W_jj for the soft-constraint
            objective  Phi_W(lambda) = logZ(lambda) - lambda^T alpha
            with stochastic gradient  g_j = W_jj * (alpha_hat_j - alpha_j).
            Higher W_jj => the solver is penalised more for missing that
            constraint => it "wins" local disagreements against low-weight
            (e.g. national-level) constraints. Defaults to all-ones, i.e.
            identical to the unweighted solver.
        sources : list[str] or None, length m
            Optional per-constraint provenance tag (e.g. "BO", "EmiliaR",
            "NorthEast", "Italy"), used only for stratified diagnostics
            (stratified_mre()). Purely informational -- does not affect
            fitting.
        """
        self.cs        = cs
        self.K         = cs.K
        self.m         = cs.m
        self.alphas    = cs.alphas_array
        self.use_numba = use_numba
        self.pool_: np.ndarray | None = None

        if weights is None:
            self.weights = np.ones(self.m, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (self.m,):
                raise ValueError(f"weights must have shape ({self.m},), got {weights.shape}")
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative")
            self.weights = weights

        if sources is not None and len(sources) != self.m:
            raise ValueError(f"sources must have length {self.m}, got {len(sources)}")
        self.sources = list(sources) if sources is not None else None

        # Precompute attr_lookup for vectorised energy accumulation
        self.lookup = cs.build_attr_lookup()

        # Numba kernel (compiled on first call if available)
        self._numba_kernel = None
        self._numba_args = None
        if use_numba:
            self._numba_kernel = _make_gibbs_numba_kernel()
            if self._numba_kernel is not None:
                self._numba_args = self._prepare_numba_lookup()

        # Sentinel-block MH moves (see block_moves.py)
        self.blocks = blocks
        self._blocks_prepared = None
        self._block_kernel = None
        if blocks:
            from istat.block_moves import prepare_block, make_block_toggle_kernel
            self._blocks_prepared = []
            for b in blocks:
                p = prepare_block(cs, b["attrs_idx"])
                p["name"] = b["name"]
                self._blocks_prepared.append(p)
            if use_numba:
                self._block_kernel = make_block_toggle_kernel()

        # Results (populated by fit)
        self.lambdas:      np.ndarray | None = None
        self.history:      list[dict]        = []
        self.final_mre:    float             = float('nan')
        self.final_mae:    float             = float('nan')
        self.final_weighted_mre: float       = float('nan')
        self.final_weighted_mae: float       = float('nan')
        self.selection_metric: str           = 'mre'
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
        rng = getattr(self, '_rng', None)
        if rng is None:
            rng = self._rng = np.random.default_rng()

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
            u    = rng.random((N, 1))
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
            lr_patience:    int   = 0,
            lr_decay:       float = 0.5,
            lr_min:         float | None = None,
            plateau_smooth: int   = 25,
            lr_schedule:    str   = 'plateau',
            selection_metric: str = 'auto',
            init_pool:      np.ndarray | None = None,
            block_move_frac: float = 1.0,
            verbose_every:  int   = 50) -> 'GibbsPCDSolver':
        """
        Fit GibbsPCDSolver using Adam optimiser with adaptive stopping and L2 Regularization.

        Reproducibility
        ---------------
        `seed` seeds ALL random streams: pool initialisation, the NumPy
        Gibbs sweeps (attribute permutation + categorical draws), and the
        per-thread Numba RNG states. With use_numba=True and parallel
        execution, runs are bit-reproducible for a FIXED thread count
        (Numba's prange uses static scheduling); pin it with e.g.
        `export NUMBA_NUM_THREADS=8` for strict reproducibility across
        sessions.

        Learning-rate schedule (reduce-on-plateau)
        ------------------------------------------
        If `lr_patience > 0`: whenever the best `selection_metric` has not
        improved for `lr_patience` consecutive iterations, the learning
        rate is multiplied by `lr_decay` (never below `lr_min`, default
        lr/20). Near the compromise equilibrium of an INCONSISTENT
        constraint set the residuals of conflicting constraints cannot all
        vanish; with a constant lr, Adam settles into a limit cycle
        (visible as late-stage oscillation of the MRE). Decaying lr on
        plateau shrinks the cycle amplitude and lets the iterate settle.

        Weighted / soft-constraint objective
        -------------------------------------
        If `self.weights` is not all-ones (pass `weights=` to __init__), the
        stochastic gradient used at every outer step is

            g_j = W_jj * (alpha_hat_j - alpha_j)

        instead of the plain g_j = alpha_hat_j - alpha_j. This implements the
        reliability-weighted relaxation

            Phi_W(lambda) = logZ(lambda) - lambda^T alpha ,
            g = W (alpha_hat(lambda) - alpha)

        i.e. a diagonal reweighting of the moment-matching residual, so that
        during optimisation, high-W (trusted, e.g. Bologna-specific)
        constraints are driven towards zero residual more aggressively than
        low-W (e.g. national-level) constraints when they are in conflict
        and the pool cannot satisfy both simultaneously.

        NOTE on the fixed point: if the constraint set were internally
        *consistent* (a lambda exists with alpha_hat(lambda) = alpha exactly
        for every j), then W would not change what the solver converges to
        -- a zero weighted gradient forces every unweighted residual to zero
        too, same as without weighting. Weighting only matters -- and it
        matters a lot here -- when the system is *inconsistent*/over-
        determined (as with mismatched-geography ISTAT tables): there is no
        exact fixed point, gradient descent settles into a compromise, and W
        controls which constraints "win" that compromise. Do not expect W to
        change results on a genuinely consistent constraint set.

        selection_metric : {'auto', 'mre', 'weighted_mre'}
            Which per-iteration score is used to pick the "best" snapshot
            of (lambda, pool) returned as self.lambdas / self.pool_.
            'auto' -> 'weighted_mre' if weights were supplied, else 'mre'.
        """
        # ── Reproducibility: one seed drives every random stream ───────────
        self.seed  = seed
        self._rng  = np.random.default_rng(seed)
        if self.use_numba and self._numba_kernel is not None:
            _seed_numba_rng(seed)

        lam  = np.zeros(self.m, dtype=np.float64)

        # ── Structural zeros are HARD constraints, not learned targets ──────
        # A target of exactly 0 encodes a logical impossibility. Learning it
        # by gradient descent does not work: Adam's normalised step has
        # magnitude ~lr per iteration, so over T iterations lambda_j can only
        # travel about lr*T. At lr=0.005 and T=1300 that is only ~-6, an odds
        # factor e^-6 = 2.5e-3 -- which at N=500,000 still leaves thousands of
        # logically impossible individuals in the pool. Worse, the gradient of
        # a zero-target constraint is alpha_hat itself, so as alpha_hat falls
        # below eps the Adam ratio g/(|g|+eps) collapses and the descent
        # stalls before the constraint is enforced.
        # Instead we pin these parameters at the clip value and freeze them:
        # e^-30 = 1e-13 makes the configuration unreachable. Both Gibbs paths
        # apply max-subtraction before exponentiating, so this is safe even
        # when several structural zeros overlap on one cell.
        structural = (self.alphas == 0.0)
        lam[structural] = -30.0

        # Pool initialisation.  A uniform pool is fine ONLY when block moves
        # are enabled: with hard structural zeros the block composition of a
        # uniform pool is otherwise frozen forever at its random initial
        # value (~0.21 non-students against a target of 0.72).  Passing a
        # legal, on-target `init_pool` (see structural_blocks.ancestral_init_pool)
        # additionally saves the block moves the work of walking the
        # composition all the way from noise to the published targets.
        if init_pool is not None:
            if init_pool.shape != (N_pool, self.K):
                raise ValueError(
                    f"init_pool must have shape ({N_pool}, {self.K}), "
                    f"got {init_pool.shape}")
            pool = np.ascontiguousarray(init_pool, dtype=np.int32).copy()
        else:
            pool = self._init_pool(N_pool, seed=seed)
            if not self._blocks_prepared:
                import warnings
                warnings.warn(
                    "Uniform pool init with hard structural zeros and no "
                    "sentinel blocks: block composition will be frozen at "
                    "its random initial value. Pass blocks= and/or init_pool=.",
                    RuntimeWarning)

        self.block_accept_: list = []
        self.block_accept_by_name_: list = []

        sweep_fn = (self._gibbs_sweep_numba
                    if (self.use_numba and self._numba_kernel is not None)
                    else self._gibbs_sweep)

        # Adam state
        m1 = np.zeros(self.m, dtype=np.float64)
        m2 = np.zeros(self.m, dtype=np.float64)

        self.history       = []
        self.stopped_early = False
        t_start            = time.time()

        weighted = not np.allclose(self.weights, 1.0)
        if selection_metric == 'auto':
            selection_metric = 'weighted_mre' if weighted else 'mre'
        if selection_metric not in ('mre', 'weighted_mre'):
            raise ValueError("selection_metric must be 'auto', 'mre' or 'weighted_mre'")

        # --- BEST TRACKER ---
        best_score = float('inf')
        best_mre   = float('inf')
        best_iter  = 0
        best_lam   = None
        best_pool  = None

        # Precompute mask for empirical constraints to protect structural zeros
        empirical_mask = self.alphas > 0.0

        # ── Learning-rate schedule ────────────────────────────────────────
        # 'plateau' : reactive reduce-on-plateau (the original behaviour).
        #             lr stays FLAT until the smoothed metric stalls, which
        #             means it is still at its initial value during the phase
        #             where the Adam limit cycle grows, and then drops in a
        #             few sudden steps that collapse the cycle abruptly.
        # 'cosine'  : lr decreases monotonically from `lr` to `lr_min` over
        #             the full n_outer iterations. Because the total
        #             parameter travel is sum_t lr_t and the oscillation
        #             amplitude scales with the CURRENT lr, a cosine schedule
        #             can deliver the same travel as a flat-then-decay
        #             schedule while never sitting at the large step size
        #             long enough for the cycle to grow. Set `lr` ~30% higher
        #             than the plateau equivalent to match travel.
        # 'exp'     : geometric decay from `lr` to `lr_min` over n_outer.
        if lr_schedule not in ('plateau', 'cosine', 'exp'):
            raise ValueError("lr_schedule must be 'plateau', 'cosine' or 'exp'")

        def _scheduled_lr(t):
            frac = (t - 1) / max(n_outer - 1, 1)
            if lr_schedule == 'cosine':
                return lr_floor + (lr - lr_floor) * 0.5 * (1.0 + np.cos(np.pi * frac))
            return lr * (lr_floor / lr) ** frac      # 'exp'

        # lr reduce-on-plateau state
        lr_eff        = lr
        score_hist: list[float] = []
        lr_floor      = lr_min if lr_min is not None else lr / 20.0
        since_improve = 0

        for t in range(1, n_outer + 1):

            # Block-toggle MH sweep: the ONLY move that can transport an
            # individual across a sentinel boundary once the structural
            # zeros are hard. Run before the Gibbs sweeps so that the
            # sweeps immediately relax the interior of any block that has
            # just been switched.
            acc_rate = float('nan')
            acc_by_block = {}
            if self._blocks_prepared:
                from istat.block_moves import block_toggle
                n_acc = n_try = 0
                for blk in self._blocks_prepared:
                    a, n = block_toggle(pool, lam, blk, self._rng,
                                        kernel=self._block_kernel,
                                        frac=block_move_frac)
                    n_acc += a
                    n_try += n
                    # Per-block rate, not just the mean. The mean can hide a
                    # block that has re-frozen: a permissive block (e.g.
                    # under3, whose attributes are loosely coupled) accepts
                    # readily and can keep the average healthy while the work
                    # block quietly drops towards zero acceptance. A rate
                    # below ~0.05 on ANY block means that block's composition
                    # is no longer mixing and its marginals are pinned at
                    # their initial values.
                    acc_by_block[blk["name"]] = a / max(n, 1)
                acc_rate = n_acc / max(n_try, 1)
                self.block_accept_.append(acc_rate)
                self.block_accept_by_name_.append(dict(acc_by_block))

            # Inner loop: Gibbs sweeps on persistent pool
            for _ in range(n_gibbs_sweeps):
                pool = sweep_fn(pool, lam)

            # Stochastic gradient estimate: grad = alpha_hat - alpha
            alpha_hat = self._estimate_expectations(pool)
            residual  = alpha_hat - self.alphas
            grad      = residual

            #  FIX: Apply L2 penalty to empirical constraints to balance contradictions
            if gamma > 0.0:
                grad[empirical_mask] += gamma * lam[empirical_mask]

            # Adam update
            #
            # IMPORTANT: the reliability weights W_jj are applied to the
            # *step*, not to the raw gradient fed into Adam's moment
            # estimates. Scaling the raw gradient by a constant per-
            # coordinate factor W_j is almost entirely undone by Adam's
            # own per-coordinate normalisation m1_hat/sqrt(m2_hat) (Adam is
            # approximately invariant to constant rescaling of the
            # gradient, since m2 absorbs the square of the same factor) --
            # verified empirically: two runs differing only by a 6.7x
            # gradient-scale factor converge to the *same* lambda after
            # 200 Adam steps. Rescaling the final step size instead gives
            # each constraint its own effective learning rate lr*W_j,
            # which does persist and genuinely biases which constraints
            # get satisfied first when the system is inconsistent.
            m1  = beta1 * m1 + (1.0 - beta1) * grad
            m2  = beta2 * m2 + (1.0 - beta2) * grad ** 2
            m1h = m1 / (1.0 - beta1 ** t)
            m2h = m2 / (1.0 - beta2 ** t)

            lam -= lr_eff * self.weights * m1h / (np.sqrt(m2h) + eps)
            lam = np.clip(lam, -30.0, 30.0)
            lam[structural] = -30.0      # keep hard constraints hard  

            # Create a mask for valid targets (ignoring the exact zeros)
            min_prob_threshold = 1e-3
            valid_mask = self.alphas > min_prob_threshold

            if np.any(valid_mask):
                abs_err = np.abs(alpha_hat[valid_mask] - self.alphas[valid_mask])
                rel_err = abs_err / self.alphas[valid_mask]
                mre = float(np.mean(rel_err))
                mae = float(np.mean(abs_err))
                w_v = self.weights[valid_mask]
                if w_v.sum() > 0:
                    weighted_mre = float(np.average(rel_err, weights=w_v))
                    weighted_mae = float(np.average(abs_err, weights=w_v))
                else:
                    weighted_mre, weighted_mae = mre, mae
            else:
                mre = mae = weighted_mre = weighted_mae = 0.0

            raw_score = weighted_mre if selection_metric == 'weighted_mre' else mre

            # ── Smoothed selection score ──────────────────────────────────
            # Every decision below (best snapshot, plateau detection, early
            # stopping) is made on a TRAILING MEAN rather than on the single
            # noisy iterate, because both decisions are pathological on a raw
            # stochastic metric:
            #
            #   * plateau detection. `since_improve` resets on any new best,
            #     and with Monte-Carlo noise a lucky trough appears every few
            #     dozen iterations by chance. With a large lr_patience the
            #     counter then never reaches the threshold, the learning rate
            #     never decays, and the run oscillates at full lr forever.
            #     Observed directly: at N=30,000 with lr_patience=150 not one
            #     decay fired in 1,440 iterations and the wMRE swing was 49%
            #     of its mean, against 19% for the same solver at N=50,000
            #     once the lr had decayed.
            #
            #   * best-snapshot selection. Taking the minimum of a noisy
            #     sequence returns whichever iterate was luckiest, not the
            #     best model: the reported metric is then a downward-biased
            #     estimate of the model actually being delivered. Smoothing
            #     first removes that bias.
            #
            # The trailing mean lags the true curve by ~smooth/2 iterations,
            # which is harmless here: the pool is a valid sample of p_lambda
            # at every iteration, so a snapshot chosen slightly late is still
            # a correct snapshot.
            score_hist.append(raw_score)
            if len(score_hist) > plateau_smooth:
                score_hist.pop(0)
            score = float(np.mean(score_hist))

            ## ------ BEST POOL TRACKER (selects on smoothed metric) ------
            if score < best_score:
                best_score = score
                best_mre   = mre
                best_iter  = t
                best_lam   = lam.copy()
                best_pool  = pool.copy()
                since_improve = 0
            else:
                since_improve += 1

            # deterministic schedules override the reactive one
            if lr_schedule != 'plateau':
                lr_eff = _scheduled_lr(t)

            # reduce-on-plateau lr decay
            if (lr_schedule == 'plateau'
                    and lr_patience > 0 and since_improve >= lr_patience
                    and lr_eff > lr_floor):
                lr_eff = max(lr_floor, lr_eff * lr_decay)
                since_improve = 0
                if verbose_every:
                    print(f"  [Gibbs] iter {t:4d}  plateau ({lr_patience} iters "
                          f"without {selection_metric} improvement) → lr = {lr_eff:.5f}")

            self.history.append({
                't':            t,
                'mre':          mre,
                'mae':          mae,
                'weighted_mre': weighted_mre,
                'weighted_mae': weighted_mae,
                'lr':           lr_eff,
                'block_accept': acc_rate,
                'block_accept_by_name': dict(acc_by_block),
                'score_smooth': score,
                'alpha_hat':    alpha_hat.copy(),
                'elapsed':      time.time() - t_start,
            })

            if verbose_every and t % verbose_every == 0:
                extra = f"  wMRE={weighted_mre:.5f}" if weighted else ""
                if self._blocks_prepared:
                    per = " ".join(f"{k[:4]}={v:.2f}"
                                   for k, v in acc_by_block.items())
                    extra += f"  blkAcc={acc_rate:.3f} [{per}]"
                print(f"  [Gibbs] iter {t:4d}  MRE={mre:.5f}  MAE={mae:.5f}{extra}  "
                      f"N={N_pool}  t={time.time()-t_start:.1f}s")

            if tol > 0.0 and t >= 2 * window:
                # Compare smoothed scores, for the same reason the plateau
                # detector does: the minimum of a raw noisy series is an
                # outlier, so `rel_improv` computed from raw minima measures
                # noise rather than progress.
                recent_min  = min(h['score_smooth'] for h in self.history[-window:])
                earlier_min = min(h['score_smooth'] for h in
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
                              f"last {selection_metric}={score:.5f}  |  "
                              f"BEST {selection_metric}={best_score:.5f} "
                              f"at iter {best_iter} (returned snapshot)")
                    break

        self.lambdas   = best_lam
        self.pool_     = best_pool         
        self.fit_time  = time.time() - t_start
        self.final_mre = best_mre
        # ALL reported "final_*" values refer to the BEST snapshot (the
        # iteration with the lowest selection_metric), whose (lambda, pool)
        # are what fit() returns — never the last iterate of the run.
        self.best_iter          = best_iter
        best_entry              = self.history[best_iter - 1]
        assert best_entry['t'] == best_iter
        self.final_mae          = best_entry['mae']
        self.final_weighted_mre = best_entry['weighted_mre']
        self.final_weighted_mae = best_entry['weighted_mae']
        self.selection_metric   = selection_metric
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

    def weighted_mre_curve(self) -> np.ndarray:
        """Return weighted-MRE values across iterations as a 1D array."""
        return np.array([h['weighted_mre'] for h in self.history])

    def stratified_mre(self, min_prob_threshold: float = 1e-3) -> dict:
        """
        Break down the final-fit MRE by constraint source group.

        Requires `sources` to have been passed to __init__ (a length-m list
        of tags, e.g. "BO"/"PBO"/"EmiliaR"/"NorthEast"/"Italy" -- see
        geo_tagging.py). Returns, for each distinct tag:

            {tag: {'mre': float, 'weighted_mre': float,
                   'mae': float, 'n_constraints': int}}

        plus an 'ALL' entry for the unstratified totals. Use this instead of
        (or alongside) a single global MRE number: it shows directly whether
        residual error is concentrated in low-trust (e.g. national) sources
        or spread evenly -- the former supports the "data inconsistency, not
        solver failure" reading; the latter would not.
        """
        if self.sources is None:
            raise ValueError(
                "stratified_mre() requires `sources` to be passed to "
                "GibbsPCDSolver(..., sources=[...]) at construction time."
            )
        if self.pool_ is None:
            raise RuntimeError("Call fit() before stratified_mre().")

        alpha_hat = self._estimate_expectations(self.pool_)
        valid = self.alphas > min_prob_threshold
        abs_err = np.abs(alpha_hat - self.alphas)
        rel_err = np.where(valid, abs_err / np.where(valid, self.alphas, 1.0), np.nan)

        n_valid_total = int(valid.sum())
        global_mre    = float(np.mean(rel_err[valid])) if n_valid_total else float('nan')

        sources_arr = np.array(self.sources)
        out = {}

        def _summ(mask):
            mask = mask & valid
            if not np.any(mask):
                return {'mre': float('nan'), 'weighted_mre': float('nan'),
                        'mae': float('nan'), 'n_constraints': 0,
                        'contribution': 0.0, 'contribution_pct': 0.0}
            w = self.weights[mask]
            mre_g = float(np.mean(rel_err[mask]))
            # contribution of this group to the GLOBAL (unweighted) MRE:
            #   global_mre = sum_g n_g * mre_g / n_valid_total
            # so contributions are additive and sum to the global figure.
            contrib = mre_g * mask.sum() / n_valid_total if n_valid_total else 0.0
            return {
                'mre':          mre_g,
                'weighted_mre': float(np.average(rel_err[mask], weights=w)) if w.sum() > 0 else mre_g,
                'mae':          float(np.mean(abs_err[mask])),
                'n_constraints': int(mask.sum()),
                'contribution':     float(contrib),
                'contribution_pct': float(100.0 * contrib / global_mre) if global_mre else 0.0,
            }

        for tag in sorted(set(self.sources)):
            out[tag] = _summ(sources_arr == tag)
        out['ALL'] = _summ(np.ones(self.m, dtype=bool))
        return out

    def _arity_array(self) -> np.ndarray:
        return np.array([len(a) for a in self.cs.attrs_list])

    def stratified_mre_arity(self, min_prob_threshold: float = 1e-3) -> dict:
        """
        Break down the final-fit MRE by constraint ARITY (1=unary marginals,
        2=binary CPT joints, 3=ternary CPT joints), with the same fields as
        stratified_mre() including additive contributions to the global MRE.
        """
        if self.pool_ is None:
            raise RuntimeError("Call fit() before stratified_mre_arity().")

        alpha_hat = self._estimate_expectations(self.pool_)
        valid   = self.alphas > min_prob_threshold
        abs_err = np.abs(alpha_hat - self.alphas)
        rel_err = np.where(valid, abs_err / np.where(valid, self.alphas, 1.0), np.nan)
        n_valid_total = int(valid.sum())
        global_mre    = float(np.mean(rel_err[valid])) if n_valid_total else float('nan')
        arity = self._arity_array()

        def _summ(mask):
            mask = mask & valid
            if not np.any(mask):
                return {'mre': float('nan'), 'weighted_mre': float('nan'),
                        'mae': float('nan'), 'n_constraints': 0,
                        'contribution': 0.0, 'contribution_pct': 0.0}
            w = self.weights[mask]
            mre_g   = float(np.mean(rel_err[mask]))
            contrib = mre_g * mask.sum() / n_valid_total
            return {
                'mre':          mre_g,
                'weighted_mre': float(np.average(rel_err[mask], weights=w)) if w.sum() > 0 else mre_g,
                'mae':          float(np.mean(abs_err[mask])),
                'n_constraints': int(mask.sum()),
                'contribution':     float(contrib),
                'contribution_pct': float(100.0 * contrib / global_mre) if global_mre else 0.0,
            }

        labels = {1: 'unary', 2: 'binary', 3: 'ternary'}
        out = {labels[a]: _summ(arity == a) for a in (1, 2, 3)}
        out['ALL'] = _summ(np.ones(self.m, dtype=bool))
        return out

    def stratified_mre_source_arity(self,
                                    min_prob_threshold: float = 1e-3) -> dict:
        """
        Source × arity cross-tab of the final-fit MRE:
            {source: {arity_label: {'mre', 'n_constraints',
                                    'contribution', 'contribution_pct'}}}
        Contributions are additive across ALL cells and sum to the global
        MRE — this identifies exactly which (source, arity) block is
        responsible for the headline number.
        """
        if self.sources is None:
            raise ValueError("requires `sources` at construction time.")
        if self.pool_ is None:
            raise RuntimeError("Call fit() before this method.")

        alpha_hat = self._estimate_expectations(self.pool_)
        valid   = self.alphas > min_prob_threshold
        rel_err = np.where(valid,
                           np.abs(alpha_hat - self.alphas)
                           / np.where(valid, self.alphas, 1.0), np.nan)
        n_valid_total = int(valid.sum())
        global_mre    = float(np.mean(rel_err[valid])) if n_valid_total else float('nan')
        arity       = self._arity_array()
        sources_arr = np.array(self.sources)
        labels = {1: 'unary', 2: 'binary', 3: 'ternary'}

        out = {}
        for tag in sorted(set(self.sources)):
            out[tag] = {}
            for a in (1, 2, 3):
                mask = (sources_arr == tag) & (arity == a) & valid
                if not np.any(mask):
                    out[tag][labels[a]] = {'mre': float('nan'), 'n_constraints': 0,
                                           'contribution': 0.0, 'contribution_pct': 0.0}
                    continue
                mre_g   = float(np.mean(rel_err[mask]))
                contrib = mre_g * mask.sum() / n_valid_total
                out[tag][labels[a]] = {
                    'mre': mre_g, 'n_constraints': int(mask.sum()),
                    'contribution':     float(contrib),
                    'contribution_pct': float(100.0 * contrib / global_mre) if global_mre else 0.0,
                }
        return out

    def __repr__(self):
        status = (f"fitted, MRE={self.final_mre:.4f}, "
                  f"{self.n_iters} iters"
                  if self.lambdas is not None else "not fitted")
        return f"GibbsPCDSolver(K={self.K}, m={self.m}, {status})"


# ------------------------------------------------------------------ #
#  Optional Numba kernel (standalone, outside class)                   #
# ------------------------------------------------------------------ #

def _seed_numba_rng(seed: int) -> None:
    """
    Seed every Numba thread's private RNG state (seed + thread_id).

    Numba's np.random state is thread-local; seeding only the main thread
    leaves prange workers unseeded. A prange loop with exactly one
    iteration per thread (static scheduling) seeds them all. Runs are
    then bit-reproducible for a fixed NUMBA_NUM_THREADS.
    No-op if numba is not installed.
    """
    try:
        import numba
        from numba import njit, prange

        n_threads = numba.get_num_threads()

        @njit(parallel=True)
        def _seed_all(base_seed, nt):
            for i in prange(nt):
                np.random.seed(base_seed + i)

        _seed_all(seed, n_threads)
    except ImportError:
        pass


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