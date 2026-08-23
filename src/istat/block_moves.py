"""
block_moves.py
--------------
Metropolis-Hastings *block* moves for sentinel-structured populations.

WHY THIS EXISTS
===============
Single-site Gibbs cannot change an individual's sentinel-block membership
once the structural zeros are pinned hard (lambda_j = -30).  To turn a
worker into a non-worker, seven attributes must leave their real values
for `NotWorker` at the same instant: any single flip lands on a state that
violates a structural table and therefore has weight e^-30.  Glauber
dynamics never takes that path, so the fraction of workers in the pool is
frozen at whatever the initialisation produced -- verified empirically:
starting from an on-target pool the worker share stays at 0.4626 for 40
sweeps to four decimal places, i.e. exactly zero transitions.

This module restores mobility with a move that jumps the whole block in
one step, never visiting an intermediate illegal state.  The e^-30 barrier
is then irrelevant, because the barrier is *between* the two basins, and
the proposal steps over it rather than through it.

THE MOVE
========
An independence sampler with a donor pool.  For individual i:

  1. draw a donor d uniformly from the pool;
  2. propose x' = x_i with the block attributes overwritten by x_d's;
  3. accept with the Metropolis-Hastings probability

         alpha = min(1, [pi(x') q(b)] / [pi(x) q(b')])

     where b, b' are the current and proposed block patterns and q is the
     empirical frequency of a pattern in the pool.

The q-ratio is the part that is easy to get wrong.  Without it the move
is a plain independence sampler against a non-uniform proposal and it
does *not* target p_lambda.  Note also that the tempting alternative -- a
symmetric *swap* of blocks between two individuals -- is not usable here:
a swap conserves the number of workers, so it can never move the very
composition we are trying to fit.

Because pi(x') / pi(x) only involves constraints that touch the block,
all other constraints cancel and never have to be evaluated.  Illegal
individuals repair themselves automatically: their current state carries
a -30 penalty, so any legal proposal is accepted with probability ~1.

VALIDITY NOTE
=============
q is the pool's own pattern distribution, so the kernel is adaptive.  The
counts are frozen at the start of each outer iteration and treated as a
fixed proposal for that iteration, which makes each individual sweep a
valid MH kernel for p_lambda.  This is the usual diminishing-adaptation
argument and is well inside the approximation already made by PCD.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ #
#  Precomputation                                                      #
# ------------------------------------------------------------------ #

def prepare_block(cs, attrs_idx: np.ndarray) -> dict:
    """
    Flatten the constraints that touch a block into C-contiguous arrays.

    Only constraints j with S_j intersecting the block can change value
    when the block is overwritten; every other constraint cancels in the
    acceptance ratio.

    Parameters
    ----------
    cs        : ConstraintSet
    attrs_idx : (b,) int32 -- attribute indices forming the block

    Returns
    -------
    dict with keys
        attrs_idx   (b,)   int32  block attributes (sorted)
        pos_in_blk  (K,)   int32  position of attribute k in the block, -1 if outside
        c_j         (J,)   int32  constraint indices touching the block
        c_off       (J+1,) int32  CSR offsets into c_attrs / c_vals
        c_attrs     (P,)   int32  attribute of each (constraint, slot)
        c_vals      (P,)   int32  required value of each (constraint, slot)
    """
    attrs_idx = np.asarray(attrs_idx, dtype=np.int32)
    in_blk = np.zeros(cs.K, dtype=bool)
    in_blk[attrs_idx] = True

    pos_in_blk = np.full(cs.K, -1, dtype=np.int32)
    for p, k in enumerate(attrs_idx):
        pos_in_blk[k] = p

    c_j = [j for j in range(cs.m) if np.any(in_blk[cs.attrs_list[j]])]

    c_off = np.zeros(len(c_j) + 1, dtype=np.int32)
    for i, j in enumerate(c_j):
        c_off[i + 1] = c_off[i] + len(cs.attrs_list[j])

    c_attrs = np.zeros(c_off[-1], dtype=np.int32)
    c_vals = np.zeros(c_off[-1], dtype=np.int32)
    for i, j in enumerate(c_j):
        s, e = c_off[i], c_off[i + 1]
        c_attrs[s:e] = cs.attrs_list[j]
        c_vals[s:e] = cs.vals_list[j]

    return {
        "attrs_idx":  attrs_idx,
        "pos_in_blk": pos_in_blk,
        "c_j":        np.array(c_j, dtype=np.int32),
        "c_off":      c_off,
        "c_attrs":    c_attrs,
        "c_vals":     c_vals,
    }


def block_pattern_counts(pool: np.ndarray, attrs_idx: np.ndarray):
    """
    Count how many individuals share each block pattern.

    Returns
    -------
    own_count : (N,) float64 -- for each individual, the number of pool
        members carrying its own block pattern.  This is N * q(b), and
        since the acceptance ratio only uses q(b)/q(b') the factor N
        cancels.
    """
    sub = pool[:, attrs_idx]
    # Encode the block pattern as bytes and count identical rows.
    view = np.ascontiguousarray(sub).view(
        np.dtype((np.void, sub.dtype.itemsize * sub.shape[1])))
    _, inv, counts = np.unique(view.ravel(), return_inverse=True,
                               return_counts=True)
    return counts[inv].astype(np.float64)


# ------------------------------------------------------------------ #
#  NumPy reference implementation                                      #
# ------------------------------------------------------------------ #

def block_toggle_numpy(pool, lam, blk, rng, subset=None):
    """
    One MH block sweep, pure NumPy.  Modifies `pool` in place.
    Returns the number of accepted moves.
    """
    N = pool.shape[0]
    idx = np.arange(N) if subset is None else subset
    donors = rng.integers(0, N, size=len(idx))

    own = block_pattern_counts(pool, blk["attrs_idx"])
    log_q = np.log(own[idx]) - np.log(own[donors])

    prop = pool[idx].copy()
    prop[:, blk["attrs_idx"]] = pool[np.ix_(donors, blk["attrs_idx"])]

    cur = pool[idx]
    e_cur = np.zeros(len(idx))
    e_prop = np.zeros(len(idx))
    for i, j in enumerate(blk["c_j"]):
        s, e = blk["c_off"][i], blk["c_off"][i + 1]
        a, v = blk["c_attrs"][s:e], blk["c_vals"][s:e]
        e_cur += lam[j] * np.all(cur[:, a] == v[np.newaxis, :], axis=1)
        e_prop += lam[j] * np.all(prop[:, a] == v[np.newaxis, :], axis=1)

    log_acc = (e_prop - e_cur) + log_q
    accept = np.log(rng.random(len(idx))) < log_acc
    pool[idx[accept]] = prop[accept]
    return int(accept.sum())


# ------------------------------------------------------------------ #
#  Numba kernel                                                        #
# ------------------------------------------------------------------ #

def block_toggle(pool, lam, blk, rng, kernel=None, frac=1.0):
    """
    One MH block sweep over `frac` of the pool.  Modifies `pool` in place.

    `idx` is drawn WITHOUT replacement so that no two threads write the
    same row.  Donors are drawn with replacement (they are only read, and
    their block values are snapshotted before the kernel runs).

    Returns (n_accepted, n_attempted).
    """
    N = pool.shape[0]
    n_move = N if frac >= 1.0 else int(N * frac)
    idx = (np.arange(N, dtype=np.int64) if n_move >= N
           else rng.choice(N, size=n_move, replace=False).astype(np.int64))
    donors = rng.integers(0, N, size=n_move)

    own = block_pattern_counts(pool, blk["attrs_idx"])
    log_q = np.log(own[idx]) - np.log(own[donors])

    if kernel is None:
        # NumPy path: recompute inside the reference implementation.
        prop = pool[idx].copy()
        prop[:, blk["attrs_idx"]] = pool[np.ix_(donors, blk["attrs_idx"])]
        cur = pool[idx]
        e_cur = np.zeros(n_move)
        e_prop = np.zeros(n_move)
        for i, j in enumerate(blk["c_j"]):
            s, e = blk["c_off"][i], blk["c_off"][i + 1]
            a, v = blk["c_attrs"][s:e], blk["c_vals"][s:e]
            e_cur += lam[j] * np.all(cur[:, a] == v[np.newaxis, :], axis=1)
            e_prop += lam[j] * np.all(prop[:, a] == v[np.newaxis, :], axis=1)
        accept = np.log(rng.random(n_move)) < (e_prop - e_cur) + log_q
        pool[idx[accept]] = prop[accept]
        return int(accept.sum()), n_move

    donor_blk = np.ascontiguousarray(pool[np.ix_(donors, blk["attrs_idx"])])
    n_acc = kernel(pool, lam, idx, donor_blk, log_q,
                   blk["pos_in_blk"], blk["c_j"], blk["c_off"],
                   blk["c_attrs"], blk["c_vals"], blk["attrs_idx"])
    return int(n_acc), n_move


def make_block_toggle_kernel():
    """Return a compiled block-toggle kernel, or None if numba is absent."""
    try:
        from numba import njit, prange
    except ImportError:
        return None

    @njit(parallel=True, cache=True)
    def _kernel(pool, lam, idx, donor_blk, log_q,
                pos_in_blk, c_j, c_off, c_attrs, c_vals, block_attrs):
        """
        `donor_blk` is an (n_move, b) SNAPSHOT of the donors' block values,
        taken before this loop starts.  Reading the donors straight out of
        `pool` inside a prange would race against the writes performed by
        other threads for accepted moves, which would silently corrupt both
        the proposal and the acceptance ratio.  `log_q` is likewise
        precomputed from the frozen pattern counts.
        """
        n_move = idx.shape[0]
        n_c = c_j.shape[0]
        n_b = block_attrs.shape[0]
        n_acc = 0

        for t in prange(n_move):
            i = idx[t]

            # Skip identical patterns: the move would be a no-op.
            same = True
            for p in range(n_b):
                if pool[i, block_attrs[p]] != donor_blk[t, p]:
                    same = False
                    break
            if same:
                continue

            e_cur = 0.0
            e_prop = 0.0
            for ci in range(n_c):
                s = c_off[ci]
                e = c_off[ci + 1]
                m_cur = True
                m_prop = True
                for q in range(s, e):
                    a = c_attrs[q]
                    v = c_vals[q]
                    if pool[i, a] != v:
                        m_cur = False
                    # proposed value: donor's if a is inside the block
                    p = pos_in_blk[a]
                    if p >= 0:
                        pv = donor_blk[t, p]
                    else:
                        pv = pool[i, a]
                    if pv != v:
                        m_prop = False
                    if (not m_cur) and (not m_prop):
                        break
                if m_cur:
                    e_cur += lam[c_j[ci]]
                if m_prop:
                    e_prop += lam[c_j[ci]]

            if np.log(np.random.random()) < (e_prop - e_cur) + log_q[t]:
                for p in range(n_b):
                    pool[i, block_attrs[p]] = donor_blk[t, p]
                n_acc += 1

        return n_acc

    return _kernel