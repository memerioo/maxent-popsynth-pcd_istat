"""
structural_blocks.py
--------------------
Sentinel *blocks*, legal pool initialisation, and legality diagnostics.

WHY THIS MODULE EXISTS
======================
Pinning the structural-zero parameters at lambda_j = -30 makes logically
impossible individuals unreachable -- which is the intended behaviour --
but it also makes large regions of X unreachable *from each other* under
single-site Gibbs.  The sentinel design of Appendix A.2 is the cause: an
individual who is `employment=FullTime` carries seven further attributes
that are all forbidden from taking their NotWorker sentinel, and an
individual who is `employment=NotInLF` is forbidden from taking anything
*but* the sentinel in those same seven attributes.  Moving between the
two therefore requires eight simultaneous flips.  Every intermediate
state violates at least one structural table, so every intermediate
state has weight e^-30 or less, and Glauber dynamics never takes the
path.

Empirically, with hard zeros and a uniform initial pool, the fraction of
"workers" in the pool freezes within five sweeps and never moves again,
whatever lambda does.  The same happens to the student block and to the
child/adult split.  The reported marginals are then a property of the
random initialisation, not of the fitted model.

THE FIX
=======
Two ingredients, both implemented here.

1. `ancestral_init_pool` -- start the chain from a pool that is already
   legal (zero structural violations) and already carries the published
   composition, instead of from uniform noise.  With the frozen blocks
   this alone fixes the marginals of the driver attributes, because a
   frozen quantity that starts on target stays on target.

2. `resolve_blocks` -- declare the blocks, so that `GibbsPCDSolver` can
   run a Metropolis-Hastings *toggle* move that flips one individual
   between the active and the inactive basin in a single step, copying
   a whole block pattern from a donor.  The move never visits an
   intermediate illegal state, so the e^-30 barrier is irrelevant to it.
   That restores mobility, and therefore restores the solver's ability
   to fit conditional structure such as P(employment | age).

Both are needed.  (1) without (2) pins each driver's marginal to its
initial value and lets no conditional structure be learned across the
sentinel boundary; (2) without (1) mixes correctly but takes very many
iterations to walk the composition from uniform noise to the target.

WHAT A BLOCK IS
===============
A block is a set of attributes governed by one *driver* attribute, such
that the driver's value determines whether the block's other attributes
take their sentinel value or a real value:

    W (work)    driver `employment`   active = {FullTime, PartTime}
    S (study)   driver `StudentStat`  active = {SchoolStudent, UniStudent}
    C (child)   driver `age`          active = {0-4, 5-14}

`ResidenceQ` and `LunchPlace` are listed in both W and S even though
they are not sentinel attributes.  They are included because structural
tables couple them to the block (H3/H4/H43/H44/H45 tie CommuteInward to
the two commute attributes, H46 ties Canteen and AtS/WPlace to being a
worker or a student).  If they were excluded, a toggle that switched an
inward commuter to non-worker would leave ResidenceQ=CommuteInward
behind and be rejected by H45.  Including them costs a little acceptance
and removes the barrier.

Block C contains W and S in full: a child must be a non-worker, and a
child aged 0-4 must be a non-student, so the toggle has to move those
attributes too or it would always be rejected.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ #
#  Block declarations (by attribute NAME -- resolved to indices below) #
# ------------------------------------------------------------------ #

_W_ATTRS = [
    "employment", "employ_stat", "Wage", "employ_commute",
    "Profession", "Occupation", "MainTranspWorker", "TranspTime_Worker",
    "ResidenceQ", "LunchPlace",
]
_W_ACTIVE = ["FullTime", "PartTime"]

_S_ATTRS = [
    "StudentStat", "Student_commute", "MainTranspStudnt", "TranspTime_Stud",
    "ResidenceQ", "LunchPlace",
]
_S_ACTIVE = ["SchoolStudent", "UniStudent"]

_ACTIVITY_ATTRS = [
    "SundayOut", "SaturdayOut", "WeekDayOut",
    "SunSocialEnterT", "SatSocialEnterT", "WeekDSocialEnterT",
    "SunSportOutD", "SatSportOutD", "WeekDSportOutD",
]

_C_EXTRA = [
    "age", "marital", "education", "BMI",
    "AlcoholCons", "Smoking", "LifeSatisfaction",
] + _ACTIVITY_ATTRS
_C_ACTIVE = ["0-4", "5-14"]

BLOCK_SPECS = [
    {"name": "work",  "driver": "employment",
     "attrs": _W_ATTRS,                          "active": _W_ACTIVE},
    {"name": "study", "driver": "StudentStat",
     "attrs": _S_ATTRS,                          "active": _S_ACTIVE},
    # The nine activity attributes each carry their own Under3yo sentinel and
    # are forced to agree with one another by the pairwise h_ tables added
    # after the profile inspection of Section 5.10.  That agreement rule makes
    # them a sentinel block in exactly the same sense as work and study: an
    # individual can only leave the "under three" basin by flipping all nine
    # at once, which single-site Gibbs can never do.  Without this block the
    # nine attributes are frozen at their initial values.
    #
    # Unlike work and study there is no separate driver attribute -- being
    # under three is expressed only through the sentinels themselves -- so
    # SundayOut stands in as the nominal driver.  The block move copies all
    # nine from a donor, which preserves agreement by construction; proposals
    # that clash with `age` (H24-H32 allow Under3yo only in the 0-4 band) are
    # rejected automatically by the acceptance ratio.
    {"name": "under3", "driver": "SundayOut",
     "attrs": _ACTIVITY_ATTRS,                   "active": ["ExitHouse", "StayIn"]},
    {"name": "child", "driver": "age",
     "attrs": sorted(set(_W_ATTRS + _S_ATTRS + _C_EXTRA)),
     "active": _C_ACTIVE},
]


def resolve_blocks(attr_names, attr_meta, names=("work", "study", "under3")):
    """
    Turn the declarative BLOCK_SPECS into index arrays for the solver.

    Parameters
    ----------
    attr_names : list[str]        -- ATTR_NAMES_SYNTH
    attr_meta  : dict             -- ATTR_META
    names      : tuple[str]       -- which blocks to activate.  The default
        omits "child": the age composition is pinned exactly on its
        published BO target by `ancestral_init_pool`, and the child block
        spans 30 of the 34 attributes, so its toggle acceptance is very
        low.  Enable it with names=("work","study","child") if you want
        the age composition to remain free during fitting.

    Returns
    -------
    list of dicts with keys:
        name, attrs_idx (int32 (b,)), driver_pos (int),
        active_vals (int32 array)
    """
    out = []
    for spec in BLOCK_SPECS:
        if spec["name"] not in names:
            continue
        attrs = [a for a in spec["attrs"] if a in attr_names]
        missing = set(spec["attrs"]) - set(attrs)
        if missing:
            raise KeyError(f"block {spec['name']}: unknown attributes {sorted(missing)}")
        idx = np.array([attr_names.index(a) for a in attrs], dtype=np.int32)
        order = np.argsort(idx)
        idx = idx[order]
        attrs = [attrs[i] for i in order]
        drv = spec["driver"]
        v2i = attr_meta[drv]["val_to_int"]
        out.append({
            "name":        spec["name"],
            "attrs":       attrs,
            "attrs_idx":   idx,
            "driver":      drv,
            "driver_pos":  attrs.index(drv),
            "active_vals": np.array([v2i[v] for v in spec["active"]], dtype=np.int32),
        })
    return out


# ------------------------------------------------------------------ #
#  Legality diagnostics                                                #
# ------------------------------------------------------------------ #

def violation_counts(cs, pool: np.ndarray) -> np.ndarray:
    """
    Number of structural (alpha_j == 0) constraints each individual
    violates.  Returns an (N,) int array.  Zero everywhere == the pool
    contains no logically impossible individual.
    """
    alphas = cs.alphas_array
    viol = np.zeros(len(pool), dtype=np.int32)
    for j in np.flatnonzero(alphas == 0.0):
        attrs = cs.attrs_list[j]
        vals = cs.vals_list[j]
        viol += np.all(pool[:, attrs] == vals[np.newaxis, :], axis=1)
    return viol


def legality_report(cs, pool: np.ndarray, attr_names=None, attr_meta=None,
                    top: int = 8) -> str:
    """Human-readable summary of structural violations in `pool`."""
    viol = violation_counts(cs, pool)
    n_bad = int((viol > 0).sum())
    lines = [f"  Illegal individuals : {n_bad:,} / {len(pool):,} "
             f"({100.0 * n_bad / len(pool):.4f}%)"]
    if n_bad and attr_names is not None:
        alphas = cs.alphas_array
        hits = []
        for j in np.flatnonzero(alphas == 0.0):
            attrs, vals = cs.attrs_list[j], cs.vals_list[j]
            c = int(np.all(pool[:, attrs] == vals[np.newaxis, :], axis=1).sum())
            if c:
                desc = ", ".join(
                    f"{attr_names[a]}={attr_meta[attr_names[a]]['vals'][v]}"
                    for a, v in zip(attrs, vals))
                hits.append((c, desc))
        hits.sort(reverse=True)
        lines.append(f"  Worst violated structural constraints:")
        for c, desc in hits[:top]:
            lines.append(f"    {c:>8,}  {desc}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Legal, on-target ancestral initialisation                           #
# ------------------------------------------------------------------ #

def _probs(marg: dict, attr: str, meta: dict, keep=None, drop=None) -> np.ndarray:
    """Marginal of `attr` as a probability vector over its domain,
    optionally restricted to `keep` / excluding `drop`, renormalised."""
    vals = meta[attr]["vals"]
    p = np.array([float(marg.get(attr, {}).get(v, 0.0)) for v in vals])
    if keep is not None:
        mask = np.array([v in keep for v in vals])
        p = p * mask
    if drop is not None:
        mask = np.array([v not in drop for v in vals])
        p = p * mask
    s = p.sum()
    if s <= 0:                       # no information -> uniform on allowed
        p = np.ones(len(vals))
        if keep is not None:
            p *= np.array([v in keep for v in vals])
        if drop is not None:
            p *= np.array([v not in drop for v in vals])
        s = p.sum()
    return p / s


def ancestral_init_pool(N: int, marginals: dict, attr_names, attr_meta,
                        seed: int = 1) -> np.ndarray:
    """
    Build an (N, K) int32 pool that is structurally legal by construction
    and whose unary marginals sit on the published targets.

    Attributes are drawn in dependency order.  Whenever a structural rule
    forces a value on a sub-population, the *free* sub-population is drawn
    from a renormalised distribution chosen so that the overall marginal
    still matches the published one -- e.g. children are forced to
    `employment=NotInLF`, so adults are activated at rate
    P(worker) / P(adult) rather than at rate P(worker).

    The result is not a sample from p_lambda; it is a starting point.
    Gibbs sweeps and the block toggles then move it towards p_lambda while
    the structural zeros keep it legal.
    """
    rng = np.random.default_rng(seed)
    K = len(attr_names)
    pool = np.zeros((N, K), dtype=np.int32)
    I = {a: i for i, a in enumerate(attr_names)}
    V = {a: attr_meta[a]["val_to_int"] for a in attr_names}

    def draw(attr, p, mask=None):
        """Sample `attr` from probability vector p, for rows in `mask`."""
        n = N if mask is None else int(mask.sum())
        if n == 0:
            return
        vals = rng.choice(len(p), size=n, p=p)
        if mask is None:
            pool[:, I[attr]] = vals
        else:
            pool[mask, I[attr]] = vals

    def rate(target: float, eligible: np.ndarray) -> float:
        """Activation rate among `eligible` so the overall share is `target`."""
        frac = eligible.mean()
        return float(np.clip(target / frac, 0.0, 1.0)) if frac > 0 else 0.0

    # ---- 1. age, and the child indicator everything else keys off ----
    draw("age", _probs(marginals, "age", attr_meta))
    age = pool[:, I["age"]]
    child = np.isin(age, [V["age"]["0-4"], V["age"]["5-14"]])
    adult = ~child
    infant = age == V["age"]["0-4"]

    # ---- 2. unconstrained attributes ----
    for a in ("sex", "citizenship", "Health", "Medication"):
        draw(a, _probs(marginals, a, attr_meta))

    # ---- 3. child-forced attributes (H1, H2, H21, H23, H33) ----
    for attr, forced in (("marital", "NeverMarried"),
                         ("education", "SecondaryAndLess"),
                         ("BMI", "UnderAge"),
                         ("Smoking", "Never"),
                         ("LifeSatisfaction", "Under14yo")):
        pool[child, I[attr]] = V[attr][forced]
    draw("marital",  _probs(marginals, "marital",  attr_meta), adult)
    draw("education", _probs(marginals, "education", attr_meta), adult)
    draw("BMI",      _probs(marginals, "BMI",      attr_meta, drop={"UnderAge"}), adult)
    draw("Smoking",  _probs(marginals, "Smoking",  attr_meta), adult)
    draw("LifeSatisfaction",
         _probs(marginals, "LifeSatisfaction", attr_meta, drop={"Under14yo"}), adult)

    # H22: 0-4 -> Never; 5-14 -> Never or Exceptionally
    pool[infant, I["AlcoholCons"]] = V["AlcoholCons"]["Never"]
    mid = child & ~infant
    draw("AlcoholCons",
         _probs(marginals, "AlcoholCons", attr_meta, keep={"Never", "Exceptionally"}), mid)
    draw("AlcoholCons", _probs(marginals, "AlcoholCons", attr_meta), adult)

    # ---- 4. activity attributes: Under3yo only inside the 0-4 band ----
    # (H24-H32).  One shared indicator: an under-3 is under 3 in all nine.
    tgt_u3 = float(marginals.get("SundayOut", {}).get("Under3yo", 0.0))
    under3 = infant & (rng.random(N) < rate(tgt_u3, infant))
    for a in _ACTIVITY_ATTRS:
        pool[under3, I[a]] = V[a]["Under3yo"]
        draw(a, _probs(marginals, a, attr_meta, drop={"Under3yo"}), ~under3)

    # ---- 5. study block (H37: no students at 0-4 or 50+) ----
    elig_s = np.isin(age, [V["age"][b] for b in ("5-14", "15-24", "25-34", "35-49")])
    tgt_student = 1.0 - float(marginals["StudentStat"].get("NotStudent", 0.0))
    is_student = elig_s & (rng.random(N) < rate(tgt_student, elig_s))
    pool[:, I["StudentStat"]] = V["StudentStat"]["NotStudent"]
    school = is_student & np.isin(age, [V["age"]["5-14"]])
    uni = is_student & np.isin(age, [V["age"][b] for b in ("25-34", "35-49")])
    teen = is_student & (age == V["age"]["15-24"])
    pool[school, I["StudentStat"]] = V["StudentStat"]["SchoolStudent"]
    pool[uni, I["StudentStat"]] = V["StudentStat"]["UniStudent"]
    p_school = float(marginals["StudentStat"].get("SchoolStudent", 0.5))
    p_uni = float(marginals["StudentStat"].get("UniStudent", 0.5))
    pick_school = rng.random(N) < p_school / max(p_school + p_uni, 1e-12)
    pool[teen & pick_school, I["StudentStat"]] = V["StudentStat"]["SchoolStudent"]
    pool[teen & ~pick_school, I["StudentStat"]] = V["StudentStat"]["UniStudent"]

    for a in ("Student_commute", "MainTranspStudnt", "TranspTime_Stud"):
        pool[:, I[a]] = V[a]["NotStudent"]
        draw(a, _probs(marginals, a, attr_meta, drop={"NotStudent"}), is_student)

    # ---- 5b. work block (H34-H36 children; H38 FullTime excludes students) ----
    m_emp = marginals["employment"]
    tgt_work = float(m_emp.get("FullTime", 0.0)) + float(m_emp.get("PartTime", 0.0))
    is_worker = adult & (rng.random(N) < rate(tgt_work, adult))
    p_ft = float(m_emp.get("FullTime", 0.0)) / max(tgt_work, 1e-12)
    full = is_worker & (rng.random(N) < p_ft) & ~is_student   # H38
    part = is_worker & ~full
    p_un = float(m_emp.get("Unemployed", 0.0))
    p_ni = float(m_emp.get("NotInLF", 0.0))
    unemp = (~is_worker) & (rng.random(N) < p_un / max(p_un + p_ni, 1e-12))
    pool[:, I["employment"]] = V["employment"]["NotInLF"]
    pool[unemp, I["employment"]] = V["employment"]["Unemployed"]
    pool[full, I["employment"]] = V["employment"]["FullTime"]
    pool[part, I["employment"]] = V["employment"]["PartTime"]

    for a in ("employ_stat", "Wage", "employ_commute",
              "Profession", "Occupation", "MainTranspWorker", "TranspTime_Worker"):
        pool[:, I[a]] = V[a]["NotWorker"]
        draw(a, _probs(marginals, a, attr_meta, drop={"NotWorker"}), is_worker)

    # ---- 6. commute geometry (H41, H3/H4, H43/H44, H45) ----
    ec, sc = I["employ_commute"], I["Student_commute"]
    # H41 row "Outward": an outward work commute forbids any study commute.
    bad = is_student & (pool[:, ec] == V["employ_commute"]["Outward"])
    pool[bad, ec] = V["employ_commute"]["InsideBO"]
    # An inward commute on either side forces the other to be inward or absent,
    # and forces ResidenceQ=CommuteInward (H3/H4/H43/H44).
    inward = ((pool[:, ec] == V["employ_commute"]["Inward"]) |
              (pool[:, sc] == V["Student_commute"]["Inward"]))
    pool[inward & is_worker,  ec] = V["employ_commute"]["Inward"]
    pool[inward & is_student, sc] = V["Student_commute"]["Inward"]
    pool[:, I["ResidenceQ"]] = 0
    draw("ResidenceQ",
         _probs(marginals, "ResidenceQ", attr_meta, drop={"CommuteInward"}), ~inward)
    pool[inward, I["ResidenceQ"]] = V["ResidenceQ"]["CommuteInward"]

    # H42: when an individual both works and studies the two main transport
    # modes must coincide.
    both = is_worker & is_student
    pool[both, I["MainTranspWorker"]] = pool[both, I["MainTranspStudnt"]]

    # ---- 7. LunchPlace (H46: no canteen for a non-worker non-student) ----
    neither = (~is_worker) & (~is_student)
    draw("LunchPlace", _probs(marginals, "LunchPlace", attr_meta), ~neither)
    draw("LunchPlace",
         _probs(marginals, "LunchPlace", attr_meta, drop={"Canteen", "AtS/WPlace"}),
         neither)

    return pool