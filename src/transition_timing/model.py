"""
Optimal-stopping model for the timing of social transition.

State variable
--------------
t : months since the start of gender affirming hormone therapy (titty skittles) (t = 0 at HRT start).

Cost streams
------------
B(t)        marginal cost per month of continuing to boymode
l(t)        marginal value of authentic time (the "life lost" term)
A * C(t, r) marginal cost per month of being socially transitioned and
            read as trans, where A is environment hostility and
            C is clock probability, higher if ur a brick, lower for passoids

Clock probability
-----------------
A person is clocked if ANY strong cue fires, so cues combine as
independent failure modes:

    C(t, r) = 1 - prod_i (1 - c_i(t, r))

Only the face cue is mostly hormone blocked; hair, brow and voice cues depend
on preparation fraction r in [0, 1]. Take Face to represent the rest of body and fat shifts as well

Objective
---------
    J(tau) = int_0^tau [B(t) + l(t)] dt  +  int_tau^T A C(t, r) dt  +  S(tau)

with S(tau) a switching cost containing step discontinuities at
context-formation dates.  The optimum is the global minimiser of J.

Smooth first-order condition (between steps of S):

    B(tau) + l(tau) = A C(tau, r)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq



# Parameters


@dataclass(frozen=True)
class Cue:
    """A non-hormonal clocking cue with a preparation lever.

    c(r) = c0 * (1 - efficacy * r)

    c0        clock probability of this cue when nothing has been done
    efficacy  fraction of c0 removed at full preparation (r = 1)
    """

    name: str
    c0: float
    efficacy: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.c0 <= 1.0:
            raise ValueError(f"{self.name}: c0 must lie in [0, 1]")
        if not 0.0 <= self.efficacy <= 1.0:
            raise ValueError(f"{self.name}: efficacy must lie in [0, 1]")

    def __call__(self, r: float) -> float:
        return self.c0 * (1.0 - self.efficacy * r)

    def d_dr(self, r: float) -> float:
        return -self.c0 * self.efficacy


@dataclass(frozen=True)
class FaceCue:
    """Hormone-gated face cue.

    c_face(t) = c_inf + (c0 - c_inf) * sigma(-k (t - t_half))

    c0       clock probability from face alone at t = 0
    c_inf    floor set by bone structure (what HRT cannot remove)
    k        rate of facial change, 1/month
    t_half   month at which half the achievable change has happened
    """

    c0: float = 0.70
    c_inf: float = 0.15
    k: float = 0.18
    t_half: float = 16.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.c_inf <= self.c0 <= 1.0:
            raise ValueError("need 0 <= c_inf <= c0 <= 1")
        if self.k <= 0:
            raise ValueError("k must be positive")

    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.c_inf + (self.c0 - self.c_inf) * _sigmoid(-self.k * (np.asarray(t) - self.t_half))

    def d_dt(self, t: float | np.ndarray) -> float | np.ndarray:
        s = _sigmoid(-self.k * (np.asarray(t) - self.t_half))
        return -(self.c0 - self.c_inf) * self.k * s * (1.0 - s)


@dataclass(frozen=True)
class Step:
    """A step in the switching cost at a context-formation date."""

    at: float
    height: float
    label: str = ""


@dataclass(frozen=True)
class Params:
    """All model parameters.  Defaults reproduce the interactive widget."""

    # boymoding cost  B(t) = B0 (1 + beta t)
    B0: float = 0.55
    beta: float = 0.03

    # life-lost term  l(t) = l0 exp(-lam t)
    l0: float = 0.0
    lam: float = 0.03

    # environment
    A: float = 0.80

    # preparation fraction for non-hormonal cues
    r: float = 0.15

    # cues
    face: FaceCue = field(default_factory=FaceCue)
    cues: tuple[Cue, ...] = (
        Cue("hair", 0.60, 0.85),
        Cue("brow", 0.40, 0.85),
        Cue("voice", 0.75, 0.80),
    )

    # switching cost
    s0: float = 0.0
    steps: tuple[Step, ...] = ()

    # horizon, months
    T: float = 36.0

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


# Model functions


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def clock_probability(t, p: Params, r: float | None = None):
    """C(t, r) = 1 - prod_i (1 - c_i)."""
    r = p.r if r is None else r
    surv = 1.0 - p.face(t)
    for cue in p.cues:
        surv = surv * (1.0 - cue(r))
    return 1.0 - surv


def survival(t, p: Params, r: float | None = None):
    """Probability of NOT being clocked: prod_i (1 - c_i)."""
    return 1.0 - clock_probability(t, p, r)


def dC_dr(t, p: Params, r: float | None = None):
    """Analytic partial of C with respect to preparation r.

    With Pi = prod (1 - c_i):
        dC/dr = -dPi/dr = -Pi * sum_i  c_i0 e_i / (1 - c_i)
    which is strictly negative whenever any efficacy is positive.
    """
    r = p.r if r is None else r
    Pi = survival(t, p, r)
    s = 0.0
    for cue in p.cues:
        s += cue.c0 * cue.efficacy / (1.0 - cue(r))
    return -Pi * s


def dC_dt(t, p: Params, r: float | None = None):
    """Analytic partial of C with respect to time (through the face cue only)."""
    r = p.r if r is None else r
    Pi = survival(t, p, r)
    return Pi * p.face.d_dt(t) / (1.0 - p.face(t))


def boymode_cost(t, p: Params):
    """B(t) = B0 (1 + beta t)."""
    return p.B0 * (1.0 + p.beta * np.asarray(t, dtype=float))


def life_lost_rate(t, p: Params):
    """l(t) = l0 exp(-lam t): the marginal value of authentic time, decreasing."""
    return p.l0 * np.exp(-p.lam * np.asarray(t, dtype=float))


def visible_cost(t, p: Params, r: float | None = None):
    """A C(t, r)."""
    return p.A * clock_probability(t, p, r)


def switching_cost(tau: float, p: Params) -> float:
    """S(tau) = s0 + sum_j height_j * H(tau - at_j).

    Right-continuous: a transition made exactly at a step date pays the step.
    """
    s = p.s0
    for st in p.steps:
        if tau >= st.at:
            s += st.height
    return s


def marginal_gap(t, p: Params, r: float | None = None):
    """g(t) = B(t) + l(t) - A C(t, r).

    The smooth FOC is g(tau*) = 0.  g < 0 means "keep waiting", g > 0
    means "should already have switched".
    """
    return boymode_cost(t, p) + life_lost_rate(t, p) - visible_cost(t, p, r)


def dg_dt(t, p: Params, r: float | None = None):
    """g'(t) = B0 beta - lam l(t) - A dC/dt.  Positive is sufficient for a unique root."""
    return p.B0 * p.beta - p.lam * life_lost_rate(t, p) - p.A * dC_dt(t, p, r)


# Objective and solver


def objective(tau: float, p: Params, r: float | None = None) -> float:
    """J(tau) evaluated by quadrature."""
    r = p.r if r is None else r
    tau = float(np.clip(tau, 0.0, p.T))
    a, _ = quad(lambda t: boymode_cost(t, p) + life_lost_rate(t, p), 0.0, tau)
    b, _ = quad(lambda t: visible_cost(t, p, r), tau, p.T)
    return a + b + switching_cost(tau, p)


def smooth_optimum(p: Params, r: float | None = None) -> float | None:
    """Root of g on [0, T] if one exists, else None.

    Returns 0.0 if g(0) >= 0 (should switch immediately) and None if
    g(T) < 0 (crossing lies beyond the horizon). Genuinely lost part of my mind working on this part.
    """
    r = p.r if r is None else r
    g0 = float(marginal_gap(0.0, p, r))
    gT = float(marginal_gap(p.T, p, r))
    if g0 >= 0.0:
        return 0.0
    if gT < 0.0:
        return None
    return float(brentq(lambda t: marginal_gap(t, p, r), 0.0, p.T, xtol=1e-9))


def optimum(p: Params, r: float | None = None) -> tuple[float | None, float]:
    """Global minimiser of J on [0, T].

    Candidates are the smooth interior root of g (if any), the endpoints,
    and the left limit of every step in S.  Returns (tau_star, J_min).
    tau_star is None when the minimum is to never switch within the horizon.
    """
    r = p.r if r is None else r
    cands: list[float] = [0.0]
    sm = smooth_optimum(p, r)
    if sm is not None:
        cands.append(sm)
    eps = 1e-6
    for st in p.steps:
        if 0.0 < st.at <= p.T:
            cands.append(max(0.0, st.at - eps))
    # "never switch" is J at tau = T with no switching cost
    never = quad(lambda t: boymode_cost(t, p) + life_lost_rate(t, p), 0.0, p.T)[0]

    best_tau: float | None = None
    best_J = never
    for c in cands:
        J = objective(c, p, r)
        if J < best_J - 1e-12:
            best_J, best_tau = J, c
    return best_tau, best_J


# Sensitivity of the smooth optimum


def sensitivity(p: Params, r: float | None = None) -> dict[str, float]:
    """Analytic partials of the smooth tau* via the implicit function theorem.

    From g(tau*, theta) = 0:   d tau*/d theta = -(dg/dtheta) / (dg/dtau).
    Returns an empty dict if no interior smooth optimum exists.
    """
    r = p.r if r is None else r
    tau = smooth_optimum(p, r)
    if tau is None or tau <= 0.0:
        return {}
    gt = float(dg_dt(tau, p, r))
    C = float(clock_probability(tau, p, r))
    Pi = 1.0 - C
    out = {
        "r": -(-p.A * float(dC_dr(tau, p, r))) / gt,
        "beta": -(p.B0 * tau) / gt,
        "B0": -(1.0 + p.beta * tau) / gt,
        "A": -(-C) / gt,
        "l0": -(np.exp(-p.lam * tau)) / gt,
        # bias in perceived c_face propagates through dC/dc_face = Pi / (1 - c_face)
        "c_face": -(-p.A * Pi / (1.0 - float(p.face(tau)))) / gt,
    }
    return out


def per_context_optima(contexts: Iterable[tuple[str, float, tuple[Step, ...]]], p: Params):
    """Solve the separable multi-context problem.

    Each context j has its own hostility A_j and step schedule; the
    boymoding and life-lost terms are shared.  Returns {name: tau_j}.
    """
    out = {}
    for name, A_j, steps_j in contexts:
        pj = p.with_(A=A_j, steps=tuple(steps_j))
        out[name] = optimum(pj)[0]
    return out

