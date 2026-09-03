import numpy as np
import pytest

from transition_timing import (
    Cue,
    FaceCue,
    Params,
    Step,
    clock_probability,
    dC_dr,
    dC_dt,
    dg_dt,
    marginal_gap,
    objective,
    optimum,
    per_context_optima,
    sensitivity,
    smooth_optimum,
    survival,
)


@pytest.fixture
def p():
    return Params()


# ---------------------------------------------------------------- limits


def test_face_cue_limits(p):
    assert p.face(0.0) == pytest.approx(0.671, abs=1e-3)
    assert p.face(1e6) == pytest.approx(p.face.c_inf, abs=1e-9)
    assert p.face(-1e6) == pytest.approx(p.face.c0, abs=1e-9)


def test_clock_probability_bounds(p):
    ts = np.linspace(0, 60, 601)
    C = clock_probability(ts, p)
    assert np.all(C >= 0) and np.all(C <= 1)
    assert np.all(np.diff(C) <= 1e-12), "C must be non-increasing in t"


def test_full_prep_removes_efficacy_fraction(p):
    for cue in p.cues:
        assert cue(1.0) == pytest.approx(cue.c0 * (1 - cue.efficacy))
        assert cue(0.0) == pytest.approx(cue.c0)


def test_survival_is_complement(p):
    assert survival(7.0, p) == pytest.approx(1 - clock_probability(7.0, p))


# ---------------------------------------------------------------- derivatives


@pytest.mark.parametrize("t", [0.0, 5.0, 16.0, 30.0])
def test_dC_dr_matches_finite_difference(p, t):
    h = 1e-6
    fd = (clock_probability(t, p, p.r + h) - clock_probability(t, p, p.r - h)) / (2 * h)
    assert dC_dr(t, p) == pytest.approx(fd, rel=1e-6)
    assert dC_dr(t, p) < 0


@pytest.mark.parametrize("t", [0.0, 5.0, 16.0, 30.0])
def test_dC_dt_matches_finite_difference(p, t):
    h = 1e-6
    fd = (clock_probability(t + h, p) - clock_probability(t - h, p)) / (2 * h)
    assert dC_dt(t, p) == pytest.approx(fd, rel=1e-5)
    assert dC_dt(t, p) <= 0


def test_dg_dt_matches_finite_difference(p):
    h = 1e-6
    for t in [1.0, 12.0, 25.0]:
        fd = (marginal_gap(t + h, p) - marginal_gap(t - h, p)) / (2 * h)
        assert dg_dt(t, p) == pytest.approx(fd, rel=1e-5)


# ---------------------------------------------------------------- optimum


def test_smooth_optimum_is_root_of_g(p):
    tau = smooth_optimum(p)
    assert tau is not None
    assert marginal_gap(tau, p) == pytest.approx(0.0, abs=1e-8)


def test_uniqueness_condition_holds_at_defaults(p):
    ts = np.linspace(0, p.T, 721)
    assert np.all(dg_dt(ts, p) > 0)


def test_objective_minimised_at_smooth_optimum(p):
    tau = smooth_optimum(p)
    J0 = objective(tau, p)
    for d in [-3.0, -1.0, 1.0, 3.0]:
        assert objective(tau + d, p) > J0


def test_switch_now_when_gap_nonnegative(p):
    assert smooth_optimum(p.with_(B0=1.5)) == 0.0


def test_no_crossing_returns_none(p):
    assert smooth_optimum(p.with_(B0=0.05, beta=0.0)) is None


# ---------------------------------------------------------------- steps


def test_step_flips_optimum_when_tall_enough(p):
    sm = smooth_optimum(p)
    gap = objective(1.0 - 1e-6, p) - objective(sm, p)
    assert gap > 0
    tau_small, _ = optimum(p.with_(steps=(Step(1.0, gap * 0.5),)))
    tau_large, _ = optimum(p.with_(steps=(Step(1.0, gap * 2.0),)))
    assert tau_small == pytest.approx(sm, abs=1e-6)
    assert tau_large == pytest.approx(1.0, abs=1e-5)
    assert tau_large < 1.0, "must switch strictly before the step"


# ---------------------------------------------------------------- sensitivity signs


def test_sensitivity_signs(p):
    s = sensitivity(p)
    assert s["r"] < 0
    assert s["beta"] < 0
    assert s["B0"] < 0
    assert s["l0"] < 0
    assert s["A"] > 0
    assert s["c_face"] > 0


def test_sensitivity_matches_finite_difference(p):
    s = sensitivity(p)
    h = 1e-4
    fd_r = (smooth_optimum(p, p.r + h) - smooth_optimum(p, p.r - h)) / (2 * h)
    fd_A = (smooth_optimum(p.with_(A=p.A + h)) - smooth_optimum(p.with_(A=p.A - h))) / (2 * h)
    assert s["r"] == pytest.approx(fd_r, rel=1e-3)
    assert s["A"] == pytest.approx(fd_A, rel=1e-3)


def test_face_bias_sensitivity_grows_with_prep(p):
    vals = [sensitivity(p.with_(r=r))["c_face"] for r in [0.15, 0.5, 0.8, 1.0]]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_life_lost_term_advances_optimum(p):
    assert smooth_optimum(p.with_(l0=0.2)) < smooth_optimum(p)


# ---------------------------------------------------------------- multi-context


def test_per_context_optima_order(p):
    ctx = [
        ("low", 0.45, (Step(1.0, 1.5),)),
        ("mid", 0.80, ()),
        ("high", 1.20, ()),
    ]
    out = per_context_optima(ctx, p)
    assert out["low"] == 0.0
    assert out["mid"] < out["high"]


# ---------------------------------------------------------------- validation


def test_cue_validation():
    with pytest.raises(ValueError):
        Cue("bad", 1.2, 0.5)
    with pytest.raises(ValueError):
        FaceCue(c0=0.1, c_inf=0.5)
