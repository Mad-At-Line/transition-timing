"""Publication figures for the model.  All functions return a matplotlib Figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .model import (
    Params,
    Step,
    boymode_cost,
    clock_probability,
    life_lost_rate,
    objective,
    optimum,
    sensitivity,
    smooth_optimum,
    visible_cost,
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#898781"
TEAL = "#1baf7a"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "axes.titleweight": "normal",
        "legend.frameon": False,
    }
)


def _grid(p: Params, n: int = 721) -> np.ndarray:
    return np.linspace(0.0, p.T, n)


def fig_crossing(p: Params | None = None) -> plt.Figure:
    """Marginal cost curves and the smooth optimum."""
    p = p or Params()
    ts = _grid(p)
    B = boymode_cost(ts, p) + life_lost_rate(ts, p)
    V = visible_cost(ts, p)
    tau = smooth_optimum(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, B, color=BLUE, lw=2, label=r"$B(t)+\ell(t)$  boymoding")
    ax.plot(ts, V, color=ORANGE, lw=2, ls="--", label=r"$A\,C(t,r)$  visibly trans")
    if tau is not None:
        ax.axvline(tau, color=GREY, lw=1, ls=":")
        ax.scatter([tau], [float(boymode_cost(tau, p) + life_lost_rate(tau, p))], color="k", zorder=5, s=30)
        ax.annotate(rf"$\tau^*\approx{tau:.1f}$ mo", (tau, ax.get_ylim()[1] * 0.95),
                    xytext=(6, 0), textcoords="offset points", va="top", fontsize=9)
    ax.set_xlabel("months on HRT")
    ax.set_ylabel("marginal cost (arbitrary units)")
    ax.set_title(f"Crossing at defaults  (r={p.r}, β={p.beta}, A={p.A}, B₀={p.B0})")
    ax.set_xlim(0, p.T)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_prep_sweep(p: Params | None = None, rs=(0.0, 0.15, 0.4, 0.6, 0.8, 1.0)) -> plt.Figure:
    """How the visible-trans curve, and hence tau*, moves with preparation r."""
    p = p or Params()
    ts = _grid(p)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.4, 1]})

    cmap = plt.get_cmap("Oranges")
    B = boymode_cost(ts, p) + life_lost_rate(ts, p)
    ax1.plot(ts, B, color=BLUE, lw=2, label="boymoding")
    taus = []
    for i, r in enumerate(rs):
        V = visible_cost(ts, p, r)
        ax1.plot(ts, V, color=cmap(0.35 + 0.6 * i / (len(rs) - 1)), lw=1.6, ls="--", label=f"r = {r:.2f}")
        taus.append(smooth_optimum(p, r))
    ax1.set_xlabel("months on HRT")
    ax1.set_ylabel("marginal cost")
    ax1.set_title("Visible-trans cost falls with preparation r")
    ax1.set_xlim(0, p.T)
    ax1.set_ylim(0, None)
    ax1.legend(fontsize=8, ncol=2)

    r_fine = np.linspace(0, 1, 101)
    tau_fine = [smooth_optimum(p, r) for r in r_fine]
    tau_fine = [np.nan if t is None else t for t in tau_fine]
    ax2.plot(r_fine, tau_fine, color="k", lw=2)
    ax2.scatter(rs, [np.nan if t is None else t for t in taus], color=ORANGE, zorder=5, s=25)
    ax2.set_xlabel("preparation fraction r")
    ax2.set_ylabel(r"$\tau^*$ (months)")
    ax2.set_title(r"$\partial\tau^*/\partial r<0$, and steepening")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, None)
    fig.tight_layout()
    return fig


def fig_step(p: Params | None = None, step_at: float = 1.0) -> plt.Figure:
    """J(tau) with a step in S at a context-formation date."""
    p = p or Params()
    sm = smooth_optimum(p)
    gap = objective(step_at - 1e-6, p) - objective(sm, p)
    heights = [0.0, gap * 0.5, gap * 1.0, gap * 2.0]
    taus = np.linspace(0, p.T, 361)

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("Purples")
    for i, h in enumerate(heights):
        ph = p.with_(steps=(Step(step_at, h, "context forms"),))
        J = [objective(t, ph) for t in taus]
        tau_star, _ = optimum(ph)
        ax.plot(taus, J, color=cmap(0.4 + 0.55 * i / (len(heights) - 1)), lw=1.8,
                label=rf"$\Delta={h:.2f}$  →  $\tau^*={tau_star:.1f}$")
    ax.axvline(step_at, color=GREY, lw=1, ls=":")
    ax.text(step_at + 0.3, ax.get_ylim()[1] * 0.98, "context\nforms", fontsize=8, va="top", color=GREY)
    ax.set_xlabel(r"switching time $\tau$ (months)")
    ax.set_ylabel(r"total cost $J(\tau)$")
    ax.set_title(rf"A step of height $\Delta>{gap:.2f}$ at month {step_at} moves the optimum to the step")
    ax.set_xlim(0, p.T)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_life_lost(p: Params | None = None, l0s=(0.0, 0.1, 0.2, 0.3)) -> plt.Figure:
    """Effect of the irreversible life-lost term."""
    p = p or Params()
    ts = _grid(p)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, visible_cost(ts, p), color=ORANGE, lw=2, ls="--", label=r"$A\,C(t)$")
    cmap = plt.get_cmap("Blues")
    for i, l0 in enumerate(l0s):
        pl = p.with_(l0=l0)
        ax.plot(ts, boymode_cost(ts, pl) + life_lost_rate(ts, pl),
                color=cmap(0.4 + 0.55 * i / (len(l0s) - 1)), lw=1.8,
                label=rf"$\ell_0={l0}$ → $\tau^*={smooth_optimum(pl):.1f}$")
    ax.set_xlabel("months on HRT")
    ax.set_ylabel("marginal cost")
    ax.set_title("Weighting early time (irreversibility) collapses the optimum")
    ax.set_xlim(0, p.T)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_bias_by_prep(p: Params | None = None) -> plt.Figure:
    """Sensitivity of tau* to face self-perception, as a function of prep r."""
    p = p or Params()
    rs = np.linspace(0.0, 1.0, 51)
    s_face, s_r = [], []
    for r in rs:
        s = sensitivity(p.with_(r=r))
        s_face.append(s.get("c_face", np.nan))
        s_r.append(s.get("r", np.nan))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rs, s_face, color=TEAL, lw=2, label=r"$\partial\tau^*/\partial c_{\rm face}$  (months per unit)")
    ax.plot(rs, -np.asarray(s_r), color=BLUE, lw=2, ls="--", label=r"$-\,\partial\tau^*/\partial r$  (months per unit)")
    ax.set_xlabel("preparation fraction r")
    ax.set_ylabel("sensitivity")
    ax.set_title("Self-perception bias matters more once the other cues are handled")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_contexts(p: Params | None = None) -> plt.Figure:
    """Per-context optima: tau is a vector."""
    p = p or Params()
    ctx = [
        ("New context\n(low A, step at Freshers)", 0.45, (Step(1.0, 1.5),)),
        ("Dublin, general\n(mid A)", 0.80, ()),
        ("Home town\n(high A)", 1.20, ()),
    ]
    names, taus = [], []
    for name, A, steps in ctx:
        names.append(name)
        t, _ = optimum(p.with_(A=A, steps=steps))
        taus.append(p.T if t is None else t)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(names, taus, color=[TEAL, BLUE, ORANGE], height=0.5)
    for y, t in enumerate(taus):
        ax.text(t + 0.4, y, "now" if t < 0.05 else f"{t:.1f} mo", va="center", fontsize=9)
    ax.set_xlabel(r"$\tau_j^*$ (months on HRT)")
    ax.set_xlim(0, p.T + 4)
    ax.set_title(r"$\tau^*$ is a vector: one optimum per context")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
