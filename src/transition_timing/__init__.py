"""transition_timing: an optimal-stopping model for social transition timing."""

from .model import (
    Cue,
    FaceCue,
    Params,
    Step,
    boymode_cost,
    clock_probability,
    dC_dr,
    dC_dt,
    dg_dt,
    life_lost_rate,
    marginal_gap,
    objective,
    optimum,
    per_context_optima,
    sensitivity,
    smooth_optimum,
    survival,
    switching_cost,
    visible_cost,
)

__all__ = [
    "Cue", "FaceCue", "Params", "Step",
    "boymode_cost", "clock_probability", "dC_dr", "dC_dt", "dg_dt",
    "life_lost_rate", "marginal_gap", "objective", "optimum",
    "per_context_optima", "sensitivity", "smooth_optimum", "survival",
    "switching_cost", "visible_cost",
]
__version__ = "0.1.0"
