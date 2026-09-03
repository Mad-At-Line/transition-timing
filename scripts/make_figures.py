"""Regenerate every figure in figures/.  Run:  python scripts/make_figures.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from transition_timing import plotting as P

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

jobs = {
    "crossing.png": P.fig_crossing,
    "prep_sweep.png": P.fig_prep_sweep,
    "step.png": P.fig_step,
    "life_lost.png": P.fig_life_lost,
    "bias_by_prep.png": P.fig_bias_by_prep,
    "contexts.png": P.fig_contexts,
}
for name, fn in jobs.items():
    fig = fn()
    fig.savefig(OUT / name, bbox_inches="tight")
    fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", OUT / name)
