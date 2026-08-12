#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic: how reliably does the model's discrepancy score track the actual
prior error as data accumulate, for the Schotten-Baumann case (Case 3)?

At every point x in the domain we compare:

    discrepancy score (model) = | GP posterior mean(x) - prior(x) |
        -> what the reconstructed kernel uses to flag where the prior and the
           data disagree (a data-driven quantity; the truth is never used).

    actual prior error (truth) = | prior(x) - ground_truth(x) |
        -> how wrong the prior really is.

For each data size n we train the reconstructed model on n observations and
report the Spearman and Pearson correlation between the two quantities over a
grid. To match the BO budget (5 initial + up to 10 iterations) we scan
n = 5..15 on a single NESTED design (n+1 = n points plus one more, fixed seed),
so the curve reflects "adding data" rather than unrelated random draws.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import scipy.stats.qmc as qmc
from scipy.stats import spearmanr, pearsonr
import plotly.graph_objects as go

from gp_model_base import train_reconstructed_gp
from schotten_baumann_CSTR import truth_ground, bounds, prior_defult


# ============================ CONFIG ============================
n_list = list(range(5, 16))    # data sizes to scan: 5,6,...,15
seed = 0                       # fixed seed for the single nested design
n_max = max(n_list)            # points drawn (nested prefixes)
grid = 40                      # grid resolution per axis for the diagnostic
# ===============================================================

bnp = np.asarray(bounds, dtype=float)
D = bnp.shape[0]


def build_grid():
    eq = np.linspace(bnp[0, 0], bnp[0, 1], grid)
    fl = np.linspace(bnp[1, 0], bnp[1, 1], grid)
    EQ, FL = np.meshgrid(eq, fl)
    return np.column_stack([EQ.ravel(), FL.ravel()])


PTS = build_grid()
PTS_T = torch.tensor(PTS, dtype=torch.float32)
PRIOR_G = prior_defult(PTS_T).numpy().reshape(-1)
TRUTH_G = np.array([truth_ground(torch.tensor(p, dtype=torch.float32)) for p in PTS])
ACTUAL_ERROR = np.abs(PRIOR_G - TRUTH_G)          # fixed: | prior - truth |


def correlations_for(X, y):
    """Train reconstructed model on (X, y); return (spearman, pearson) between
    the model discrepancy score and the actual prior error over the grid."""
    _, _, rbf2, _ = train_reconstructed_gp(X, y, False, prior_defult,
                                           return_all_flag=True)
    with torch.no_grad():
        gp_pred = rbf2.posterior(PTS_T).mean.numpy().reshape(-1)
    discrepancy = np.abs(gp_pred - PRIOR_G)       # | GP - prior |
    rs, _ = spearmanr(ACTUAL_ERROR, discrepancy)
    rp, _ = pearsonr(ACTUAL_ERROR, discrepancy)
    return rs, rp


def main():
    # one reproducible nested design of n_max points
    lhs = qmc.LatinHypercube(d=D, seed=seed)
    Xfull = qmc.scale(lhs.random(n=n_max), bnp[:, 0], bnp[:, 1])
    yfull = np.array([truth_ground(torch.tensor(p, dtype=torch.float32)) for p in Xfull])

    sp = np.zeros(len(n_list))
    pe = np.zeros(len(n_list))
    for j, n in enumerate(n_list):
        sp[j], pe[j] = correlations_for(Xfull[:n], yfull[:n])

    print("Discrepancy-vs-error correlation (single nested design, Case 3):")
    print(f"{'n':>3} | {'Spearman':>9} | {'Pearson':>8}")
    for j, n in enumerate(n_list):
        print(f"{n:>3} | {sp[j]:>9.3f} | {pe[j]:>8.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_list, y=sp, mode="lines+markers",
                             name="Spearman", line=dict(color="#0fa107", width=3),
                             marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=n_list, y=pe, mode="lines+markers",
                             name="Pearson", line=dict(color="#8B4513", width=3),
                             marker=dict(size=10)))
    fig.update_layout(
        font=dict(family="Times New Roman", size=24),
        title=dict(text="<b>Discrepancy-error correlation vs data size</b>",
                   x=0.5, xanchor="center", font=dict(size=30)),
        plot_bgcolor="white", template="plotly_white",
        legend=dict(x=0.98, y=0.05, xanchor="right", yanchor="bottom", font=dict(size=30)),
        width=820, height=560, showlegend=True,
        xaxis=dict(
            title=dict(text="<b>Number of observations  n</b>", font=dict(size=30)),
            showgrid=True, gridcolor="lightgray", tickfont=dict(size=24),
        ),
        yaxis=dict(
            title=dict(text="<b>Correlation with prior error</b>", font=dict(size=30)),
            range=[0, 1.02], showgrid=True, gridcolor="lightgray", tickfont=dict(size=24),
        ),
    )
    os.makedirs("BO", exist_ok=True)
    out_html = "BO/discrepancy_correlation_vs_n.html"
    out_png = "BO/discrepancy_correlation_vs_n.png"
    fig.write_html(out_html)
    try:
        fig.write_image(out_png, scale=4)
    except Exception as e:
        print(f"[png export skipped: {e}]")
    fig.show()
    print(f"\nSaved: {out_html}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
