#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined severity-sweep figure: two horizontal panels (EI | UCB), styled to
match the convergence plots (Times New Roman, bold titles, error bars, grid).

Near-zero means (a method reaching the optimum to solver precision) are floored
to `floor` so the log axis can render them; such points read as "optimum
reached", not a precise regret. Writes one HTML + one high-res PNG.
"""

import os
import glob
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================ CONFIG ============================
file_ei = None          # None -> newest BO/severity_scan_*EI*.npy
file_ucb = None         # None -> newest BO/severity_scan_*UCB*.npy
out_base = "BO/severity_scan_combined"
floor = 1e-4            # log-axis floor for zero/near-zero values
use_sem = True          # True -> std/sqrt(N) (matches convergence plots)
png_scale = 4
# ===============================================================

LABELS = {
    "RBF_zeroprior": "zero-prior",
    "RBF_nonzeroprior": "mech-prior",
    "multi_fidelity": "multi-fidelity",
    "reconstructed": "modified",
}
COLOR_BY_TYPE = {
    "RBF_zeroprior": "#4682B4",
    "RBF_nonzeroprior": "#8B4513",
    "multi_fidelity": "#C9A227",
    "reconstructed": "#0fa107",
}
FALLBACK = ['#7D3C98', '#6A5ACD', '#008B8B', '#B22222']


def pick(pattern):
    c = sorted(glob.glob(pattern))
    return c[-1] if c else None


def add_panel(fig, path, col, show_legend):
    d = np.load(path, allow_pickle=True).item()
    s_list = np.asarray(d["s_list"], dtype=float)
    regret = d["regret"]
    model_list = d.get("model_type_list", list(regret.keys()))
    for i, m in enumerate(model_list):
        arr = np.asarray(regret[m], dtype=float)
        mean = arr.mean(axis=1)
        spread = arr.std(axis=1)
        if use_sem:
            spread = spread / np.sqrt(arr.shape[1])
        mean_plot = np.maximum(mean, floor)
        upper = mean + spread
        lower = np.maximum(mean - spread, floor)
        err_plus = np.maximum(upper - mean_plot, 0.0)
        err_minus = np.maximum(mean_plot - lower, 0.0)
        # points that reached the optimum sit at the floor -> no error bar
        at_floor = mean <= floor
        err_plus = np.where(at_floor, 0.0, err_plus)
        err_minus = np.where(at_floor, 0.0, err_minus)
        color = COLOR_BY_TYPE.get(m, FALLBACK[i % len(FALLBACK)])
        fig.add_trace(go.Scatter(
            x=s_list, y=mean_plot, mode="lines+markers",
            name=LABELS.get(m, m), legendgroup=m, showlegend=show_legend,
            line=dict(color=color), marker=dict(size=10), connectgaps=True,
            error_y=dict(type="data", symmetric=False,
                         array=err_plus, arrayminus=err_minus,
                         visible=True, color=color),
        ), row=1, col=col)


def main():
    ei = file_ei or pick("BO/severity_scan_*EI*.npy")
    ucb = file_ucb or pick("BO/severity_scan_*UCB*.npy")
    if ei is None or ucb is None:
        raise FileNotFoundError("Need one EI and one UCB severity_scan_*.npy in BO/.")

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("<b>UCB</b>", "<b>EI</b>"))
    add_panel(fig, ucb, col=1, show_legend=True)
    add_panel(fig, ei, col=2, show_legend=False)

    # y-axis: log, horizontal gridlines only at the decades (0.001, 0.01, 0.1, ...)
    fig.update_yaxes(type="log", dtick=1, showgrid=True, gridcolor="lightgray",
                     tickfont=dict(size=24), row=1, col=1)
    fig.update_yaxes(type="log", dtick=1, showgrid=True, gridcolor="lightgray", row=1, col=2)
    fig.update_yaxes(title=dict(text="<b>Final regret</b>", font=dict(size=30)), row=1, col=1)
    for c in (1, 2):
        fig.update_xaxes(title=dict(text="<b>Misspecification severity  s</b>", font=dict(size=30)),
                         showgrid=True, gridcolor="lightgray", tickfont=dict(size=24), row=1, col=c)

    fig.update_layout(
        font=dict(family="Times New Roman", size=24),
        plot_bgcolor="white", template="plotly_white",
        # legend in the empty top-left area so it does not cover any curve
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top", font=dict(size=26)),
        width=1300, height=620, showlegend=True,
    )
    for ann in fig.layout.annotations:            # subplot titles
        ann.font = dict(size=30, family="Times New Roman")

    out_html, out_png = out_base + ".html", out_base + ".png"
    fig.write_html(out_html)
    try:
        fig.write_image(out_png, scale=png_scale)
    except Exception as e:
        print(f"[png export skipped: {e}]")
    fig.show()
    print(f"EI  : {ei}")
    print(f"UCB : {ucb}")
    print(f"Saved: {out_html}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
