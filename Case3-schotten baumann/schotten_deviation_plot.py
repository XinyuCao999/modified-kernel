#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the deviation between truth ground and the mechanistic (prior) model
for the Schotten-Baumann reaction.

Truth ground  : uses theta_cal(flowrate) — flow-rate-dependent k1, k2
Mechanistic   : uses fixed theta = [60, 6.5]  (prior_schotten default)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from schotten_baumann_CSTR import schotten_truth_ground, prior_schotten, bounds


# ------------------------------------------------------------------
# Build a grid over the input space
# ------------------------------------------------------------------
N = 50
x_equal_vals  = np.linspace(bounds[0, 0], bounds[0, 1], N)   # [1.0, 1.5]
y_flowrate_vals = np.linspace(bounds[1, 0], bounds[1, 1], N)  # [0.1, 6.0]

XX, YY = np.meshgrid(x_equal_vals, y_flowrate_vals)           # (N, N)

Z_truth = np.zeros_like(XX)
Z_prior = np.zeros_like(XX)

print("Computing truth ground and mechanistic model on grid ...")
for i in range(N):
    for j in range(N):
        xe = XX[i, j]
        yf = YY[i, j]
        Z_truth[i, j] = schotten_truth_ground(xe, yf)
        Z_prior[i, j] = prior_schotten(xe, yf)

Z_dev = Z_truth - Z_prior   # positive → truth > prior

print(f"Deviation range: [{Z_dev.min():.4f}, {Z_dev.max():.4f}]")


# ------------------------------------------------------------------
# Color palette (matches reference figure style)
# ------------------------------------------------------------------
# Truth ground  → solid black   (like "True Function" in reference)
# Prior model   → semi-transparent muted teal  (like "prior" in reference)
# Deviation     → diverging: dark charcoal (negative) → light gray (zero) → teal (positive)

CS_BLACK = [[0, "rgb(15,15,15)"],  [1, "rgb(15,15,15)"]]
CS_TEAL  = [[0, "rgb(72,140,140)"], [1, "rgb(72,140,140)"]]
CS_DIV   = [
    [0.0,  "rgb(30, 30, 30)"],     # most negative → near black
    [0.25, "rgb(90, 105, 105)"],   # negative       → dark slate
    [0.5,  "rgb(210, 215, 215)"],  # zero           → light gray
    [0.75, "rgb(80, 160, 155)"],   # positive       → muted teal
    [1.0,  "rgb(30, 110, 105)"],   # most positive  → deep teal
]

scene_cfg = dict(
    xaxis=dict(title="initial concentration", backgroundcolor="rgb(230,235,240)",
               gridcolor="white", showbackground=True),
    yaxis=dict(title="flowrate", backgroundcolor="rgb(230,235,240)",
               gridcolor="white", showbackground=True),
    zaxis=dict(title="yield", backgroundcolor="rgb(215,222,228)",
               gridcolor="white", showbackground=True),
    bgcolor="rgb(240,244,248)",
)


# ------------------------------------------------------------------
# Plot – side-by-side: truth ground / mechanistic model / deviation
# ------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
    subplot_titles=["Truth Ground", "Mechanistic Model (Prior)", "Deviation (Truth − Prior)"],
    horizontal_spacing=0.02,
)

fig.add_trace(
    go.Surface(x=XX, y=YY, z=Z_truth,
               colorscale=CS_BLACK, showscale=False,
               opacity=0.92, name="Truth Ground",
               lighting=dict(diffuse=0.6, specular=0.3, roughness=0.6)),
    row=1, col=1,
)

fig.add_trace(
    go.Surface(x=XX, y=YY, z=Z_prior,
               colorscale=CS_TEAL, showscale=False,
               opacity=0.75, name="Mechanistic Model",
               lighting=dict(diffuse=0.6, specular=0.2, roughness=0.7)),
    row=1, col=2,
)

fig.add_trace(
    go.Surface(x=XX, y=YY, z=Z_dev,
               colorscale=CS_DIV, showscale=True,
               cmid=0, opacity=0.88, name="Deviation",
               colorbar=dict(
                   x=1.01, len=0.7, thickness=14,
                   title=dict(text="Δ yield", side="right"),
                   tickfont=dict(size=11),
               ),
               lighting=dict(diffuse=0.6, specular=0.2, roughness=0.6)),
    row=1, col=3,
)

for i in range(1, 4):
    sk = "scene" if i == 1 else f"scene{i}"
    fig.update_layout(**{sk: scene_cfg})

fig.update_layout(
    title=dict(
        text="Schotten-Baumann: Truth Ground vs Mechanistic Model",
        font=dict(size=16, color="rgb(50,50,50)"),
        x=0.5,
    ),
    paper_bgcolor="rgb(248,250,252)",
    width=1600,
    height=680,
    font=dict(family="Arial", color="rgb(60,60,60)"),
)

fig.show()
fig.write_html(os.path.join(os.path.dirname(__file__), "schotten_deviation.html"))
print("Saved: schotten_deviation.html")
