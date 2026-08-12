#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Severity sweep for the Schotten-Baumann case (Case 3).

We vary the SEVERITY of the (regime-dependent) prior misspecification while
keeping the prior fixed at the intrinsic, well-mixed rate constants [60, 6.5].
A severity factor s in [0, 1] interpolates the TRUE apparent rate constants:

    s = 0 : intrinsic kinetics everywhere   -> truth == prior (zero mismatch)
    s = 1 : full flow-dependent suppression -> the paper's ground truth

    k_true(flow; s) = (1 - s) * k_intrinsic + s * k_full(flow)

(s is kept within [0, 1]; extrapolating past 1 would drive the low-flow rate
constants negative, so the physical severity range is exactly [0, 1].)

For each s we (i) build the corresponding ground-truth function, (ii) measure a
scalar mismatch delta = ||f_prior - f_true|| / ||f_true|| over the design grid,
and (iii) run BO for the requested models and report the final regret
(|reference optimum(s) - best found|), aggregated over repetitions.

Initial designs (input locations) are read from the pre-generated
BO/initial_sample_{initial_point_number}.npz file used by the main experiments,
so the sweep is consistent with them. Only the y-values are recomputed under
each severity's ground truth, so every method starts from the identical design
at every severity level.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import datetime
import plotly.graph_objects as go
from scipy.optimize import minimize

from BO_base import run_BO
from schotten_baumann_CSTR import theta_cal, schotten_CSTR, x0_base, bounds, prior_defult


# ============================ CONFIG ============================
# Edit these freely, just like a main.py.
s_list = [0.02, 0.2, 0.4, 0.6, 0.8, 1.0]      # severity levels to scan (physical range [0, 1])
model_type_list = [ "RBF_zeroprior", "RBF_nonzeroprior", "reconstructed"]
acquisition_function = "EI"                  # "UCB" or "EI"
beta = 2                                      # UCB exploration weight (ignored for EI)
decay_factor = 0.9                            # only affects the reconstructed model
BO_repete_num = 50                            # repetitions per (s, model)
max_iter = 10                                 # BO iterations
initial_point_number = 5                      # initial LHS points (matches the stored file)
grid_delta = 40                               # grid resolution for the delta metric
grid_ref = 60                                 # grid resolution for the reference optimum
# ===============================================================


INTRINSIC = [60.0, 6.5]   # prior = intrinsic, well-mixed rate constants

LABELS = {
    "RBF_zeroprior": "zero-prior",
    "RBF_nonzeroprior": "mech-prior",
    "multi_fidelity": "multi-fidelity",
    "reconstructed": "modified",
}


def make_truth_ground(s):
    """Ground-truth yield function with mass-transfer suppression severity s."""
    def truth_s(X):
        x_equal, y_flowrate = X.squeeze().tolist()
        k_full = theta_cal(y_flowrate)
        theta = [(1.0 - s) * ki + s * kf for ki, kf in zip(INTRINSIC, k_full)]
        resident_time = 0.5 / y_flowrate * 60
        x0 = x0_base.copy()
        x0[0] = x0[0] * x_equal
        x0[3] = x0[3] * x_equal
        output = schotten_CSTR(theta, x0, min(300, resident_time))
        return output / 0.3
    return truth_s


def eval_on_grid(fn, n_grid):
    eq = np.linspace(bounds[0, 0], bounds[0, 1], n_grid)
    fl = np.linspace(bounds[1, 0], bounds[1, 1], n_grid)
    EQ, FL = np.meshgrid(eq, fl)
    pts = np.column_stack([EQ.ravel(), FL.ravel()])
    vals = np.array([fn(p) for p in pts])
    return pts, vals


def compute_delta(truth_fn):
    """Normalized L2 mismatch between the fixed prior and the severity-s truth."""
    pts, truth_vals = eval_on_grid(truth_fn, grid_delta)
    prior_vals = prior_defult(pts).numpy().reshape(-1)
    return float(np.linalg.norm(prior_vals - truth_vals) / np.linalg.norm(truth_vals))


def compute_reference_max(truth_fn):
    """True optimum of the severity-s ground truth: dense grid to locate the
    basin, then continuous refinement so the reference is the true optimum and
    the BO best cannot exceed it (which would zero-out the regret on a log axis)."""
    pts, vals = eval_on_grid(truth_fn, grid_ref)
    x0 = pts[int(np.argmax(vals))]
    res = minimize(lambda x: -float(truth_fn(np.asarray(x, dtype=float))),
                   x0=x0, method="L-BFGS-B",
                   bounds=list(zip(bounds[:, 0], bounds[:, 1])))
    return max(float(np.max(vals)), float(-res.fun))


def sample_y(u_design, truth_fn):
    return np.array([truth_fn(np.array(pt)) for pt in u_design])


def main():
    date_str = datetime.date.today().strftime("%m%d")
    acq_tag = f"UCB_beta{beta}" if acquisition_function == "UCB" else acquisition_function

    # Read the pre-generated initial designs (input locations) used by the main
    # experiments, so the sweep is consistent with them. Only the input locations
    # are reused; the y-values are recomputed per severity below, because each
    # severity has a different ground truth.
    sample_path = os.path.join("BO", f"initial_sample_{initial_point_number}.npz")
    u_all = np.load(sample_path)["u_series"]   # shape [N_stored, initial_point_number, d]
    if BO_repete_num > len(u_all):
        raise ValueError(
            f"BO_repete_num={BO_repete_num} exceeds the {len(u_all)} stored designs "
            f"in {sample_path}."
        )
    designs = [u_all[rep] for rep in range(BO_repete_num)]

    # final regret arrays: [n_s, n_rep] per model (best-found values are kept too,
    # so the reference optimum can be revised later without re-running BO)
    regret = {m: np.zeros((len(s_list), BO_repete_num)) for m in model_type_list}
    best_all = {m: np.zeros((len(s_list), BO_repete_num)) for m in model_type_list}
    delta_list = np.zeros(len(s_list))
    ref_list = np.zeros(len(s_list))

    for si, s in enumerate(s_list):
        truth_s = make_truth_ground(s)
        delta_list[si] = compute_delta(truth_s)
        ref_max = compute_reference_max(truth_s)
        ref_list[si] = ref_max
        y_designs = [sample_y(u, truth_s) for u in designs]
        print(f"[s={s:.2f}] delta={delta_list[si]:.4f}, reference_max={ref_max:.4f}")

        for m in model_type_list:
            for rep in range(BO_repete_num):
                info = run_BO(
                    m, initial_point_number, max_iter, acquisition_function,
                    initial_sampling_fun=None, bounds=bounds, truth_ground=truth_s,
                    initial_sampling=(designs[rep], y_designs[rep]),
                    machanistic_fun=prior_defult, beta=beta,
                    decay_factor=decay_factor, print_flag=False,
                )
                best = float(info["y"].reshape(-1).max())
                best_all[m][si, rep] = best
                # reference is the true continuous optimum, so best <= ref_max and
                # the regret is a clean positive quantity.
                regret[m][si, rep] = max(ref_max - best, 0.0)
            print(f"    {LABELS.get(m, m):14s} mean final regret = {regret[m][si].mean():.5f}")

    # ---- save raw results ----
    save_dict = {
        "s_list": np.array(s_list),
        "delta_list": delta_list,
        "ref_list": ref_list,             # reference optimum per severity
        "regret": regret,                 # dict: model -> [n_s, n_rep]
        "best_all": best_all,             # dict: model -> [n_s, n_rep] best-found values
        "model_type_list": model_type_list,
        "acquisition_function": acquisition_function,
        "beta": beta,
        "decay_factor": decay_factor,
        "BO_repete_num": BO_repete_num,
        "max_iter": max_iter,
        "initial_point": initial_point_number,
    }
    os.makedirs("BO", exist_ok=True)
    out_npy = os.path.join("BO", f"severity_scan_{date_str}_{acq_tag}.npy")
    np.save(out_npy, save_dict)
    print(f"\nSaved raw results to: {out_npy}")

    # ---- plot: final regret vs delta ----
    color_list = ['#4682B4', '#8B4513', '#0fa107', '#7D3C98', '#FF8C00', '#B22222']
    fig = go.Figure()
    for i, m in enumerate(model_type_list):
        mean = regret[m].mean(axis=1)
        std = regret[m].std(axis=1)
        color = color_list[i % len(color_list)]
        fig.add_trace(go.Scatter(
            x=s_list, y=mean, mode='lines+markers',
            name=LABELS.get(m, m), line=dict(color=color), marker=dict(size=8),
            error_y=dict(type='data', array=std, visible=True),
        ))
    fig.update_layout(
        title=f"Case 3 severity sweep (acq: {acq_tag})",
        xaxis_title="Misspecification severity  s  (0 = prior correct, 1 = full mismatch)",
        yaxis_title="Final regret  |reference optimum - best found|",
        yaxis_type="log", template="plotly_white",
        width=800, height=600,
    )
    out_html = os.path.join("BO", f"severity_scan_{date_str}_{acq_tag}.html")
    fig.write_html(out_html)
    fig.show()
    print(f"Saved plot to: {out_html}")


if __name__ == "__main__":
    main()
