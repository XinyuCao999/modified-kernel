# -*- coding: utf-8 -*-
"""
3 行 × 3 列图（参照论文排版）：
  列 0：LHS 初始采样后的 3D GP 曲面（均值 + 95% CI + 真实曲面）
  列 1：第 1 次 BO 迭代的 2D 采样点
  列 2：第 3 次 BO 迭代的 2D 采样点
行：(1) zero-prior  (2) mech-prior  (3) modified
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D   # noqa
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman"],
    "mathtext.fontset": "stix",
})

from gp_model_base import model_prediction
from BO_base import run_BO
from schotten_baumann_CSTR import (
    initial_sampling_fun, truth_ground, bounds,
    prior_defult, schotten_truth_ground,
)

# ── 参数 ───────────────────────────────────────────────────────────────────────
INITIAL_N  = 5
MAX_ITER   = 5
SHOW_ITERS = [0, 2]       # 第 1、第 3 次 BO 迭代（0-based）
GRID_N     = 35

MODEL_TYPES = ["RBF_zeroprior", "RBF_nonzeroprior", "reconstructed"]
ROW_LABELS  = ["(1) zero-prior", "(2) mech-prior", "(3) modified"]

TRUE_OPT = np.array([1.5, 0.1])
C_PREV   = "#90EE90"
C_NEXT   = "#2E7D32"
C_OPT    = "#8B0000"

FONT_LABEL = 17   # 轴标签
FONT_TICK  = 14   # 刻度
FONT_TITLE = 17   # 列标题
FONT_ROW   = 15   # 行标签


# ── 真实曲面（只算一次）────────────────────────────────────────────────────────
x1g = np.linspace(bounds[0][0], bounds[0][1], GRID_N)
x2g = np.linspace(bounds[1][0], bounds[1][1], GRID_N)
X1, X2  = np.meshgrid(x1g, x2g)
pts_np  = np.stack([X1.ravel(), X2.ravel()], axis=1)

print("Computing true surface ...")
truth_surf = np.array([
    schotten_truth_ground(float(p[0]), float(p[1])) for p in pts_np
]).reshape(X1.shape)


# ── 运行 BO ────────────────────────────────────────────────────────────────────
all_opts = []
for mtype in MODEL_TYPES:
    print(f"Running BO ({mtype}) ...")
    opt = run_BO(
        mtype, INITIAL_N,
        max_iter=MAX_ITER,
        acquisition_function="UCB",
        initial_sampling_fun=initial_sampling_fun,
        bounds=bounds,
        truth_ground=truth_ground,
        beta=2, print_flag=False,
        machanistic_fun=prior_defult,
    )
    all_opts.append(opt)
    print("  Done.")


# ── 3D 绘制 ────────────────────────────────────────────────────────────────────
def draw_3d(ax, opt):
    model_dict = opt["all_model_list"][0]
    X_init = np.array(opt["X"])[:INITIAL_N]
    y_init = np.array(opt["y"])[:INITIAL_N].ravel()

    gp_mean, gp_std = model_prediction(pts_np, model_dict)
    mean_surf = gp_mean.reshape(X1.shape)
    std_surf  = gp_std.reshape(X1.shape)

    # 轴互换：X2=flowrate 作 x 轴（朝前），X1=initial concentration 作 y 轴
    # 真实曲面（深灰）
    ax.plot_surface(X2, X1, truth_surf,
                    color="#444444", alpha=0.85,
                    linewidth=0, antialiased=True, zorder=1)
    # CI 上下界（浅灰半透明）
    ax.plot_surface(X2, X1, mean_surf + 1.96 * std_surf,
                    color="#CCCCCC", alpha=0.30,
                    linewidth=0, antialiased=True, zorder=2)
    ax.plot_surface(X2, X1, mean_surf - 1.96 * std_surf,
                    color="#CCCCCC", alpha=0.30,
                    linewidth=0, antialiased=True, zorder=2)
    # GP 均值（青色）
    ax.plot_surface(X2, X1, mean_surf,
                    color="#29A3A3", alpha=0.72,
                    linewidth=0, antialiased=True, zorder=3)

    ax.set_xlabel("flowrate",               fontsize=FONT_LABEL - 1, labelpad=3)
    ax.set_ylabel("initial\nconcentration", fontsize=FONT_LABEL - 1, labelpad=3)
    ax.set_zlabel("yield",                  fontsize=FONT_LABEL - 1, labelpad=2)
    ax.tick_params(labelsize=FONT_TICK - 2, pad=1)
    ax.view_init(elev=22, azim=-50)

    # 去掉多余的边框感
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#CCCCCC")
    ax.yaxis.pane.set_edgecolor("#CCCCCC")
    ax.zaxis.pane.set_edgecolor("#CCCCCC")


# ── 2D 散点绘制 ────────────────────────────────────────────────────────────────
def draw_scatter(ax, X_all, iter_idx,
                 show_xlabel=True, show_ylabel=True):
    sample_end = INITIAL_N + iter_idx
    prev_pts   = X_all[:sample_end]
    next_pt    = X_all[sample_end]

    for spine in ax.spines.values():
        spine.set_edgecolor("#7B3F8C")
        spine.set_linewidth(1.8)

    ax.set_facecolor("white")
    ax.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)

    ax.scatter(prev_pts[:, 0], prev_pts[:, 1],
               color=C_PREV, s=120, zorder=3, alpha=0.75, linewidths=0)
    ax.scatter(next_pt[0], next_pt[1],
               color=C_NEXT, s=120, zorder=4, linewidths=0)
    ax.scatter(TRUE_OPT[0], TRUE_OPT[1],
               marker="*", color=C_OPT, s=280, zorder=5, linewidths=0)

    ax.set_xlim(bounds[0][0] - 0.02, bounds[0][1] + 0.02)
    ax.set_ylim(bounds[1][0] - 0.3,  bounds[1][1] + 0.3)

    ax.set_xlabel("initial concentration" if show_xlabel else "",
                  fontsize=FONT_LABEL)
    ax.set_ylabel("flowrate" if show_ylabel else "",
                  fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)


# ── 图布局：外层 gs 分左右，左=3D列，右=2列2D ─────────────────────────────────
fig = plt.figure(figsize=(14, 12))
fig.patch.set_facecolor("white")

# 顶层：左列（3D）更宽，右侧两列 2D 收窄
outer = gridspec.GridSpec(
    1, 2,
    width_ratios=[1.7, 1.6],
    left=0.09, right=0.97,
    top=0.91, bottom=0.17,   # 底部留更多空间
    wspace=0.05,
)

# 左侧：3 行 3D 子图
gs_left = gridspec.GridSpecFromSubplotSpec(
    3, 1,
    subplot_spec=outer[0],
    hspace=0.10,
)

# 右侧：3 行 × 2 列 2D 子图
gs_right = gridspec.GridSpecFromSubplotSpec(
    3, 2,
    subplot_spec=outer[1],
    hspace=0.10,
    wspace=0.22,
)

col_titles = ["1$^{st}$ iteration of BO", "3$^{rd}$ iteration of BO"]

for row, (opt, row_label) in enumerate(zip(all_opts, ROW_LABELS)):
    X_all = np.array(opt["X"])

    # ── 3D 子图 ───────────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs_left[row, 0], projection="3d")
    draw_3d(ax3d, opt)

    # 行标签放在 3D 子图左侧
    ax3d.annotate(
        row_label,
        xy=(-0.18, 0.5), xycoords="axes fraction",
        fontsize=FONT_ROW, fontweight="bold",
        va="center", ha="center", rotation=90,
        annotation_clip=False,
    )

    # ── 2D 子图 ───────────────────────────────────────────────────────────────
    for col, iter_idx in enumerate(SHOW_ITERS):
        ax2d = fig.add_subplot(gs_right[row, col])
        draw_scatter(
            ax2d, X_all, iter_idx,
            show_xlabel=(row == 2),
            show_ylabel=(col == 0),
        )
        # 列标题只在第一行显示
        if row == 0:
            ax2d.set_title(col_titles[col],
                           fontsize=FONT_TITLE, fontweight="bold", pad=7)

# ── 列脚注（3D 列正下方居中）─────────────────────────────────────────────────
left_pos = outer[0].get_position(fig)
fig.text(
    left_pos.x0 + left_pos.width / 2, 0.13,
    "The model after\nLatin hypercube sampling",
    ha="center", va="top",
    fontsize=FONT_TITLE - 1, style="italic",
)

# ── 3D 图例（3D 列下方，脚注下面一行）────────────────────────────────────────
legend_3d = [
    mpatches.Patch(color="#444444", alpha=0.85, label="truth ground"),
    mpatches.Patch(color="#29A3A3", alpha=0.72, label="surrogate"),
    mpatches.Patch(color="#CCCCCC", alpha=0.60, label="uncertainty"),
]
fig.legend(
    handles=legend_3d,
    ncol=3, fontsize=12, frameon=False,
    bbox_to_anchor=(left_pos.x0 + left_pos.width / 2, 0.04),
    loc="lower center",
)

# ── 2D 图例（右侧列正下方居中）────────────────────────────────────────────────
right_pos = outer[1].get_position(fig)
legend_2d = [
    mpatches.Patch(color=C_PREV, alpha=0.75, label="previous sampling point"),
    mpatches.Patch(color=C_NEXT,             label="next point"),
    plt.Line2D([0], [0], marker="*", color="w",
               markerfacecolor=C_OPT, markersize=13, label="optimum"),
]
fig.legend(
    handles=legend_2d,
    ncol=3, fontsize=12, frameon=False,
    bbox_to_anchor=(right_pos.x0 + right_pos.width / 2, 0.04),
    loc="lower center",
)

plt.savefig("schotten_iter_all_models.png", dpi=180, bbox_inches="tight")
print("Saved: schotten_iter_all_models.png")
plt.show()
