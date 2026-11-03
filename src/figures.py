"""Report figures, drawn only from the CSV tables written by src/stats.py.

  results/figures/fig1_forgetting_trajectories.{png,pdf}   MMLU and GSM8K forgetting after each stage
  results/figures/fig2_gain_vs_retention.{png,pdf}         domain gain vs MMLU retention, every stage end
  results/figures/fig3_mmlu_confusion.{png,pdf}            gold x predicted letter, Stage 0 vs Stage 3

Usage:  python src/stats.py && python src/figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, load_experiment_config  # noqa: E402

TABLES = RESULTS_DIR / "tables"
FIGS = RESULTS_DIR / "figures"
METHODS = ["full", "lora", "adapter"]
LABEL = {"full": "Full fine-tuning", "lora": "LoRA", "adapter": "Bottleneck adapter", "base": "Stage 0 (base)"}
# fixed categorical order (validated palette, slots 1-3); identity also carried by marker + line style
COLOR = {"full": "#2a78d6", "lora": "#eb6834", "adapter": "#1baf7a"}
MARKER = {"full": "o", "lora": "s", "adapter": "^"}
LINESTYLE = {"full": "-", "lora": "--", "adapter": ":"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
DOMAIN_METRIC = {"medmcqa": "medmcqa", "casehold": "casehold", "code": "mbpp"}
DOMAIN_LABEL = {"medmcqa": "medicine", "casehold": "law", "code": "code"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", FIGS / f"{name}.png")


def fig_trajectories(traj: pd.DataFrame, order: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    ticks = ["Stage 0\n(base)"] + [f"Stage {k}\n({DOMAIN_LABEL[d]})" for k, d in enumerate(order, 1)]
    for ax, metric, title in zip(axes, ["mmlu_primary", "gsm8k"],
                                 ["MMLU (1,260 items, 5-shot)", "GSM8K (500 items, 5-shot, flexible-extract)"]):
        for i, m in enumerate(METHODS):
            t = traj[(traj.metric == metric) & (traj.method == m)].sort_values("stage")
            if len(t) < 2:
                continue
            x = t.stage.to_numpy() + (i - 1) * 0.06
            ax.errorbar(x, t.forgetting_mean, yerr=t.forgetting_sd.fillna(0), color=COLOR[m], marker=MARKER[m],
                        ls=LINESTYLE[m], lw=2, ms=6, capsize=2.5, elinewidth=1, label=LABEL[m])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xticks(range(len(ticks)), ticks, fontsize=7.5)
        ax.set_ylabel("Forgetting F = S0 − S (pp)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.text(0.0, -0.06, "Mean ± SD over 3 seeds; positive = score lost relative to the base model. "
             "Source: results/tables/trajectories.csv", fontsize=7, color=MUTED)
    save(fig, "fig1_forgetting_trajectories")


def fig_gain_vs_retention(traj: pd.DataFrame, order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for m in METHODS:
        xs, ys = [], []
        for k, dom in enumerate(order, 1):
            dm = DOMAIN_METRIC[dom]
            s0 = traj[(traj.metric == dm) & (traj.method == m) & (traj.stage == 0)].score_mean
            sk = traj[(traj.metric == dm) & (traj.method == m) & (traj.stage == k)].score_mean
            rk = traj[(traj.metric == "mmlu_primary") & (traj.method == m) & (traj.stage == k)].retention_mean
            if s0.empty or sk.empty or rk.empty:
                continue
            x, y = float(sk.iloc[0] - s0.iloc[0]), float(rk.iloc[0])
            xs.append(x)
            ys.append(y)
            ax.scatter(x, y, s=55, marker=MARKER[m], color=COLOR[m] if k != 3 else "white",
                       edgecolors=COLOR[m], linewidths=1.8, zorder=3)
            ax.annotate(str(k), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7, color=MUTED)
        if not xs:
            continue
        ax.plot(xs, ys, color=COLOR[m], ls=LINESTYLE[m], lw=1.2, alpha=0.8, label=LABEL[m])
    ax.axhline(100, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Gain on the domain just trained (pp vs Stage 0)\nMedMCQA / CaseHOLD acc., MBPP pass@1")
    ax.set_ylabel("MMLU retention R = S / S0 (%)")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(ax.get_legend_handles_labels()[0]), handlelength=2.5)
    fig.text(0.0, -0.13, "Labels = stage number; seed means. Up-right is better. "
             "Source: results/tables/trajectories.csv", fontsize=7, color=MUTED)
    save(fig, "fig2_gain_vs_retention")


def fig_confusion(conf: pd.DataFrame) -> None:
    panels = ["base", *[m for m in METHODS if (conf.method == m).any()]]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.25 * len(panels), 2.6))
    letters = list("ABCD")
    vmax = 0
    mats = {}
    for p in panels:
        c = conf[conf.method == p]
        M = np.zeros((4, 4))
        for r in c.itertuples():
            M[letters.index(r.gold), letters.index(r.pred)] += r.count
        M = M / M.sum() * 100 if M.sum() else M
        mats[p] = M
        vmax = max(vmax, M.max())
    for ax, p in zip(axes, panels):
        M = mats[p]
        ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.6 * vmax else INK)
        ax.set_xticks(range(4), letters)
        ax.set_yticks(range(4), letters)
        ax.grid(False)
        ax.set_title(LABEL[p] + ("" if p == "base" else ", Stage 3"), fontsize=8, color=INK)
        ax.set_xlabel("Predicted")
        if p == "base":
            ax.set_ylabel("Gold")
    fig.text(0.0, -0.08, "MMLU primary subset, % of all answers (Stage 3 pooled over 3 seeds). "
             "Source: results/tables/mmlu_confusion.csv", fontsize=7, color=MUTED)
    save(fig, "fig3_mmlu_confusion")


def main() -> int:
    cfg = load_experiment_config()
    order = cfg["orders"]["MLC"]
    traj = pd.read_csv(TABLES / "trajectories.csv")
    fig_trajectories(traj, order)
    fig_gain_vs_retention(traj, order)
    conf_path = TABLES / "mmlu_confusion.csv"
    if conf_path.exists():
        fig_confusion(pd.read_csv(conf_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
