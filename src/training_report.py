"""Tables and figures from the training logs (no test data is read here).

Inputs: results/raw/<run_id>/{run_meta.json, run_summary.json, dev_metrics.jsonl, train_log.jsonl},
        results/lr_search/*/dev_metrics.jsonl, data/manifest.json
Outputs (results/tables/):
  training_efficiency.csv    run x stage: steps, tokens, training time, throughput, peak GPU memory
  efficiency_summary.csv     method: trainable parameters, time and memory (mean +- SD over seeds)
  dev_trajectories.csv       method x checkpoint: development metrics, mean +- SD over seeds
  lr_search.csv              method x learning rate: MedMCQA-dev after Stage 1 (selected marked)
  data_pipeline.csv          domain: pool size, removed duplicates / overlaps, final train and dev size
Outputs (results/figures/):
  fig4_dev_trajectories      development metrics at all 13 checkpoints (learning and dev-level forgetting)
  fig5_training_loss         training loss across the three stages
  fig6_lr_sensitivity        MedMCQA-dev after Stage 1 for each learning rate in the grid

Usage: python src/training_report.py [--order MLC] [--model q25-05b]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, read_json, read_jsonl  # noqa: E402

TABLES = RESULTS_DIR / "tables"
FIGS = RESULTS_DIR / "figures"
METHODS = ["full", "lora", "adapter"]
LABEL = {"full": "Full fine-tuning", "lora": "LoRA", "adapter": "Bottleneck adapter"}
COLOR = {"full": "#2a78d6", "lora": "#eb6834", "adapter": "#1baf7a"}  # same palette as figures.py
MARKER = {"full": "o", "lora": "s", "adapter": "^"}
LINESTYLE = {"full": "-", "lora": "--", "adapter": ":"}
INK, MUTED, GRID, BAND = "#1f1f1e", "#6b6a64", "#e4e3de", "#f3f2ee"
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


def write(df: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False, float_format="%.6g")
    print("wrote", TABLES / name)


def runs(model: str, order: str) -> list[Path]:
    out = sorted((RESULTS_DIR / "raw").glob(f"{model}_*_{order}_s*"))
    if not out:
        raise SystemExit(f"no runs matching {model}_*_{order}_s* in results/raw")
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def efficiency(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    params = {}
    for d in run_dirs:
        meta, summ = read_json(d / "run_meta.json"), read_json(d / "run_summary.json")
        params[meta["method"]] = meta["params"]
        for st in summ["stages"]:
            rows.append({
                "run_id": meta["run_id"], "method": meta["method"], "seed": meta["seed"],
                "stage": st["stage"], "domain": st["domain"], "steps": st["max_steps"],
                "tokens": st["tokens"], "response_tokens": st["response_tokens"],
                "train_min": st["train_seconds"] / 60, "dev_eval_min": st["dev_eval_seconds"] / 60,
                "sec_per_step": st["train_seconds"] / st["max_steps"],
                "tokens_per_sec": st["tokens"] * 2 / st["train_seconds"],  # 2 epochs per stage
                "peak_gpu_gib": st["peak_gpu_memory_gib"], "final_train_loss": st["final_train_loss"],
            })
    per_stage = pd.DataFrame(rows).sort_values(["method", "seed", "stage"])
    per_run = per_stage.groupby(["method", "seed"]).agg(train_min=("train_min", "sum"),
                                                       sec_per_step=("sec_per_step", "mean"),
                                                       peak_gpu_gib=("peak_gpu_gib", "max")).reset_index()
    summ = []
    for m in METHODS:
        r = per_run[per_run.method == m]
        if r.empty:
            continue
        p = params[m]
        summ.append({
            "method": m, "trainable_params": p["trainable"], "trainable_pct": p["trainable_pct"],
            "n_runs": len(r),
            "train_min_mean": r.train_min.mean(), "train_min_sd": r.train_min.std(ddof=1),
            "sec_per_step_mean": r.sec_per_step.mean(), "sec_per_step_sd": r.sec_per_step.std(ddof=1),
            "peak_gpu_gib": r.peak_gpu_gib.max(),
        })
    s = pd.DataFrame(summ)
    full = s.set_index("method").loc["full"]
    s["time_vs_full"] = s.train_min_mean / full.train_min_mean
    s["memory_vs_full"] = s.peak_gpu_gib / full.peak_gpu_gib
    return per_stage, s


def dev_trajectories(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        meta = read_json(d / "run_meta.json")
        for i, r in enumerate(read_jsonl(d / "dev_metrics.jsonl")):
            rows.append({"method": meta["method"], "seed": meta["seed"], "index": i, "checkpoint": r["checkpoint"],
                         "stage": r["stage"], "domain": r["domain"], "fraction": r["fraction"],
                         "medmcqa_dev_acc": r["medmcqa_dev_acc"], "casehold_dev_acc": r["casehold_dev_acc"],
                         "code_dev_loss": r["code_dev_loss"], "mbpp_dev_pass_at_1": r["mbpp_dev_pass_at_1"]})
    df = pd.DataFrame(rows)
    metrics = ["medmcqa_dev_acc", "casehold_dev_acc", "code_dev_loss", "mbpp_dev_pass_at_1"]
    g = df.groupby(["method", "index", "checkpoint", "stage", "domain", "fraction"], dropna=False)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b.replace('std', 'sd')}" for a, b in out.columns]
    out = out.reset_index()
    out["n_seeds"] = g.size().to_numpy()
    out["method"] = pd.Categorical(out.method, METHODS, ordered=True)
    return out.sort_values(["method", "index"]).reset_index(drop=True)


def lr_table() -> pd.DataFrame:
    rows = []
    for d in sorted((RESULTS_DIR / "lr_search").glob("lrsearch_*")):
        meta = read_json(d / "run_meta.json")
        final = [r for r in read_jsonl(d / "dev_metrics.jsonl") if r["stage"] == 1][-1]
        rows.append({"method": meta["method"], "lr": meta["lr"], "medmcqa_dev_acc": final["medmcqa_dev_acc"],
                     "casehold_dev_acc": final["casehold_dev_acc"], "code_dev_loss": final["code_dev_loss"]})
    df = pd.DataFrame(rows)
    sel = read_json(RESULTS_DIR.parent / "configs" / "lr_selected.json")
    chosen = sel[sel["_applies_to_all_models_from"]]
    df["selected"] = [abs(r.lr - chosen[r.method]) < 1e-12 for r in df.itertuples()]
    df["method"] = pd.Categorical(df.method, METHODS, ordered=True)
    return df.sort_values(["method", "lr"]).reset_index(drop=True)


def data_table() -> pd.DataFrame:
    st = read_json(DATA_DIR / "manifest.json")["stats"]["domains"]
    rows = []
    for dom, d in st.items():
        removed = d["removed_duplicates"] + d["removed_test_overlap"] + d["removed_dev_overlap"]
        rows.append({"domain": dom, "pool_raw": d["pool_raw"], "removed_duplicates": d["removed_duplicates"],
                     "removed_test_overlap": d["removed_test_overlap"],
                     "removed_dev_overlap": d["removed_dev_overlap"], "removed_pct": 100 * removed / d["pool_raw"],
                     "pool_clean": d["pool_clean"], "train": d["train"], "dev": d["dev"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def shade_stages(ax, bounds: list[float]) -> None:
    for k in range(1, len(bounds) - 1, 2):
        ax.axvspan(bounds[k], bounds[k + 1], color=BAND, zorder=0, lw=0)


def fig_dev(traj: pd.DataFrame, order: list[str]) -> None:
    panels = [("medmcqa_dev_acc", "MedMCQA-dev accuracy (medicine)"),
              ("casehold_dev_acc", "CaseHOLD-dev accuracy (law)"),
              ("code_dev_loss", "Code-dev loss (lower = better)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharex=True)
    n = int(traj["index"].max()) + 1
    bounds = [0.5 + 4 * k for k in range(len(order) + 1)]
    for ax, (metric, title) in zip(axes, panels):
        shade_stages(ax, bounds)
        for m in METHODS:
            t = traj[traj.method == m]
            if t.empty:
                continue
            x, y, sd = t["index"].to_numpy(), t[f"{metric}_mean"].to_numpy(), t[f"{metric}_sd"].fillna(0).to_numpy()
            ax.fill_between(x, y - sd, y + sd, color=COLOR[m], alpha=0.15, lw=0)
            ax.plot(x, y, color=COLOR[m], marker=MARKER[m], ls=LINESTYLE[m], lw=1.6, ms=3.5, label=LABEL[m])
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xticks([0] + [2.5 + 4 * k for k in range(len(order))],
                      ["0"] + [f"S{k}\n{DOMAIN_LABEL[d]}" for k, d in enumerate(order, 1)], fontsize=7.5)
        ax.set_xlim(-0.5, n - 0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=7.5)
    fig.text(0.0, -0.08, "Development data only, 200 items per domain; checkpoints at 25/50/75/100% of each stage. "
             "Line = mean over 3 seeds, band = ±1 SD. Source: results/tables/dev_trajectories.csv",
             fontsize=6.8, color=MUTED)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_dev_trajectories")


def fig_loss(run_dirs: list[Path], order: list[str]) -> None:
    rows = []
    for d in run_dirs:
        meta = read_json(d / "run_meta.json")
        steps_per_stage = {s["stage"]: s["max_steps"] for s in read_json(d / "run_summary.json")["stages"]}
        offset = {k: sum(steps_per_stage[j] for j in range(1, k)) for k in steps_per_stage}
        for r in read_jsonl(d / "train_log.jsonl"):
            if "loss" in r:
                rows.append({"method": meta["method"], "seed": meta["seed"], "stage": r["stage"],
                             "x": offset[r["stage"]] + r["step"], "loss": r["loss"]})
    df = pd.DataFrame(rows)
    g = df.groupby(["method", "stage", "x"]).loss.agg(["mean", "std"]).reset_index()
    total = int(df.x.max())
    stage_len = total / len(order)
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    bounds = [k * stage_len for k in range(len(order) + 1)]
    shade_stages(ax, bounds)
    for m in METHODS:
        for k in range(1, len(order) + 1):
            t = g[(g.method == m) & (g.stage == k)]
            if t.empty:
                continue
            ax.fill_between(t.x, t["mean"] - t["std"].fillna(0), t["mean"] + t["std"].fillna(0),
                            color=COLOR[m], alpha=0.15, lw=0)
            ax.plot(t.x, t["mean"], color=COLOR[m], ls=LINESTYLE[m], lw=1.6, label=LABEL[m] if k == 1 else None)
    for k, d in enumerate(order, 1):
        ax.text(bounds[k - 1] + stage_len / 2, 1.0, f"Stage {k}: {DOMAIN_LABEL[d]}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(0, total)
    ax.set_xlabel("Optimiser step (cumulative over stages)")
    ax.set_ylabel("Training loss (response tokens)")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.text(0.0, -0.1, "Mean over 3 seeds, band = ±1 SD; logged every 10 steps. "
             "Source: results/raw/*/train_log.jsonl", fontsize=7, color=MUTED)
    save(fig, "fig5_training_loss")


def fig_lr(lr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    ymin = lr.medmcqa_dev_acc.min() - 0.03
    for ax, m in zip(axes, METHODS):
        t = lr[lr.method == m].sort_values("lr")
        ax.plot(t.lr, t.medmcqa_dev_acc, color=COLOR[m], marker=MARKER[m], ls=LINESTYLE[m], lw=1.6, ms=5)
        s = t[t.selected]
        ax.scatter(s.lr, s.medmcqa_dev_acc, s=110, facecolors="none", edgecolors=INK, lw=1.2, zorder=5)
        for r in t.itertuples():
            ax.annotate(f"{r.medmcqa_dev_acc:.3f}", (r.lr, r.medmcqa_dev_acc), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=INK if r.selected else MUTED)
        ax.set_xscale("log")
        ax.set_xticks(t.lr.tolist(), [f"{v:.0e}".replace("e-0", "e-") for v in t.lr], fontsize=7.5)
        ax.minorticks_off()
        ax.set_title(LABEL[m], fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Learning rate")
        ax.set_ylim(ymin, lr.medmcqa_dev_acc.max() + 0.04)
        ax.set_xlim(t.lr.min() / 1.6, t.lr.max() * 1.6)
    axes[0].set_ylabel("MedMCQA-dev accuracy\nafter Stage 1")
    fig.text(0.0, -0.12, "Seed 0, Stage 1 only, 200 development items (1 item = 0.005). Circled = selected "
             "(highest accuracy; ties to the smaller rate). Source: results/tables/lr_search.csv",
             fontsize=6.8, color=MUTED)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig6_lr_sensitivity")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="q25-05b")
    ap.add_argument("--order", default="MLC")
    args = ap.parse_args()

    run_dirs = runs(args.model, args.order)
    order = read_json(run_dirs[0] / "run_meta.json")["order"]

    per_stage, summary = efficiency(run_dirs)
    write(per_stage, "training_efficiency.csv")
    write(summary, "efficiency_summary.csv")
    traj = dev_trajectories(run_dirs)
    write(traj, "dev_trajectories.csv")
    lr = lr_table()
    write(lr, "lr_search.csv")
    write(data_table(), "data_pipeline.csv")

    fig_dev(traj, order)
    fig_loss(run_dirs, order)
    fig_lr(lr)

    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
