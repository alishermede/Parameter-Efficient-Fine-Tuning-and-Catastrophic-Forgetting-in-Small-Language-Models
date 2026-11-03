"""All test-set numbers of the report, computed from the per-item files of src/eval.py.

Nothing in the report is typed by hand: every table in results/tables/ is written here, and
src/figures.py draws the figures from these tables.

Definitions (Appendix A):
  S_s   score after stage s (Stage 0 = untouched base model, shared by all methods)
  F_s   = S_0 - S_s                 forgetting, percentage points (pp); positive = forgot
  R_s   = S_s / S_0 x 100 %         retention
  Pairwise H1 tests on F_3 (LoRA-FFT, Adapter-FFT, LoRA-Adapter): paired bootstrap over test
  items, 10,000 resamples, on seed-averaged per-item correctness; two-sided p; Holm across the
  three comparisons, separately for MMLU (primary) and GSM8K (flexible-extract).
  F_A - F_B = S_B - S_A, so the shared S_0 cancels: negative = A forgets less than B.
  H2: MMLU retention at the matched Stage-1 checkpoint, LoRA vs Adapter and LoRA vs FFT, same
  bootstrap (S_0 resampled with the items), Holm across the two comparisons.

Outputs (results/tables/):
  scores_long.csv          every run x checkpoint x metric (score, forgetting, retention)
  trajectories.csv         method x stage x metric: mean, SD over seeds, bootstrap 95% CI
  main_table.csv           report Table (Stage 3, mean +- SD), and main_table_numeric.csv
  pairwise_tests.csv       H1 comparisons (+ secondary metrics, exploratory per-seed t-tests)
  h2_tests.csv             matched-checkpoint comparisons
  hypotheses.csv           verdicts by the frozen criteria
  mmlu_by_category.csv     subgroup scores (STEM / humanities / social sciences / other, overlap)
  mmlu_confusion.csv       gold x predicted answer letter, Stage 0 vs Stage 3
  report_numbers.md        the key numbers as sentences, each with its source file

Usage:
  python src/stats.py [--model primary] [--order MLC] [--boot 10000]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA_DIR, RESULTS_DIR, load_experiment_config, read_json, read_jsonl  # noqa: E402

EVAL_DIR = RESULTS_DIR / "eval"
TABLES = RESULTS_DIR / "tables"
METHODS = ["full", "lora", "adapter"]
METHOD_LABEL = {"base": "No adaptation (Stage 0)", "full": "Full fine-tuning", "lora": "LoRA",
                "adapter": "Bottleneck adapter"}
# metric -> (per-item file, correctness field)
METRICS = {
    "mmlu_primary": ("mmlu_primary", "correct"),
    "mmlu_overlap": ("mmlu_overlap", "correct"),
    "gsm8k": ("gsm8k", "correct"),                 # flexible-extract (primary GSM8K metric)
    "gsm8k_strict": ("gsm8k", "correct_strict"),
    "medmcqa": ("medmcqa", "correct"),
    "casehold": ("casehold", "correct"),
    "mbpp": ("mbpp", "correct"),
}
RETENTION_METRICS = ["mmlu_primary", "mmlu_overlap", "gsm8k", "gsm8k_strict"]
COMPARISONS = [("lora", "full"), ("adapter", "full"), ("lora", "adapter")]
ALPHA = 0.05
RUN_RE = re.compile(r"^(?P<model>[^_]+)_(?P<method>full|lora|adapter)_(?P<order>[A-Z]+)_s(?P<seed>\d+)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def stage_of(ckpt: str) -> int:
    return 0 if ckpt == "stage0" else int(ckpt[5])


def load_items(eval_dir: Path) -> pd.DataFrame:
    """One row per (run, checkpoint, metric, item)."""
    frames = []
    for summary in sorted(eval_dir.glob("*/*/summary.json")):
        ckpt_dir = summary.parent
        run, ckpt = ckpt_dir.parent.name, ckpt_dir.name
        if run.endswith("_base"):
            info = {"model": run[:-5], "method": "base", "order": None, "seed": 0}
        else:
            m = RUN_RE.match(run)
            if not m:
                continue
            info = {**m.groupdict(), "seed": int(m["seed"])}
        for metric, (fname, field) in METRICS.items():
            p = ckpt_dir / f"{fname}.jsonl"
            if not p.exists():
                continue
            rows = read_jsonl(p)
            df = pd.DataFrame({"item_id": [r["item_id"] for r in rows],
                               "correct": [float(r[field]) for r in rows]})
            if metric.startswith("mmlu"):
                df["subject"] = [r["subject"] for r in rows]
                df["pred"] = [r["pred"] for r in rows]
                df["gold"] = [r["gold"] for r in rows]
            df = df.assign(run_id=run, checkpoint=ckpt, stage=stage_of(ckpt), metric=metric, **info)
            frames.append(df)
    if not frames:
        raise SystemExit(f"no evaluation results under {eval_dir}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def boot_indices(n: int, B: int, rng: np.random.Generator, chunk: int = 1000):
    for start in range(0, B, chunk):
        yield rng.integers(0, n, size=(min(chunk, B - start), n))


def bootstrap(stat, arrays: list[np.ndarray], B: int, seed: int) -> np.ndarray:
    """Item bootstrap of stat(*resampled_means)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    out = []
    for idx in boot_indices(n, B, rng):
        out.append(stat(*[a[idx].mean(axis=1) for a in arrays]))
    return np.concatenate(out)


def two_sided_p(boot: np.ndarray) -> float:
    B = len(boot)
    lo = (np.sum(boot <= 0) + 1) / (B + 1)
    hi = (np.sum(boot >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def mean_sd(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def fmt(mean: float, sd: float, d: int = 1) -> str:
    return f"{mean:.{d}f}" if np.isnan(sd) else f"{mean:.{d}f} ± {sd:.{d}f}"


def item_matrix(items: pd.DataFrame, metric: str, keys: list[tuple[str, str]]) -> tuple[list[str], np.ndarray]:
    """Rows = items (sorted ids), columns = (run_id, checkpoint) pairs; values 0/1 (or pass@1)."""
    sub = items[items.metric == metric]
    cols = []
    ids = None
    for run, ckpt in keys:
        s = sub[(sub.run_id == run) & (sub.checkpoint == ckpt)].set_index("item_id")["correct"].sort_index()
        if s.empty:
            raise KeyError(f"missing {metric} for {run}/{ckpt}")
        if ids is None:
            ids = s.index
        elif not s.index.equals(ids):
            raise AssertionError(f"{metric}: item sets differ between checkpoints ({run}/{ckpt})")
        cols.append(s.to_numpy())
    return list(ids), np.column_stack(cols)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def scores_long(items: pd.DataFrame) -> pd.DataFrame:
    g = items.groupby(["run_id", "model", "method", "order", "seed", "checkpoint", "stage", "metric"],
                      dropna=False)["correct"]
    df = g.agg(score="mean", n="size").reset_index()
    df["score"] *= 100
    base = df[df.method == "base"].set_index(["model", "metric"])["score"]
    df["S0"] = [base.get((m, k), np.nan) for m, k in zip(df.model, df.metric)]
    df["forgetting_pp"] = df.S0 - df.score
    df["retention_pct"] = df.score / df.S0 * 100
    return df.sort_values(["model", "order", "method", "seed", "stage", "checkpoint", "metric"])


def run_ids(model: str, order: str, method: str, seeds: list[int]) -> list[str]:
    return [f"{model}_{method}_{order}_s{s}" for s in seeds]


def trajectories(long: pd.DataFrame, items: pd.DataFrame, model: str, order: str, seeds, B: int) -> pd.DataFrame:
    """Stage ends only (stage0 + *_100), mean +- SD over seeds and an item-bootstrap CI."""
    rows = []
    base = long[(long.method == "base") & (long.model == model)]
    for metric in METRICS:
        b = base[base.metric == metric]
        if b.empty:
            continue
        s0 = float(b.score.iloc[0])
        for method in METHODS:
            rows.append({"model": model, "order": order, "method": method, "stage": 0, "checkpoint": "stage0",
                         "metric": metric, "n_seeds": 1, "score_mean": s0, "score_sd": 0.0,
                         "forgetting_mean": 0.0, "forgetting_sd": 0.0, "retention_mean": 100.0,
                         "retention_sd": 0.0, "score_ci_low": np.nan, "score_ci_high": np.nan})
            sel = long[(long.model == model) & (long.order == order) & (long.method == method)
                       & (long.metric == metric) & long.checkpoint.str.endswith("_100")
                       & long.seed.isin(seeds)]
            for (stage, ckpt), grp in sel.groupby(["stage", "checkpoint"]):
                sm, ssd = mean_sd(grp.score)
                fm, fsd = mean_sd(grp.forgetting_pp)
                rm, rsd = mean_sd(grp.retention_pct)
                keys = [(r, ckpt) for r in grp.run_id]
                _, X = item_matrix(items, metric, keys)
                bt = bootstrap(lambda a: 100 * a, [X.mean(axis=1)], B, seed=2026)
                rows.append({"model": model, "order": order, "method": method, "stage": int(stage),
                             "checkpoint": ckpt, "metric": metric, "n_seeds": len(grp),
                             "score_mean": sm, "score_sd": ssd, "forgetting_mean": fm, "forgetting_sd": fsd,
                             "retention_mean": rm, "retention_sd": rsd,
                             "score_ci_low": float(np.percentile(bt, 2.5)),
                             "score_ci_high": float(np.percentile(bt, 97.5))})
    return pd.DataFrame(rows)


def final_ckpt(order_domains: list[str]) -> str:
    return f"stage3_{order_domains[2]}_100"


def pairwise(items, long, model, order, seeds, ckpt_of, B, metrics, family_note):
    """ckpt_of(run_id) -> checkpoint compared. Returns rows for each metric x comparison."""
    rows = []
    for metric in metrics:
        base_key = [(f"{model}_base", "stage0")]
        _, S0 = item_matrix(items, metric, base_key)
        s0 = S0[:, 0]
        res = []
        for a, b in COMPARISONS:
            ra, rb = run_ids(model, order, a, seeds), run_ids(model, order, b, seeds)
            _, A = item_matrix(items, metric, [(r, ckpt_of(r)) for r in ra])
            _, Bm = item_matrix(items, metric, [(r, ckpt_of(r)) for r in rb])
            a_avg, b_avg = A.mean(axis=1), Bm.mean(axis=1)
            # forgetting difference F_a - F_b = S_b - S_a  (pp)
            diff = 100 * (b_avg.mean() - a_avg.mean())
            bt = bootstrap(lambda x, y: 100 * (y - x), [a_avg, b_avg], B, seed=2026)
            per_seed = [100 * (Bm[:, i].mean() - A[:, i].mean()) for i in range(len(seeds))]
            Fa, Fb = 100 * (s0.mean() - A.mean(axis=0)), 100 * (s0.mean() - Bm.mean(axis=0))
            try:
                from scipy.stats import ttest_rel
                t = ttest_rel(Fa, Fb)
                t_stat, t_p = float(t.statistic), float(t.pvalue)
            except Exception:  # noqa: BLE001
                t_stat = t_p = np.nan
            fb_mean = float(Fb.mean())
            res.append({
                "metric": metric, "comparison": f"{a} - {b}", "method_a": a, "method_b": b,
                "checkpoint": ckpt_of(ra[0]), "F_a_mean": float(Fa.mean()), "F_b_mean": fb_mean,
                "diff_F_pp": diff, "ci_low": float(np.percentile(bt, 2.5)), "ci_high": float(np.percentile(bt, 97.5)),
                "p_boot": two_sided_p(bt), "n_items": len(s0), "n_boot": B,
                "relative_reduction_pct": (-diff / fb_mean * 100) if fb_mean else np.nan,
                **{f"diff_seed{s}": d for s, d in zip(seeds, per_seed)},
                "same_sign_all_seeds": int(all(np.sign(d) == np.sign(diff) and d != 0 for d in per_seed)),
                "t_stat_exploratory": t_stat, "t_p_exploratory": t_p, "family": family_note,
            })
        for r, ph in zip(res, holm([r["p_boot"] for r in res])):
            r["p_holm"] = ph
            r["significant"] = int(ph < ALPHA and (r["ci_high"] < 0 or r["ci_low"] > 0))
        rows.extend(res)
    return pd.DataFrame(rows)


def h2_tests(items, model, order, seeds, matched: pd.DataFrame, B: int) -> pd.DataFrame:
    ck = dict(zip(matched.run_id, matched.matched_checkpoint))
    _, S0 = item_matrix(items, "mmlu_primary", [(f"{model}_base", "stage0")])
    s0 = S0[:, 0]
    avg = {}
    for m in METHODS:
        rids = run_ids(model, order, m, seeds)
        _, X = item_matrix(items, "mmlu_primary", [(r, ck[r]) for r in rids])
        avg[m] = X
    rows = []
    for a, b in [("lora", "adapter"), ("lora", "full")]:
        A, Bm = avg[a].mean(axis=1), avg[b].mean(axis=1)
        diff = (A.mean() - Bm.mean()) / s0.mean() * 100   # retention difference, pp of S0
        bt = bootstrap(lambda x, y, z: (x - y) / z * 100, [A, Bm, s0], B, seed=2026)
        per_seed = [(avg[a][:, i].mean() - avg[b][:, i].mean()) / s0.mean() * 100 for i in range(len(seeds))]
        rows.append({"metric": "mmlu_primary_retention", "comparison": f"{a} - {b}",
                     "R_a_mean": float(A.mean() / s0.mean() * 100), "R_b_mean": float(Bm.mean() / s0.mean() * 100),
                     "diff_R_pp": float(diff), "ci_low": float(np.percentile(bt, 2.5)),
                     "ci_high": float(np.percentile(bt, 97.5)), "p_boot": two_sided_p(bt),
                     **{f"diff_seed{s}": d for s, d in zip(seeds, per_seed)}, "n_boot": B})
    for r, ph in zip(rows, holm([r["p_boot"] for r in rows])):
        r["p_holm"] = ph
        r["significant"] = int(ph < ALPHA and r["ci_low"] > 0)
    return pd.DataFrame(rows)


def verdicts(pw: pd.DataFrame, h2: pd.DataFrame | None, h2_testable: bool | None) -> pd.DataFrame:
    """Frozen criteria (Appendix A)."""
    def ok(method, metric):
        r = pw[(pw.metric == metric) & (pw.method_a == method) & (pw.method_b == "full")].iloc[0]
        return bool(r.diff_F_pp < 0 and r.p_holm < ALPHA and r.ci_high < 0 and r.same_sign_all_seeds)

    def ev(method, metric):
        r = pw[(pw.metric == metric) & (pw.method_a == method) & (pw.method_b == "full")].iloc[0]
        return (f"{method}-FFT {metric}: ΔF3 = {r.diff_F_pp:+.2f} pp [{r.ci_low:+.2f}, {r.ci_high:+.2f}], "
                f"p_Holm = {r.p_holm:.3g}, same sign in all seeds: {'yes' if r.same_sign_all_seeds else 'no'}")

    peft = [m for m in ("lora", "adapter") if m in set(pw.method_a)]
    checks = {(m, k): ok(m, k) for m in peft for k in ("mmlu_primary", "gsm8k")}
    n_ok = sum(checks.values())
    h1 = "Supported" if n_ok == 4 else ("Not supported" if n_ok == 0 else "Partly supported")
    if len(peft) < 2:
        h1 = f"PRELIMINARY ({'/'.join(peft)} only): " + h1
    rows = [{"hypothesis": "H1", "evidence": "pairwise_tests.csv; Fig. 1",
             "result_in_numbers": " | ".join(ev(m, k) for m in peft for k in ("mmlu_primary", "gsm8k")),
             "criteria_met": "; ".join(f"{m}/{k}: {'yes' if v else 'no'}" for (m, k), v in checks.items()),
             "verdict": h1}]
    if h2_testable is False:
        rows.append({"hypothesis": "H2", "evidence": "matched_summary.json", "result_in_numbers":
                     "no matched checkpoint within ±3 pp for all methods", "criteria_met": "-", "verdict": "Not testable"})
    elif h2 is not None and len(h2):
        n = int(h2.significant.sum())
        h2v = "Supported" if n == 2 else ("Partly supported" if n == 1 else "Not supported")
        rows.append({"hypothesis": "H2", "evidence": "h2_tests.csv; matched_checkpoints.csv",
                     "result_in_numbers": " | ".join(
                         f"{r.comparison} MMLU retention: {r.diff_R_pp:+.2f} pp [{r.ci_low:+.2f}, {r.ci_high:+.2f}], "
                         f"p_Holm = {r.p_holm:.3g}" for r in h2.itertuples()),
                     "criteria_met": "; ".join(f"{r.comparison}: {'yes' if r.significant else 'no'}" for r in h2.itertuples()),
                     "verdict": h2v})
    return pd.DataFrame(rows)


def main_table(traj: pd.DataFrame, long: pd.DataFrame, model: str, order: str, seeds) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = {}
    for m in METHODS:
        vals = []
        for r in run_ids(model, order, m, seeds):
            p = RESULTS_DIR / "raw" / r / "run_meta.json"
            if p.exists():
                vals.append(read_json(p)["params"])
        if vals:
            params[m] = vals[0]
    cols = [("medmcqa", "score", "MedMCQA"), ("casehold", "score", "CaseHOLD"), ("mbpp", "score", "MBPP pass@1"),
            ("mmlu_primary", "score", "MMLU S3"), ("mmlu_primary", "forgetting", "MMLU F3 (pp)"),
            ("gsm8k", "score", "GSM8K S3"), ("gsm8k", "forgetting", "GSM8K F3 (pp)")]
    fmt_rows, num_rows = [], []
    t3 = traj[traj.stage == 3]
    t0 = traj[(traj.stage == 0) & (traj.method == "full")]
    row_f, row_n = {"Condition": METHOD_LABEL["base"], "Trainable params": "0"}, {"condition": "base", "trainable_pct": 0.0}
    for metric, kind, label in cols:
        r = t0[t0.metric == metric]
        v = float(r.score_mean.iloc[0]) if len(r) else np.nan
        row_f[label] = "0" if kind == "forgetting" else (f"{v:.1f}" if not np.isnan(v) else "")
        row_n[f"{metric}_{kind}_mean"] = 0.0 if kind == "forgetting" else v
    fmt_rows.append(row_f)
    num_rows.append(row_n)
    for m in METHODS:
        p = params.get(m, {})
        row_f = {"Condition": METHOD_LABEL[m],
                 "Trainable params": (f"{p['trainable']/1e6:.1f} M ({p['trainable_pct']:.2f}%)" if p else "")}
        row_n = {"condition": m, "trainable": p.get("trainable"), "trainable_pct": p.get("trainable_pct")}
        for metric, kind, label in cols:
            r = t3[(t3.method == m) & (t3.metric == metric)]
            if r.empty:
                row_f[label] = ""
                continue
            mean, sd = float(r[f"{kind}_mean"].iloc[0]), float(r[f"{kind}_sd"].iloc[0])
            row_f[label] = fmt(mean, sd)
            row_n[f"{metric}_{kind}_mean"], row_n[f"{metric}_{kind}_sd"] = mean, sd
        fmt_rows.append(row_f)
        num_rows.append(row_n)
    return pd.DataFrame(fmt_rows), pd.DataFrame(num_rows)


def mmlu_subgroups(items: pd.DataFrame, model: str, order: str, seeds, final: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cats = read_json(DATA_DIR / "mmlu_categories.json")["category"]
    mm = items[items.metric.isin(["mmlu_primary", "mmlu_overlap"]) & (items.model == model)].copy()
    mm["category"] = mm.subject.map(cats)
    mm.loc[mm.metric == "mmlu_overlap", "category"] = "overlap_" + mm.loc[mm.metric == "mmlu_overlap", "category"]
    keep = (mm.method == "base") | ((mm.order == order) & mm.seed.isin(seeds) & (mm.checkpoint == final))
    mm = mm[keep]
    by_seed = mm.groupby(["method", "seed", "category"])["correct"].mean().mul(100).reset_index()
    sub = by_seed.groupby(["method", "category"])["correct"].agg(["mean", "std", "count"]).reset_index()
    base = sub[sub.method == "base"].set_index("category")["mean"]
    sub["forgetting_pp"] = sub.category.map(base) - sub["mean"]
    sub = sub.rename(columns={"mean": "score_mean", "std": "score_sd", "count": "n_seeds"})
    sub["n_items"] = sub.category.map(mm[mm.method == "base"].groupby("category").size())
    prim = mm[mm.metric == "mmlu_primary"]
    conf = prim.groupby(["method", "gold", "pred"]).size().rename("count").reset_index()
    conf["stage"] = np.where(conf.method == "base", 0, 3)
    return sub, conf


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def report_md(main_f, pw, h2, hyp, traj, matched_sum, B) -> str:
    L = ["# Key numbers (generated by src/stats.py — do not edit by hand)", ""]
    L.append("Source tables are in `results/tables/`. Means ± SD over 3 seeds; ΔF = F_A − F_B "
             f"(negative = A forgets less); paired item bootstrap, {B} resamples, Holm per benchmark.")
    L += ["", "## Main table (Stage 3) — main_table.csv", "", md_table(main_f), ""]
    L += ["## Pairwise tests — pairwise_tests.csv", ""]
    for r in pw.itertuples():
        L.append(f"- **{r.metric}, {r.comparison}**: F_a = {r.F_a_mean:.2f} pp, F_b = {r.F_b_mean:.2f} pp, "
                 f"ΔF3 = {r.diff_F_pp:+.2f} pp (95% CI {r.ci_low:+.2f} to {r.ci_high:+.2f}), "
                 f"p = {r.p_boot:.4f}, Holm p = {r.p_holm:.4f}; relative reduction {r.relative_reduction_pct:.1f}%; "
                 f"per seed {', '.join(f'{getattr(r, c):+.2f}' for c in pw.columns if c.startswith('diff_seed'))}; "
                 f"exploratory paired t (n=3): t = {r.t_stat_exploratory:.2f}, p = {r.t_p_exploratory:.3f}  [{r.family}]")
    if matched_sum:
        L += ["", "## Matched checkpoint (H2) — matched_summary.json, h2_tests.csv", "",
              f"T = {100*matched_sum['T']:.1f}% (MedMCQA-dev, {matched_sum['T_method']}); testable: {matched_sum['h2_testable']}"]
        for m, v in matched_sum["per_method"].items():
            L.append(f"- {m}: seed-mean matched dev acc {100*v['mean_matched_dev_acc']:.1f}% (|diff| {v['abs_diff_pp']:.1f} pp)")
    if h2 is not None and len(h2):
        for r in h2.itertuples():
            L.append(f"- **{r.comparison}**: R_a = {r.R_a_mean:.1f}%, R_b = {r.R_b_mean:.1f}%, ΔR = {r.diff_R_pp:+.2f} pp "
                     f"(95% CI {r.ci_low:+.2f} to {r.ci_high:+.2f}), Holm p = {r.p_holm:.4f}")
    L += ["", "## Hypotheses — hypotheses.csv", ""]
    for r in hyp.itertuples():
        L.append(f"- **{r.hypothesis}: {r.verdict}** — {r.result_in_numbers}")
    L += ["", "## Trajectories (MMLU primary, GSM8K) — trajectories.csv", ""]
    t = traj[traj.metric.isin(["mmlu_primary", "gsm8k"])]
    for r in t.itertuples():
        L.append(f"- {r.metric} {r.method} stage {r.stage}: S = {fmt(r.score_mean, r.score_sd)}, "
                 f"F = {fmt(r.forgetting_mean, r.forgetting_sd, 2)} pp, R = {fmt(r.retention_mean, r.retention_sd)}%")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="primary")
    ap.add_argument("--order", default="MLC")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--eval_dir", default=str(EVAL_DIR))
    ap.add_argument("--out", default=str(TABLES))
    args = ap.parse_args(argv)

    cfg = load_experiment_config()
    model = cfg["models"][args.model]["short"]
    seeds = cfg["seeds"]["primary"] if args.model == "primary" and args.order == "MLC" else [1]
    final = final_ckpt(cfg["orders"][args.order])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if (args.model, args.order) == ("primary", "MLC") else f"_{model}_{args.order}"

    items = load_items(Path(args.eval_dir))
    # methods with all seeds at the final stage; comparisons restricted to them (preliminary runs)
    global METHODS, COMPARISONS
    have = [m for m in METHODS if items[(items.method == m) & (items.model == model) & (items.order == args.order)
                                         & (items.checkpoint == final)].seed.nunique() == len(seeds)]
    if have != METHODS:
        print(f"PRELIMINARY: only {have} have all {len(seeds)} seeds at {final}; others skipped")
    METHODS = have
    COMPARISONS = [(a, b) for a, b in COMPARISONS if a in have and b in have]
    long = scores_long(items)
    long.to_csv(out / "scores_long.csv", index=False, float_format="%.4f")   # all models/orders
    traj = trajectories(long, items, model, args.order, seeds, args.boot)
    traj.to_csv(out / f"trajectories{suffix}.csv", index=False, float_format="%.4f")
    main_f, main_n = main_table(traj, long, model, args.order, seeds)
    main_f.to_csv(out / f"main_table{suffix}.csv", index=False)
    main_n.to_csv(out / f"main_table_numeric{suffix}.csv", index=False, float_format="%.4f")
    sub, conf = mmlu_subgroups(items, model, args.order, seeds, final)
    sub.to_csv(out / f"mmlu_by_category{suffix}.csv", index=False, float_format="%.4f")
    conf.to_csv(out / f"mmlu_confusion{suffix}.csv", index=False)

    if len(seeds) < 2:        # replication / reversed order: descriptive only (1 seed)
        print("1 seed: descriptive tables only ->", out)
        return 0

    pw = pd.concat([
        pairwise(items, long, model, args.order, seeds, lambda r: final, args.boot,
                 ["mmlu_primary", "gsm8k"], "primary (H1)"),
        pairwise(items, long, model, args.order, seeds, lambda r: final, args.boot,
                 ["gsm8k_strict", "mmlu_overlap"], "secondary (exploratory)"),
    ], ignore_index=True)
    pw.to_csv(out / "pairwise_tests.csv", index=False, float_format="%.6f")

    matched_path, msum_path = out / "matched_checkpoints.csv", out / "matched_summary.json"
    h2, msum = None, None
    if matched_path.exists():
        msum = json.loads(msum_path.read_text())
        try:
            if len(METHODS) < 3:
                raise KeyError("needs all three methods")
            h2 = h2_tests(items, model, args.order, seeds, pd.read_csv(matched_path), args.boot)
            h2.to_csv(out / "h2_tests.csv", index=False, float_format="%.6f")
        except KeyError as e:
            print("H2 skipped (matched checkpoint not evaluated yet):", e)
    hyp = verdicts(pw, h2, msum["h2_testable"] if msum else None)
    hyp.to_csv(out / "hypotheses.csv", index=False)
    (out / "report_numbers.md").write_text(report_md(main_f, pw, h2, hyp, traj, msum, args.boot))
    print(main_f.to_string(index=False))
    print(hyp[["hypothesis", "verdict"]].to_string(index=False))
    print("tables ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
