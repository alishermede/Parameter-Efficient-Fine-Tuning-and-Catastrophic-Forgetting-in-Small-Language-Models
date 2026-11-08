"""Replication analysis (Qwen3-0.6B-Base, MLC order, seed 1).

Same quantities as the primary analysis, for one seed: scores after every stage, forgetting
F_s = S0 - S_s and retention R_s = S_s / S0 on MMLU (primary) and GSM8K, target-domain scores,
pairwise paired item bootstrap (10,000 resamples) on Stage-3 forgetting with Holm correction
within each benchmark, and MMLU retention at the matched checkpoint (deviation 13).
With a single seed there is no seed variability; results are descriptive of robustness.

Writes results/tables/replication_main.csv, replication_trajectories.csv,
replication_pairwise.csv, replication_matched.csv and replication_numbers.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "eval"
OUT = ROOT / "results" / "tables"
METHODS = {"full": "Full fine-tuning", "lora": "LoRA", "adapter": "Bottleneck adapter"}
STAGES = {1: "stage1_medmcqa_100", 2: "stage2_casehold_100", 3: "stage3_code_100"}
MATCHED = {"full": "stage1_medmcqa_100", "lora": "stage1_medmcqa_050", "adapter": "stage1_medmcqa_100"}
TASKS = ["mmlu_primary", "mmlu_overlap", "gsm8k", "medmcqa", "casehold", "mbpp"]
B, SEED = 10_000, 0


def load(run: str, ckpt: str) -> dict[str, pd.Series]:
    d = EVAL / run / ckpt
    out = {}
    for t in TASKS:
        rows = [json.loads(l) for l in open(d / f"{t}.jsonl")]
        out[t] = pd.Series({r["item_id"]: float(r["correct"]) for r in rows})
        if t == "gsm8k":
            out["gsm8k_strict"] = pd.Series({r["item_id"]: float(r["correct_strict"]) for r in rows})
    return out


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p); m = len(p); adj = [0.0] * m; running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def boot_diff(a: pd.Series, b: pd.Series, rng) -> tuple[float, float, float, float]:
    """Difference in forgetting F_A - F_B = (S0 - S_A) - (S0 - S_B) = S_B - S_A, paired over items."""
    d = (b - a).to_numpy()
    n = len(d); est = d.mean()
    idx = rng.integers(0, n, size=(B, n)); bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return 100 * est, 100 * lo, 100 * hi, max(p, 1 / B)


def main() -> None:
    base = load("q3-06b_base", "stage0")
    s0 = {t: v.mean() for t, v in base.items()}
    traj, main_rows = [], []
    final = {}
    for m in METHODS:
        run = f"q3-06b_{m}_MLC_s1"
        for st, ck in [(0, None)] + list(STAGES.items()):
            sc = base if ck is None else load(run, ck)
            if st == 3:
                final[m] = sc
            for t, v in sc.items():
                S = v.mean()
                traj.append({"method": m, "stage": st, "metric": t, "score": 100 * S,
                             "forgetting_pp": 100 * (s0[t] - S), "retention_pct": 100 * S / s0[t]})
        f = final[m]
        main_rows.append({
            "condition": METHODS[m],
            "MedMCQA": 100 * f["medmcqa"].mean(), "CaseHOLD": 100 * f["casehold"].mean(), "MBPP": 100 * f["mbpp"].mean(),
            "MMLU_S3": 100 * f["mmlu_primary"].mean(), "MMLU_F3_pp": 100 * (s0["mmlu_primary"] - f["mmlu_primary"].mean()),
            "MMLU_R3_pct": 100 * f["mmlu_primary"].mean() / s0["mmlu_primary"],
            "GSM8K_S3": 100 * f["gsm8k"].mean(), "GSM8K_F3_pp": 100 * (s0["gsm8k"] - f["gsm8k"].mean()),
            "GSM8K_R3_pct": 100 * f["gsm8k"].mean() / s0["gsm8k"],
            "GSM8K_strict_S3": 100 * f["gsm8k_strict"].mean()})
    main_rows.insert(0, {"condition": "No adaptation (Stage 0)", "MedMCQA": 100 * s0["medmcqa"], "CaseHOLD": 100 * s0["casehold"],
                         "MBPP": 100 * s0["mbpp"], "MMLU_S3": 100 * s0["mmlu_primary"], "MMLU_F3_pp": 0.0, "MMLU_R3_pct": 100.0,
                         "GSM8K_S3": 100 * s0["gsm8k"], "GSM8K_F3_pp": 0.0, "GSM8K_R3_pct": 100.0,
                         "GSM8K_strict_S3": 100 * s0["gsm8k_strict"]})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(main_rows).round(2).to_csv(OUT / "replication_main.csv", index=False)
    pd.DataFrame(traj).round(2).to_csv(OUT / "replication_trajectories.csv", index=False)

    rng = np.random.default_rng(SEED); pw = []
    pairs = [("lora", "full"), ("adapter", "full"), ("lora", "adapter")]
    for bench in ["mmlu_primary", "gsm8k"]:
        res = [boot_diff(final[a][bench], final[b][bench], rng) for a, b in pairs]
        adj = holm([r[3] for r in res])
        for (a, b), r, pa in zip(pairs, res, adj):
            pw.append({"benchmark": bench, "comparison": f"{a} - {b}", "dF3_pp": r[0], "ci_lo": r[1], "ci_hi": r[2],
                       "p_boot": r[3], "p_holm": pa})
    pwd = pd.DataFrame(pw); pwd.round(4).to_csv(OUT / "replication_pairwise.csv", index=False)

    mt = []
    matched = {m: load(f"q3-06b_{m}_MLC_s1", MATCHED[m]) for m in METHODS}
    for m in METHODS:
        S = matched[m]["mmlu_primary"].mean()
        mt.append({"method": m, "checkpoint": MATCHED[m], "MMLU": 100 * S, "MMLU_retention_pct": 100 * S / s0["mmlu_primary"]})
    res = [boot_diff(matched[b]["mmlu_primary"], matched[a]["mmlu_primary"], rng) for a, b in [("lora", "full"), ("lora", "adapter")]]
    # here the difference is retention_A - retention_B in pp of accuracy (S_A - S_B)
    adj = holm([r[3] for r in res])
    mtd = pd.DataFrame(mt)
    mtd.round(2).to_csv(OUT / "replication_matched.csv", index=False)

    md = ["# Replication numbers (Qwen3-0.6B-Base, MLC, seed 1)", "",
          pd.DataFrame(main_rows).round(1).to_markdown(index=False), "",
          "## Pairwise Stage-3 forgetting (paired item bootstrap, Holm within benchmark)", "",
          pwd.round(3).to_markdown(index=False), "", "## Matched checkpoint (T = 47.5% MedMCQA-dev)", "",
          mtd.round(1).to_markdown(index=False), ""]
    for (a, b), r, pa in zip([("lora", "full"), ("lora", "adapter")], res, adj):
        md.append(f"- MMLU at matched checkpoint, {a} - {b}: {r[0]:+.2f} pp [{r[1]:+.2f}, {r[2]:+.2f}], Holm p = {pa:.4f}")
    (OUT / "replication_numbers.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
