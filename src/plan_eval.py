"""Which checkpoints get the full test evaluation, and the matched checkpoint for H2.

Protocol (Appendix A):
  * full test set at Stage 0 (once per base model) and at the end of every stage (*_100);
  * H2 matched checkpoint, chosen from DEVELOPMENT metrics only:
      T = lowest final Stage-1 MedMCQA-dev accuracy among the three methods (mean over seeds);
      for each run, the Stage-1 checkpoint whose MedMCQA-dev accuracy is closest to T.
    Tie-break (not specified in the freeze form; recorded in docs/deviations.md):
      equal distance -> the later checkpoint (more training, closer to the end-of-stage model).
    H2 is testable only if, for every method, the seed-mean dev accuracy of its matched
    checkpoints is within +-3 pp of T.

Outputs:
  results/tables/matched_checkpoints.csv  one row per primary run
  results/tables/matched_summary.json     T, per-method means, testability
  results/eval/jobs/<name>.txt            job lists for src/eval.py (optionally split per GPU)

Usage:
  python src/plan_eval.py                                   # matching + job list of all primary runs
  python src/plan_eval.py --methods full,lora --name participant2 --gpus 2 --stage0
  python src/plan_eval.py --model replication --methods full,lora,adapter --seeds 1 --name replication --stage0
  python src/plan_eval.py --order CLM --methods full,lora,adapter --seeds 1 --name reversed
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS_DIR, load_experiment_config, read_jsonl, write_json  # noqa: E402

METHODS = ("full", "lora", "adapter")
TOL_PP = 3.0


def run_id(model_short: str, method: str, order: str, seed: int) -> str:
    return f"{model_short}_{method}_{order}_s{seed}"


def stage_end_names(order: list[str]) -> list[str]:
    return [f"stage{k}_{dom}_100" for k, dom in enumerate(order, 1)]


def match(model_short: str, order_name: str, seeds: list[int], raw_dir: Path) -> dict:
    """Matched Stage-1 checkpoints for every primary run (dev data only)."""
    dev = {}
    for m in METHODS:
        for s in seeds:
            rid = run_id(model_short, m, order_name, s)
            f = raw_dir / rid / "dev_metrics.jsonl"
            if not f.exists():
                raise SystemExit(f"{f} missing - pull the training logs first (src/pull_results.py)")
            dev[rid] = [r for r in read_jsonl(f) if r["stage"] == 1]
    finals = {}
    for m in METHODS:
        vals = []
        for s in seeds:
            rows = dev[run_id(model_short, m, order_name, s)]
            vals.append(max(rows, key=lambda r: r["fraction"])["medmcqa_dev_acc"])
        finals[m] = sum(vals) / len(vals)
    t_method = min(finals, key=finals.get)
    T = finals[t_method]

    rows_out = []
    for m in METHODS:
        for s in seeds:
            rid = run_id(model_short, m, order_name, s)
            cands = sorted(dev[rid], key=lambda r: r["fraction"])
            # distance rounded to remove float noise (accuracies are k/200); ties -> later checkpoint
            best = min(cands, key=lambda r: (round(abs(r["medmcqa_dev_acc"] - T), 9), -r["fraction"]))
            dists = [round(abs(r["medmcqa_dev_acc"] - T), 9) for r in cands]
            rows_out.append({
                "run_id": rid, "method": m, "seed": s, "matched_checkpoint": best["checkpoint"],
                "matched_dev_acc": best["medmcqa_dev_acc"], "target_T": round(T, 6),
                "abs_diff_pp": round(100 * abs(best["medmcqa_dev_acc"] - T), 3),
                "tie_broken": int(dists.count(min(dists)) > 1),
                "candidates": " ".join(f"{r['checkpoint']}={r['medmcqa_dev_acc']}" for r in cands),
            })
    per_method = {}
    for m in METHODS:
        accs = [r["matched_dev_acc"] for r in rows_out if r["method"] == m]
        mean = sum(accs) / len(accs)
        per_method[m] = {"mean_matched_dev_acc": round(mean, 6), "abs_diff_pp": round(100 * abs(mean - T), 3),
                         "within_tol": abs(mean - T) * 100 <= TOL_PP + 1e-9,
                         "max_run_diff_pp": max(r["abs_diff_pp"] for r in rows_out if r["method"] == m)}
    summary = {
        "rule": "T = min over methods of seed-mean final Stage-1 MedMCQA-dev acc; per run the Stage-1 "
                "checkpoint closest to T; ties -> later checkpoint; testable if every method's seed-mean "
                f"matched dev acc is within +-{TOL_PP} pp of T",
        "final_stage1_dev_acc_mean": {m: round(v, 6) for m, v in finals.items()},
        "T": round(T, 6), "T_method": t_method, "per_method": per_method,
        "h2_testable": all(v["within_tol"] for v in per_method.values()),
    }
    return {"rows": rows_out, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="primary", choices=["primary", "replication"])
    ap.add_argument("--order", default="MLC")
    ap.add_argument("--methods", default="full,lora,adapter")
    ap.add_argument("--seeds", default=None, help="default: primary seeds 1,2,3")
    ap.add_argument("--stage0", action="store_true", help="add the Stage-0 job (once per base model)")
    ap.add_argument("--name", default=None, help="job file name (results/eval/jobs/<name>.txt)")
    ap.add_argument("--gpus", type=int, default=1, help="split the jobs into N balanced files")
    args = ap.parse_args(argv)

    cfg = load_experiment_config()
    short = cfg["models"][args.model]["short"]
    order = cfg["orders"][args.order]
    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]["primary"]

    matched = {}
    if args.model == "primary" and args.order == "MLC":
        res = match(short, "MLC", cfg["seeds"]["primary"], RESULTS_DIR / "raw")
        tables = RESULTS_DIR / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        with open(tables / "matched_checkpoints.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["rows"][0]), lineterminator="\n")
            w.writeheader()
            w.writerows(res["rows"])
        write_json(tables / "matched_summary.json", res["summary"])
        print(json.dumps(res["summary"], indent=2))
        for r in res["rows"]:
            print(f"  {r['run_id']:26s} -> {r['matched_checkpoint']:20s} dev {r['matched_dev_acc']:.3f} "
                  f"(|diff| {r['abs_diff_pp']:.1f} pp{', tie' if r['tie_broken'] else ''})")
        matched = {r["run_id"]: r["matched_checkpoint"] for r in res["rows"]}

    jobs = []
    if args.stage0:
        jobs.append((run_id(short, methods[0], args.order, seeds[0]), "stage0"))
    for m in methods:
        for s in seeds:
            rid = run_id(short, m, args.order, s)
            names = stage_end_names(order)
            extra = matched.get(rid)
            if extra and extra not in names:
                names = [names[0], extra, *names[1:]]
            jobs.extend((rid, n) for n in names)

    name = args.name or f"{short}_{args.order}_{'-'.join(methods)}"
    out = RESULTS_DIR / "eval" / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    parts = [jobs[i::args.gpus] for i in range(args.gpus)]
    for i, part in enumerate(parts):
        p = out / (f"{name}.txt" if args.gpus == 1 else f"{name}_gpu{i}.txt")
        p.write_text("".join(f"{r} {c}\n" for r, c in part))
        print(f"{p}: {len(part)} jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
