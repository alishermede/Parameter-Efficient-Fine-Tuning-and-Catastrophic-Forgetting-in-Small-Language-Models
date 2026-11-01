"""Equal learning-rate budget: three values per method, Stage 1 (medicine) only, seed 0.

Selection uses MedMCQA-dev accuracy at the end of Stage 1 - never retention or test data.
Ties are broken towards the smaller learning rate.

  python src/lr_search.py --methods full,lora,adapter      # run (can be split across accounts)
  python src/lr_search.py --select                         # write configs/lr_selected.json
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CONFIG_DIR, RESULTS_DIR, load_experiment_config, read_jsonl, write_json  # noqa: E402
import train as T  # noqa: E402

LR_DIR = RESULTS_DIR / "lr_search"


def run_id(short: str, method: str, lr: float, seed: int) -> str:
    return f"lrsearch_{short}_{method}_lr{lr:.0e}_s{seed}".replace("e-0", "e-")


def run_all(args, exp) -> None:
    mcfg = T.resolve_model(exp, args.model)
    seed = exp["seeds"]["lr_search"]
    first_domain = exp["orders"]["MLC"][0]
    for method in args.methods.split(","):
        for lr in exp["lr_grid"][method]:
            rid = run_id(mcfg["short"], method, lr, seed)
            out = LR_DIR / rid
            if (out / "run_summary.json").exists() and not args.overwrite:
                print(f"skip {rid}: already done")
                continue
            argv = ["--method", method, "--seed", str(seed), "--model", args.model, "--stages", first_domain,
                    "--lr", str(lr), "--run_id", rid, "--out", str(out), "--ckpt_fracs", "1.0", "--no_save",
                    "--overwrite"]
            if args.config:
                argv += ["--config", args.config]
            if args.hub_repo:
                argv += ["--hub_repo", args.hub_repo]
            else:
                argv += ["--no_push_results"]
            if args.train_n:
                argv += ["--train_n", str(args.train_n), "--dev_n", str(args.dev_n)]
            T.run(T.build_parser().parse_args(argv))


def select(args, exp) -> None:
    mcfg = T.resolve_model(exp, args.model)
    rows, chosen = [], {}
    for method in ("full", "lora", "adapter"):
        best = None
        for lr in exp["lr_grid"][method]:
            path = LR_DIR / run_id(mcfg["short"], method, lr, exp["seeds"]["lr_search"]) / "dev_metrics.jsonl"
            if not path.exists():
                raise SystemExit(f"missing LR-search result: {path}")
            final = [m for m in read_jsonl(path) if m["stage"] == 1][-1]
            acc = final["medmcqa_dev_acc"]
            rows.append({"model": mcfg["short"], "method": method, "lr": lr, "medmcqa_dev_acc": acc,
                         "source": str(path.relative_to(RESULTS_DIR.parent))})
            if best is None or acc > best[1] + 1e-12 or (abs(acc - best[1]) <= 1e-12 and lr < best[0]):
                best = (lr, acc)
        chosen[method] = best[0]
    for r in rows:
        r["selected"] = chosen[r["method"]] == r["lr"]
    LR_DIR.mkdir(parents=True, exist_ok=True)
    with open(LR_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sel = {
        mcfg["short"]: chosen,
        "_applies_to_all_models_from": mcfg["short"],
        "_rule": "max MedMCQA-dev accuracy after Stage 1, seed 0; ties -> smaller LR",
    }
    write_json(CONFIG_DIR / "lr_selected.json", sel)
    for r in rows:
        print(f"{r['method']:8s} lr={r['lr']:.0e}  medmcqa_dev={r['medmcqa_dev_acc']:.3f}  {'<- selected' if r['selected'] else ''}")
    print("written", CONFIG_DIR / "lr_selected.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="full,lora,adapter")
    ap.add_argument("--model", default="primary")
    ap.add_argument("--config", default=None)
    ap.add_argument("--hub_repo", default=None)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--train_n", type=int, default=0, help="pilot only")
    ap.add_argument("--dev_n", type=int, default=0, help="pilot only")
    args = ap.parse_args()
    exp = load_experiment_config(args.config)
    select(args, exp) if args.select else run_all(args, exp)


if __name__ == "__main__":
    main()
