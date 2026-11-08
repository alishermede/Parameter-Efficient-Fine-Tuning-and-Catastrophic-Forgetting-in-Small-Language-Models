"""Test-set evaluation of one checkpoint (or a queue of checkpoints) with lm-evaluation-harness.

For every checkpoint it downloads only that folder from the Hugging Face Hub, loads it with
modeling.load_checkpoint() (LoRA merged, adapter active, fp16), runs the frozen task settings
from configs/eval.yaml on the fixed ID subsets in data/test_subsets/, and stores the answer to
EVERY test item:

  results/eval/<run_id>/<checkpoint>/<file>.jsonl   one line per item: item_id, pred, gold, correct, ...
      mmlu_primary.jsonl (1,260)  mmlu_overlap.jsonl (450)  gsm8k.jsonl (500)
      medmcqa.jsonl (500)         casehold.jsonl (500)      mbpp.jsonl (500)
  results/eval/<run_id>/<checkpoint>/lm_eval_<group>.json   lm-eval aggregate output (no samples)
  results/eval/<run_id>/<checkpoint>/summary.json           scores, timings, environment, hashes

Stage 0 is identical for all methods (checked in training: init_max_logit_change == 0), so it is
evaluated once per base model and stored as results/eval/<model>_base/stage0/.

The test set is used ONLY for final numbers. Nothing here feeds back into training or selection.

Usage (from anywhere; the script works from the repository root):
  python src/eval.py --run_id q25-05b_lora_MLC_s1 --checkpoint stage3_code_100
  python src/eval.py --run_id q25-05b_full_MLC_s1 --checkpoint stage0           # -> q25-05b_base/stage0
  python src/eval.py --jobs results/eval/jobs/participant2_gpu0.txt --push_repo <hf-user>/rmt-peft-ckpts
  python src/eval.py --run_id q25-05b_full_MLC_s1 --checkpoint stage0 --pilot 50  # pilot, never reported
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA_DIR, RESULTS_DIR, ROOT, environment_info, load_yaml, read_json, sha256_file, write_json, write_jsonl  # noqa: E402

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")      # MBPP executes generated code (Kaggle sandbox)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EVAL_CFG = ROOT / "configs" / "eval.yaml"
TASKS_DIR = ROOT / "tasks"
SUBSETS_DIR = DATA_DIR / "test_subsets"
LETTERS = "ABCDE"

# output file name -> (group, subset file)
OUTPUT_FILES = {
    "mmlu_primary": ("mmlu", "mmlu_primary"),
    "mmlu_overlap": ("mmlu", "mmlu_overlap"),
    "gsm8k": ("gsm8k", "gsm8k"),
    "medmcqa": ("medmcqa", "medmcqa_test"),
    "casehold": ("casehold", "casehold_test"),
    "mbpp": ("mbpp", "mbpp_test"),
}
GROUP_ORDER = ["mmlu", "medmcqa", "casehold", "gsm8k", "mbpp"]   # cheap log-likelihood tasks first


# ---------------------------------------------------------------------------
# Names and paths
# ---------------------------------------------------------------------------
def output_run_id(run_id: str, checkpoint: str) -> str:
    """Stage 0 is shared by all runs of a base model."""
    return f"{run_id.split('_')[0]}_base" if checkpoint == "stage0" else run_id


def output_dir(run_id: str, checkpoint: str, pilot: int | None) -> Path:
    base = RESULTS_DIR / ("eval_pilot" if pilot else "eval")
    return base / output_run_id(run_id, checkpoint) / checkpoint


def load_subset(name: str) -> dict:
    return read_json(SUBSETS_DIR / f"{name}.json")


def pilot_cut(indices: list[int], n: int) -> list[int]:
    return sorted(indices)[:n]


# ---------------------------------------------------------------------------
# Checkpoint download / loading
# ---------------------------------------------------------------------------
def fetch_checkpoint(hub_repo: str, run_id: str, checkpoint: str, cache: Path) -> Path:
    from huggingface_hub import snapshot_download

    pattern = f"{run_id}/{checkpoint}/*"
    local = snapshot_download(repo_id=hub_repo, allow_patterns=[pattern], local_dir=str(cache),
                              token=os.environ.get("HF_TOKEN") or None)
    path = Path(local) / run_id / checkpoint
    if not (path / "checkpoint_meta.json").exists():
        raise FileNotFoundError(f"{hub_repo}: {run_id}/{checkpoint}/checkpoint_meta.json not found "
                                "(check the run id, the checkpoint name and that the repo is public)")
    return path


def load_model(ckpt: Path, dtype_name: str):
    import torch
    from modeling import load_checkpoint

    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    tok, model = load_checkpoint(ckpt, device="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype)
    return tok, model


def sanity_check(model, tok) -> dict:
    """fp16 Qwen sometimes overflows; fail loudly instead of reporting garbage."""
    import torch

    ids = tok("The capital of France is", return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**ids).logits.float()
    finite = bool(torch.isfinite(logits).all())
    top = tok.decode(logits[0, -1].argmax().item())
    if not finite:
        raise RuntimeError("non-finite logits - rerun with --dtype float32 and log a deviation")
    return {"logits_finite": finite, "probe_next_token": top}


# ---------------------------------------------------------------------------
# lm-eval
# ---------------------------------------------------------------------------
def group_request(group: str, cfg: dict, pilot: int | None) -> tuple[list[str], dict[str, list[int]] | None]:
    """lm-eval task names and the `samples` dict (task -> doc indices) for one group."""
    g = cfg["groups"][group]
    if group == "mmlu":
        samples: dict[str, list[int]] = {}
        for sub in g["subsets"]:
            for task, idx in load_subset(sub)["lm_eval_doc_indices"].items():
                samples[task] = pilot_cut(idx, max(1, pilot // 25)) if pilot else sorted(idx)
        return sorted(samples), samples
    if group in ("gsm8k", "medmcqa"):
        idx = next(iter(load_subset(g["subsets"][0])["lm_eval_doc_indices"].values()))
        idx = pilot_cut(idx, pilot) if pilot else sorted(idx)
        return g["tasks"], {g["tasks"][0]: idx}
    if group == "casehold":   # the local file holds exactly the 500 test items
        return g["tasks"], ({g["tasks"][0]: list(range(pilot))} if pilot else None)
    if group == "mbpp":       # lm-eval's MBPP test split is the full 500-task test split
        return g["tasks"], ({g["tasks"][0]: list(range(pilot))} if pilot else None)
    raise ValueError(group)


def run_group(lm, group: str, cfg: dict, pilot: int | None, task_manager) -> dict:
    from lm_eval import simple_evaluate

    tasks, samples = group_request(group, cfg, pilot)
    s = cfg["seeds"]
    return simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=cfg["groups"][group]["num_fewshot"],
        samples=samples,
        log_samples=True,
        task_manager=task_manager,
        confirm_run_unsafe_code=(group == "mbpp"),
        random_seed=s["random"], numpy_random_seed=s["numpy"],
        torch_random_seed=s["torch"], fewshot_random_seed=s["fewshot"],
        bootstrap_iters=0,
    )


# ---------------------------------------------------------------------------
# Per-item records (the input of stats.py and of the error analysis)
# ---------------------------------------------------------------------------
def _mc_pred(sample: dict) -> tuple[int, list[float]]:
    lls = [float(r[0]) for r in sample["filtered_resps"]]
    return max(range(len(lls)), key=lambda i: lls[i]), lls


def mc_record(item_id: str, sample: dict, extra: dict | None = None) -> dict:
    pred, lls = _mc_pred(sample)
    gold = int(sample["target"])
    rec = {"item_id": item_id, "pred": LETTERS[pred], "gold": LETTERS[gold],
           "correct": int(pred == gold), "doc_id": int(sample["doc_id"]),
           "loglik": [round(x, 4) for x in lls]}
    if int(round(float(sample["acc"]))) != rec["correct"]:
        raise AssertionError(f"{item_id}: argmax disagrees with lm-eval acc")
    rec.update(extra or {})
    return rec


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def _first(x):
    while isinstance(x, (list, tuple)) and x:
        x = x[0]
    return x


def parse_samples(group: str, out: dict) -> dict[str, list[dict]]:
    """lm-eval logged samples -> {output file: [records]}."""
    samples = out["samples"]
    files: dict[str, list[dict]] = {}
    if group == "mmlu":
        primary_tasks = set(load_subset("mmlu_primary")["lm_eval_doc_indices"])
        for task, rows in samples.items():
            subject = task[len("mmlu_"):]
            name = "mmlu_primary" if task in primary_tasks else "mmlu_overlap"
            for s in rows:
                files.setdefault(name, []).append(
                    mc_record(f"mmlu-{subject}-{s['doc_id']}", s, {"subject": subject}))
    elif group == "medmcqa":
        for s in samples["medmcqa"]:
            files.setdefault("medmcqa", []).append(
                mc_record(f"medmcqa-{s['doc']['id']}", s, {"subject": s["doc"].get("subject_name")}))
    elif group == "casehold":
        for s in samples["casehold_test"]:
            files.setdefault("casehold", []).append(mc_record(s["doc"]["uid"], s))
    elif group == "gsm8k":
        by_doc: dict[int, dict] = {}
        for s in samples["gsm8k"]:
            d = by_doc.setdefault(int(s["doc_id"]), {"doc_id": int(s["doc_id"]), "target": s["target"],
                                                     "response": _first(s["resps"])})
            d[s["filter"]] = (str(_first(s["filtered_resps"])), int(round(float(s["exact_match"]))))
        recs = []
        for doc_id in sorted(by_doc):
            d = by_doc[doc_id]
            flex, strict = d["flexible-extract"], d["strict-match"]
            recs.append({"item_id": f"gsm8k-test-{doc_id}", "pred": flex[0], "gold": gsm8k_gold(d["target"]),
                         "correct": flex[1], "pred_strict": strict[0], "correct_strict": strict[1],
                         "doc_id": doc_id, "response": d["response"]})
        files["gsm8k"] = recs
    elif group == "mbpp":
        for s in samples["mbpp"]:
            files.setdefault("mbpp", []).append({
                "item_id": f"mbpp-{s['doc']['task_id']}", "pred": str(_first(s["filtered_resps"])),
                "gold": "\n".join(s["doc"]["test_list"]),
                "correct": int(round(float(s["pass_at_1"]))), "doc_id": int(s["doc_id"])})
    else:
        raise ValueError(group)
    for recs in files.values():
        recs.sort(key=lambda r: r["doc_id"] if "subject" not in r else (r["subject"], r["doc_id"]))
    return files


def check_coverage(name: str, recs: list[dict], pilot: int | None) -> None:
    """Every frozen test item answered exactly once (full runs only)."""
    ids = [r["item_id"] for r in recs]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{name}: duplicate item ids")
    if pilot:
        return
    expected = set(load_subset(OUTPUT_FILES[name][1])["uids"])
    got = set(ids)
    if got != expected:
        raise AssertionError(f"{name}: {len(expected - got)} frozen items missing, "
                             f"{len(got - expected)} unexpected (dataset revision changed?)")


def aggregate_scores(files: dict[str, list[dict]]) -> dict:
    out = {}
    for name, recs in files.items():
        n = len(recs)
        out[name] = {"n": n, "score": sum(r["correct"] for r in recs) / n}
        if name == "gsm8k":
            out[name]["score_strict"] = sum(r["correct_strict"] for r in recs) / n
    return out


def strip_samples(out: dict) -> dict:
    keep = {k: v for k, v in out.items() if k not in ("samples",)}
    return json.loads(json.dumps(keep, default=str))


# ---------------------------------------------------------------------------
# One checkpoint
# ---------------------------------------------------------------------------
def evaluate_checkpoint(run_id: str, checkpoint: str, args, cfg: dict) -> Path:
    import torch
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    out_dir = output_dir(run_id, checkpoint, args.pilot)
    groups = [g for g in GROUP_ORDER if g in args.groups]
    done = {g for g in groups
            if all((out_dir / f"{f}.jsonl").exists() for f, (gg, _) in OUTPUT_FILES.items() if gg == g)}
    if not args.overwrite and done == set(groups) and (out_dir / "summary.json").exists():
        print(f"[skip] {out_dir} already complete")
        return out_dir
    todo = groups if args.overwrite else [g for g in groups if g not in done]
    print(f"\n=== {run_id}/{checkpoint} -> {out_dir}  groups: {todo}")

    t0 = time.time()
    cache = Path(args.cache_dir)
    ckpt = fetch_checkpoint(args.hub_repo, run_id, checkpoint, cache)
    meta = read_json(ckpt / "checkpoint_meta.json")
    tok, model = load_model(ckpt, args.dtype)
    if args.strict_gen_cap:
        # Qwen's generation_config sets max_new_tokens=2048, which overrides lm-eval's max_gen_toks.
        # Clearing it applies the frozen 256-token limit (deviation 14; used for the replication runs).
        model.generation_config.max_new_tokens = None
    sanity = sanity_check(model, tok)
    load_s = time.time() - t0

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=cfg["batch_size"], max_length=cfg["max_length"])
    # generation: fixed batch (lm-eval "auto" sizes it for a 4096-token prompt and ends up at ~1)
    lm_gen = HFLM(pretrained=model, tokenizer=tok, batch_size=args.gen_batch_size or cfg["gen_batch_size"], max_length=cfg["max_length"])
    tm = TaskManager(include_path=str(TASKS_DIR))
    summary_path = out_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() and not args.overwrite else {}
    summary.setdefault("scores", {})
    summary.setdefault("seconds", {})

    cwd = os.getcwd()
    os.chdir(ROOT)                      # tasks/casehold.yaml uses a path relative to the repo root
    try:
        for group in todo:
            t = time.time()
            if args.limit_check:
                print(f"  [{group}] request:", {k: len(v) for k, v in (group_request(group, cfg, args.pilot)[1] or {}).items()})
            out = run_group(lm_gen if group in ("gsm8k", "mbpp") else lm, group, cfg, args.pilot, tm)
            files = parse_samples(group, out)
            for name, recs in files.items():
                check_coverage(name, recs, args.pilot)
                write_jsonl(out_dir / f"{name}.jsonl", recs)
            write_json(out_dir / f"lm_eval_{group}.json", strip_samples(out))
            summary["scores"].update(aggregate_scores(files))
            summary["seconds"][group] = round(time.time() - t, 1)
            print(f"  [{group}] {time.time() - t:6.0f}s  " +
                  "  ".join(f"{k}={v['score']:.4f}" for k, v in aggregate_scores(files).items()))
            write_json(summary_path, summary)   # partial progress survives a crash
    finally:
        os.chdir(cwd)

    summary.update({
        "run_id": output_run_id(run_id, checkpoint), "source_run_id": run_id, "checkpoint": checkpoint,
        "hub_repo": args.hub_repo, "checkpoint_meta": meta, "dtype": args.dtype, "pilot": args.pilot,
        "eval_config": cfg, "sanity": sanity, "load_seconds": round(load_s, 1),
        "total_seconds": round(sum(summary["seconds"].values()) + load_s, 1),
        "subset_sha256": {f"test_subsets/{s}.json": sha256_file(SUBSETS_DIR / f"{s}.json")
                          for _, s in OUTPUT_FILES.values()},
        "casehold_file_sha256": sha256_file(DATA_DIR / "test" / "casehold_test.jsonl"),
        "environment": environment_info(), "argv": sys.argv,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    write_json(summary_path, summary)

    del lm, lm_gen, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not args.keep_ckpt and meta.get("method") != "base":
        shutil.rmtree(cache / run_id / checkpoint, ignore_errors=True)

    if args.push_repo and not args.pilot:
        push(args.push_repo, out_dir, f"eval/{output_run_id(run_id, checkpoint)}/{checkpoint}")
    return out_dir


def push(repo: str, folder: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for attempt in range(3):
        try:
            api.upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=path_in_repo)
            print(f"  [hub] pushed -> {repo}/{path_in_repo}")
            return
        except Exception as e:  # noqa: BLE001
            print(f"  [hub] upload failed (attempt {attempt + 1}): {e!r}")
            time.sleep(10 * (attempt + 1))
    print("  [hub] giving up; results are still in", folder)


def read_jobs(path: str) -> list[tuple[str, str]]:
    jobs = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            run_id, ckpt = line.split()[:2]
            jobs.append((run_id, ckpt))
    return jobs


def verify_test_files() -> None:
    sums = dict(reversed(l.split("  ", 1)) for l in (DATA_DIR / "SHA256SUMS").read_text().strip().splitlines())
    for rel in ("test/casehold_test.jsonl", *[f"test_subsets/{s}.json" for _, s in OUTPUT_FILES.values()]):
        p = DATA_DIR / rel
        if not p.exists():
            raise SystemExit(f"{p} missing - run `python src/prepare_data.py` first")
        if sha256_file(p) != sums[rel]:
            raise SystemExit(f"{rel}: SHA-256 differs from data/SHA256SUMS - test set changed, stop")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_id")
    ap.add_argument("--checkpoint")
    ap.add_argument("--jobs", help="text file with lines '<run_id> <checkpoint>'")
    ap.add_argument("--groups", default=",".join(GROUP_ORDER), help="subset of " + ",".join(GROUP_ORDER))
    ap.add_argument("--hub_repo", default=None, help="where the checkpoints are (default: configs/eval.yaml)")
    ap.add_argument("--push_repo", default=None, help="your HF repo: results are uploaded to eval/<run_id>/<ckpt>")
    ap.add_argument("--cache_dir", default=os.environ.get("CKPT_CACHE", "/tmp/ckpt_cache"))
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--pilot", type=int, default=None, help="evaluate only N items per task -> results/eval_pilot")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep_ckpt", action="store_true")
    ap.add_argument("--keep_going", action="store_true", help="in --jobs mode, continue after a failed job")
    ap.add_argument("--gen_batch_size", type=int, default=None, help="override gen_batch_size (replication: 16, deviation 14)")
    ap.add_argument("--strict_gen_cap", action="store_true",
                    help="clear the model's generation_config.max_new_tokens so max_gen_toks=256 applies (replication, deviation 14)")
    ap.add_argument("--limit_check", action="store_true", help="print request sizes")
    args = ap.parse_args(argv)

    cfg = load_yaml(EVAL_CFG)
    args.hub_repo = args.hub_repo or cfg["hub_repo"]
    args.dtype = args.dtype or cfg["dtype"]
    args.groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    verify_test_files()

    jobs = read_jobs(args.jobs) if args.jobs else [(args.run_id, args.checkpoint)]
    if not all(j[0] and j[1] for j in jobs):
        ap.error("give --run_id and --checkpoint, or --jobs")
    failed = []
    for i, (run_id, ckpt) in enumerate(jobs, 1):
        print(f"\n##### job {i}/{len(jobs)}: {run_id} {ckpt}")
        try:
            evaluate_checkpoint(run_id, ckpt, args, cfg)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            failed.append((run_id, ckpt))
            if not args.keep_going:
                break
    if failed:
        print("FAILED:", *[f"{r} {c}" for r, c in failed], sep="\n  ")
        return 1
    print("all jobs done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
