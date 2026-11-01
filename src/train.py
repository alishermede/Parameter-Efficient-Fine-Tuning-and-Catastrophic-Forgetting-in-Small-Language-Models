"""Sequential domain adaptation run: one model x one method x one seed x one domain order.

Example (primary run, LoRA, seed 1, medicine -> law -> code):
  python src/train.py --method lora --seed 1 --model primary --order MLC \
      --ckpt_root /kaggle/working/ckpts --hub_repo <user>/rmt-peft-ckpts

Writes to results/raw/<run_id>/:
  run_meta.json        configuration, library versions, GPU, parameter counts, learning rate
  dev_metrics.jsonl    one line per checkpoint (stage 0 + 25/50/75/100% of every stage)
  dev_preds/*.jsonl    per-item dev predictions of every checkpoint
  train_log.jsonl      Trainer loss / learning-rate history per stage
  run_summary.json     per-stage training time, peak GPU memory, checkpoints written
Checkpoints go to <ckpt_root>/<run_id>/<checkpoint name>/ and optionally to the Hub.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on 16 GB T4

import torch  # noqa: E402

from common import (  # noqa: E402
    CONFIG_DIR, DATA_DIR, RESULTS_DIR, environment_info, load_experiment_config, read_json,
    read_jsonl, set_seed, write_json,
)
from dev_eval import mbpp_pass_at_1, mcq_accuracy, response_loss  # noqa: E402
from modeling import (  # noqa: E402
    apply_method, checkpoint_steps, enable_gradient_checkpointing, load_base, save_checkpoint,
    write_base_marker,
)
from tokenize_data import PadCollator, tokenize_example  # noqa: E402


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _jsonl_append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def resolve_model(exp_cfg: dict, key: str) -> dict:
    if key in exp_cfg["models"]:
        return dict(exp_cfg["models"][key])
    return {"id": key, "revision": "main", "short": key.split("/")[-1].lower()}


def resolve_revision_sha(model_id: str, revision: str) -> str | None:
    if Path(model_id).exists():
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id, revision=revision).sha
    except Exception:  # noqa: BLE001
        return None


def selected_lr(model_short: str, method: str) -> float:
    path = CONFIG_DIR / "lr_selected.json"
    if not path.exists():
        raise SystemExit("configs/lr_selected.json not found: run src/lr_search.py first or pass --lr")
    sel = read_json(path)
    table = sel.get(model_short) or sel.get(sel.get("_applies_to_all_models_from", ""), {})
    if method not in table:
        raise SystemExit(f"no selected learning rate for {model_short}/{method} in {path}")
    return float(table[method])


# ---------------------------------------------------------------------------
# Hub upload (optional)
# ---------------------------------------------------------------------------
def push_folder(repo_id: str, folder: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    for attempt in range(3):
        try:
            api.upload_folder(repo_id=repo_id, folder_path=str(folder), path_in_repo=path_in_repo)
            return
        except Exception as e:  # noqa: BLE001
            print(f"[hub] upload failed (attempt {attempt + 1}): {e!r}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not upload {folder} to {repo_id}")


# ---------------------------------------------------------------------------
# Resume helper: load trainable weights from a stage-end checkpoint
# ---------------------------------------------------------------------------
def load_trainable_weights(model, method: str, ckpt: Path) -> None:
    from safetensors.torch import load_file

    if method == "full":
        sd = load_file(str(ckpt / "model.safetensors"))
        missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        missing = [k for k in missing if k != "lm_head.weight"]
        if missing or unexpected:
            raise RuntimeError(f"resume mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    elif method == "lora":
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, load_file(str(ckpt / "adapter_model.safetensors")))
    elif method == "adapter":
        files = list(ckpt.glob("pytorch_adapter.*"))
        if not files:
            raise FileNotFoundError(f"no adapter weights in {ckpt}")
        f = files[0]
        sd = load_file(str(f)) if f.suffix == ".safetensors" else torch.load(f, map_location="cpu")
        own = model.state_dict()
        missing = [k for k in sd if k not in own]
        if missing:
            raise RuntimeError(f"adapter keys not in model: {missing[:5]}")
        model.load_state_dict(sd, strict=False)
    else:
        raise ValueError(method)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run(args) -> dict:
    exp = load_experiment_config(args.config)
    tcfg, dcfg = exp["training"], exp["dev_eval"]
    mcfg = resolve_model(exp, args.model)

    if args.stages:
        order = args.stages.split(",")
        order_name = args.order_name or "".join(d[0].upper() for d in order)
    else:
        order = exp["orders"][args.order]
        order_name = args.order
    lr = args.lr if args.lr is not None else selected_lr(mcfg["short"], args.method)
    run_id = args.run_id or f"{mcfg['short']}_{args.method}_{order_name}_s{args.seed}"
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "raw" / run_id
    ckpt_root = Path(args.ckpt_root) / run_id
    fractions = [float(x) for x in args.ckpt_fracs.split(",")] if args.ckpt_fracs else tcfg["checkpoint_fractions"]
    stage1_fractions = (
        [float(x) for x in args.stage1_ckpt_fracs.split(",")] if args.stage1_ckpt_fracs else fractions
    )
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume_stage:
        if not args.overwrite:
            raise SystemExit(f"{out_dir} already has results; use --overwrite or --resume_stage")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(tcfg["fp16"]) and device.type == "cuda"

    # ---- data ------------------------------------------------------------
    def limit(rows, n):
        return rows[:n] if n else rows

    train_rows = {d: limit(read_jsonl(data_dir / "train" / f"{d}.jsonl"), args.train_n) for d in order}
    dev = {
        "medmcqa": limit(read_jsonl(data_dir / "dev" / "medmcqa.jsonl"), args.dev_n),
        "casehold": limit(read_jsonl(data_dir / "dev" / "casehold.jsonl"), args.dev_n),
        "code": limit(read_jsonl(data_dir / "dev" / "code.jsonl"), args.dev_n),
        "mbpp": limit(read_jsonl(data_dir / "dev" / "mbpp_validation.jsonl"), args.dev_n),
    }

    # ---- model -----------------------------------------------------------
    set_seed(args.seed)
    tok, model = load_base(mcfg["id"], mcfg.get("revision", "main"), dtype=torch.float32)
    model, pinfo = apply_method(model, args.method, exp)
    model.to(device)
    if tcfg["gradient_checkpointing"]:
        enable_gradient_checkpointing(model, args.method)

    ckpt_meta_base = {
        "base_model": mcfg["id"], "base_revision": mcfg.get("revision", "main"), "run_id": run_id,
        "seed": args.seed, "lr": lr, "order": order,
    }
    meta = {
        "run_id": run_id, "method": args.method, "seed": args.seed, "lr": lr,
        "model": mcfg, "model_revision_sha": resolve_revision_sha(mcfg["id"], mcfg.get("revision", "main")),
        "order": order, "order_name": order_name, "checkpoint_fractions": fractions,
        "stage1_checkpoint_fractions": stage1_fractions, "params": pinfo, "training": tcfg,
        "dev_eval": dcfg, "lora": exp["lora"], "adapter": exp["adapter"], "fp16": use_fp16,
        "pilot_limits": {"train_n": args.train_n, "dev_n": args.dev_n},
        "data_sha256": read_json(data_dir / "manifest.json")["sha256"] if (data_dir / "manifest.json").exists() else None,
        "environment": environment_info(), "argv": sys.argv,
        "resumed_from_stage": args.resume_stage or None,
    }
    write_json(out_dir / ("run_meta.json" if not args.resume_stage else f"run_meta_resume{args.resume_stage}.json"), meta)
    print(json.dumps({k: meta[k] for k in ("run_id", "method", "lr", "order", "params")}, indent=2))

    summary = {"run_id": run_id, "stages": [], "checkpoints": []}

    def evaluate_dev(stage_domain: str | None, with_mbpp: bool) -> dict:
        t0 = time.time()
        was_training = model.training
        model.eval()
        res = {
            "medmcqa_dev": mcq_accuracy(model, tok, dev["medmcqa"], dcfg["max_length"], dcfg["batch_size"]),
            "casehold_dev": mcq_accuracy(model, tok, dev["casehold"], dcfg["max_length"], dcfg["batch_size"]),
            "code_dev": response_loss(model, tok, dev["code"], dcfg["max_length"], dcfg["batch_size"]),
        }
        if with_mbpp:
            res["mbpp_dev"] = mbpp_pass_at_1(
                model, tok, dev["mbpp"], dcfg["mbpp_max_new_tokens"], dcfg["mbpp_timeout_s"], dcfg["batch_size"]
            )
        if was_training:
            model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        res["_seconds"] = time.time() - t0
        return res

    def record_checkpoint(name: str, stage: int, domain: str | None, frac: float, step: int,
                          cum_step: int, dev_res: dict, ckpt_dir: Path | None) -> None:
        line = {
            "checkpoint": name, "stage": stage, "domain": domain, "fraction": frac, "step": step,
            "cumulative_step": cum_step, "medmcqa_dev_acc": dev_res["medmcqa_dev"]["acc"],
            "casehold_dev_acc": dev_res["casehold_dev"]["acc"], "code_dev_loss": dev_res["code_dev"]["loss"],
            "mbpp_dev_pass_at_1": dev_res.get("mbpp_dev", {}).get("pass_at_1"),
            "dev_eval_seconds": round(dev_res["_seconds"], 1),
            "checkpoint_dir": str(ckpt_dir) if ckpt_dir else None,
        }
        _jsonl_append(out_dir / "dev_metrics.jsonl", line)
        preds_path = out_dir / "dev_preds" / f"{name}.jsonl"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preds_path, "w") as f:
            for key in ("medmcqa_dev", "casehold_dev"):
                for p in dev_res[key]["preds"]:
                    f.write(json.dumps({"set": key, **p}) + "\n")
            for p in dev_res.get("mbpp_dev", {}).get("per_task", []):
                f.write(json.dumps({"set": "mbpp_dev", **p}) + "\n")
        summary["checkpoints"].append(line)
        print(f"[{name}] medmcqa={line['medmcqa_dev_acc']:.3f} casehold={line['casehold_dev_acc']:.3f} "
              f"code_loss={line['code_dev_loss']:.3f} mbpp={line['mbpp_dev_pass_at_1']}")

    def maybe_push(ckpt_dir: Path, name: str) -> None:
        if args.hub_repo and ckpt_dir is not None:
            push_folder(args.hub_repo, ckpt_dir, f"{run_id}/{name}")
            if args.delete_after_push:
                shutil.rmtree(ckpt_dir, ignore_errors=True)

    # ---- stage 0 -----------------------------------------------------------
    start_stage = 1
    cum_step = 0
    if args.resume_stage:
        resume_dir = Path(args.resume_ckpt)
        load_trainable_weights(model, args.method, resume_dir)
        start_stage = args.resume_stage + 1
        print(f"resumed from {resume_dir}; continuing with stage {start_stage}")
    else:
        name = "stage0"
        ckpt_dir = write_base_marker(ckpt_root / name, {**ckpt_meta_base, "stage": 0, "step": 0})
        dev_res = evaluate_dev(None, with_mbpp=dcfg["mbpp_dev_on_code_stage"] and "code" in order)
        record_checkpoint(name, 0, None, 0.0, 0, 0, dev_res, ckpt_dir)
        maybe_push(ckpt_dir, name)

    # ---- sequential stages ------------------------------------------------
    from transformers import Trainer, TrainerCallback, TrainingArguments

    collator = PadCollator(tok.pad_token_id)
    for stage, domain in enumerate(order, start=1):
        if stage < start_stage:
            continue
        feats = [
            tokenize_example(tok, r, tcfg["max_length"], tcfg.get("medmcqa_include_explanation", True))
            for r in train_rows[domain]
        ]
        n_tokens = sum(len(f["input_ids"]) for f in feats)
        n_resp = sum(sum(1 for x in f["labels"] if x != -100) for f in feats)
        stage_fracs = stage1_fractions if stage == 1 else fractions
        stage_state = {"eval_seconds": 0.0, "save_seconds": 0.0, "steps": []}

        class CheckpointCallback(TrainerCallback):
            def on_train_begin(self, targs, state, control, **kw):
                stage_state["steps"] = checkpoint_steps(state.max_steps, stage_fracs)
                stage_state["max_steps"] = state.max_steps

            def on_step_end(self, targs, state, control, model=None, **kw):
                if state.global_step not in stage_state["steps"]:
                    return
                frac = state.global_step / stage_state["max_steps"]
                pct = int(round(100 * frac))
                name = f"stage{stage}_{domain}_{pct:03d}"
                ckpt_dir = None
                t0 = time.time()
                if not args.no_save:
                    ckpt_dir = save_checkpoint(model, args.method, ckpt_root / name, {
                        **ckpt_meta_base, "stage": stage, "domain": domain, "step": state.global_step,
                        "fraction": frac,
                    })
                stage_state["save_seconds"] += time.time() - t0
                with_mbpp = bool(dcfg["mbpp_dev_on_code_stage"]) and domain == "code"
                dev_res = evaluate_dev(domain, with_mbpp)
                stage_state["eval_seconds"] += dev_res["_seconds"]
                record_checkpoint(name, stage, domain, frac, state.global_step,
                                  cum_step + state.global_step, dev_res, ckpt_dir)
                t0 = time.time()
                if ckpt_dir is not None:
                    maybe_push(ckpt_dir, name)
                stage_state["save_seconds"] += time.time() - t0

        targs = TrainingArguments(
            output_dir=str(out_dir / "_trainer_tmp"),
            per_device_train_batch_size=tcfg["per_device_train_batch_size"],
            gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
            num_train_epochs=args.epochs if args.epochs else tcfg["epochs_per_stage"],
            max_steps=args.max_steps if args.max_steps else -1,
            learning_rate=lr,
            lr_scheduler_type=tcfg["lr_scheduler"],
            warmup_ratio=tcfg["warmup_ratio"],
            weight_decay=tcfg["weight_decay"],
            max_grad_norm=tcfg["max_grad_norm"],
            optim=tcfg["optimizer"],
            fp16=use_fp16,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            disable_tqdm=args.quiet,
        )
        trainer = Trainer(
            model=model, args=targs, train_dataset=ListDataset(feats), data_collator=collator,
            callbacks=[CheckpointCallback()],
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        trainer.train()
        wall = time.time() - t_start
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None

        for entry in trainer.state.log_history:
            _jsonl_append(out_dir / "train_log.jsonl", {"stage": stage, "domain": domain, **entry})
        stage_info = {
            "stage": stage, "domain": domain, "examples": len(feats), "tokens": n_tokens,
            "response_tokens": n_resp, "max_steps": stage_state.get("max_steps"),
            "checkpoint_steps": stage_state["steps"], "wall_seconds": round(wall, 1),
            "train_seconds": round(wall - stage_state["eval_seconds"] - stage_state["save_seconds"], 1),
            "dev_eval_seconds": round(stage_state["eval_seconds"], 1),
            "save_upload_seconds": round(stage_state["save_seconds"], 1),
            "peak_gpu_memory_gib": round(peak, 3) if peak is not None else None,
            "final_train_loss": next(
                (e["train_loss"] for e in reversed(trainer.state.log_history) if "train_loss" in e), None
            ),
        }
        summary["stages"].append(stage_info)
        print(f"[stage {stage} {domain}] {stage_info}")
        cum_step += stage_state.get("max_steps") or 0
        write_json(out_dir / ("run_summary.json" if not args.resume_stage else
                              f"run_summary_resume{args.resume_stage}.json"), summary)
        del trainer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    shutil.rmtree(out_dir / "_trainer_tmp", ignore_errors=True)
    if args.hub_repo and not args.no_push_results:
        push_folder(args.hub_repo, out_dir, f"results/{run_id}")
    print(f"done: {out_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["full", "lora", "adapter"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="primary", help="'primary', 'replication' or a HF id / local path")
    ap.add_argument("--order", default="MLC", help="key in configs/experiment.yaml:orders")
    ap.add_argument("--stages", default=None, help="explicit comma-separated domains (overrides --order)")
    ap.add_argument("--order_name", default=None)
    ap.add_argument("--lr", type=float, default=None, help="default: configs/lr_selected.json")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--ckpt_root", default=str(Path("checkpoints")))
    ap.add_argument("--ckpt_fracs", default=None, help="e.g. 0.25,0.5,0.75,1.0")
    ap.add_argument("--stage1_ckpt_fracs", default=None, help="denser checkpoints in stage 1 (H2 matching)")
    ap.add_argument("--hub_repo", default=None, help="private HF repo for checkpoints and logs")
    ap.add_argument("--delete_after_push", action="store_true")
    ap.add_argument("--no_push_results", action="store_true")
    ap.add_argument("--no_save", action="store_true", help="do not write checkpoints (LR search)")
    ap.add_argument("--resume_stage", type=int, default=0, help="last completed stage")
    ap.add_argument("--resume_ckpt", default=None, help="checkpoint dir of that stage's 100%% point")
    ap.add_argument("--overwrite", action="store_true")
    # pilot / debugging only - never used for reported runs
    ap.add_argument("--train_n", type=int, default=0)
    ap.add_argument("--dev_n", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
