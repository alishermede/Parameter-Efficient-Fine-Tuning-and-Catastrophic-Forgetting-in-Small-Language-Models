"""End-to-end smoke test without network access or GPU.

Builds a tiny random Qwen2 model and a tiny BPE tokenizer, writes synthetic data in the
exact format of prepare_data.py, then for every method: runs three sequential stages,
checks checkpoints, reloads them with load_checkpoint(), verifies that the adapter /
LoRA really changes the forward pass, and resumes a run from a stage-end checkpoint.

  python tests/smoke_test.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from common import read_jsonl, write_jsonl  # noqa: E402

LETTERS = "ABCDE"


def make_tokenizer(path: Path, corpus: list[str]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    fast.save_pretrained(path)
    return fast


def make_model(path: Path, vocab: int):
    from transformers import AutoModelForCausalLM, Qwen2Config

    cfg = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                      max_position_embeddings=1024)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)


def synth_mcq(domain: str, n: int, n_opt: int, rng: random.Random, prefix: str) -> list[dict]:
    rows = []
    for i in range(n):
        label = rng.randrange(n_opt)
        rows.append({
            "uid": f"{prefix}-{i}", "domain": domain, "doc_index": i,
            "question": f"Question {i} about topic {rng.randrange(9)} with some words " * rng.randint(1, 6),
            "options": [f"option {LETTERS[k]} text {rng.randrange(50)}" for k in range(n_opt)],
            "label": label, "explanation": "Because reasons " * rng.randint(0, 5),
            "group": f"subj{i % 3}",
        })
    return rows


def synth_code(n: int, prefix: str) -> list[dict]:
    return [{"uid": f"{prefix}-{i}", "domain": "code", "instruction": f"Write a function f{i} that adds two numbers.",
             "input": "" if i % 2 else "a=1, b=2", "output": f"def f{i}(a, b):\n    return a + b"} for i in range(n)]


def synth_mbpp(n: int) -> list[dict]:
    return [{"uid": f"mbpp-{511 + i}", "domain": "code", "task_id": 511 + i,
             "text": "Write a function to add two numbers.", "code": "def add(a, b):\n    return a + b",
             "test_list": ["assert add(1, 2) == 3"], "test_setup_code": ""} for i in range(n)]


def main() -> int:
    from dev_eval import run_tests
    from modeling import load_checkpoint

    assert run_tests("def add(a, b):\n    return a + b", "", ["assert add(1, 2) == 3"], 5)
    assert not run_tests("def add(a, b):\n    return a - b", "", ["assert add(1, 2) == 3"], 5)
    assert not run_tests("while True: pass", "", [], 1)
    print("mbpp execution harness: OK")

    tmp = Path(tempfile.mkdtemp(prefix="rmt_smoke_"))
    try:
        rng = random.Random(0)
        data = tmp / "data"
        train = {"medmcqa": synth_mcq("medmcqa", 64, 4, rng, "med"),
                 "casehold": synth_mcq("casehold", 64, 5, rng, "case"),
                 "code": synth_code(64, "ca")}
        dev = {"medmcqa": synth_mcq("medmcqa", 12, 4, rng, "medd"),
               "casehold": synth_mcq("casehold", 12, 5, rng, "cased"),
               "code": synth_code(8, "cad"), "mbpp_validation": synth_mbpp(4)}
        for d, rows in train.items():
            write_jsonl(data / "train" / f"{d}.jsonl", rows)
        for d, rows in dev.items():
            write_jsonl(data / "dev" / f"{d}.jsonl", rows)

        corpus = []
        for rows in list(train.values()) + list(dev.values()):
            for r in rows:
                corpus.append(json.dumps(r))
        corpus += [f"### Instruction:\n### Response:\nThe answer is {L}. Explanation:" for L in LETTERS] * 50
        tok_dir = tmp / "tiny"
        tok = make_tokenizer(tok_dir, corpus)
        make_model(tok_dir, len(tok))

        exp = yaml.safe_load(open(ROOT / "configs" / "experiment.yaml"))
        exp["models"]["tiny"] = {"id": str(tok_dir), "revision": "main", "short": "tiny"}
        exp["dev_eval"]["mbpp_max_new_tokens"] = 24
        exp["dev_eval"]["mbpp_timeout_s"] = 3
        exp["adapter"]["match_tolerance"] = 0.25  # tiny model: integer bottleneck is coarse
        cfg_path = tmp / "experiment.yaml"
        yaml.safe_dump(exp, open(cfg_path, "w"))

        import train as T

        for method in ("full", "lora", "adapter"):
            out = tmp / "results" / method
            args = T.build_parser().parse_args([
                "--method", method, "--seed", "1", "--model", "tiny", "--order", "MLC",
                "--lr", "1e-3", "--config", str(cfg_path), "--data_dir", str(data), "--out", str(out),
                "--ckpt_root", str(tmp / "ckpts"), "--epochs", "1", "--quiet",
            ])
            summary = T.run(args)
            metrics = read_jsonl(out / "dev_metrics.jsonl")
            names = [m["checkpoint"] for m in metrics]
            assert names[0] == "stage0" and len(names) == 1 + 3 * 4, names
            assert [s["domain"] for s in summary["stages"]] == ["medmcqa", "casehold", "code"]
            assert all(m["mbpp_dev_pass_at_1"] is not None for m in metrics if m["domain"] in (None, "code"))
            run_id = summary["run_id"]
            last = tmp / "ckpts" / run_id / "stage3_code_100"
            assert last.exists(), last

            # reload the final checkpoint and compare with the untouched base model
            _, trained = load_checkpoint(last, dtype=torch.float32)
            _, base = load_checkpoint(tmp / "ckpts" / run_id / "stage0", dtype=torch.float32)
            x = torch.tensor([tok("The answer is A.").input_ids])
            delta = (trained(x).logits - base(x).logits).abs().max().item()
            assert delta > 1e-4, f"{method}: checkpoint does not change the forward pass ({delta})"
            if method == "adapter":
                trained.set_active_adapters(None)
                d_off = (trained(x).logits - base(x).logits).abs().max().item()
                assert d_off < 1e-5, f"adapter inactive output differs from base ({d_off})"
            print(f"{method}: 13 checkpoints, reload OK, max |logit delta| vs base = {delta:.4f}")

            # resume from the end of stage 1 and run stages 2-3 again
            res_out = tmp / "results" / f"{method}_resume"
            args = T.build_parser().parse_args([
                "--method", method, "--seed", "1", "--model", "tiny", "--order", "MLC",
                "--lr", "1e-3", "--config", str(cfg_path), "--data_dir", str(data), "--out", str(res_out),
                "--ckpt_root", str(tmp / "ckpts_resume"), "--epochs", "1", "--quiet",
                "--resume_stage", "1", "--resume_ckpt", str(tmp / "ckpts" / run_id / "stage1_medmcqa_100"),
            ])
            T.run(args)
            res_metrics = read_jsonl(res_out / "dev_metrics.jsonl")
            assert res_metrics[0]["checkpoint"] == "stage2_casehold_025", res_metrics[0]
            print(f"{method}: resume from stage 1 OK")

        # LR-search style run: stage 1 only, no checkpoints written
        out = tmp / "results" / "lrsearch"
        args = T.build_parser().parse_args([
            "--method", "lora", "--seed", "0", "--model", "tiny", "--stages", "medmcqa", "--lr", "1e-3",
            "--config", str(cfg_path), "--data_dir", str(data), "--out", str(out),
            "--ckpt_root", str(tmp / "ckpts_lr"), "--epochs", "1", "--ckpt_fracs", "1.0", "--no_save", "--quiet",
        ])
        T.run(args)
        assert [m["checkpoint"] for m in read_jsonl(out / "dev_metrics.jsonl")] == ["stage0", "stage1_medmcqa_100"]
        print("lr-search mode: OK")
        print("SMOKE TEST PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
