"""Development-set evaluation run inside training (never on test data).

  mcq_accuracy  - MedMCQA-dev / CaseHOLD-dev: log-likelihood of each answer letter after
                  "...### Response:\\nThe answer is", the same template used in training.
  response_loss - mean response-token cross-entropy on the CodeAlpaca dev set.
  mbpp_pass_at_1 - greedy generation on the 90 MBPP validation tasks, executed against the
                  unit tests in a subprocess with a timeout.

These numbers drive learning-rate selection and checkpoint matching only.
"""
from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

from templates import ANSWER_PREFIX, LETTERS, mbpp_dev_prompt
from tokenize_data import fit_mcq_prompt, tokenize_example


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _left_pad(seqs: list[list[int]], pad_id: int, device):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s)
        mask[i, width - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)


@torch.no_grad()
def mcq_accuracy(model, tok, items: list[dict], max_length: int = 512, batch_size: int = 8) -> dict:
    device = next(model.parameters()).device
    n_opt = len(items[0]["options"])
    letter_ids = [tok(" " + L, add_special_tokens=False).input_ids for L in LETTERS[:n_opt]]
    if any(len(t) != 1 for t in letter_ids):
        raise ValueError(f"answer letters are not single tokens: {letter_ids}")
    letter_ids = torch.tensor([t[0] for t in letter_ids], device=device)

    contexts = []
    for it in items:
        prompt_ids = fit_mcq_prompt(tok, it, max_length - 8)
        contexts.append(prompt_ids + tok(ANSWER_PREFIX, add_special_tokens=False).input_ids)

    correct, preds = 0, []
    for start in range(0, len(items), batch_size):
        batch = contexts[start:start + batch_size]
        ids, mask, pos = _left_pad(batch, tok.pad_token_id, device)
        with _autocast(device):
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -1, :]
        scores = F.log_softmax(logits.float(), dim=-1)[:, letter_ids]
        pred = scores.argmax(-1).tolist()
        for j, p in enumerate(pred):
            it = items[start + j]
            preds.append({"uid": it["uid"], "pred": int(p), "gold": int(it["label"])})
            correct += int(p == it["label"])
    return {"acc": correct / len(items), "n": len(items), "preds": preds}


@torch.no_grad()
def response_loss(model, tok, items: list[dict], max_length: int = 512, batch_size: int = 8) -> dict:
    device = next(model.parameters()).device
    total_loss, total_tokens = 0.0, 0
    feats = [tokenize_example(tok, it, max_length) for it in items]
    for start in range(0, len(feats), batch_size):
        batch = feats[start:start + batch_size]
        width = max(len(f["input_ids"]) for f in batch)
        ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, f in enumerate(batch):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            lab[i, :n] = torch.tensor(f["labels"])
            mask[i, :n] = 1
        ids, lab, mask = ids.to(device), lab.to(device), mask.to(device)
        with _autocast(device):
            logits = model(input_ids=ids, attention_mask=mask).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = lab[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += int((shift_labels != -100).sum())
    return {"loss": total_loss / max(total_tokens, 1), "tokens": total_tokens, "n": len(items)}


# ---------------------------------------------------------------------------
# MBPP validation pass@1
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(completion: str) -> str:
    m = _FENCE.search(completion)
    code = m.group(1) if m else completion
    return code.split("### Instruction:")[0].strip()


def run_tests(code: str, setup: str, tests: list[str], timeout: float) -> bool:
    program = "\n".join([setup, code, *tests]) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            res = subprocess.run(
                [sys.executable, "-I", path], cwd=tmp, capture_output=True, timeout=timeout,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            return False
    return res.returncode == 0


@torch.no_grad()
def mbpp_pass_at_1(model, tok, tasks: list[dict], max_new_tokens: int = 384, timeout: float = 10.0,
                   batch_size: int = 8) -> dict:
    device = next(model.parameters()).device
    was_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    old_side = tok.padding_side
    tok.padding_side = "left"
    results = []
    try:
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            prompts = [mbpp_dev_prompt(t["text"], t["test_list"]) for t in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
            with _autocast(device):
                out = model.generate(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
            completions = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for t, comp in zip(batch, completions):
                ok = run_tests(extract_code(comp), t["test_setup_code"], t["test_list"], timeout)
                results.append({"uid": t["uid"], "passed": ok})
    finally:
        tok.padding_side = old_side
        model.config.use_cache = was_cache
    passed = sum(r["passed"] for r in results)
    return {"pass_at_1": passed / len(results), "n": len(results), "per_task": results}
