"""Turning records into token ids with response-only loss masking.

Loss is computed only on the response tokens (prompt tokens get label -100), and an EOS
token closes every response. Long CaseHOLD excerpts are shortened from the start of the
excerpt so that prompt + response fit `max_length` without cutting the answer options.
"""
from __future__ import annotations

from templates import code_prompt, mcq_prompt, mcq_response


def _ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False).input_ids


def fit_mcq_prompt(tok, item: dict, budget: int) -> list[int]:
    """Prompt ids of an MCQ item, trimming the question/excerpt from its start if needed."""
    question = item["question"]
    ids = _ids(tok, mcq_prompt(item["domain"], question, item["options"]))
    if len(ids) <= budget:
        return ids
    words = question.split()

    def build(k: int) -> list[int]:  # keep the last k words of the excerpt
        return _ids(tok, mcq_prompt(item["domain"], "... " + " ".join(words[len(words) - k:]), item["options"]))

    lo, hi, best = 0, len(words), build(0)
    while lo <= hi:  # largest k whose prompt fits the budget
        mid = (lo + hi) // 2
        cand = build(mid)
        if len(cand) <= budget:
            best, lo = cand, mid + 1
        else:
            hi = mid - 1
    return best[-budget:] if len(best) > budget else best


def prompt_and_response(item: dict, include_explanation: bool = True) -> tuple[str, str]:
    if item["domain"] in ("medmcqa", "casehold"):
        exp = item.get("explanation") if include_explanation else None
        return (mcq_prompt(item["domain"], item["question"], item["options"]),
                mcq_response(item["options"], item["label"], exp))
    if item["domain"] == "code":
        return code_prompt(item["instruction"], item.get("input")), item["output"]
    raise ValueError(item["domain"])


def tokenize_example(tok, item: dict, max_length: int, include_explanation: bool = True) -> dict:
    prompt, response = prompt_and_response(item, include_explanation)
    resp_ids = _ids(tok, response) + [tok.eos_token_id]
    if item["domain"] in ("medmcqa", "casehold"):
        # keep at least the answer line of the response, trim the excerpt first
        answer_line = _ids(tok, response.split("\n")[0])
        budget = max_length - min(len(resp_ids), max(len(answer_line) + 1, max_length // 4))
        prompt_ids = fit_mcq_prompt(tok, item, budget)
    else:
        prompt_ids = _ids(tok, prompt)
        if len(prompt_ids) > max_length // 2:
            prompt_ids = prompt_ids[: max_length // 2]
    resp_ids = resp_ids[: max_length - len(prompt_ids)]
    input_ids = prompt_ids + resp_ids
    labels = [-100] * len(prompt_ids) + resp_ids
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


class PadCollator:
    """Right padding for training; labels padded with -100."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]):
        import torch

        width = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = width - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
