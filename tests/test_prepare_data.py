"""Offline test of prepare_data.py with synthetic datasets that mimic the real schemas.

Checks: subset sizes, determinism (identical SHA-256 on rebuild), train/dev/test
disjointness, 13-gram decontamination, stratification, lm-eval index export.
  python tests/test_prepare_data.py
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset  # noqa: E402

import data_sources as S  # noqa: E402
import prepare_data as P  # noqa: E402
from common import load_data_config, read_json, read_jsonl  # noqa: E402

LEAK = "the patient presented with a rare unilateral painless swelling of the parotid gland region today"


def words(rng, n):
    return " ".join(f"w{rng.randrange(5000)}" for _ in range(n))


def fake_load_dataset(repo, config=None, split=None, revision=None):
    rng = random.Random(f"{repo}-{config}-{split}")
    if repo == "openlifescienceai/medmcqa":
        n = 3000 if split == "train" else 800
        rows = []
        for i in range(n):
            q = words(rng, 20)
            if split == "validation" and i == 5:
                q = LEAK  # this test item will be sampled or not; leak is planted below too
            rows.append({"id": f"{split}{i}", "question": q, "opa": "a", "opb": "b", "opc": "c", "opd": "d",
                         "cop": rng.randrange(4), "exp": words(rng, 10),
                         "subject_name": f"Subject{i % 7 if i % 11 else 0}", "choice_type": "single"})
        if split == "train":
            rows[0]["question"] = rows[1]["question"]  # exact duplicate
        return Dataset.from_list(rows)
    if repo == "coastalcph/lex_glue":
        n = 3000 if split == "train" else 700
        return Dataset.from_list([{"context": words(rng, 60) + " (<HOLDING>) " + words(rng, 20),
                                   "endings": [words(rng, 12) for _ in range(5)],
                                   "label": rng.randrange(5)} for _ in range(n)])
    if repo == "sahil2801/CodeAlpaca-20k":
        return Dataset.from_list([{"instruction": words(rng, 12), "input": "" if i % 3 else words(rng, 4),
                                   "output": "def f():\n    return " + words(rng, 5)} for i in range(3000)])
    if repo == "google-research-datasets/mbpp":
        start, n = {"test": (11, 500), "validation": (511, 90)}[split]
        return Dataset.from_list([{"task_id": start + i, "text": words(rng, 15), "code": "def g(): pass",
                                   "test_list": ["assert g() is None"], "test_setup_code": "",
                                   "challenge_test_list": []} for i in range(n)])
    if repo == "cais/mmlu":
        return Dataset.from_list([{"question": words(rng, 15), "subject": config, "choices": ["1", "2", "3", "4"],
                                   "answer": rng.randrange(4)} for _ in range(100)])
    if repo == "openai/gsm8k":
        return Dataset.from_list([{"question": words(rng, 30), "answer": "#### 4"} for _ in range(1319)])
    raise ValueError(repo)


def main() -> int:
    cfg = load_data_config()
    built = P.build(cfg, fake_load_dataset)
    st = built["stats"]
    print("stats:", st["domains"])
    assert st["test_sizes"] == {"mmlu_primary": 1260, "mmlu_overlap": 450, "gsm8k": 500,
                                "medmcqa_test": 500, "casehold_test": 500, "mbpp_test": 500}, st["test_sizes"]
    for d in ("medmcqa", "casehold", "code"):
        assert len(built["train"][d]) == 2000 and len(built["dev"][d]) == 200
        tr, dv = set(P.uids(built["train"][d])), set(P.uids(built["dev"][d]))
        assert not tr & dv, d
    assert st["domains"]["medmcqa"]["removed_duplicates"] >= 1
    assert len(built["dev"]["mbpp_validation"]) == 90
    test_uids = {u for items in built["test_sets"].values() for u in P.uids(items)}
    for d in ("medmcqa", "casehold"):
        assert not set(P.uids(built["train"][d])) & test_uids

    # decontamination: plant the leak in a training example and rebuild the index
    idx = P.NgramIndex(13)
    idx.add_items([{"question": LEAK}])
    assert idx.contaminated({"question": "Q: " + LEAK + " which nerve?"})
    assert not idx.contaminated({"question": "the patient presented with a rare swelling"})

    # stratification: subject proportions of train close to the pool
    from collections import Counter

    c = Counter(it["group"] for it in built["train"]["medmcqa"])
    assert len(c) == 7 and max(c.values()) - min(c.values()) < 400, c

    # determinism of written files
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        h1 = P.write_outputs(built, Path(a), cfg, {})
        h2 = P.write_outputs(P.build(cfg, fake_load_dataset), Path(b), cfg, {})
        assert h1 == h2, [k for k in h1 if h1[k] != h2.get(k)]
        rec = read_json(Path(a) / "test_subsets" / "mmlu_primary.json")
        assert rec["n"] == 1260 and len(rec["lm_eval_doc_indices"]) == 42
        assert all(len(v) == 30 for v in rec["lm_eval_doc_indices"].values())
        assert set(rec["lm_eval_doc_indices"]) == {f"mmlu_{s}" for s in S.MMLU_PRIMARY_SUBJECTS}
        g = read_json(Path(a) / "test_subsets" / "gsm8k.json")
        assert len(g["lm_eval_doc_indices"]["gsm8k"]) == 500
        assert len(read_jsonl(Path(a) / "test" / "medmcqa_test.jsonl")) == 500
        print(f"{len(h1)} files, hashes identical on rebuild")
    print("PREPARE_DATA TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
