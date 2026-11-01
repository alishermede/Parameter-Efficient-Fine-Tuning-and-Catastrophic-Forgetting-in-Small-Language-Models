"""Build every data subset used in the study and record SHA-256 hashes.

Outputs (all deterministic for a fixed dataset revision and seed):
  data/train/{medmcqa,casehold,code}.jsonl     2,000 training examples per domain
  data/dev/{medmcqa,casehold,code}.jsonl       200 development examples per domain
  data/dev/mbpp_validation.jsonl               90 MBPP validation tasks (code dev, pass@1)
  data/test_subsets/*.json                     ID lists of every test subset (for lm-eval)
  data/test/{medmcqa,casehold}_test.jsonl      full text of the two domain test sets
  data/manifest.json                           sizes, decontamination counts, revisions, hashes
  data/SHA256SUMS                              hashes of all files above

Usage:
  python src/prepare_data.py            # build and write
  python src/prepare_data.py --verify   # rebuild in memory and compare with data/SHA256SUMS
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA_DIR, environment_info, load_data_config, sha256_file, write_json, write_jsonl  # noqa: E402
import data_sources as S  # noqa: E402

_WORD = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Decontamination
# ---------------------------------------------------------------------------
def ngrams(text: str, n: int) -> set[int]:
    words = _WORD.findall(text.lower())
    return {hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)}


class NgramIndex:
    def __init__(self, n: int):
        self.n = n
        self.grams: set[int] = set()

    def add_items(self, items: list[dict]) -> None:
        for it in items:
            self.grams |= ngrams(S.item_text(it), self.n)

    def contaminated(self, item: dict) -> bool:
        return not ngrams(S.item_text(item), self.n).isdisjoint(self.grams)


def decontaminate(pool: list[dict], index: NgramIndex) -> tuple[list[dict], int]:
    kept = [it for it in pool if not index.contaminated(it)]
    return kept, len(pool) - len(kept)


def dedupe(pool: list[dict], key) -> tuple[list[dict], int]:
    seen, out = set(), []
    for it in pool:
        k = re.sub(r"\s+", " ", key(it).strip().lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out, len(pool) - len(out)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k > len(items):
        raise ValueError(f"cannot sample {k} from {len(items)} items")
    idx = sorted(rng.sample(range(len(items)), k))
    return [items[i] for i in idx]


def stratified_sample(items: list[dict], k: int, rng: random.Random, key: str = "group") -> list[dict]:
    """Proportional allocation with largest remainder; deterministic for a given rng state."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[it[key]].append(it)
    names = sorted(groups)
    total = len(items)
    quotas = {g: k * len(groups[g]) / total for g in names}
    alloc = {g: min(int(quotas[g]), len(groups[g])) for g in names}
    remainder = k - sum(alloc.values())
    for g in sorted(names, key=lambda g: (-(quotas[g] - int(quotas[g])), g)):
        if remainder == 0:
            break
        if alloc[g] < len(groups[g]):
            alloc[g] += 1
            remainder -= 1
    out = []
    for g in names:
        if alloc[g]:
            out.extend(sample(groups[g], alloc[g], rng))
    return sorted(out, key=lambda it: it["uid"])


def uids(items: list[dict]) -> list[str]:
    return [it["uid"] for it in items]


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build(cfg: dict, load_dataset) -> dict:
    seed = cfg["seed"]
    n_train = cfg["sizes"]["train_per_domain"]
    n_dev = cfg["sizes"]["dev_per_domain"]
    src = cfg["sources"]
    stats: dict = {"seed": seed, "ngram": cfg["ngram"], "domains": {}}

    print("[1/4] loading sources")
    med_train_pool = S.load_medmcqa(load_dataset, src["medmcqa"], "train")
    med_test_pool = S.load_medmcqa(load_dataset, src["medmcqa"], "validation")
    case_train_pool = S.load_casehold(load_dataset, src["casehold"], "train")
    case_test_pool = S.load_casehold(load_dataset, src["casehold"], "test")
    code_pool = S.load_codealpaca(load_dataset, src["codealpaca"])
    mbpp_test = S.load_mbpp(load_dataset, src["mbpp"], "test")
    mbpp_val = S.load_mbpp(load_dataset, src["mbpp"], "validation")
    gsm8k_test = S.load_gsm8k_test(load_dataset, src["gsm8k"])
    mmlu = {s: S.load_mmlu_subject(load_dataset, src["mmlu"], s) for s in S.MMLU_SUBJECTS}

    print("[2/4] fixing test subsets")
    rng = random.Random(seed)
    mmlu_primary, mmlu_overlap = [], []
    for subj in S.MMLU_SUBJECTS:
        chosen = sample(mmlu[subj], cfg["sizes"]["mmlu_per_subject"], rng)
        (mmlu_overlap if subj in S.MMLU_OVERLAP_SUBJECTS else mmlu_primary).extend(chosen)
    gsm8k_sub = sample(gsm8k_test, cfg["sizes"]["test_gsm8k"], rng)
    med_test = stratified_sample(med_test_pool, cfg["sizes"]["test_medmcqa"], rng)
    case_test = sample(case_test_pool, cfg["sizes"]["test_casehold"], rng)

    test_sets = {
        "mmlu_primary": mmlu_primary,
        "mmlu_overlap": mmlu_overlap,
        "gsm8k": gsm8k_sub,
        "medmcqa_test": med_test,
        "casehold_test": case_test,
        "mbpp_test": mbpp_test,
    }

    print("[3/4] decontaminating and sampling train/dev")
    test_index = NgramIndex(cfg["ngram"])
    for items in test_sets.values():
        test_index.add_items(items)
    test_index.add_items(mbpp_val)  # code dev set must not leak into training either

    def make_split(name, pool, key, stratify):
        d = {"pool_raw": len(pool)}
        pool, d["removed_duplicates"] = dedupe(pool, key)
        pool, d["removed_test_overlap"] = decontaminate(pool, test_index)
        r = random.Random(f"{seed}-{name}")
        pick = stratified_sample if stratify else sample
        dev = pick(pool, n_dev, r)
        dev_ids = set(uids(dev))
        rest = [it for it in pool if it["uid"] not in dev_ids]
        dev_index = NgramIndex(cfg["ngram"])
        dev_index.add_items(dev)
        rest, d["removed_dev_overlap"] = decontaminate(rest, dev_index)
        train = pick(rest, n_train, r)
        d["pool_clean"] = len(rest)
        d["train"], d["dev"] = len(train), len(dev)
        if stratify:
            d["train_groups"] = len({it["group"] for it in train})
        stats["domains"][name] = d
        return train, dev

    med_train, med_dev = make_split("medmcqa", med_train_pool, lambda it: it["question"], True)
    case_train, case_dev = make_split("casehold", case_train_pool, lambda it: it["question"], False)
    code_train, code_dev = make_split(
        "code", code_pool, lambda it: it["instruction"] + "\n" + it["input"], False
    )

    stats["test_sizes"] = {k: len(v) for k, v in test_sets.items()}
    stats["mbpp_validation"] = len(mbpp_val)
    return {
        "train": {"medmcqa": med_train, "casehold": case_train, "code": code_train},
        "dev": {"medmcqa": med_dev, "casehold": case_dev, "code": code_dev, "mbpp_validation": mbpp_val},
        "test_sets": test_sets,
        "stats": stats,
    }


def test_subset_record(name: str, items: list[dict]) -> dict:
    rec = {"name": name, "n": len(items), "uids": uids(items)}
    if items and "doc_index" in items[0]:
        if "subject" in items[0]:
            by_task: dict[str, list[int]] = defaultdict(list)
            for it in items:
                by_task[f"mmlu_{it['subject']}"].append(it["doc_index"])
            rec["lm_eval_doc_indices"] = {k: sorted(v) for k, v in sorted(by_task.items())}
        else:
            task = {"gsm8k": "gsm8k", "medmcqa_test": "medmcqa_validation_split",
                    "casehold_test": "casehold_test_split"}.get(name, name)
            rec["lm_eval_doc_indices"] = {task: sorted(it["doc_index"] for it in items)}
    if name == "mbpp_test":
        rec["task_ids"] = sorted(it["task_id"] for it in items)
    return rec


def write_outputs(built: dict, out_dir: Path, cfg: dict, revisions: dict) -> dict[str, str]:
    files: dict[str, Path] = {}
    for dom, rows in built["train"].items():
        files[f"train/{dom}.jsonl"] = out_dir / "train" / f"{dom}.jsonl"
        write_jsonl(files[f"train/{dom}.jsonl"], rows)
    for dom, rows in built["dev"].items():
        files[f"dev/{dom}.jsonl"] = out_dir / "dev" / f"{dom}.jsonl"
        write_jsonl(files[f"dev/{dom}.jsonl"], rows)
    for name, items in built["test_sets"].items():
        p = out_dir / "test_subsets" / f"{name}.json"
        write_json(p, test_subset_record(name, items))
        files[f"test_subsets/{name}.json"] = p
    # Full text of the two domain test sets, so custom lm-eval tasks can read local JSONL
    for name in ("medmcqa_test", "casehold_test"):
        p = out_dir / "test" / f"{name}.jsonl"
        write_jsonl(p, built["test_sets"][name])
        files[f"test/{name}.jsonl"] = p
    # ID lists of train/dev (committed to git even if the text files are not)
    for split in ("train", "dev"):
        for dom, rows in built[split].items():
            p = out_dir / "ids" / f"{split}_{dom}.json"
            write_json(p, uids(rows))
            files[f"ids/{split}_{dom}.json"] = p
    p = out_dir / "mmlu_categories.json"
    write_json(p, {"category": S.MMLU_CATEGORY, "overlap_groups": S.MMLU_OVERLAP})
    files["mmlu_categories.json"] = p

    hashes = {rel: sha256_file(path) for rel, path in sorted(files.items())}
    manifest = {
        "config": cfg,
        "resolved_revisions": revisions,
        "stats": built["stats"],
        "sha256": hashes,
        "environment": environment_info(),
    }
    write_json(out_dir / "manifest.json", manifest)
    with open(out_dir / "SHA256SUMS", "w") as f:
        for rel, h in hashes.items():
            f.write(f"{h}  {rel}\n")
    return hashes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(DATA_DIR))
    ap.add_argument("--verify", action="store_true", help="rebuild and compare with existing SHA256SUMS")
    args = ap.parse_args(argv)

    from datasets import load_dataset

    cfg = load_data_config(args.config)
    revisions = {k: S.resolve_revision(v["repo"], v["revision"]) for k, v in cfg["sources"].items()}
    print("resolved dataset revisions:", revisions)
    built = build(cfg, load_dataset)

    out = Path(args.out)
    if args.verify:
        reference = {}
        with open(out / "SHA256SUMS") as f:
            for line in f:
                h, rel = line.strip().split("  ", 1)
                reference[rel] = h
        with tempfile.TemporaryDirectory() as tmp:
            hashes = write_outputs(built, Path(tmp), cfg, revisions)
        diff = sorted(rel for rel in reference if reference[rel] != hashes.get(rel))
        if diff:
            print("MISMATCH in:", *diff, sep="\n  ")
            return 1
        print(f"OK: all {len(reference)} files reproduce exactly")
        return 0

    reference_text = (out / "SHA256SUMS").read_text() if (out / "SHA256SUMS").exists() else None
    hashes = write_outputs(built, out, cfg, revisions)
    if reference_text is not None:
        reference = dict(reversed(line.split("  ", 1)) for line in reference_text.strip().splitlines())
        diff = sorted(rel for rel in reference if reference[rel] != hashes.get(rel))
        if diff:
            (out / "SHA256SUMS").write_text(reference_text)  # keep the committed reference
            print("ERROR: rebuilt files differ from the committed data/SHA256SUMS:", *diff, sep="\n  ")
            print("Check dataset revisions in configs/data.yaml (see data/manifest.json).")
            return 1
        print(f"verified: all {len(reference)} files match the committed SHA256SUMS")
    print("[4/4] written to", out)
    for k, v in built["stats"]["domains"].items():
        print(f"  {k}: {v}")
    print("  test sizes:", built["stats"]["test_sizes"])
    print(f"  {len(hashes)} files hashed -> {out / 'SHA256SUMS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
