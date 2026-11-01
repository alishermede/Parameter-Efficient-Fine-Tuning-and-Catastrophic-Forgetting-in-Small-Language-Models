"""Loaders that turn each Hugging Face dataset into a list of normalised records.

Every record has a stable `uid` so that subsets can be stored as ID lists and hashed.
Schemas are checked explicitly: if a dataset revision changes its columns, loading fails
with a clear message instead of silently producing different data.
"""
from __future__ import annotations

from typing import Callable

# ---------------------------------------------------------------------------
# MMLU subject lists (Hendrycks et al., 2021)
# ---------------------------------------------------------------------------
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
]

# Subjects in the health, law and computer-science subcategories overlap the adaptation
# domains; they are excluded from the primary score and reported separately (A3, Sec. 3).
MMLU_OVERLAP = {
    "health": ["anatomy", "clinical_knowledge", "college_medicine", "human_aging",
               "medical_genetics", "nutrition", "professional_medicine", "virology"],
    "law": ["international_law", "jurisprudence", "professional_law"],
    "computer_science": ["college_computer_science", "computer_security",
                         "high_school_computer_science", "machine_learning"],
}
MMLU_OVERLAP_SUBJECTS = sorted(s for group in MMLU_OVERLAP.values() for s in group)
MMLU_PRIMARY_SUBJECTS = sorted(s for s in MMLU_SUBJECTS if s not in MMLU_OVERLAP_SUBJECTS)

# Four top-level categories used for the subgroup analysis.
MMLU_CATEGORY = {}
for _s in ["abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
           "college_computer_science", "college_mathematics", "college_physics",
           "computer_security", "conceptual_physics", "electrical_engineering",
           "elementary_mathematics", "high_school_biology", "high_school_chemistry",
           "high_school_computer_science", "high_school_mathematics", "high_school_physics",
           "high_school_statistics", "machine_learning"]:
    MMLU_CATEGORY[_s] = "stem"
for _s in ["formal_logic", "high_school_european_history", "high_school_us_history",
           "high_school_world_history", "international_law", "jurisprudence",
           "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
           "prehistory", "professional_law", "world_religions"]:
    MMLU_CATEGORY[_s] = "humanities"
for _s in ["econometrics", "high_school_geography", "high_school_government_and_politics",
           "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
           "human_sexuality", "professional_psychology", "public_relations",
           "security_studies", "sociology", "us_foreign_policy"]:
    MMLU_CATEGORY[_s] = "social_sciences"
for _s in ["business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
           "human_aging", "management", "marketing", "medical_genetics", "miscellaneous",
           "nutrition", "professional_accounting", "professional_medicine", "virology"]:
    MMLU_CATEGORY[_s] = "other"

assert len(MMLU_SUBJECTS) == 57, len(MMLU_SUBJECTS)
assert len(MMLU_OVERLAP_SUBJECTS) == 15
assert len(MMLU_PRIMARY_SUBJECTS) == 42
assert set(MMLU_CATEGORY) == set(MMLU_SUBJECTS)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
LoadFn = Callable[..., object]


def _load(load_dataset: LoadFn, repo: str, config: str | None, split: str, revision: str):
    if config:
        return load_dataset(repo, config, split=split, revision=revision)
    return load_dataset(repo, split=split, revision=revision)


def _require(ds, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in ds.column_names]
    if missing:
        raise ValueError(
            f"{name}: expected columns {columns}, missing {missing}; found {ds.column_names}. "
            "The dataset revision may have changed - pin `revision` in configs/data.yaml."
        )


def resolve_revision(repo: str, revision: str) -> str | None:
    """Commit hash of a dataset revision (None if the Hub cannot be reached)."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo, revision=revision).sha
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Domain datasets
# ---------------------------------------------------------------------------
def load_medmcqa(load_dataset: LoadFn, src: dict, split: str) -> list[dict]:
    """split: 'train' (training pool) or 'validation' (our test pool)."""
    ds = _load(load_dataset, src["repo"], src.get("config"), split, src["revision"])
    _require(ds, ["id", "question", "opa", "opb", "opc", "opd", "cop", "exp", "subject_name"], "MedMCQA")
    out = []
    for i, row in enumerate(ds):
        label = row["cop"]
        if label is None or int(label) not in (0, 1, 2, 3):
            continue
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]
        if any(o is None or not str(o).strip() for o in options) or not str(row["question"]).strip():
            continue
        out.append({
            "uid": f"medmcqa-{row['id']}",
            "doc_index": i,
            "domain": "medmcqa",
            "question": str(row["question"]),
            "options": [str(o) for o in options],
            "label": int(label),
            "explanation": str(row["exp"] or ""),
            "group": str(row["subject_name"] or "unknown"),
        })
    return out


def load_casehold(load_dataset: LoadFn, src: dict, split: str) -> list[dict]:
    try:
        ds = _load(load_dataset, src["repo"], src.get("config"), split, src["revision"])
        _require(ds, ["context", "endings", "label"], "CaseHOLD (lex_glue)")
        rows = [(r["context"], list(r["endings"]), r["label"]) for r in ds]
        origin = src["repo"]
    except Exception as primary_error:  # noqa: BLE001
        if not src.get("fallback_repo"):
            raise
        print(f"[casehold] primary source failed ({primary_error!r}); trying {src['fallback_repo']}")
        ds = _load(load_dataset, src["fallback_repo"], src.get("fallback_config"), split, "main")
        cols = ["citing_prompt", "holding_0", "holding_1", "holding_2", "holding_3", "holding_4", "label"]
        _require(ds, cols, "CaseHOLD (casehold/casehold)")
        rows = [(r["citing_prompt"], [r[f"holding_{i}"] for i in range(5)], r["label"]) for r in ds]
        origin = src["fallback_repo"]
    out = []
    for i, (context, endings, label) in enumerate(rows):
        if label is None or str(label).strip() == "":
            continue
        label = int(label)
        if label not in range(5) or len(endings) != 5:
            continue
        out.append({
            "uid": f"casehold-{split}-{i}",
            "doc_index": i,
            "domain": "casehold",
            "question": str(context),
            "options": [str(e) for e in endings],
            "label": label,
            "group": "casehold",
            "source": origin,
        })
    return out


def load_codealpaca(load_dataset: LoadFn, src: dict) -> list[dict]:
    ds = _load(load_dataset, src["repo"], src.get("config"), "train", src["revision"])
    _require(ds, ["instruction", "input", "output"], "CodeAlpaca-20k")
    out = []
    for i, row in enumerate(ds):
        if not str(row["instruction"]).strip() or not str(row["output"]).strip():
            continue
        out.append({
            "uid": f"codealpaca-{i}",
            "domain": "code",
            "instruction": str(row["instruction"]),
            "input": str(row["input"] or ""),
            "output": str(row["output"]),
        })
    return out


def load_mbpp(load_dataset: LoadFn, src: dict, split: str) -> list[dict]:
    try:
        ds = _load(load_dataset, src["repo"], src.get("config"), split, src["revision"])
    except Exception as e:  # noqa: BLE001  legacy alias
        print(f"[mbpp] {src['repo']} failed ({e!r}); trying legacy id 'mbpp'")
        ds = _load(load_dataset, "mbpp", src.get("config"), split, "main")
    _require(ds, ["task_id", "text", "code", "test_list", "test_setup_code"], "MBPP")
    return [{
        "uid": f"mbpp-{row['task_id']}",
        "domain": "code",
        "task_id": int(row["task_id"]),
        "text": str(row["text"]),
        "code": str(row["code"]),
        "test_list": list(row["test_list"]),
        "test_setup_code": str(row["test_setup_code"] or ""),
    } for row in ds]


# ---------------------------------------------------------------------------
# Retention benchmarks (only needed for test subsets and decontamination)
# ---------------------------------------------------------------------------
def load_mmlu_subject(load_dataset: LoadFn, src: dict, subject: str) -> list[dict]:
    ds = load_dataset(src["repo"], subject, split="test", revision=src["revision"])
    _require(ds, ["question", "choices", "answer"], f"MMLU/{subject}")
    return [{
        "uid": f"mmlu-{subject}-{i}",
        "subject": subject,
        "doc_index": i,  # row index inside lm-eval's per-subject task
        "question": str(row["question"]),
        "options": [str(c) for c in row["choices"]],
        "label": int(row["answer"]),
    } for i, row in enumerate(ds)]


def load_gsm8k_test(load_dataset: LoadFn, src: dict) -> list[dict]:
    ds = _load(load_dataset, src["repo"], src.get("config"), "test", src["revision"])
    _require(ds, ["question", "answer"], "GSM8K")
    return [{
        "uid": f"gsm8k-test-{i}",
        "doc_index": i,
        "question": str(row["question"]),
        "answer": str(row["answer"]),
    } for i, row in enumerate(ds)]


def item_text(item: dict) -> str:
    """All text of an item that could leak into training data."""
    parts = []
    for key in ("question", "instruction", "input", "output", "text", "code", "answer", "explanation"):
        if item.get(key):
            parts.append(str(item[key]))
    parts.extend(str(o) for o in item.get("options", []))
    parts.extend(str(t) for t in item.get("test_list", []))
    return "\n".join(parts)
