"""Offline test of src/stats.py and src/figures.py on SYNTHETIC per-item files.

Writes fake evaluation outputs in the exact format of src/eval.py into a temporary copy of
results/, with a known forgetting pattern (FFT forgets most), and checks that the pipeline
recovers it. Nothing written here is a result; the temporary directory is deleted.

  python tests/test_stats.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import read_json, write_json, write_jsonl  # noqa: E402

TRUE_S = {  # (metric file -> stage -> method -> mean accuracy)
    "mmlu": {0: .47, 1: {"full": .44, "lora": .465, "adapter": .46}, 2: {"full": .42, "lora": .46, "adapter": .455},
             3: {"full": .39, "lora": .455, "adapter": .45}},
    "gsm8k": {0: .36, 1: {"full": .30, "lora": .34, "adapter": .33}, 2: {"full": .26, "lora": .33, "adapter": .32},
              3: {"full": .20, "lora": .32, "adapter": .30}},
}


def fake(root: Path) -> None:
    rng = np.random.default_rng(0)
    subs = {n: read_json(ROOT / "data" / "test_subsets" / f"{n}.json") for n in
            ["mmlu_primary", "mmlu_overlap", "gsm8k", "medmcqa_test", "casehold_test", "mbpp_test"]}
    diff = {n: rng.normal(0, 1.2, len(s["uids"])) for n, s in subs.items()}   # item difficulty

    def recs(name, p, kind):
        uids = subs[name]["uids"]
        logit = np.log(p / (1 - p)) - diff[name] + rng.normal(0, 0.6, len(uids))
        c = (rng.random(len(uids)) < 1 / (1 + np.exp(-logit))).astype(int)
        out = []
        for u, ci in zip(uids, c):
            r = {"item_id": u, "correct": int(ci), "doc_id": 0}
            if kind == "mmlu":
                gold = "ABCD"[hash(u) % 4]
                r.update(subject=u.split("-")[1], gold=gold, pred=gold if ci else "ABCD"[(hash(u) + 1) % 4])
            if kind == "gsm8k":
                r.update(correct_strict=int(ci and rng.random() < .9), pred="1", gold="1")
            out.append(r)
        return out

    def write(run, ckpt, stage, method):
        d = root / "eval" / run / ckpt
        get = (lambda k: TRUE_S[k][0]) if stage == 0 else (lambda k: TRUE_S[k][stage][method])
        write_jsonl(d / "mmlu_primary.jsonl", recs("mmlu_primary", get("mmlu"), "mmlu"))
        write_jsonl(d / "mmlu_overlap.jsonl", recs("mmlu_overlap", get("mmlu") - .02, "mmlu"))
        write_jsonl(d / "gsm8k.jsonl", recs("gsm8k", get("gsm8k"), "gsm8k"))
        dom = {0: (.36, .25, .28), 1: (.45, .26, .27), 2: (.43, .55, .26), 3: (.42, .5, .33)}[stage]
        write_jsonl(d / "medmcqa.jsonl", recs("medmcqa_test", dom[0], ""))
        write_jsonl(d / "casehold.jsonl", recs("casehold_test", dom[1], ""))
        write_jsonl(d / "mbpp.jsonl", recs("mbpp_test", dom[2], ""))
        write_json(d / "summary.json", {"synthetic": True})

    write("q25-05b_base", "stage0", 0, "base")
    for m in ("full", "lora", "adapter"):
        for s in (1, 2, 3):
            run = f"q25-05b_{m}_MLC_s{s}"
            for k, name in enumerate(["stage1_medmcqa_100", "stage2_casehold_100", "stage3_code_100"], 1):
                write(run, name, k, m)
            for extra in ("stage1_medmcqa_025", "stage1_medmcqa_075"):
                write(run, extra, 1, m)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", "eval", "figures"))
        for d in ("tables",):
            shutil.rmtree(repo / "results" / d, ignore_errors=True)
        fake(repo / "results")
        py = [sys.executable]
        subprocess.run(py + ["src/plan_eval.py", "--name", "test"], cwd=repo, check=True, capture_output=True)
        subprocess.run(py + ["src/stats.py", "--boot", "2000"], cwd=repo, check=True)
        subprocess.run(py + ["src/figures.py"], cwd=repo, check=True)
        t = repo / "results" / "tables"
        import pandas as pd

        pw = pd.read_csv(t / "pairwise_tests.csv")
        r = pw[(pw.metric == "mmlu_primary") & (pw.comparison == "lora - full")].iloc[0]
        assert r.diff_F_pp < -3 and r.p_holm < 0.05, r
        hyp = pd.read_csv(t / "hypotheses.csv")
        print(hyp[["hypothesis", "verdict"]])
        assert hyp.iloc[0].verdict in ("Supported", "Partly supported")
        for f in ["fig1_forgetting_trajectories.png", "fig2_gain_vs_retention.png", "fig3_mmlu_confusion.png"]:
            assert (repo / "results" / "figures" / f).stat().st_size > 10_000
        out = ROOT / "tests" / "_synthetic_preview"
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(repo / "results" / "figures", out)
        shutil.copy(t / "report_numbers.md", out / "report_numbers.md")
        print("OK - synthetic pipeline test passed; previews (NOT results) in", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
