"""Rebuild per-item jsonl for adapter checkpoints from compact grabs (tools_fetch.js output).

Item order/ids, gold and subject come from a reference checkpoint evaluated on the same items
(item_id order verified by hash). Only the fields stats.py reads are written.
"""
import base64, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "results/eval/q25-05b_full_MLC_s1/stage3_code_100"


def h(s: str) -> int:
    x = 5381
    for ch in s:
        x = ((x * 33) ^ ord(ch)) & 0xFFFFFFFF
    return x


def unpack(b64: str, n: int) -> list[int]:
    b = base64.b64decode(b64)
    return [(b[i >> 3] >> (i & 7)) & 1 for i in range(n)]


def rebuild(grab: dict) -> Path:
    out = ROOT / "results/eval" / grab["run"] / grab["ckpt"]
    out.mkdir(parents=True, exist_ok=True)
    for t in ["mmlu_primary", "mmlu_overlap", "gsm8k", "medmcqa", "casehold", "mbpp"]:
        g = grab.get(t)
        if g is None:
            raise SystemExit(f"{grab['run']}/{grab['ckpt']}: missing {t}")
        ref = [json.loads(l) for l in open(REF / f"{t}.jsonl")]
        ids = [r["item_id"] for r in ref]
        assert len(ids) == g["n"], (t, len(ids), g["n"])
        assert h("|".join(ids)) == g["h"], f"item order differs for {t}"
        c = unpack(g["c"], g["n"])
        cs = unpack(g["cs"], g["n"]) if "cs" in g else None
        with open(out / f"{t}.jsonl", "w") as f:
            for i, r in enumerate(ref):
                row = {"item_id": r["item_id"], "correct": c[i], "gold": r.get("gold")}
                if cs is not None:
                    row["correct_strict"] = cs[i]
                if t.startswith("mmlu"):
                    row["subject"] = r["subject"]
                    row["pred"] = g["p"][i]
                f.write(json.dumps(row) + "\n")
    summ = grab.get("summary") or {}
    summ.setdefault("run_id", grab["run"]); summ.setdefault("checkpoint", grab["ckpt"])
    summ["note"] = "rebuilt from Kaggle output (correctness only; responses/logliks in eval_results.zip)"
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        for line in open(p):
            if line.strip():
                print(rebuild(json.loads(line)))
