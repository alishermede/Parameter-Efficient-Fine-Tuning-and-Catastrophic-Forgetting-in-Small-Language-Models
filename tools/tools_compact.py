import base64, glob, json, os
os.makedirs("/kaggle/working/compact", exist_ok=True)

def h(s):
    x = 5381
    for ch in s:
        x = ((x * 33) ^ ord(ch)) & 0xFFFFFFFF
    return x

def pack(a):
    b = bytearray((len(a) + 7) // 8)
    for i, v in enumerate(a):
        if v:
            b[i >> 3] |= 1 << (i & 7)
    return base64.b64encode(bytes(b)).decode()

dirs = sorted(set(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/results/eval/q25-05b_adapter_*/*/summary.json", recursive=True)))
print("found", len(dirs), "adapter checkpoints")
for d in dirs:
    run, ckpt = d.split("/")[-2:]
    s = json.load(open(f"{d}/summary.json")); s.pop("eval_config", None)
    out = {"run": run, "ckpt": ckpt, "summary": s}
    for t in ["mmlu_primary", "mmlu_overlap", "gsm8k", "medmcqa", "casehold", "mbpp"]:
        rows = [json.loads(l) for l in open(f"{d}/{t}.jsonl")]
        o = {"n": len(rows), "h": h("|".join(r["item_id"] for r in rows)), "c": pack([int(r["correct"]) for r in rows])}
        if t == "gsm8k":
            o["cs"] = pack([int(r["correct_strict"]) for r in rows])
        if t.startswith("mmlu"):
            o["p"] = "".join(str(r["pred"]) for r in rows)
        out[t] = o
    line = json.dumps(out, separators=(",", ":"))
    open(f"/kaggle/working/compact/{run}__{ckpt}.txt", "w").write(line)
    print(run, ckpt, len(line), {k: round(v["score"], 4) for k, v in s.get("scores", {}).items()})
