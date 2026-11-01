"""Check that the bottleneck adapter matches LoRA's trainable-parameter count (+-5%).

Offline mode builds the two models from their published architecture values with random
weights on the 'meta' device (no download). On Kaggle run with --from_hub to use the real
configs:  python tests/check_param_matching.py --from_hub
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config  # noqa: E402

from common import load_experiment_config  # noqa: E402
from modeling import apply_method, lora_param_count  # noqa: E402

OFFLINE_CONFIGS = {
    # Qwen2.5-0.5B: config.json of Qwen/Qwen2.5-0.5B
    "Qwen/Qwen2.5-0.5B": dict(
        model_type="qwen2", hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
        num_attention_heads=14, num_key_value_heads=2, vocab_size=151936, tie_word_embeddings=True,
    ),
    # Qwen3-0.6B-Base: config.json of Qwen/Qwen3-0.6B-Base
    "Qwen/Qwen3-0.6B-Base": dict(
        model_type="qwen3", hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128, vocab_size=151936,
        tie_word_embeddings=True,
    ),
}


def build(model_id: str, from_hub: bool):
    if from_hub:
        cfg = AutoConfig.from_pretrained(model_id)
    else:
        kw = dict(OFFLINE_CONFIGS[model_id])
        mtype = kw.pop("model_type")
        if mtype == "qwen2":
            cfg = Qwen2Config(**kw)
        else:
            from transformers import Qwen3Config

            cfg = Qwen3Config(**kw)
    model_cls = AutoModelForCausalLM._model_mapping[type(cfg)]  # import outside the meta context
    with torch.device("meta"):
        return model_cls(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_hub", action="store_true")
    args = ap.parse_args()
    exp = load_experiment_config()
    ok = True
    for model_id in OFFLINE_CONFIGS:
        rows = {}
        for method in ("lora", "adapter"):
            model = build(model_id, args.from_hub)
            target = lora_param_count(model, exp["lora"]["r"], exp["lora"]["targets"])
            model, info = apply_method(model, method, exp, check_identity=False)  # meta device: no forward
            rows[method] = info
        lo, ad = rows["lora"], rows["adapter"]
        diff = (ad["trainable"] - lo["trainable"]) / lo["trainable"]
        ok &= abs(diff) <= exp["adapter"]["match_tolerance"] and lo["trainable"] == target
        print(f"{model_id}: total={lo['total'] - lo['trainable']:,}  "
              f"LoRA={lo['trainable']:,} ({lo['trainable_pct']:.2f}%)  "
              f"adapter={ad['trainable']:,} ({ad['trainable_pct']:.2f}%, bottleneck {ad['adapter_bottleneck']})  "
              f"diff={diff:+.3%}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
