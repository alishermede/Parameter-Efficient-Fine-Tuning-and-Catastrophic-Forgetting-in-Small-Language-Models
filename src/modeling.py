"""Model loading, the three adaptation methods, and checkpoint save/load.

Methods (one set of trainable weights carried through all stages):
  full    - every weight is updated
  lora    - LoRA (Hu et al., 2022), r=16, alpha=32, on all attention and MLP projections
  adapter - sequential bottleneck adapter after each feed-forward block (Pfeiffer config),
            bottleneck width chosen so trainable parameters match LoRA within +-5%

`load_checkpoint()` returns a ready-to-evaluate model for any saved checkpoint; the
evaluation code (eval.py, lm-eval HFLM) should use it so all methods load identically.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

CKPT_META = "checkpoint_meta.json"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
def load_base(model_id: str, revision: str = "main", dtype=torch.float32, device: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    model.config.use_cache = False
    if device:
        model.to(device)
    return tok, model


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": 100.0 * trainable / total}


# ---------------------------------------------------------------------------
# Parameter matching between LoRA and the bottleneck adapter
# ---------------------------------------------------------------------------
def lora_param_count(model: nn.Module, r: int, targets: list[str]) -> int:
    n = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            n += r * (module.in_features + module.out_features)
    if n == 0:
        raise ValueError(f"no Linear modules named {targets} found")
    return n


def adapter_bottleneck_for(model: nn.Module, target_params: int) -> int:
    """Width b of a down/up bottleneck adapter (with biases) per layer: params = L * (2*h*b + b + h)."""
    h = model.config.hidden_size
    layers = model.config.num_hidden_layers
    b = (target_params / layers - h) / (2 * h + 1)
    return max(1, int(round(b)))


# ---------------------------------------------------------------------------
# Applying a method
# ---------------------------------------------------------------------------
@torch.no_grad()
def _probe_logits(model: nn.Module) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = torch.arange(3, 19, device=device).unsqueeze(0) % model.config.vocab_size
    was_training = model.training
    model.eval()
    out = model(input_ids=ids).logits.float().cpu()
    if was_training:
        model.train()
    return out


def apply_method(model: nn.Module, method: str, cfg: dict, check_identity: bool = True):
    """Returns (model, info). `cfg` is the experiment config (needs `lora` and `adapter`).

    With check_identity, verifies that the freshly initialised method leaves the base model's
    output unchanged, so Stage 0 is the same model for every condition.
    """
    lcfg = cfg["lora"]
    before = _probe_logits(model) if check_identity else None
    target = lora_param_count(model, lcfg["r"], lcfg["targets"])
    info = {"method": method, "lora_equivalent_params": target}

    if method == "full":
        for p in model.parameters():
            p.requires_grad_(True)

    elif method == "lora":
        from peft import LoraConfig, get_peft_model

        pcfg = LoraConfig(
            r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
            target_modules=list(lcfg["targets"]), bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, pcfg)

    elif method == "adapter":
        import adapters
        from adapters import SeqBnConfig

        b = adapter_bottleneck_for(model, target)
        h = model.config.hidden_size
        # Adapters computes width as int(h // reduction_factor); h/(b+0.5) avoids float round-down.
        # init_weights="mam_adapter": down projection Kaiming-uniform, up projection zero - the same
        # scheme PEFT uses for LoRA (A Kaiming, B zero), so both start as the unchanged base model.
        acfg = SeqBnConfig(reduction_factor=h / (b + 0.5), non_linearity=cfg["adapter"]["non_linearity"],
                           init_weights=cfg["adapter"].get("init_weights", "mam_adapter"))
        adapters.init(model)
        model.add_adapter("domain", config=acfg)
        model.train_adapter("domain")
        info["adapter_bottleneck"] = b
        info["adapter_reduction_factor"] = h / (b + 0.5)

    else:
        raise ValueError(f"unknown method {method!r}")

    if check_identity:
        diff = (_probe_logits(model) - before).abs().max().item()
        info["init_max_logit_change"] = diff
        if diff > 1e-3:
            raise AssertionError(f"{method}: initialised model differs from the base model (max |dlogit| {diff:.4g})")
    info.update(count_params(model))
    if method in ("lora", "adapter"):
        rel = abs(info["trainable"] - target) / target
        info["param_match_rel_diff"] = rel
        if method == "adapter":
            tol = cfg["adapter"]["match_tolerance"]
            if rel > tol:
                raise AssertionError(
                    f"adapter has {info['trainable']} trainable params vs LoRA {target} "
                    f"({rel:.1%} > {tol:.0%})"
                )
    return model, info


def enable_gradient_checkpointing(model: nn.Module, method: str) -> None:
    kwargs = {"use_reentrant": False}
    base = model.get_base_model() if method == "lora" else model
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    base.config.use_cache = False


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, method: str, out_dir: str | Path, meta: dict) -> Path:
    """Saves only what differs from the base model; full fine-tuning is stored in fp16."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if method == "full":
        from safetensors.torch import save_file

        sd = {k: v.detach().to("cpu", torch.float16).contiguous() for k, v in model.state_dict().items()}
        if getattr(model.config, "tie_word_embeddings", False):
            sd.pop("lm_head.weight", None)
        save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
        cfg = model.config.to_dict()
        cfg["torch_dtype"] = "float16"
        cfg["use_cache"] = True
        with open(out / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(out)
    elif method == "lora":
        model.save_pretrained(out)
    elif method == "adapter":
        model.save_adapter(str(out), "domain")
    else:
        raise ValueError(method)
    with open(out / CKPT_META, "w") as f:
        json.dump({"method": method, **meta}, f, indent=2)
    return out


def load_checkpoint(ckpt_dir: str | Path, device: str | None = None, dtype=torch.float16, merge_lora: bool = True):
    """Load any checkpoint written by save_checkpoint (or `stage0` for the untouched base).

    Returns (tokenizer, model) in eval mode. LoRA is merged into the weights by default so
    every method runs through the same plain forward pass in lm-eval.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = Path(ckpt_dir)
    meta = json.loads((ckpt / CKPT_META).read_text())
    base_id, rev, method = meta["base_model"], meta.get("base_revision", "main"), meta["method"]
    tok = AutoTokenizer.from_pretrained(base_id, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if method == "full":
        model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_id, revision=rev, dtype=dtype)
        if method == "lora":
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, ckpt)
            if merge_lora:
                model = model.merge_and_unload()
        elif method == "adapter":
            import adapters

            adapters.init(model)
            name = model.load_adapter(str(ckpt), load_as="domain", set_active=True)
            model.set_active_adapters(name)
            model.to(dtype)  # adapter weights load as fp32; match the base dtype
        elif method == "base":
            pass
        else:
            raise ValueError(method)
    model.config.use_cache = True
    if device:
        model.to(device)
    model.eval()
    return tok, model


def write_base_marker(out_dir: str | Path, meta: dict) -> Path:
    """A 'checkpoint' directory for Stage 0 that just points at the base model."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / CKPT_META, "w") as f:
        json.dump({"method": "base", **meta}, f, indent=2)
    return out


def checkpoint_steps(max_steps: int, fractions: list[float]) -> list[int]:
    steps = sorted({max(1, min(max_steps, math.ceil(f * max_steps))) for f in fractions})
    if steps[-1] != max_steps:
        steps.append(max_steps)
    return steps
