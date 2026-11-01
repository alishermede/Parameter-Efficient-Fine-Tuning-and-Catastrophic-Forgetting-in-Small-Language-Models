# Parameter-Efficient Fine-Tuning and Catastrophic Forgetting in Small Language Models

Research Methods (Astana IT University), Assignment 4 — Rakhat Dosmagambetov, Danila Bussygin, Alisher Medebayev, group SE-2414.

**Research question.** To what extent do LoRA and adapters retain the general knowledge and reasoning performance of small language models compared with full fine-tuning during sequential domain adaptation?

Each run adapts one base model with one method through three domains in sequence (medicine → law → programming by default). General knowledge (MMLU) and reasoning (GSM8K) are measured before adaptation and after every stage; development-set scores are logged at 25/50/75/100% of every stage.

| | |
|---|---|
| Base models | `Qwen/Qwen2.5-0.5B` (primary, 3 seeds), `Qwen/Qwen3-0.6B-Base` (replication, 1 seed) |
| Methods | full fine-tuning; LoRA r=16, α=32 on q,k,v,o,gate,up,down; sequential bottleneck adapter after each FFN block, sized to LoRA's parameter count (±5%) |
| Domains | MedMCQA → CaseHOLD → CodeAlpaca-20k, 2,000 training examples each, 2 epochs |
| Retention | MMLU (1,260 questions, 42 non-overlapping subjects), GSM8K (500 problems) — evaluated with lm-evaluation-harness (`eval/`, Participant 2) |
| Hardware | one NVIDIA T4 (Kaggle), fp16 mixed precision |

## Repository structure

```
configs/experiment.yaml   models, domain orders, training hyper-parameters, LR grid, seeds
configs/data.yaml         dataset sources, revisions, subset sizes, decontamination n-gram
configs/lr_selected.json  learning rates chosen by src/lr_search.py (written after the search)
src/prepare_data.py       builds all train/dev/test subsets, 13-gram decontamination, SHA-256
src/data_sources.py       dataset loaders with schema checks; MMLU subject lists and categories
src/templates.py          the single prompt template shared by all methods
src/tokenize_data.py      tokenisation with response-only loss masking
src/modeling.py           the three methods, adapter/LoRA parameter matching, checkpoint save/load
src/dev_eval.py           development metrics used inside training (never test data)
src/train.py              one sequential run (model x method x seed x order)
src/lr_search.py          equal 3-value learning-rate budget per method on Stage-1 dev data
src/pull_results.py       collects run logs pushed to the Hub by the Kaggle sessions
notebooks/kaggle_run.ipynb  runner for Kaggle (pilot, LR search, final runs)
tests/                    offline smoke test, data-pipeline test, parameter-matching check
data/                     ID lists, manifest and SHA256SUMS (text files are regenerated, not committed)
results/raw/<run_id>/     logs of every run (see "Outputs")
docs/                     protocol freeze form, deviations log
```

## Installation

```bash
git clone https://github.com/alishermede/parameter-efficient-fine-tuning-and-catastrophic-forgetting-in-small-language-models.git
cd parameter-efficient-fine-tuning-and-catastrophic-forgetting-in-small-language-models
pip install -r requirements.txt        # torch: use the CUDA build already installed (Kaggle/Colab)
export HF_TOKEN=<token with write access>   # only needed to push checkpoints to the Hub
```

`adapters==1.3.0` requires `transformers~=4.57.6`; all versions are pinned in `requirements.txt`, and every run stores the exact library versions, GPU and git commit in `run_meta.json`.

## Reproducing the experiment

All commands are run from the repository root. On Kaggle the notebook `notebooks/kaggle_run.ipynb` runs the same commands (select **GPU T4 x2**; the notebook uses one GPU).

**1. Data (≈5 min, CPU).** Rebuilds every subset and checks it against the committed hashes:
```bash
python src/prepare_data.py
```

**2. Checks (CPU, no download needed for the first two).**
```bash
python tests/smoke_test.py                  # whole training pipeline on a tiny random model
python tests/test_prepare_data.py           # data pipeline on synthetic datasets
python tests/check_param_matching.py --from_hub   # adapter vs LoRA trainable parameters
```

**3. Learning-rate search (Stage 1 only, seed 0, MedMCQA-dev).**
```bash
python src/lr_search.py --methods full,lora,adapter --hub_repo <hf-user>/rmt-peft-ckpts
python src/lr_search.py --select            # writes configs/lr_selected.json
```

**4. Final runs (15).** One command per run:
```bash
python src/train.py --method <full|lora|adapter> --seed <1|2|3> --model primary --order MLC \
    --ckpt_root /kaggle/working/ckpts --hub_repo <hf-user>/rmt-peft-ckpts --delete_after_push
```

| Account | Runs |
|---|---|
| Participant 1 | LR search (9 short runs); `full` seeds 1, 2, 3 (primary, MLC) |
| Participant 2 | `lora` seeds 1, 2, 3 (primary, MLC); `full`, `lora`, `adapter` seed 1 with `--model replication` |
| Participant 3 | `adapter` seeds 1, 2, 3 (primary, MLC); `full`, `lora`, `adapter` seed 1 with `--order CLM` |

**5. Collect logs** from all accounts into `results/`:
```bash
python src/pull_results.py --repos <hf-user1>/rmt-peft-ckpts <hf-user2>/rmt-peft-ckpts <hf-user3>/rmt-peft-ckpts
```

**6. Test-set evaluation and statistics** — see `eval/README.md` (Participant 2).

If a Kaggle session stops mid-run, continue from the last completed stage (the run reseeds identically at every stage start):
```bash
python src/train.py ... --resume_stage 1 --resume_ckpt <ckpt_root>/<run_id>/stage1_medmcqa_100
```

## Outputs

`results/raw/<run_id>/` with `run_id = <model>_<method>_<order>_s<seed>`, e.g. `q25-05b_lora_MLC_s1`:

| File | Content |
|---|---|
| `run_meta.json` | full configuration, learning rate, parameter counts, library versions, GPU, git commit, data hashes |
| `dev_metrics.jsonl` | one line per checkpoint: MedMCQA-dev and CaseHOLD-dev accuracy, code-dev loss, MBPP-validation pass@1 |
| `dev_preds/<checkpoint>.jsonl` | per-item development predictions |
| `train_log.jsonl` | loss and learning rate every 10 optimiser steps |
| `run_summary.json` | per stage: training time, dev-evaluation time, peak GPU memory, steps, checkpoint steps |

Checkpoints (`stage0`, `stage<k>_<domain>_<pct>`) are pushed to the private Hub repo under `<run_id>/`. Load any of them for evaluation with:
```python
from modeling import load_checkpoint
tok, model = load_checkpoint("<dir>")   # full, LoRA (merged) or adapter; fp16, eval mode
```

## Data and licences

Only public data are used; no personal data are processed. MedMCQA (`openlifescienceai/medmcqa`), CaseHOLD (`coastalcph/lex_glue`, config `case_hold`), CodeAlpaca-20k (`sahil2801/CodeAlpaca-20k`), MBPP (`google-research-datasets/mbpp`), MMLU (`cais/mmlu`), GSM8K (`openai/gsm8k`). The resolved dataset revisions are recorded in `data/manifest.json`; licences are those stated on each dataset card at that revision. Training subsets are not redistributed: `src/prepare_data.py` regenerates them and verifies them against `data/SHA256SUMS`.

## Licence

Code: MIT (see `LICENSE`).
