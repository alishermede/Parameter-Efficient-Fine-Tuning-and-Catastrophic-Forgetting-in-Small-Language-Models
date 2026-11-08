# Parameter-Efficient Fine-Tuning and Catastrophic Forgetting in Small Language Models

Research Methods (Astana IT University), Assignment 4 — Rakhat Dosmagambetov, Danila Bussygin, Alisher Medebayev, group SE-2414.

**Research question.** To what extent do LoRA and adapters retain the general knowledge and reasoning performance of small language models compared with full fine-tuning during sequential domain adaptation?

Each run adapts one base model with one method through three domains in sequence (medicine → law → programming by default). General knowledge (MMLU) and reasoning (GSM8K) are measured before adaptation and after every stage; development-set scores are logged at 25/50/75/100% of every stage.

| | |
|---|---|
| Base models | `Qwen/Qwen2.5-0.5B` (primary, 3 seeds), `Qwen/Qwen3-0.6B-Base` (replication, 1 seed) |
| Methods | full fine-tuning; LoRA r=16, α=32 on q,k,v,o,gate,up,down; sequential bottleneck adapter after each FFN block, sized to LoRA's parameter count (±5%) |
| Domains | MedMCQA → CaseHOLD → CodeAlpaca-20k, 2,000 training examples each, 2 epochs |
| Retention | MMLU (1,260 questions, 42 non-overlapping subjects), GSM8K (500 problems) — evaluated with lm-evaluation-harness (`src/eval.py`) |
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
src/plan_eval.py          H2 matched checkpoints (dev data only) and evaluation job lists
src/eval.py               test-set evaluation of a checkpoint with lm-eval, answers stored per item
src/stats.py              forgetting, retention, paired bootstrap + Holm, hypothesis verdicts, all tables
src/figures.py            report figures from the tables
src/training_report.py    training-log tables and figures (efficiency, dev trajectories, LR search, data pipeline)
tasks/casehold.yaml       custom lm-eval task for the CaseHOLD test subset
configs/eval.yaml         frozen evaluation settings
notebooks/kaggle_eval.ipynb runner for test-set evaluation on Kaggle
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
pip uninstall -y torchao               # Kaggle's torchao 0.10 breaks peft 0.21; not used here
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

| Account | Runs | ≈ GPU-h |
|---|---|---|
| Participant 1 | LR search (done); all 9 primary runs: `full`, `lora`, `adapter` × seeds 1, 2, 3 (MLC) — three sessions, one method each with `SEED = [1, 2, 3]` | 10–12 |
| Participant 2 | 3 replication runs: `full`, `lora`, `adapter`, seed 1, `--model replication`; test-set evaluation (share) | 4 + eval |
| Participant 3 | 3 reversed-order runs: `full`, `lora`, `adapter`, seed 1, `--order CLM`; test-set evaluation (share); reproduction test | 4 + eval |

**5. Collect logs** from all accounts into `results/`:
```bash
python src/pull_results.py --repos <hf-user1>/rmt-peft-ckpts <hf-user2>/rmt-peft-ckpts <hf-user3>/rmt-peft-ckpts
```

**6. Test-set evaluation (Participant 2 and 3).** Full test sets at Stage 0 (once per base model), at the end of each stage and at the H2 matched checkpoint. `notebooks/kaggle_eval.ipynb` runs the same commands on 2 × T4.
```bash
python src/prepare_data.py                                   # rebuilds data/test/ and checks hashes
python src/plan_eval.py --methods full,lora --stage0 --name participant2 --gpus 2
                                                             # matched checkpoints (dev data only) + job lists
python src/eval.py --run_id q25-05b_full_MLC_s1 --checkpoint stage0 --pilot 50     # pilot, never reported
python src/eval.py --jobs results/eval/jobs/participant2_gpu0.txt                  # one worker per GPU
```
Settings are frozen in `configs/eval.yaml` (lm-eval 0.4.13; MMLU 5-shot and GSM8K 5-shot greedy as in the freeze form; MedMCQA/CaseHOLD 0-shot log-likelihood; MBPP 3-shot pass@1). CaseHOLD is a custom task, `tasks/casehold.yaml`. Every answer is stored per item:

| File in `results/eval/<run_id>/<checkpoint>/` | Content |
|---|---|
| `mmlu_primary.jsonl`, `mmlu_overlap.jsonl`, `gsm8k.jsonl`, `medmcqa.jsonl`, `casehold.jsonl`, `mbpp.jsonl` | one line per test item: `item_id`, `pred`, `gold`, `correct` (+ subject, log-likelihoods; GSM8K: strict-match and raw response) |
| `lm_eval_<group>.json` | lm-eval aggregate output |
| `summary.json` | scores, timing, checkpoint metadata, environment, subset hashes |

Stage 0 is stored once as `results/eval/<model>_base/stage0/`.

**7. Statistics, tables and figures** — all report numbers come from these two commands:
```bash
python src/training_report.py                                # training logs -> efficiency, dev trajectories, LR search, data pipeline (fig4-6)
python src/stats.py && python src/figures.py                 # -> results/tables/, results/figures/
python src/stats.py --model replication && python src/stats.py --order CLM   # descriptive, 1 seed
python tests/test_stats.py                                   # offline check of the pipeline on synthetic data
```
`results/tables/report_numbers.md` lists every key number with its source table.

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

## 8. Replication (Qwen3-0.6B-Base)

Test-set results of the three replication runs are in `results/eval/q3-06b_*` (Stage 0, end of every stage,
and the LoRA matched checkpoint `stage1_medmcqa_050`, deviation 13). Run `python src/replication.py` to produce
`results/tables/replication_*.csv` and `replication_numbers.md` (one seed: scores, forgetting, retention, paired item
bootstrap with Holm, matched-checkpoint retention). Generation used the frozen 256-token limit and batch 16 (deviation 14): `python src/eval.py --jobs <file> --hub_repo Rakhat3110/rmt-peft-ckpts --strict_gen_cap --gen_batch_size 16` (or run `notebooks/kaggle_eval_replication.ipynb`).
