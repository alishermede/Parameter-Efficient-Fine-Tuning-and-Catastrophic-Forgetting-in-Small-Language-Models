# Report sections — Participant 1

Sections 3, 4 and 9 of the Assignment 4 report, final text. Tables 1–4 precede Table 5 in
Section 5. Every number is read from a file in the repository; the source is named in each caption.

---

## 3. Deviations from the Frozen Protocol

The Protocol Freeze form was uploaded to Moodle on 2 November 2026 at 03:38 (UTC+5), before any final run. Every training run reported here started after the upload, and the training code, configuration and data did not change afterwards: later commits added only logs, analysis scripts and documentation. Table 1 lists the decisions that differ from Assignment 3, all fixed before the freeze, and the details clarified after it; the full log is `docs/deviations.md`.

The largest change concerns the initial implementation in Assignment 3, which did not follow its own design: it trained an instruction-tuned 1.5B model on code and then on GSM8K. Instruction-tuned models were excluded by our selection criteria, and training on GSM8K would contaminate a retention benchmark, so the final pipeline uses the planned base models and domain order. The two clarifications after the freeze concern details the form left open; both were decided without looking at test results and apply identically to all methods.

Table 1. Deviations from Assignment 3 and clarifications of the frozen protocol (source: `docs/deviations.md`).

| # | When | Change | Reason |
|---|---|---|---|
| 1 | Before freeze | Base models, domains and precision brought in line with the A3 design (Qwen2.5-0.5B and Qwen3-0.6B-Base; medicine → law → code; fp16) | The A3 pilot used an excluded instruct model and trained on a retention benchmark |
| 2 | Before freeze | Test evaluation at Stage 0, at stage ends and at the matched checkpoint, not at every quarter-stage | Generative evaluation takes about one GPU-hour per checkpoint |
| 3 | Before freeze | Item-level paired bootstrap with Holm correction instead of a repeated-measures model | Tests over three seeds cannot reach significance |
| 4 | Before freeze | One primary metric: MMLU forgetting after Stage 3 | Required by Assignment 4 |
| 5 | Before freeze | Adapter up-projection initialised to zero | The library default changed the model before training (MedMCQA-dev 0.12 vs 0.30) |
| 6 | Before freeze | Learning rates searched on the primary model only | Compute budget |
| 7 | After freeze | Matched-checkpoint ties resolved towards the later checkpoint | The rule left ties undefined |
| 8 | After freeze | Few-shot settings not fixed in the form: MedMCQA and CaseHOLD 0-shot, MBPP 3-shot | Same settings for every checkpoint |
| 9 | After freeze | [fill: Qwen3-0.6B replication runs — completed as planned / not completed, and why] | |

---

## 4. Final Experimental Setup

**Data.** Each stage trains on 2,000 examples drawn with a fixed seed: MedMCQA (stratified by its 21 subjects), CaseHOLD (LexGLUE) and CodeAlpaca-20k. Before sampling, duplicates and every example sharing a word 13-gram with a test or development item were removed (Table 2); this removed 39% of the CaseHOLD pool, mostly shared legal boilerplate. Test subsets are fixed ID lists with SHA-256 hashes (`data/SHA256SUMS`), identical for every checkpoint.

**Conditions.** Baseline 1 is the untouched base model (Stage 0); Baseline 2 is full fine-tuning (FFT). The proposed conditions are LoRA (r = 16, α = 32 on all seven attention and MLP projections) and a bottleneck adapter after every feed-forward block, sized to LoRA: 8,800,032 against 8,798,208 trainable parameters. One set of trainable weights is carried through all stages.

**Fairness.** All conditions share the base checkpoint, data, domain order, prompt template, schedule, precision, hardware and seeds; only the adaptation mechanism differs. Each method had the same three-value learning-rate budget, selected on MedMCQA-dev after Stage 1, and every method starts from the identical base model, which is checked at run time.

**Implementation.** Tables 3 and 4 list the configuration; each run stores its library versions, GPU, commit and data hashes in `run_meta.json`.

Table 2. Data used in each stage and for evaluation (sources: `data/manifest.json`, `results/tables/data_pipeline.csv`).

| Domain / benchmark | Pool | Removed (duplicate / test overlap / dev overlap) | Train | Dev | Test |
|---|---|---|---|---|---|
| Medicine: MedMCQA | 182,822 | 73 / 155 / 528 | 2,000 | 200 | 500 (official validation split) |
| Law: CaseHOLD | 45,000 | 2 / 13,421 / 4,123 | 2,000 | 200 | 500 |
| Code: CodeAlpaca-20k; MBPP | 20,016 | 0 / 106 / 91 | 2,000 | 200 + 90 MBPP validation | 500 MBPP |
| General knowledge: MMLU | — | — | — | — | 1,260 (42 non-overlapping subjects × 30) + 450 (15 overlapping) |
| Reasoning: GSM8K | — | — | — | — | 500 of 1,319 |

Table 3. Training setup (sources: `configs/experiment.yaml`, `results/raw/*/run_summary.json`).

| Item | Setting |
|---|---|
| Base models | Qwen/Qwen2.5-0.5B, revision 060db64 (primary); Qwen/Qwen3-0.6B-Base, revision da87bfb (replication) |
| Runs | 9 primary (3 methods × seeds 1, 2, 3; medicine → law → code); 3 reversed order (code → law → medicine, seed 1); 3 replication on Qwen3 (seed 1) |
| Trainable parameters | FFT 494.0 M (100%); LoRA 8.80 M (1.75%); adapter 8.80 M (1.75%) |
| Training | 2 epochs per stage, effective batch 16 (2 × 8 accumulation), 250 optimiser steps per stage, max length 512, loss on response tokens only |
| Optimiser | AdamW, linear schedule, 3% warm-up, restarted at every stage; weight decay 0; gradient clipping 1.0 |
| Learning rate | FFT 2e-5, LoRA 2e-4, adapter 3e-4, chosen from {1e-5, 2e-5, 5e-5}, {1e-4, 2e-4, 5e-4}, {1e-4, 3e-4, 1e-3} (`results/lr_search/summary.csv`) |
| Checkpoints | Steps 63, 125, 188 and 250 of every stage (25, 50, 75, 100%) |
| Hardware and precision | One NVIDIA T4 16 GB (Kaggle); fp16 mixed precision with fp32 master weights; gradient checkpointing for all methods |
| Software | Python 3.13.15, PyTorch 2.11.0 (CUDA 12.8), transformers 4.57.6, peft 0.21.2, adapters 1.3.0, datasets 4.8.5, accelerate 1.12.0 |
| Training time, 3 stages (mean ± SD, 3 seeds) | FFT 29.8 ± 0.2 min; LoRA 38.6 ± 0.5 min; adapter 25.1 ± 0.2 min |
| Peak GPU memory | FFT 9.84 GiB; LoRA 4.07 GiB; adapter 4.07 GiB |

Table 4. Evaluation setup: lm-evaluation-harness 0.4.13, fp16, greedy decoding, few-shot seed 1234 (source: `results/eval/*/summary.json`).

| Benchmark | Items | Shots | Scoring |
|---|---|---|---|
| MMLU (primary metric) | 1,260 + 450 overlapping | 5 | Log-likelihood accuracy |
| GSM8K | 500 | 5 | Exact match, flexible-extract (strict-match also reported); max 256 generated tokens |
| MedMCQA | 500 | 0 | Log-likelihood accuracy |
| CaseHOLD | 500 | 0 | Log-likelihood accuracy |
| MBPP | 500 | 3 | pass@1, generated code run against unit tests |

---

## 9. Reproducibility Statement

All code, configuration files, data-subset identifiers with SHA-256 hashes, raw logs and analysis scripts are available at https://github.com/alishermede/parameter-efficient-fine-tuning-and-catastrophic-forgetting-in-small-language-models, release `assignment-4`. `src/prepare_data.py` rebuilds every data subset from the public datasets and checks it against the committed hashes; `src/train.py` reproduces any training run from `configs/`, and `notebooks/kaggle_run.ipynb` runs the same commands on a free Kaggle T4. `src/eval.py` produces the per-item test outputs in `results/eval/`, and `src/stats.py`, `src/figures.py` and `src/training_report.py` regenerate every table and figure from the committed files. All checkpoints are public on the Hugging Face Hub (`alishermede/rmt-peft-ckpts`, `dddddffwf/rmt-peft-ckpts`). The reproduction test (Appendix C) was performed by [fill: team member] on [fill: date] in a clean Kaggle environment.
