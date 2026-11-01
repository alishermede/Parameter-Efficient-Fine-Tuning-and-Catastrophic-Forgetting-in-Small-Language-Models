# Report sections — Participant 1 (draft)

Sections 3, 4 and 9 of the Assignment 4 report. `[fill: …]` marks values that come from the
runs; each one names the file it is read from, so every number stays traceable.

---

## 3. Deviations from the Frozen Protocol

The final experiment follows the design of Assignment 3 (Sections 1–6). Seven decisions were changed or fixed before the final runs and were declared in the Protocol Freeze form uploaded on [fill: date and time of the Moodle upload]; they are listed in Table 3 and in `docs/deviations.md`.

First, the initial implementation reported in Assignment 3 (Part 7) did not match the planned design: it used an instruction-tuned 1.5B model, adapted it to code and then to GSM8K, and trained in bf16. The final pipeline uses the planned base models and domain order, because instruction-tuned models were excluded by our selection criteria and training on GSM8K would contaminate one of the two retention benchmarks. Second, the test benchmarks are evaluated at the end of each stage and at the matched checkpoint rather than at every quarter of a stage, because the full schedule would have needed about three times our GPU budget. Third, the repeated-measures analysis over three seeds was replaced by an item-level paired bootstrap, since tests over three runs cannot reach significance. Fourth, MMLU forgetting was fixed as the single primary metric, as Assignment 4 requires.

[fill: any deviation that happens during the final runs — for example a resumed or incomplete run — or state "No deviations occurred after the freeze."]

Table 3. Deviations from Assignment 3 and the frozen protocol (summary of `docs/deviations.md`).

---

## 4. Final Experimental Setup

**Data.** Each adaptation stage uses 2,000 training examples drawn with a fixed seed: MedMCQA (training split, stratified by its 21 subjects), CaseHOLD (LexGLUE training split) and CodeAlpaca-20k. Before sampling we removed exact duplicates and every example sharing a word 13-gram with any test or development item ([fill: removed counts per domain, `data/manifest.json` → stats]). Development sets of 200 examples per domain are disjoint from training data; the 90 MBPP validation tasks serve as code development data. Test subsets are fixed as ID lists with SHA-256 hashes (`data/SHA256SUMS`): MMLU 1,260 questions (30 from each of 42 subjects that do not overlap the adaptation domains) plus 450 from the 15 overlapping subjects, GSM8K 500 of 1,319, MedMCQA 500 (official validation split), CaseHOLD 500 and MBPP 500.

**Conditions.** Baseline 1 is the untouched base model (Stage 0). Baseline 2 is full fine-tuning of all weights. The proposed conditions are LoRA (r = 16, α = 32, dropout 0.05, on all seven attention and MLP projections) and a sequential bottleneck adapter after every feed-forward block, whose width is computed so that its trainable parameters match LoRA: on Qwen2.5-0.5B, 8,800,032 adapter parameters (bottleneck 204) against 8,798,208 for LoRA, about 1.8% of the 494 M base parameters. One set of trainable weights is carried through all three stages.

**Fairness.** All conditions share the base checkpoint, data files, domain order, prompt template, epochs, batch size, sequence length, optimiser, schedule, precision, hardware and seeds; only the adaptation mechanism differs. Each method received the same learning-rate budget of three values, selected on MedMCQA-dev accuracy after Stage 1 only (Table 4).

**Implementation.** Table 4 lists the configuration. Every run records library versions, GPU, git commit and data hashes in `run_meta.json`.

Table 4. Final experimental setup.

| Item | Setting |
|---|---|
| Base models | Qwen/Qwen2.5-0.5B, revision [fill: run_meta.json → model_revision_sha] (primary, seeds 1–3); Qwen/Qwen3-0.6B-Base, revision [fill] (replication, seed 1) |
| Domain order | Medicine → law → code; reversed order (code → law → medicine) for one seed per method |
| Trainable parameters | Full: 494.0 M (100%); LoRA: 8.80 M (1.78%); adapter: 8.80 M (1.78%) — Qwen2.5-0.5B |
| Training | 2 epochs per stage, effective batch 16 (4 × 4 accumulation), 250 optimiser steps per stage, max length 512 tokens, loss on response tokens only |
| Optimiser | AdamW, linear schedule with 3% warm-up restarted at every stage, weight decay 0, gradient clipping 1.0 |
| Learning rate | Full [fill], LoRA [fill], adapter [fill] — chosen from {1e-5, 2e-5, 5e-5}, {1e-4, 2e-4, 5e-4}, {1e-4, 3e-4, 1e-3} (`configs/lr_selected.json`, `results/lr_search/summary.csv`) |
| Checkpoints | 25, 50, 75 and 100% of every stage (steps 63, 125, 188, 250) |
| Precision and hardware | fp16 mixed precision with fp32 master weights; gradient checkpointing for all methods; one NVIDIA T4 16 GB (Kaggle) |
| Software | Python [fill], PyTorch [fill], transformers 4.57.6, peft 0.21.2, adapters 1.3.0, datasets 4.8.5, accelerate 1.12.0, lm-evaluation-harness 0.4.13 |
| Seeds | 1, 2, 3 (primary); 1 (replication, reversed order); 0 (learning-rate search) |
| Run time | [fill: mean training time per run per method, `run_summary.json` → train_seconds] |

---

## 9. Reproducibility Statement

All code, configuration files, data-subset identifiers with SHA-256 hashes, raw logs and analysis scripts are available at https://github.com/alishermede/parameter-efficient-fine-tuning-and-catastrophic-forgetting-in-small-language-models, release `assignment-4`. `src/prepare_data.py` regenerates every data subset from the public datasets and verifies it against the committed hashes; `src/train.py` reproduces any run from the configuration in `configs/` with the seeds listed in Table 4, and `notebooks/kaggle_run.ipynb` runs the same commands on a free Kaggle T4. The README lists the exact command for each of the 15 runs. Checkpoints are stored in a private Hugging Face repository and are available on request. The reproduction test (Appendix C) was performed by [fill: team member] in a clean Kaggle environment.
