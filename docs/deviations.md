# Deviations from the frozen protocol

Every change relative to Assignment 3 or to the Protocol Freeze form is logged here with its
date and reason. Changes dated before the freeze upload were decided before any final result
was seen; they are listed so the report can show how the final setup differs from A3.

Freeze form uploaded to Moodle: 2 November 2026, 03:38 (UTC+5). The nine primary runs started after the upload (commit f1b58e6).

| # | Date | What changed | From (A3 / freeze) | To (final) | Reason | Effect on conclusions |
|---|---|---|---|---|---|---|
| 1 | before freeze | Implementation aligned with the A3 methodology | A3 Part 7 pilot: Qwen2.5-1.5B-Instruct, code → GSM8K, bf16, LoRA on q/k/v/o only, adapter reduction factor 16 | Qwen2.5-0.5B base + Qwen3-0.6B-Base; medicine → law → code; fp16 mixed precision; LoRA on all 7 projections; adapter matched to LoRA's parameter count | The pilot did not follow A3 Sections 1–6: an instruct model is excluded by our criteria, and training on GSM8K would contaminate a retention benchmark; T4 GPUs do not support bf16 | None: the final setup is the A3 design; pilot numbers are not reported |
| 2 | before freeze | Test-set evaluation schedule | Full test evaluation at 25/50/75/100% of every stage | Development metrics at every checkpoint; full test evaluation at Stage 0, at the end of each stage and at the matched checkpoint | Realistic cost is 55–75 T4 GPU-hours instead of the ~20 estimated in A3, mostly generative evaluation | Retention trajectories have 4 points per run (Stages 0–3) instead of 13; checkpoint matching (H2) still uses dev metrics at every checkpoint as planned |
| 3 | before freeze | Statistical analysis | Repeated-measures model over runs with Holm correction | Item-level paired bootstrap (10,000 resamples) with Holm correction; mean ± SD over seeds; per-seed tests exploratory | With n = 3 seeds a test over runs has almost no power (Wilcoxon cannot reach p < 0.25) | More informative inference; seed variability still reported |
| 4 | before freeze | Primary outcome | Two primary outcomes (MMLU, GSM8K) | One primary metric: MMLU forgetting after Stage 3; GSM8K forgetting secondary | Assignment 4 requires a single primary metric | GSM8K results remain reported in full |
| 5 | before freeze | Learning-rate search scope | "equal 3-value search per method" (model not specified) | Search on the primary model only; the replication model uses the same learning rates | Compute budget | Replication results may be slightly sub-optimal for Qwen3; stated as a limitation |
| 6 | before freeze | Effective batch size (not fixed in A3) | — (pilot used 4) | 16 (2 per device × 8 accumulation), 250 optimiser steps per stage | Fits full fine-tuning in 16 GB T4 memory; identical for all methods | None |
| 7 | before freeze | Profiling hardware | A3 Part 8 mentioned A100 / RTX 4090 | One NVIDIA T4 (Kaggle) for all runs | Matches A3 Sections 2 and 6; one GPU type keeps time and memory comparable | None |
| 8 | before freeze (pilot, 2026-11-02) | Adapter initialisation | Adapters library default (`bert`: both projections random) | `mam_adapter`: up-projection zero, down-projection Kaiming-uniform, as LoRA | The pilot showed the default changes the model before training (MedMCQA-dev 0.12 vs 0.30, MBPP 0.00 vs 0.24 at Stage 0); with zero up-projection all methods start from the identical base model (checked at run time) | Removes a confound; no reported run used the old initialisation |
| 9 | | [later deviations, if any — replication or reversed-order runs] | | | | |
