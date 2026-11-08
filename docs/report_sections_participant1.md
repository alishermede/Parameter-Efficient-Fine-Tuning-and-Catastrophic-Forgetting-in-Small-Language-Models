# Report sections — Participant 1 (final text, as in the submitted report)

## 3. Deviations from the Frozen Protocol

The Protocol Freeze form was uploaded to Moodle on 2 November 2026 at 03:38 (UTC+5), before any final run. Every training run reported here started after the upload, and the training code, configuration and data did not change afterwards: later commits added only logs, analysis scripts and documentation. Table 1 lists the decisions that differ from Assignment 3, all fixed before the freeze, and the details clarified after it; the full log is `docs/deviations.md`.

The largest change concerns the initial implementation in Assignment 3, which did not follow its own design: it trained an instruction-tuned 1.5B model on code and then on GSM8K. Instruction-tuned models were excluded by our selection criteria, and training on GSM8K would contaminate a retention benchmark, so the final pipeline uses the planned base models and domain order. The changes after the freeze settle details the form left open or fix problems found while evaluating; none altered how any condition was trained, and each applies identically to all methods. The most consequential is row 11: GSM8K and MBPP scores of the primary runs were generated with a longer output limit than planned, so they are comparable across methods but not exactly to a 256-token limit.

Table 1. Deviations from Assignment 3 and from the frozen protocol (source: `docs/deviations.md`).

| # | When | Change | Reason |
|---|---|---|---|
| 1 | Before freeze | Base models, domains and precision brought in line with the A3 design (Qwen2.5-0.5B and Qwen3-0.6B-Base; medicine → law → code; fp16) | The A3 pilot used an excluded instruct model and trained on a retention benchmark |
| 2 | Before freeze | Test evaluation at Stage 0, at stage ends and at the matched checkpoint, not at every quarter-stage | Generative evaluation takes about one GPU-hour per checkpoint |
| 3 | Before freeze | Item-level paired bootstrap with Holm correction instead of a repeated-measures model | Tests over three seeds cannot reach significance |
| 4 | Before freeze | One primary metric: MMLU forgetting after Stage 3 | Required by Assignment 4 |
| 5 | Before freeze | Adapter up-projection initialised to zero | The library default changed the model before training (MedMCQA-dev 0.12 vs 0.30) |
| 6 | Before freeze | Learning rates searched on the primary model only | Compute budget |
| 7 | After freeze, before any test result | Evaluation settings not fixed in the form: MedMCQA and CaseHOLD 0-shot, MBPP 3-shot (lm-eval default) | Needed to run the evaluation; identical for every checkpoint |
| 8 | After freeze, before any test result | Matched-checkpoint ties resolved towards the later checkpoint; ±3 pp tolerance applied to seed means | Dev accuracies are multiples of 0.5 pp, so ties occurred in 3 of 9 runs |
| 9 | During evaluation | Test evaluation on both T4 GPUs of a Kaggle session, one checkpoint per GPU | Halves wall-clock time; scores unaffected |
| 10 | During evaluation | Adapter checkpoints cast to fp16 after loading | Adapter weights loaded in fp32 and the fp16 forward pass failed; no adapter score existed before the fix |
| 11 | During evaluation | GSM8K and MBPP generation in the primary runs could reach 2,048 new tokens (Qwen's generation config overrode the 256-token limit); the replication used 256 | Found when Qwen3 ran out of memory; about 6% of GSM8K responses exceed ~256 tokens |
| 12 | Replication | Matching rule applied to the single Qwen3 seed (T = 47.5%) | Only one replication seed; result descriptive |
| 13 | Reversed order | Reversed-order runs evaluated on development data only; the planned test-set evaluation was not completed | Evaluation time ran out before the deadline; Section 6.4 therefore reports development metrics only |

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

Table 4. Evaluation setup: lm-evaluation-harness 0.4.13, fp16, greedy decoding, few-shot seed 1234 (sources: `configs/eval.yaml`, `results/eval/*/summary.json`).

| Benchmark | Items | Shots | Scoring |
|---|---|---|---|
| MMLU (primary metric) | 1,260 + 450 overlapping | 5 | Log-likelihood accuracy |
| GSM8K | 500 | 5 | Exact match, flexible-extract (strict-match also reported); up to 2,048 generated tokens in the primary runs, 256 in the replication (Table 1, row 11) |
| MedMCQA | 500 | 0 | Log-likelihood accuracy |
| CaseHOLD | 500 | 0 | Log-likelihood accuracy |
| MBPP | 500 | 3 | pass@1, generated code run against unit tests; generation limit as for GSM8K |

---

## 9. Reproducibility Statement

All code, configuration files, data-subset identifiers with SHA-256 hashes, raw logs and analysis scripts are available at https://github.com/alishermede/parameter-efficient-fine-tuning-and-catastrophic-forgetting-in-small-language-models, release `assignment-4`. `src/prepare_data.py` rebuilds every data subset from the public datasets and checks it against the committed hashes; `src/train.py` reproduces any training run from `configs/`, and `notebooks/kaggle_run.ipynb` runs the same commands on a free Kaggle T4. `src/eval.py` produces the per-item test outputs in `results/eval/` with the settings in `configs/eval.yaml`, and `src/stats.py`, `src/replication.py`, `src/figures.py` and `src/training_report.py` regenerate every table and figure from the committed files. All checkpoints are public on the Hugging Face Hub (`alishermede/rmt-peft-ckpts`, `dddddffwf/rmt-peft-ckpts`). The reproduction test (Appendix C) was performed by [fill: team member] on [fill: date] in a clean Kaggle environment.


---

## Appendix A. Experiment Protocol Freeze Form

The form below was uploaded to Moodle on 2 November 2026 at 03:38 (UTC+5), before any final run; it is reproduced unchanged.

Table A1. Experiment Protocol Freeze Form as uploaded (source: Moodle submission; docs/deviations.md lists every later change).

| Item | Your decision |
|---|---|
| Primary research question | To what extent do LoRA and adapters retain the general knowledge and reasoning performance of small language models compared with full fine-tuning during sequential domain adaptation? |
| Hypotheses / expected comparisons | H1: After sequential adaptation (medicine → law → programming), models adapted with LoRA and with bottleneck adapters show lower forgetting of general performance than full fine-tuning. H2: When the methods reach comparable target-domain performance, LoRA retains general performance better than bottleneck adapters and full fine-tuning. Pairwise comparisons: LoRA vs FFT, Adapter vs FFT, LoRA vs Adapter. |
| Primary outcome / metric and justification | MMLU forgetting after Stage 3, F₃ = S₀ − S₃ (percentage points), on the fixed 1,260-question subset (30 questions × 42 subjects that do not overlap the adaptation domains), 5-shot, log-likelihood scoring. Justification: it measures the general-knowledge construct directly; log-likelihood scoring is deterministic and independent of output format, so it separates knowledge loss from formatting drift; it is the largest evaluation set (standard error ≈ 1.4 pp); and it is expressed relative to each model’s own baseline, which cancels contamination. |
| Secondary metrics | GSM8K forgetting (exact match, flexible-extract, 500 problems, 5-shot, greedy; strict-match also reported); retention Rₛ = Sₛ / S₀ × 100% for MMLU and GSM8K after each stage; target-domain scores (MedMCQA and CaseHOLD accuracy, MBPP pass@1) and domain gain; backward transfer on earlier domains; MMLU on the 15 overlapping subjects (450 questions); trainable parameters, peak GPU memory, training time. |
| Baseline 1 (simple) | No adaptation: the pretrained base checkpoint (Stage 0) evaluated with the same harness, subsets and prompts. Chance level (25% on MMLU) is reported as a floor. |
| Baseline 2 (strong / literature) | Full fine-tuning of all weights, the standard practice and the reference condition in Biderman et al. (2024), with the same data, domain order, epochs, batch size, sequence length, optimiser, schedule and an equal three-value learning-rate budget. |
| Proposed method / condition | (a) LoRA: r = 16, α = 32, dropout 0.05, on q, k, v, o, gate, up and down projections (8,798,208 trainable parameters, ≈ 1.8% of Qwen2.5-0.5B). (b) Sequential bottleneck adapter after each feed-forward block (Pfeiffer configuration), width 204 so trainable parameters match LoRA within ±5% (8,800,032); the up-projection is initialised to zero, like LoRA’s B matrix, so every condition starts from the identical base model. One set of trainable weights is carried through all stages. Training: 2 epochs per stage, effective batch 16 (2 × 8 accumulation), 250 optimiser steps per stage, max length 512, AdamW with linear schedule and 3% warm-up, fp16 mixed precision on one NVIDIA T4. Learning rate chosen from three values per method on Stage-1 MedMCQA-dev accuracy only (seed 0): FFT {1e-5, 2e-5, 5e-5} → 2e-5; LoRA {1e-4, 2e-4, 5e-4} → 2e-4; Adapter {1e-4, 3e-4, 1e-3} → 3e-4. |
| Final dataset version, size, and test-set identifier (count / hash) | Models: Qwen/Qwen2.5-0.5B, revision 060db64 (primary); Qwen/Qwen3-0.6B-Base, revision da87bfb (replication). Training per stage: 2,000 examples each from MedMCQA (train), CaseHOLD (LexGLUE train), CodeAlpaca-20k, after removing duplicates and examples sharing a word 13-gram with any test or development item. Development: 200 per domain; MBPP validation (90 tasks) for code. Test: MMLU 1,260 (+450 overlapping), GSM8K 500 of 1,319, MedMCQA 500 (official validation split), CaseHOLD 500, MBPP 500. SHA-256 (first 12 hex digits; full values in data/SHA256SUMS of the repository): MMLU primary 6e387f74e96d, MMLU overlap 3f7fbc4e6558, GSM8K 2b8b7e11faca, MedMCQA 367aa387559c, CaseHOLD 9118e63570c1, MBPP d62b8b1b6a6e. Dataset revisions: MedMCQA 91c6572, LexGLUE c23fdff, CodeAlpaca-20k 152bb5e, MBPP 4bb6404, MMLU c30699e, GSM8K 740312a. |
| Number of runs, seeds, folds, trials, or participants | 15 sequential runs: 9 primary (3 methods × seeds 1, 2, 3; Qwen2.5-0.5B; order medicine → law → code); 3 replication (Qwen3-0.6B-Base, seed 1); 3 reversed order (code → law → medicine; Qwen2.5-0.5B; seed 1). Checkpoints at 25/50/75/100% of each stage: development evaluation at every checkpoint; full test evaluation at Stage 0, at the end of each stage and at the matched checkpoint. |
| Statistical test(s) and multiple-comparison correction | Each condition: mean ± SD over 3 seeds. Pairwise differences in Stage-3 forgetting: paired bootstrap over test items (10,000 resamples) on seed-averaged per-item correctness, giving the difference in pp with a 95% CI and a two-sided p-value; Holm correction across the three comparisons, separately for MMLU and GSM8K. Effect size: absolute difference in pp and relative reduction in forgetting. Per-seed paired t-tests are reported as exploratory only (n = 3). Stage trajectories are reported descriptively with bootstrap CIs; no significance is claimed for the method × stage interaction. |
| Result that would SUPPORT each hypothesis | H1: both LoRA and adapters have lower MMLU F₃ than FFT, Holm-adjusted p < 0.05 and a 95% CI of the difference that excludes 0, with the same direction in all 3 seeds. Partly supported: only one PEFT method meets this, or the effect holds on MMLU but not on GSM8K (or the reverse). H2: at the matched checkpoint, LoRA’s MMLU retention exceeds both adapters and FFT (Holm p < 0.05, CI excludes 0); partly supported if it exceeds only one of them. |
| Result that would NOT support each hypothesis | H1: the CI of the difference includes 0 for both PEFT methods, or PEFT methods forget more than FFT. H2: LoRA does not exceed either comparator. If no matched checkpoint within ±3 pp is reachable for all three methods, H2 is reported as not testable. Matching rule: target T = lowest final Stage-1 MedMCQA-dev accuracy among the three methods (mean over seeds); for each method and seed, the Stage-1 checkpoint whose dev accuracy is closest to T. |
| Planned additional analyses (error analysis + ...) | (1) Error analysis: 30 forgotten GSM8K items (correct at Stage 0, wrong after Stage 3) for FFT and 30 for LoRA, seed 1, coded into predefined categories: answer-format loss, arithmetic slip, wrong reasoning step, incomplete or repetitive output, off-task output. (2) Subgroups: MMLU by category (STEM, humanities, social sciences, other) and the 15 overlapping subjects; backward transfer per domain. (3) Sensitivity: reversed domain order. (4) Robustness: replication on Qwen3-0.6B-Base. (5) Efficiency: trainable parameters, peak memory, training time; Pareto front of domain gain × retention. (6) MMLU answer-letter confusion matrix, Stage 0 vs Stage 3. |

---

## Appendix D. Additional Tables and Figures

The tables and figures below use training logs and development data only; they support Sections 4 and 6. All are generated by `src/training_report.py`.

Table D1. Training efficiency on Qwen2.5-0.5B, three stages, mean ± SD over three seeds (source: `results/tables/efficiency_summary.csv`).

| Method | Trainable parameters | Training time (min) | Time per step (s) | Peak GPU memory (GiB) | Time / memory vs FFT |
|---|---|---|---|---|---|
| Full fine-tuning | 494.0 M (100%) | 29.8 ± 0.2 | 2.39 ± 0.01 | 9.84 | 1.00 / 1.00 |
| LoRA | 8.80 M (1.75%) | 38.6 ± 0.5 | 3.09 ± 0.04 | 4.07 | 1.29 / 0.41 |
| Bottleneck adapter | 8.80 M (1.75%) | 25.1 ± 0.2 | 2.01 ± 0.01 | 4.07 | 0.84 / 0.41 |

![](fig4_dev_trajectories.png)

Figure D1. Development metrics at all 13 checkpoints of the primary runs: MedMCQA-dev and CaseHOLD-dev accuracy and code-dev loss, mean ± SD over three seeds; the shaded band marks the law stage (source: `results/tables/dev_trajectories.csv`).

![](fig5_training_loss.png)

Figure D2. Training loss on response tokens across the three stages, mean ± SD over three seeds (source: `results/raw/*/train_log.jsonl`).

![](fig6_lr_sensitivity.png)

Figure D3. MedMCQA-dev accuracy after Stage 1 for each learning rate in the grid, seed 0; the circled point is the selected rate (source: `results/tables/lr_search.csv`).
