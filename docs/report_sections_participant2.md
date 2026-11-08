# Report sections — Participant 2 (draft)

Sections 5 and 7 of the Assignment 4 report. Every `[fill: …]` names the generated file the
value is copied from (`results/tables/report_numbers.md` collects them in one place). The
sentences in *italics* depend on the direction of the result and must be rewritten once the
numbers exist; do not keep a sentence the data do not support.

---

## 5. Main Results

Table 5 reports every condition after the last stage (Stage 3) on the primary model. The untouched base model scores [fill: MMLU S0, main_table.csv] on the MMLU primary subset and [fill: GSM8K S0] on GSM8K. After sequential adaptation to medicine, law and code, full fine-tuning lost [fill: FFT MMLU F3 ± SD] pp on MMLU, LoRA [fill: LoRA F3 ± SD] pp and the bottleneck adapter [fill: adapter F3 ± SD] pp; on GSM8K the losses were [fill: three GSM8K F3 values] pp. Seed-to-seed variability was [fill: largest SD] pp at most, *small/large* relative to the differences between methods.

Table 5. Results after Stage 3, Qwen2.5-0.5B, mean ± SD over three seeds (source: `results/tables/main_table.csv`). S3 in %, F3 = S0 − S3 in percentage points.

[paste main_table.csv]

**Statistical comparison.** Because all conditions answer the same test items, we compared forgetting with a paired bootstrap over items (10,000 resamples) on per-item correctness averaged over the three seeds, with Holm correction across the three pairwise comparisons, separately for MMLU and GSM8K (Table 6). On MMLU, LoRA forgot [fill: |ΔF3|] pp *less/more* than full fine-tuning (95% CI [fill], Holm-adjusted p = [fill]), and the adapter [fill] pp (95% CI [fill], p = [fill]). LoRA and the adapter differed by [fill] pp (95% CI [fill], p = [fill]). On GSM8K the corresponding differences were [fill] pp (LoRA − FFT, p = [fill]), [fill] pp (adapter − FFT, p = [fill]) and [fill] pp (LoRA − adapter, p = [fill]). The direction of the LoRA − FFT difference was [fill: the same in all three seeds / not consistent] (per-seed values in Table 6); per-seed paired t-tests (n = 3) are reported only as exploratory.

Table 6. Pairwise differences in Stage-3 forgetting, ΔF3 = F_A − F_B in pp (negative: A forgets less). Paired bootstrap, 10,000 resamples; Holm correction within each benchmark (source: `results/tables/pairwise_tests.csv`).

| Benchmark | Comparison | ΔF3 (pp) | 95% CI | p | p (Holm) | Per seed (s1, s2, s3) |
|---|---|---|---|---|---|---|
| MMLU | LoRA − FFT | | | | | |
| MMLU | Adapter − FFT | | | | | |
| MMLU | LoRA − Adapter | | | | | |
| GSM8K | LoRA − FFT | | | | | |
| GSM8K | Adapter − FFT | | | | | |
| GSM8K | LoRA − Adapter | | | | | |

**Practical significance.** In relative terms, LoRA reduced MMLU forgetting by [fill: relative_reduction_pct]% compared with full fine-tuning, and the adapter by [fill]%. *An absolute difference of [x] pp on MMLU corresponds to about [x × 12.6] of the 1,260 questions; the standard error of a single MMLU score on this subset is about 1.4 pp.* GSM8K strict-match, which requires the "#### answer" format, fell by [fill] pp for full fine-tuning against [fill] pp with flexible extraction, *indicating that part of the GSM8K loss is a loss of answer format rather than of reasoning* (examined in Section 6).

**Trajectories and the gain–retention trade-off.** Figure 1 shows forgetting after each stage. *Most of the loss for full fine-tuning appears after Stage [k] ([domain]), while LoRA stays within [x] pp of the base model throughout.* Figure 2 plots, for every stage, the gain on the domain just trained against MMLU retention: *full fine-tuning reaches the largest in-domain gains ([fill]) at the lowest retention ([fill]%), whereas the PEFT methods achieve [fill] pp gains at [fill]% retention*. Target-domain scores at the end of each stage are in `results/tables/trajectories.csv`.

**Matched checkpoint (H2).** The matching target was T = [fill: matched_summary.json] MedMCQA-dev accuracy (the lowest final Stage-1 mean, [fill: method]). Each method's seed-mean matched dev accuracy was within [fill] pp of T, so H2 is testable. At these checkpoints MMLU retention was [fill: R for FFT, LoRA, adapter]%; LoRA − adapter = [fill] pp (95% CI [fill], Holm p = [fill]) and LoRA − FFT = [fill] pp (95% CI [fill], Holm p = [fill]) (`results/tables/h2_tests.csv`).

Figure 1. Forgetting of MMLU (left) and GSM8K (right) after each stage, mean ± SD over three seeds (source: `results/tables/trajectories.csv`; `results/figures/fig1_forgetting_trajectories.png`).

Figure 2. Gain on the domain just trained versus MMLU retention after every stage, seed means (source: `results/tables/trajectories.csv`; `results/figures/fig2_gain_vs_retention.png`).

---

## 7. Answering the Research Question and Hypotheses

Table 7. Hypotheses, evidence and verdicts by the criteria fixed in the Protocol Freeze form (source: `results/tables/hypotheses.csv`).

| Hypothesis | Evidence | Result in numbers | Supported / Partly / Not supported |
|---|---|---|---|
| H1: after sequential adaptation, LoRA and bottleneck adapters show lower forgetting of general performance than full fine-tuning | Table 6, Figure 1 | MMLU ΔF3 LoRA − FFT [fill] pp [CI], p_Holm [fill]; Adapter − FFT [fill] pp [CI], p_Holm [fill]; GSM8K [fill]; same sign in all seeds: [fill] | [fill: verdict] |
| H2: at comparable target-domain performance, LoRA retains general performance better than adapters and full fine-tuning | h2_tests.csv, matched_checkpoints.csv | MMLU retention LoRA − adapter [fill] pp [CI], p_Holm [fill]; LoRA − FFT [fill] pp [CI], p_Holm [fill] | [fill: verdict] |

Criteria (frozen): H1 is supported if both PEFT methods have lower MMLU F3 than full fine-tuning with Holm-adjusted p < 0.05, a 95% CI excluding zero and the same direction in all three seeds; partly supported if only one method meets this or the effect holds on MMLU but not on GSM8K (or the reverse). H2 is supported if LoRA's MMLU retention at the matched checkpoint exceeds both comparators under the same criteria, partly supported if it exceeds one, and not testable if no matched checkpoint within ±3 pp exists for all methods.

**Answer to the research question (100–150 words).** *Template — rewrite to the actual verdicts:* In sequential adaptation of Qwen2.5-0.5B to medicine, law and code, [LoRA and bottleneck adapters] retained [x]% and [y]% of the base model's MMLU accuracy, against [z]% for full fine-tuning, i.e. they forgot [a] and [b] pp instead of [c] pp; the difference [was / was not] statistically reliable after correction. On GSM8K the pattern was [similar / weaker / reversed] ([numbers]). When methods were compared at matched medical-domain performance, LoRA [did / did not] retain more general knowledge than [the adapter / full fine-tuning]. The extent of the advantage is therefore [state it in pp], obtained with [~1.8%] of trainable parameters, for one model family below one billion parameters and three domains; [replication / reversed order] results in Section 6 indicate whether it generalises.
