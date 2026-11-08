# Replication numbers (Qwen3-0.6B-Base, MLC, seed 1)

| condition               |   MedMCQA |   CaseHOLD |   MBPP |   MMLU_S3 |   MMLU_F3_pp |   MMLU_R3_pct |   GSM8K_S3 |   GSM8K_F3_pp |   GSM8K_R3_pct |   GSM8K_strict_S3 |
|:------------------------|----------:|-----------:|-------:|----------:|-------------:|--------------:|-----------:|--------------:|---------------:|------------------:|
| No adaptation (Stage 0) |      35.8 |       44.8 |   39.2 |      54.1 |          0   |         100   |       50   |           0   |          100   |              48.2 |
| Full fine-tuning        |      34   |       75.2 |   32.6 |      39.7 |         14.4 |          73.3 |       29.6 |          20.4 |           59.2 |              14   |
| LoRA                    |      39.2 |       73.2 |   39.6 |      50.5 |          3.7 |          93.3 |       43.4 |           6.6 |           86.8 |              28.2 |
| Bottleneck adapter      |      39.4 |       70.4 |   37.8 |      54   |          0.1 |          99.9 |       47   |           3   |           94   |              41.2 |

## Pairwise Stage-3 forgetting (paired item bootstrap, Holm within benchmark)

| benchmark    | comparison     |   dF3_pp |   ci_lo |   ci_hi |   p_boot |   p_holm |
|:-------------|:---------------|---------:|--------:|--------:|---------:|---------:|
| mmlu_primary | lora - full    |  -10.794 | -13.413 |  -8.175 |    0     |    0     |
| mmlu_primary | adapter - full |  -14.365 | -17.302 | -11.429 |    0     |    0     |
| mmlu_primary | lora - adapter |    3.571 |   1.587 |   5.558 |    0.001 |    0.001 |
| gsm8k        | lora - full    |  -13.8   | -18.2   |  -9.4   |    0     |    0     |
| gsm8k        | adapter - full |  -17.4   | -21.8   | -13     |    0     |    0     |
| gsm8k        | lora - adapter |    3.6   |  -0.4   |   7.6   |    0.093 |    0.093 |

## Matched checkpoint (T = 47.5% MedMCQA-dev)

| method   | checkpoint         |   MMLU |   MMLU_retention_pct |
|:---------|:-------------------|-------:|---------------------:|
| full     | stage1_medmcqa_100 |   53.9 |                 99.6 |
| lora     | stage1_medmcqa_050 |   54.4 |                100.4 |
| adapter  | stage1_medmcqa_100 |   52.6 |                 97.2 |

- MMLU at matched checkpoint, lora - full: +0.48 pp [-1.51, +2.38], Holm p = 0.6600
- MMLU at matched checkpoint, lora - adapter: +1.75 pp [+0.00, +3.49], Holm p = 0.1032
