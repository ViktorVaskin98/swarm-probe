# PLAN — swarm-probe

## Questions
- **Q1 (swarm of small models).** Does majority voting over *k* samples of a small model
  (self-consistency) catch up with one larger model at an equal FLOPs budget?
- **Q2 (failure prediction).** Does a linear probe on hidden states predict that an answer will be
  wrong — before generation and from the first answer tokens? Is it better than mean log-probability
  and answer length?
- **Q3 (stretch, links Q1 and Q2).** Does voting weighted by the probe's confidence improve the swarm
  over plain majority voting?

## Setup
- GSM8K test, 500 problems (seed 0); numeric answer, exact-match scoring.
- Qwen2.5-Instruct 0.5B / 1.5B / 3B, HF `generate`, Kaggle T4, temperature 0.7, top-p 0.95.
- Cost: `2 · N_params · (prompt + generated tokens)` per sample.
- Probe: logistic regression over a layer × prefix grid (0 / 16 / 32 / 64 tokens), `GroupKFold` by
  problem, ROC-AUC with bootstrap intervals; baselines — mean log-probability, answer length.

## Fixed before the full results (2026-10-15, 21:00 MSK)
- **Q2, primary metric:** probe ROC-AUC over the whole layer × prefix grid, every cell reported.
  The "best layer" curve in the figure is picked post hoc and labelled as such.
- **Q3, primary cell:** prefix `full` (last answer token), layer 16, logistic regression C = 0.1,
  out-of-fold scores from `GroupKFold(5)`. All other cells are exploratory only.
- **Q1:** comparison at equal FLOPs; the main contrast is 0.5B × *k* against 1.5B and 3B greedy.

## Outcome (2026-10-16)
Q1 — the swarm loses at equal compute, but its pass@k matches the larger model: the bottleneck is
aggregation. Q2 — the probe beats log-probability, but before the answer is complete it detects
hard problems rather than failing samples. Q3 — the preregistered cell gives +3.6 pp over majority
voting (95% CI +1.2…+6.0). Details in the [README](README.md).
