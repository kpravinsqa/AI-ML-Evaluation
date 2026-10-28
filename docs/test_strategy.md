# AI / ML Model Testing Strategy

Testing a model is not just "accuracy ≥ X". This suite treats the model as a product with eight test layers, each tied
to a gate in `config/quality_gates.json` that risk and compliance sign off.

| Layer | What it catches | Examples |
|---|---|---|
| Data validation | bad inputs before training | schema/range, nulls, duplicates, train/test overlap, class balance, **target leakage**, protected attrs not used, group representation |
| Performance | a model that is not good enough | ROC-AUC, precision/recall for defaulters, beats naive baselines, **business cost** vs approve-all, AUC per segment, champion vs challenger |
| Fairness | discriminatory outcomes | disparate impact (four-fifths rule), equal opportunity, chi-square on false declines, calibration within groups |
| Robustness | brittle behaviour | **directional expectations** (more missed payments → higher PD), **invariance** (gender/name flip → same score), ±2% noise flip rate, extreme valid inputs, missing values |
| Calibration | PDs that cannot be used as probabilities | ECE, Brier, mean PD vs observed, monotonic risk deciles |
| Drift | the world changing after go-live | PSI and KS per feature, prediction drift; stable batch passes, downturn batch flagged |
| Inference contract | integration failures | input validation messages, output schema, **threshold boundaries 0.1999/0.20/0.3999/0.40**, single = batch, p95 latency, reproducible training |
| Explainability | wrong reasons for right answers | permutation importance (top drivers plausible, no single feature > 60%), adverse-action reason codes |

## Seeded-defect proof

`INJECT_DEFECTS=1 pytest` trains on data with three classic real-world defects. The suite fails 6 checks:

| # | Defect | Caught by |
|---|---|---|
| D1 | Historical label bias against women | disparate impact (gender) |
| D2 | Leaky post-outcome feature `collections_flag` | target-leakage check, explainability (one feature dominates) |
| D3 | Gender used as a model input | protected-attribute check, invariance test (gender flip changes PD) |

**Lesson built into the suite:** equal-opportunity metrics computed on *biased labels* can look fair. That's why
invariance and protected-attribute checks are needed alongside outcome metrics.

## Testing note

A fixed "≤ 5 percentage points" gap between groups proved **flaky** on ~600 rows per group. It was replaced by a
chi-square significance test, which separates real bias from sampling noise.
