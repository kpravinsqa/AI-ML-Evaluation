# AI / ML Model Evaluation — Credit Risk Model Testing

![Tests](https://github.com/kpravinsqa/ai-ml-model-evaluation/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E) ![Pytest](https://img.shields.io/badge/pytest-8.x-0A9EDC)

A test framework for **machine-learning models as products**, applied to a credit-risk (probability-of-default) model
for personal-loan decisions. It goes beyond accuracy to test **data quality, performance gates, fairness, robustness,
calibration, drift, the inference API contract and explainability**, with every threshold version-controlled as a
release gate.

> Synthetic data and an original model built for this portfolio. No real customer data.

## Highlights

* **77 automated checks** across 8 layers ([test strategy](docs/test_strategy.md)), with gates in [`config/quality_gates.json`](config/quality_gates.json)
* **Fairness testing**: four-fifths rule, equal opportunity, chi-square on false declines, calibration within groups
* **Behavioural testing**: directional expectations, invariance to protected attributes, noise perturbation, extreme inputs
* **Monitoring tests**: PSI and KS data drift plus prediction drift; a stable batch passes, a downturn batch gets flagged
* **Champion / challenger**: the gradient-boosting challenger is *not* promoted, since it shows no uplift over logistic regression
* **Inference contract**: validation messages, decision-threshold boundaries, single vs batch parity, p95 latency, reproducibility
* **Seeded-defect proof**: `INJECT_DEFECTS=1` injects label bias, target leakage and a protected feature; the suite catches all three (CI asserts it)
* **Model card** and an **Excel evaluation workbook** for model-risk sign-off

## Quick start

```bash
git clone https://github.com/kpravinsqa/ai-ml-model-evaluation.git
cd ai-ml-model-evaluation
pip install -r requirements.txt

pytest                                   # all gates + reports/report.html
pytest -m fairness                       # one layer
INJECT_DEFECTS=1 pytest                  # watch seeded model defects get caught
python -m reports_tool.model_eval_report # reports/model_evaluation_report.xlsx
python -m src.train                      # train champion + challenger into models/
```

## Structure

```
data/generate_data.py       synthetic applications, train/test split, stable + downturn production batches
src/features.py             preprocessing pipeline        src/train.py     champion + challenger
src/predict.py              scoring API: validation, decision policy, reason codes
src/metrics.py              fairness, calibration (ECE), PSI / KS drift, promotion rule
config/quality_gates.json   release thresholds
tests/                      data · performance · fairness · robustness · calibration · drift · contract · explainability
docs/                       model card, test strategy
```

## Author

**Pravin Khedkar**, QA Lead / SDET (AI QA) · [@kpravinsqa](https://github.com/kpravinsqa)
