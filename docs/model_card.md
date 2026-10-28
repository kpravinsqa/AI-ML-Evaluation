# Model Card: Credit Risk PD Model (`credit-risk-pd-1.4.0`)

> Synthetic portfolio model. No real customer data.

| Item | Detail |
|---|---|
| Purpose | Estimate the 12-month probability of default (PD) for unsecured personal-loan applications |
| Decision policy | PD < 0.20 → APPROVE · 0.20–0.40 → REFER to underwriter · ≥ 0.40 → DECLINE (with reason codes) |
| Champion | Logistic regression (interpretable, well calibrated) |
| Challenger | Histogram gradient boosting, not promoted: no AUC uplift on hold-out |
| Inputs | income, loan amount and term, credit history, DTI, utilisation, missed payments, employment, home ownership |
| Excluded | gender, age band (protected attributes are audited for fairness but never used as inputs) |
| Training data | 4,500 synthetic applications; hold-out 1,500; default rate ≈ 22% |
| Performance | ROC-AUC ≈ 0.84, ECE < 0.05 (see `reports/model_evaluation_report.xlsx`) |
| Fairness | Four-fifths rule on approvals for gender and age band; equal opportunity diff ≤ 0.10; no significant false-decline gap |
| Monitoring | PSI per feature and on predictions: < 0.10 stable, 0.10–0.25 watch, ≥ 0.25 drift / retrain review |
| Limitations | Not validated for secured or business lending; downturn conditions shift DTI and utilisation; retrain review required |
