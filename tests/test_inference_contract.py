"""Scoring API contract: input validation, output schema, thresholds, latency, consistency, reproducibility."""
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from data.generate_data import TARGET
from src.features import MODEL_INPUTS
from src.predict import ValidationError, score, score_batch
from src.train import build


@pytest.mark.contract
@pytest.mark.smoke
def test_output_schema(champion, good_applicant):
    result = score(champion, good_applicant)
    assert set(result) == {"probability_of_default", "decision", "reason_codes"}
    assert result["decision"] in {"APPROVE", "REFER", "DECLINE"}
    assert isinstance(result["reason_codes"], list)


@pytest.mark.contract
def test_good_applicant_approved_without_reasons(champion, good_applicant):
    result = score(champion, good_applicant)
    assert result["decision"] == "APPROVE" and result["reason_codes"] == []


@pytest.mark.contract
def test_risky_applicant_declined_with_reasons(champion, risky_applicant):
    result = score(champion, risky_applicant)
    assert result["decision"] == "DECLINE"
    assert result["reason_codes"] == ["High credit utilisation", "Recent missed payments", "High debt-to-income ratio"]


@pytest.mark.contract
@pytest.mark.parametrize("field, value, message", [
    ("age", 17, "must be between 18 and 100"),
    ("age", 30.5, "must be a whole number"),
    ("credit_utilisation", 1.2, "must be between 0 and 1"),
    ("annual_income", "high", "must be numeric"),
    ("missed_payments_12m", True, "must be numeric"),
    ("loan_amount", None, "is required"),
    ("home_ownership", "LEASE", "must be one of"),
], ids=["age-17", "age-decimal", "util-1.2", "income-string", "bool", "missing", "bad-category"])
def test_input_validation(champion, good_applicant, field, value, message):
    with pytest.raises(ValidationError) as err:
        score(champion, {**good_applicant, field: value})
    assert message in err.value.errors[field]


@pytest.mark.contract
@pytest.mark.parametrize("pd_value, expected", [(0.1999, "APPROVE"), (0.20, "REFER"), (0.3999, "REFER"), (0.40, "DECLINE")])
def test_decision_threshold_boundaries(good_applicant, pd_value, expected):
    class FixedModel:
        def predict_proba(self, X):
            return np.array([[1 - pd_value, pd_value]])
    assert score(FixedModel(), good_applicant)["decision"] == expected


@pytest.mark.contract
def test_single_and_batch_scoring_agree(champion, data):
    rows = data["test"].head(20)
    batch = score_batch(champion, rows)
    single = [score(champion, {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()})["probability_of_default"]
              for r in rows[MODEL_INPUTS].to_dict("records")]
    assert np.allclose(batch, single, atol=1e-6)


@pytest.mark.contract
def test_p95_latency(champion, good_applicant, gates):
    timings = []
    for _ in range(200):
        start = time.perf_counter()
        score(champion, good_applicant)
        timings.append((time.perf_counter() - start) * 1000)
    assert np.percentile(timings, 95) <= gates["inference"]["p95_latency_ms_max"]


@pytest.mark.contract
def test_training_is_reproducible(data):
    """Same data + same seed = identical model scores (needed for audit / model risk management)."""
    train, test = data["train"], data["test"]
    a = build("challenger").fit(train[MODEL_INPUTS], train[TARGET]).predict_proba(test[MODEL_INPUTS])[:, 1]
    b = build("challenger").fit(train[MODEL_INPUTS], train[TARGET]).predict_proba(test[MODEL_INPUTS])[:, 1]
    assert np.array_equal(a, b)
