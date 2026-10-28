"""Behavioural testing (CheckList-style): invariance, directional expectations, perturbation, edge cases."""
import numpy as np
import pandas as pd
import pytest

from src.features import MODEL_INPUTS
from src.predict import score


@pytest.mark.robustness
@pytest.mark.smoke
def test_risky_applicant_scores_higher_than_good(champion, good_applicant, risky_applicant):
    assert score(champion, risky_applicant)["probability_of_default"] > score(champion, good_applicant)["probability_of_default"]


@pytest.mark.robustness
@pytest.mark.parametrize("field, worse", [
    ("missed_payments_12m", 3), ("debt_to_income", 0.65), ("credit_utilisation", 0.85), ("credit_history_years", 1.0),
], ids=["more-missed-payments", "higher-dti", "higher-utilisation", "shorter-history"])
def test_directional_expectation_risk_increases(champion, good_applicant, field, worse):
    before = score(champion, good_applicant)["probability_of_default"]
    after = score(champion, {**good_applicant, field: worse})["probability_of_default"]
    assert after > before, f"{field} worsened but PD went {before:.4f} -> {after:.4f}"


@pytest.mark.robustness
def test_directional_expectation_income_lowers_risk(champion, risky_applicant):
    before = score(champion, risky_applicant)["probability_of_default"]
    after = score(champion, {**risky_applicant, "annual_income": 3_000_000.0})["probability_of_default"]
    assert after < before


@pytest.mark.robustness
@pytest.mark.fairness
@pytest.mark.parametrize("attr, values", [("gender", ["F", "M"]), ("applicant_name", ["Priya Sharma", "Rahul Verma"])])
def test_invariance_to_irrelevant_attributes(champion, good_applicant, attr, values):
    """Changing protected / irrelevant fields must not change the score."""
    scores = {score(champion, {**good_applicant, attr: v})["probability_of_default"] for v in values}
    assert len(scores) == 1


@pytest.mark.robustness
def test_small_noise_rarely_flips_decisions(champion, data, gates):
    rng = np.random.default_rng(0)
    X = data["test"][MODEL_INPUTS].copy()
    base = champion.predict_proba(X)[:, 1] < 0.2
    noisy = X.copy()
    pct = gates["robustness"]["noise_pct"]
    for col in ["annual_income", "loan_amount", "debt_to_income", "credit_utilisation"]:
        noisy[col] = noisy[col] * (1 + rng.normal(0, pct, len(noisy)))
    noisy["credit_utilisation"] = noisy["credit_utilisation"].clip(0, 1)
    flips = np.mean(base != (champion.predict_proba(noisy)[:, 1] < 0.2))
    assert flips <= gates["robustness"]["noise_flip_rate_max"], f"{flips:.1%} decisions flipped under ±{pct:.0%} noise"


@pytest.mark.robustness
@pytest.mark.parametrize("override", [
    {"annual_income": 0.0}, {"annual_income": 99_000_000.0}, {"credit_history_years": 0.0},
    {"debt_to_income": 0.0}, {"missed_payments_12m": 50}, {"employment_years": 0.0},
], ids=["zero-income", "very-high-income", "no-history", "zero-dti", "max-missed", "unemployed"])
def test_extreme_but_valid_inputs_give_valid_probability(champion, good_applicant, override):
    result = score(champion, {**good_applicant, **override})
    assert 0.0 <= result["probability_of_default"] <= 1.0
    assert not np.isnan(result["probability_of_default"])


@pytest.mark.robustness
def test_missing_values_in_batch_are_imputed(champion, data):
    X = data["test"][MODEL_INPUTS].head(50).copy()
    X.loc[X.index[:10], "employment_years"] = np.nan
    probs = champion.predict_proba(X)[:, 1]
    assert not np.isnan(probs).any()
