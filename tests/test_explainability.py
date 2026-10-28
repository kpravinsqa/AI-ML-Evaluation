import pytest
from sklearn.inspection import permutation_importance

from data.generate_data import TARGET
from src.features import MODEL_INPUTS


@pytest.fixture(scope="module")
def importances(champion, data):
    sample = data["test"].sample(800, random_state=1)
    result = permutation_importance(champion, sample[MODEL_INPUTS], sample[TARGET], scoring="roc_auc",
                                    n_repeats=5, random_state=0)
    return dict(sorted(zip(MODEL_INPUTS, result.importances_mean), key=lambda kv: -kv[1]))


@pytest.mark.explainability
def test_top_drivers_are_business_plausible(importances):
    top3 = list(importances)[:3]
    assert "missed_payments_12m" in top3, importances
    assert {"debt_to_income", "credit_utilisation"} & set(top3), importances


@pytest.mark.explainability
def test_no_single_feature_dominates(importances):
    total = sum(max(v, 0) for v in importances.values())
    assert max(importances.values()) / total < 0.6, "model over-relies on one feature"


@pytest.mark.explainability
def test_irrelevant_feature_has_negligible_importance(importances):
    assert importances["loan_term_months"] < 0.01
