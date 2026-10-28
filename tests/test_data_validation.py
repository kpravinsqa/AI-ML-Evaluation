"""Data quality and leakage checks on the training data (before any model is trained)."""
import numpy as np
import pytest

from data.generate_data import FEATURES, SENSITIVE, TARGET
from src.features import MODEL_INPUTS
from src.predict import SCHEMA


@pytest.mark.data
@pytest.mark.smoke
def test_expected_columns_present(data):
    missing = set(FEATURES + SENSITIVE + [TARGET, "application_id"]) - set(data["train"].columns)
    assert not missing, f"missing columns {missing}"


@pytest.mark.data
@pytest.mark.parametrize("column", list(SCHEMA))
def test_numeric_ranges_respect_schema(data, column):
    _, lo, hi = SCHEMA[column]
    col = data["train"][column]
    assert col.between(lo, hi).all(), f"{column} outside [{lo}, {hi}]: {col[~col.between(lo, hi)].head().tolist()}"


@pytest.mark.data
def test_no_nulls_in_model_inputs_or_target(data):
    nulls = data["train"][MODEL_INPUTS + [TARGET]].isna().sum()
    assert nulls.sum() == 0, nulls[nulls > 0].to_dict()


@pytest.mark.data
def test_categorical_domain(data):
    assert set(data["train"]["home_ownership"]) <= {"OWN", "MORTGAGE", "RENT"}
    assert set(data["train"][TARGET]) == {0, 1}


@pytest.mark.data
def test_no_duplicate_applications(data):
    assert data["train"]["application_id"].is_unique


@pytest.mark.data
@pytest.mark.smoke
def test_no_train_test_overlap(data):
    assert not set(data["train"]["application_id"]) & set(data["test"]["application_id"])


@pytest.mark.data
def test_class_balance_is_plausible(data):
    rate = data["train"][TARGET].mean()
    assert 0.05 <= rate <= 0.40, f"default rate {rate:.1%} outside the plausible 5-40% band"


@pytest.mark.data
def test_train_and_test_have_similar_label_rate(data):
    assert abs(data["train"][TARGET].mean() - data["test"][TARGET].mean()) < 0.03


@pytest.mark.data
@pytest.mark.smoke
def test_no_target_leakage(data, gates):
    """No single numeric input may be (almost) a proxy of the label."""
    limit = gates["leakage"]["max_abs_correlation_with_target"]
    numeric = data["train"][MODEL_INPUTS].select_dtypes("number")
    corr = numeric.corrwith(data["train"][TARGET]).abs()
    assert (corr < limit).all(), f"suspected leakage: {corr[corr >= limit].round(3).to_dict()}"


@pytest.mark.data
@pytest.mark.fairness
def test_protected_attributes_not_used_as_model_inputs(gates):
    used = set(gates["fairness"]["protected_attributes"]) & set(MODEL_INPUTS)
    assert not used, f"protected attributes used as features: {used}"


@pytest.mark.data
def test_protected_groups_are_represented(data):
    for attr in ("gender", "age_band"):
        shares = data["train"][attr].value_counts(normalize=True)
        assert shares.min() >= 0.15, f"{attr} has an under-represented group: {shares.round(3).to_dict()}"
