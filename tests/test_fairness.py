"""Fairness checks across protected attributes (attributes are audited, never used as inputs)."""
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from data.generate_data import TARGET
from src.metrics import disparate_impact_ratio, equal_opportunity_difference


@pytest.mark.fairness
@pytest.mark.smoke
@pytest.mark.parametrize("attr", ["gender", "age_band"])
def test_disparate_impact_four_fifths_rule(scored_test, gates, attr):
    ratio = disparate_impact_ratio(scored_test["approved"].values, scored_test[attr].values)
    assert ratio >= gates["fairness"]["disparate_impact_min"], f"{attr}: DI ratio {ratio:.2f}"


@pytest.mark.fairness
@pytest.mark.parametrize("attr", ["gender"])
def test_equal_opportunity(scored_test, gates, attr):
    """Creditworthy applicants must be approved at similar rates regardless of group."""
    diff = equal_opportunity_difference(scored_test[TARGET].values, scored_test["approved"].values, scored_test[attr].values)
    assert diff <= gates["fairness"]["equal_opportunity_diff_max"], f"{attr}: EO difference {diff:.3f}"


@pytest.mark.fairness
def test_false_decline_gap_not_statistically_significant(scored_test):
    """Chi-square test on false declines among good borrowers: a gap that could be noise is acceptable,
    a statistically significant one (p < 0.01) is a fairness defect. A fixed % cut-off is flaky on ~600 rows per group."""
    good = scored_test[scored_test[TARGET] == 0]
    table = pd.crosstab(good["gender"], good["approved"])
    p_value = chi2_contingency(table)[1]
    assert p_value >= 0.01, f"significant false-decline gap by gender (p={p_value:.4f}): {table.to_dict()}"


@pytest.mark.fairness
def test_calibration_within_groups(scored_test):
    """Predicted PD should match observed default rate within each gender (calibration fairness)."""
    for g, grp in scored_test.groupby("gender"):
        assert abs(grp["pd"].mean() - grp[TARGET].mean()) <= 0.03, f"group {g} mis-calibrated"
