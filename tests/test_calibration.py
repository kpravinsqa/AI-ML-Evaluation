import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from data.generate_data import TARGET
from src.metrics import expected_calibration_error


@pytest.mark.calibration
@pytest.mark.smoke
def test_expected_calibration_error(scored_test, gates):
    ece = expected_calibration_error(scored_test[TARGET], scored_test["pd"])
    assert ece <= gates["calibration"]["ece_max"], f"ECE {ece:.3f}"


@pytest.mark.calibration
def test_brier_score(scored_test, gates):
    assert brier_score_loss(scored_test[TARGET], scored_test["pd"]) <= gates["calibration"]["brier_max"]


@pytest.mark.calibration
def test_mean_predicted_pd_matches_observed_rate(scored_test):
    assert abs(scored_test["pd"].mean() - scored_test[TARGET].mean()) <= 0.02


@pytest.mark.calibration
def test_risk_bands_are_monotonic(scored_test):
    """Observed default rate must rise across predicted-risk deciles (rank ordering)."""
    deciles = scored_test.assign(band=np.digitize(scored_test["pd"], np.quantile(scored_test["pd"], np.linspace(0.1, 0.9, 9))))
    rates = deciles.groupby("band")[TARGET].mean().values
    violations = sum(b < a - 0.03 for a, b in zip(rates, rates[1:]))
    assert violations <= 1, f"non-monotonic risk bands: {np.round(rates, 3)}"
