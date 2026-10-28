"""Production monitoring: data drift (PSI / KS) and prediction drift on recent scoring batches."""
import pytest

from src.features import MODEL_INPUTS
from src.metrics import feature_drift_report, psi

MONITORED = ["debt_to_income", "credit_utilisation", "missed_payments_12m", "annual_income", "loan_amount"]


@pytest.mark.drift
@pytest.mark.smoke
def test_stable_batch_has_no_feature_drift(data, prod_batches, gates):
    report = feature_drift_report(data["train"], prod_batches["stable"], MONITORED)
    assert (report["psi"] < gates["drift"]["psi_watch"]).all(), report.to_dict("records")


@pytest.mark.drift
def test_downturn_batch_is_flagged(data, prod_batches, gates):
    report = feature_drift_report(data["train"], prod_batches["drifted"], MONITORED)
    drifted = set(report.loc[report["psi"] >= gates["drift"]["psi_drift"], "feature"])
    assert {"debt_to_income", "credit_utilisation"} <= drifted, report.to_dict("records")


@pytest.mark.drift
def test_ks_test_detects_shift(data, prod_batches):
    stable = feature_drift_report(data["train"], prod_batches["stable"], ["debt_to_income"])["ks_pvalue"].iloc[0]
    drift = feature_drift_report(data["train"], prod_batches["drifted"], ["debt_to_income"])["ks_pvalue"].iloc[0]
    assert stable > 0.01 and drift < 0.001


@pytest.mark.drift
def test_prediction_drift(champion, data, prod_batches, gates):
    ref = champion.predict_proba(data["test"][MODEL_INPUTS])[:, 1]
    stable = champion.predict_proba(prod_batches["stable"][MODEL_INPUTS])[:, 1]
    drifted = champion.predict_proba(prod_batches["drifted"][MODEL_INPUTS])[:, 1]
    assert psi(ref, stable) < gates["drift"]["psi_watch"]
    assert psi(ref, drifted) >= gates["drift"]["psi_drift"]


@pytest.mark.drift
def test_psi_of_identical_distributions_is_zero(data):
    assert psi(data["train"]["debt_to_income"], data["train"]["debt_to_income"]) == pytest.approx(0, abs=1e-9)
