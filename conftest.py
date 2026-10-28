import json
from pathlib import Path

import numpy as np
import pytest

from data.generate_data import TARGET, production_batches, splits
from src.features import MODEL_INPUTS
from src.train import train_all

ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def gates():
    return json.loads((ROOT / "config" / "quality_gates.json").read_text())


@pytest.fixture(scope="session")
def data():
    train, test = splits()
    return {"train": train, "test": test}


@pytest.fixture(scope="session")
def models(tmp_path_factory):
    import joblib

    out = tmp_path_factory.mktemp("models")
    train_all(out)
    return {k: joblib.load(out / f"{k}.joblib") for k in ("baseline", "challenger")}


@pytest.fixture(scope="session")
def champion(models):
    """The production model (champion) is the logistic-regression baseline."""
    return models["baseline"]


@pytest.fixture(scope="session")
def scored_test(data, champion, gates):
    test = data["test"].copy()
    test["pd"] = champion.predict_proba(test[MODEL_INPUTS])[:, 1]
    test["approved"] = (test["pd"] < gates["inference"]["decision_threshold_approve"]).astype(int)
    test["pred_default"] = (test["pd"] >= 0.5).astype(int)
    return test


@pytest.fixture(scope="session")
def prod_batches():
    stable, drifted = production_batches()
    return {"stable": stable, "drifted": drifted}


@pytest.fixture
def good_applicant():
    return {"age": 35, "annual_income": 1_500_000.0, "loan_amount": 300_000.0, "loan_term_months": 36,
            "credit_history_years": 12.0, "debt_to_income": 0.18, "credit_utilisation": 0.15,
            "missed_payments_12m": 0, "employment_years": 8.0, "home_ownership": "OWN"}


@pytest.fixture
def risky_applicant():
    return {"age": 24, "annual_income": 350_000.0, "loan_amount": 900_000.0, "loan_term_months": 60,
            "credit_history_years": 1.5, "debt_to_income": 0.75, "credit_utilisation": 0.9,
            "missed_payments_12m": 4, "employment_years": 0.5, "home_ownership": "RENT"}
