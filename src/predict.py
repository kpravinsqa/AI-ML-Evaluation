"""
Inference service contract: validate an application, score it, return a decision with reason codes.
This mirrors what a scoring API exposes to the loan-origination system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import MODEL_INPUTS

APPROVE_BELOW = 0.20        # probability of default
REFER_BELOW = 0.40
SCHEMA = {
    "age": (int, 18, 100), "annual_income": (float, 0, 1e8), "loan_amount": (float, 10_000, 5_000_000),
    "loan_term_months": (int, 6, 120), "credit_history_years": (float, 0, 80), "debt_to_income": (float, 0, 5),
    "credit_utilisation": (float, 0, 1), "missed_payments_12m": (int, 0, 50), "employment_years": (float, 0, 60),
}
CATEGORIES = {"home_ownership": {"OWN", "MORTGAGE", "RENT"}}
REASONS = {
    "missed_payments_12m": "Recent missed payments", "debt_to_income": "High debt-to-income ratio",
    "credit_utilisation": "High credit utilisation", "credit_history_years": "Short credit history",
    "employment_years": "Short employment tenure", "annual_income": "Income relative to loan",
}


class ValidationError(ValueError):
    def __init__(self, errors: dict):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def validate(app: dict) -> dict:
    errors = {}
    clean = {}
    for field, (typ, lo, hi) in SCHEMA.items():
        if field not in app or app[field] is None:
            errors[field] = "is required"
            continue
        val = app[field]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errors[field] = "must be numeric"
            continue
        if typ is int and float(val) != int(val):
            errors[field] = "must be a whole number"
            continue
        if not lo <= val <= hi:
            errors[field] = f"must be between {lo:g} and {hi:g}"
            continue
        clean[field] = typ(val)
    for field, allowed in CATEGORIES.items():
        if app.get(field) not in allowed:
            errors[field] = f"must be one of {sorted(allowed)}"
        else:
            clean[field] = app[field]
    if errors:
        raise ValidationError(errors)
    for extra in ("gender", "collections_flag"):          # only present when a defective model expects them
        if extra in MODEL_INPUTS:
            clean[extra] = app.get(extra, "M" if extra == "gender" else 0)
    return clean


def score(model, app: dict) -> dict:
    clean = validate(app)
    row = pd.DataFrame([clean])[MODEL_INPUTS]
    pd_ = float(model.predict_proba(row)[0, 1])
    decision = "APPROVE" if pd_ < APPROVE_BELOW else "REFER" if pd_ < REFER_BELOW else "DECLINE"
    return {"probability_of_default": round(pd_, 6), "decision": decision,
            "reason_codes": reason_codes(clean) if decision != "APPROVE" else []}


def reason_codes(clean: dict, top: int = 3) -> list[str]:
    """Simple rule-based adverse-action reasons, ordered by how far each driver is from a healthy value."""
    risk = {
        "missed_payments_12m": clean["missed_payments_12m"] / 2,
        "debt_to_income": max(0.0, clean["debt_to_income"] - 0.35) / 0.2,
        "credit_utilisation": max(0.0, clean["credit_utilisation"] - 0.4) / 0.2,
        "credit_history_years": max(0.0, 5 - clean["credit_history_years"]) / 3,
        "employment_years": max(0.0, 2 - clean["employment_years"]) / 2,
        "annual_income": max(0.0, clean["loan_amount"] / max(clean["annual_income"], 1) - 1),
    }
    ranked = sorted((v, k) for k, v in risk.items() if v > 0)[::-1]
    return [REASONS[k] for _, k in ranked[:top]]


def score_batch(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[MODEL_INPUTS])[:, 1]
