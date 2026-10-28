"""
Synthetic loan-application data for a credit-risk (probability of default) model.

The ground-truth default probability depends only on financial behaviour
(credit history, debt-to-income, utilisation, income, employment, missed payments).
Gender and age band are recorded for fairness auditing but have NO causal effect.

With INJECT_DEFECTS=1 the generator reproduces two classic real-world failures:
  * historical label bias against one group (women labelled "default" more often)
  * a leaky feature (`collections_flag`) that is only known AFTER the default happens
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEFECTS = os.getenv("INJECT_DEFECTS") == "1"
FEATURES = ["age", "annual_income", "loan_amount", "loan_term_months", "credit_history_years",
            "debt_to_income", "credit_utilisation", "missed_payments_12m", "employment_years", "home_ownership"]
SENSITIVE = ["gender", "age_band"]
TARGET = "defaulted"


def _true_pd(df: pd.DataFrame) -> np.ndarray:
    z = (-3.4
         + 5.0 * (df["debt_to_income"] - 0.35)
         + 3.5 * (df["credit_utilisation"] - 0.4)
         + 0.95 * df["missed_payments_12m"]
         - 0.10 * (df["credit_history_years"] - 8)
         - 1.2 * np.log(df["annual_income"] / 900_000)
         - 0.08 * (df["employment_years"] - 5)
         + 0.35 * (df["home_ownership"] == "RENT")
         + 0.6 * (df["loan_amount"] / df["annual_income"]))
    return 1 / (1 + np.exp(-z))


def generate(n: int = 6000, seed: int = 42, shift: float = 0.0) -> pd.DataFrame:
    """`shift` > 0 simulates a population drift (e.g. a downturn: higher DTI and utilisation)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, n)
    df = pd.DataFrame({
        "application_id": [f"APP{seed:03d}{i:06d}" for i in range(n)],
        "age": age,
        "annual_income": np.round(rng.lognormal(13.6, 0.45, n), -3),          # ~INR 8 lakh median
        "loan_amount": np.round(rng.uniform(50_000, 1_500_000, n), -3),
        "loan_term_months": rng.choice([12, 24, 36, 48, 60], n),
        "credit_history_years": np.clip(np.round(rng.normal(8, 4, n), 1), 0, 40),
        "debt_to_income": np.clip(rng.beta(3, 5, n) + shift * 0.25, 0, 1.5).round(3),
        "credit_utilisation": np.clip(rng.beta(2, 3, n) + shift * 0.3, 0, 1).round(3),
        "missed_payments_12m": rng.poisson(0.6 + shift * 1.2, n),
        "employment_years": np.clip(np.round(rng.exponential(5, n), 1), 0, 40),
        "home_ownership": rng.choice(["OWN", "MORTGAGE", "RENT"], n, p=[0.3, 0.3, 0.4]),
        "gender": rng.choice(["F", "M"], n),
    })
    df["age_band"] = pd.cut(df["age"], [0, 30, 45, 120], labels=["21-30", "31-45", "46+"]).astype(str)
    p = _true_pd(df)
    if DEFECTS:
        p = np.where(df["gender"] == "F", np.clip(p * 3.0, 0, 1), p)          # D1 historical label bias
    df[TARGET] = (rng.random(n) < p).astype(int)
    if DEFECTS:
        df["collections_flag"] = ((df[TARGET] == 1) & (rng.random(n) < 0.9)).astype(int)   # D2 leakage
    return df


def splits(seed: int = 42):
    df = generate(6000, seed)
    test = df.sample(frac=0.25, random_state=seed)
    train = df.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def production_batches():
    """A stable recent batch and a drifted (downturn) batch for monitoring tests."""
    return generate(2000, seed=7, shift=0.0), generate(2000, seed=8, shift=1.0)
