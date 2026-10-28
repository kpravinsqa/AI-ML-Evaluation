"""Evaluation helpers: fairness, calibration and drift statistics (no external fairness libs needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def selection_rate(decisions: np.ndarray) -> float:
    return float(np.mean(decisions))


def disparate_impact_ratio(approved: np.ndarray, group: np.ndarray) -> float:
    """min / max approval rate across groups (four-fifths rule: should be >= 0.8)."""
    rates = [approved[group == g].mean() for g in np.unique(group)]
    return float(min(rates) / max(rates))


def equal_opportunity_difference(y_true, y_pred_approve, group) -> float:
    """Max gap in approval rate among *good* borrowers (true non-defaulters) across groups."""
    good = y_true == 0
    rates = [y_pred_approve[good & (group == g)].mean() for g in np.unique(group)]
    return float(max(rates) - min(rates))


def expected_calibration_error(y_true, y_prob, bins: int = 10) -> float:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.any():
            ece += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using quantile bins of the expected (training) distribution."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_pvalue(expected, actual) -> float:
    return float(stats.ks_2samp(expected, actual).pvalue)


def feature_drift_report(train: pd.DataFrame, prod: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "psi": round(psi(train[f], prod[f]), 4), "ks_pvalue": ks_pvalue(train[f], prod[f])} for f in features]
    df = pd.DataFrame(rows)
    df["status"] = np.select([df["psi"] >= 0.25, df["psi"] >= 0.10], ["DRIFT", "WATCH"], "STABLE")
    return df


def promotion_decision(champion_auc: float, challenger_auc: float, min_uplift: float = 0.005) -> str:
    """Champion/challenger rule: promote only on a meaningful AUC uplift."""
    return "PROMOTE_CHALLENGER" if challenger_auc - champion_auc >= min_uplift else "KEEP_CHAMPION"
