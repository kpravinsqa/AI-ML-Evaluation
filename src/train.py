"""Train the champion (logistic regression) and challenger (gradient boosting) models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from data.generate_data import TARGET, splits
from src.features import MODEL_INPUTS, preprocessor

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_VERSION = "credit-risk-pd-1.4.0"


def build(kind: str) -> Pipeline:
    if kind == "baseline":
        clf = LogisticRegression(max_iter=1000, C=1.0)
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=42)
    return Pipeline([("prep", preprocessor()), ("clf", clf)])


def train_all(out: Path = MODEL_DIR) -> dict:
    train, test = splits()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind in ("baseline", "challenger"):
        model = build(kind).fit(train[MODEL_INPUTS], train[TARGET])
        joblib.dump(model, out / f"{kind}.joblib")
        results[kind] = round(roc_auc_score(test[TARGET], model.predict_proba(test[MODEL_INPUTS])[:, 1]), 4)
    (out / "training_summary.json").write_text(json.dumps({"model_version": MODEL_VERSION, "roc_auc": results}, indent=2))
    return results


if __name__ == "__main__":
    print(train_all())
