"""Model quality gates on the hold-out set, per segment, and champion vs challenger."""
import pytest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from data.generate_data import TARGET
from src.features import MODEL_INPUTS
from src.metrics import promotion_decision


@pytest.mark.performance
@pytest.mark.smoke
def test_roc_auc_meets_gate(scored_test, gates):
    auc = roc_auc_score(scored_test[TARGET], scored_test["pd"])
    assert auc >= gates["performance"]["roc_auc_min"], f"AUC {auc:.3f}"


@pytest.mark.performance
def test_recall_and_precision_for_defaulters(scored_test, gates):
    y, yhat = scored_test[TARGET], (scored_test["pd"] >= 0.3).astype(int)
    assert recall_score(y, yhat) >= gates["performance"]["recall_default_min"]
    assert precision_score(y, yhat) >= gates["performance"]["precision_default_min"]


@pytest.mark.performance
def test_beats_naive_baselines(scored_test):
    """Must beat always-predict-majority (accuracy) and random ranking (AUC 0.5) by a clear margin."""
    y = scored_test[TARGET]
    majority_acc = 1 - y.mean()
    model_acc = (scored_test["pred_default"] == y).mean()
    assert model_acc > majority_acc
    assert roc_auc_score(y, scored_test["pd"]) > 0.65


@pytest.mark.performance
def test_business_cost_lower_than_approve_all(scored_test):
    """Cost model: a missed defaulter costs 5x a wrongly declined good customer."""
    y = scored_test[TARGET]
    declined = 1 - scored_test["approved"]
    tn, fp, fn, tp = confusion_matrix(y, declined).ravel()
    model_cost = 5 * fn + 1 * fp
    approve_all_cost = 5 * y.sum()
    assert model_cost < 0.7 * approve_all_cost


@pytest.mark.performance
@pytest.mark.parametrize("segment", ["age_band", "home_ownership"])
def test_auc_holds_in_every_segment(scored_test, gates, segment):
    floor = gates["segment"]["roc_auc_min_per_segment"]
    for value, grp in scored_test.groupby(segment):
        auc = roc_auc_score(grp[TARGET], grp["pd"])
        assert auc >= floor, f"{segment}={value}: AUC {auc:.3f} < {floor}"


@pytest.mark.performance
def test_champion_challenger_decision(models, data, gates):
    test = data["test"]
    champ = roc_auc_score(test[TARGET], models["baseline"].predict_proba(test[MODEL_INPUTS])[:, 1])
    chall = roc_auc_score(test[TARGET], models["challenger"].predict_proba(test[MODEL_INPUTS])[:, 1])
    assert chall - champ >= gates["performance"]["challenger_min_auc_uplift"], "challenger is materially worse"
    # On this data the simpler model is as good: the challenger must NOT be promoted.
    assert promotion_decision(champ, chall) == "KEEP_CHAMPION"


@pytest.mark.performance
def test_f1_is_reported_and_reasonable(scored_test):
    assert f1_score(scored_test[TARGET], (scored_test["pd"] >= 0.3).astype(int)) >= 0.5
