from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.generate_data import DEFECTS

NUMERIC = ["age", "annual_income", "loan_amount", "loan_term_months", "credit_history_years",
           "debt_to_income", "credit_utilisation", "missed_payments_12m", "employment_years"]
CATEGORICAL = ["home_ownership"]
if DEFECTS:
    NUMERIC = NUMERIC + ["collections_flag"]       # D2: leaky post-outcome feature used for training
    CATEGORICAL = CATEGORICAL + ["gender"]         # D3: protected attribute used as a feature
MODEL_INPUTS = NUMERIC + CATEGORICAL


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
