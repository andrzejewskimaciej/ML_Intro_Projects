import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def load_set(path: str, with_FS=False) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df["default"]
    X = df.drop(columns="default")
    if with_FS:
        with open("models/selected_features.pkl", "rb") as f:
            selected = pickle.load(f)
        if not set(selected).issubset(set(X.columns)):
            raise ValueError("W ramce danych nie ma wyselekcjonowanych kolumn")
        X = X[selected]
    return X, y


def load_fitted_models():
    with open("models/LogisticRegression.pkl", "rb") as f:
        LR = pickle.load(f)
    with open("models/XGBoost.pkl", "rb") as f:
        XGB = pickle.load(f)
    return LR, XGB


def load_unfitted_models():
    with open("models/LR_params.pkl", "rb") as f:
        LR_params = pickle.load(f)
    with open("models/XGB_params.pkl", "rb") as f:
        XGB_params = pickle.load(f)

    LR = LogisticRegression(**LR_params)
    XGB = XGBClassifier(**XGB_params)

    return LR, XGB


def load_calibrated_models():
    with open("models/CalibratedLogisticRegression.pkl", "rb") as f:
        CLR = pickle.load(f)
    with open("models/CalibratedXGBoost.pkl", "rb") as f:
        CXGB = pickle.load(f)
    return CLR, CXGB
