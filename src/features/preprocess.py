import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Time -> hour sin/cos, Amount -> log1p. Raw Time/Amount hata deta hai."""
    X = X.copy()
    hour = (X["Time"] // 3600) % 24
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["log_amount"] = np.log1p(X["Amount"])
    return X.drop(columns=["Time", "Amount"])


def build_preprocessor() -> Pipeline:
    """Scaler sirf train par fit hoga (pipeline ke through, leakage nahi)."""
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            (
                "scale",
                ColumnTransformer(
                    [("amount", RobustScaler(), ["log_amount"])],
                    remainder="passthrough",
                ),
            ),
        ]
    )
