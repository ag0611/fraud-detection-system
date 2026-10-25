import numpy as np
import pandas as pd

from src.features.preprocess import build_preprocessor
from src.models.metrics import recall_at_precision


def test_preprocessor_shape_and_no_raw_cols():
    df = pd.DataFrame({"Time": [0, 3600, 7200], "Amount": [1.0, 10.0, 100.0],
                       **{f"V{i}": [0.1, 0.2, 0.3] for i in range(1, 29)}})
    out = build_preprocessor().fit_transform(df)
    assert out.shape == (3, 28 + 3)  # V1-28 + hour_sin/cos + log_amount


def test_recall_at_precision_perfect_model():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    assert recall_at_precision(y, s, 0.8) == 1.0
